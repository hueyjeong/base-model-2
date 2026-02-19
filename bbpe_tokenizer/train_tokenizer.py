"""BBPE 토크나이저 학습 스크립트 (HuggingFace tokenizers 라이브러리)

사용법 (plain text):
    python -m bbpe_tokenizer.train_tokenizer \\
        --input /path/to/corpus.txt \\
        --vocab_size 64000

사용법 (JSONL):
    python -m bbpe_tokenizer.train_tokenizer \\
        --input /path/to/corpus.jsonl \\
        --text_key text \\
        --vocab_size 64000

지원 형식:
  - .txt: 한 줄에 한 문장
  - .jsonl: 각 줄이 JSON 객체, --text_key로 텍스트 필드 지정

ByteLevel BPE (BBPE) 방식으로 서브워드 학습을 진행한다.
출력: HuggingFace tokenizers JSON (다른 토크나이저와 동일 형식)
"""
import argparse
import json
import os
import sys
from typing import Optional, Iterator

from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, AddedToken


# 이 스크립트의 디렉토리 (bbpe_tokenizer/)
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 공통 special tokens 목록
SPECIAL_TOKENS = [
    "[PAD]", "[UNK]", "[BOS]", "[EOS]", "[SEP]", "[CLS]", "[MASK]",
] + [f"[UNUSED{i}]" for i in range(10)]


def _detect_text_key(jsonl_path: str) -> str:
    """JSONL 파일의 첫 번째 줄에서 텍스트 필드 키를 자동 감지"""
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            candidates = ["text", "content", "sentence", "document", "body"]
            for key in candidates:
                if key in obj:
                    return key
            for key, val in obj.items():
                if isinstance(val, str) and len(val) > 10:
                    return key
            raise ValueError(
                f"텍스트 필드를 찾을 수 없습니다. "
                f"사용 가능한 키: {list(obj.keys())}. "
                f"--text_key 옵션으로 지정해 주세요."
            )
    raise ValueError("빈 JSONL 파일입니다.")


def jsonl_iter(
    jsonl_path: str,
    text_key: str,
    max_lines: Optional[int] = None,
) -> Iterator[str]:
    """JSONL 파일에서 텍스트를 스트리밍으로 yield

    Args:
        jsonl_path: 입력 JSONL 파일 경로
        text_key: JSON 객체에서 텍스트를 담고 있는 키 이름
        max_lines: 최대 처리 줄 수 (None이면 전부)

    Yields:
        정제된 텍스트 문장
    """
    count = 0
    errors = 0
    with open(jsonl_path, "r", encoding="utf-8") as fin:
        for line_no, line in enumerate(fin, 1):
            if max_lines and count >= max_lines:
                break
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                text = obj.get(text_key, "")
                if isinstance(text, str) and text.strip():
                    text = text.replace("\n", " ").replace("\r", " ").strip()
                    yield text
                    count += 1
            except (json.JSONDecodeError, KeyError):
                errors += 1
                if errors <= 5:
                    print(f"[WARN] line {line_no}: JSON 파싱 실패, 건너뜀",
                          file=sys.stderr)

            if count % 5_000_000 == 0 and count > 0:
                print(f"[INFO] JSONL 읽는 중... {count:,}줄 처리됨")

    if errors > 5:
        print(f"[WARN] 총 {errors}줄 파싱 실패", file=sys.stderr)

    print(f"[INFO] JSONL 읽기 완료: {count:,}줄")


def txt_iter(txt_path: str) -> Iterator[str]:
    """TXT 파일에서 한 줄씩 yield"""
    count = 0
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield line
                count += 1
                if count % 5_000_000 == 0:
                    print(f"[INFO] TXT 읽는 중... {count:,}줄 처리됨")
    print(f"[INFO] TXT 읽기 완료: {count:,}줄")


def train_bbpe_tokenizer(
    input_path: str,
    output_path: str,
    vocab_size: int = 64000,
    min_frequency: int = 2,
    text_key: Optional[str] = None,
    max_lines: Optional[int] = None,
) -> str:
    """HuggingFace ByteLevel BPE 토크나이저 학습

    Args:
        input_path: 학습 코퍼스 파일 경로 (.txt 또는 .jsonl)
        output_path: 출력 JSON 파일 경로
        vocab_size: 어휘 크기
        min_frequency: 최소 출현 빈도
        text_key: JSONL에서 텍스트 필드 키 (None이면 자동 감지)
        max_lines: 최대 처리 줄 수

    Returns:
        저장된 토크나이저 JSON 파일 경로
    """
    # --- 토크나이저 구성 ---
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    # --- 트레이너 설정 ---
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=SPECIAL_TOKENS,
        show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )

    # --- 코퍼스 읽기 & 학습 ---
    is_jsonl = input_path.endswith(".jsonl") or input_path.endswith(".json")

    if is_jsonl:
        if text_key is None:
            text_key = _detect_text_key(input_path)
            print(f"[INFO] 자동 감지된 텍스트 키: '{text_key}'")
        print(f"[INFO] JSONL 스트리밍 학습 시작 (text_key='{text_key}')...")
        iterator = jsonl_iter(input_path, text_key, max_lines)
    else:
        print(f"[INFO] TXT 학습 시작...")
        iterator = txt_iter(input_path)

    print(f"[INFO] BPE 학습 시작 (vocab_size={vocab_size})...")
    tokenizer.train_from_iterator(iterator, trainer=trainer)

    # --- special tokens를 added_tokens로 등록 ---
    tokenizer.add_special_tokens([
        AddedToken(tok, special=True) for tok in SPECIAL_TOKENS
    ])

    # --- 저장 ---
    tokenizer.save(output_path)
    vocab_size_actual = tokenizer.get_vocab_size()
    print(f"[INFO] 토크나이저 학습 완료: {output_path}")
    print(f"[INFO] Vocab size: {vocab_size_actual:,}")

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="BBPE 토크나이저 학습 (HuggingFace tokenizers)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  # plain text 코퍼스
  python -m bbpe_tokenizer.train_tokenizer -i corpus.txt

  # JSONL 코퍼스 (텍스트 키 자동 감지)
  python -m bbpe_tokenizer.train_tokenizer -i corpus.jsonl

  # JSONL 코퍼스 (텍스트 키 지정)
  python -m bbpe_tokenizer.train_tokenizer -i corpus.jsonl --text_key content
        """,
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="학습 코퍼스 파일 경로 (.txt 또는 .jsonl)",
    )
    parser.add_argument(
        "--output", "-o",
        default=os.path.join(_SCRIPT_DIR, "bbpe.json"),
        help="출력 토크나이저 JSON 경로 (default: bbpe_tokenizer/bbpe.json)",
    )
    parser.add_argument(
        "--vocab_size", "-v",
        type=int,
        default=64000,
        help="어휘 크기 (default: 64000)",
    )
    parser.add_argument(
        "--text_key",
        type=str,
        default=None,
        help="JSONL에서 텍스트를 담고 있는 키 (default: 자동 감지)",
    )
    parser.add_argument(
        "--min_frequency",
        type=int,
        default=2,
        help="최소 출현 빈도 (default: 2)",
    )
    parser.add_argument(
        "--max_lines",
        type=int,
        default=None,
        help="JSONL에서 최대 처리 줄 수 (default: 전부)",
    )

    args = parser.parse_args()

    train_bbpe_tokenizer(
        input_path=args.input,
        output_path=args.output,
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        text_key=args.text_key,
        max_lines=args.max_lines,
    )


if __name__ == "__main__":
    main()
