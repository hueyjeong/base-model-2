"""SentencePiece BBPE + MeCab 토크나이저 학습 스크립트

사용법 (plain text):
    python -m mecab_bbpe_tokenizer.train_tokenizer \
        --input /path/to/corpus.txt \
        --model_prefix mecab_bbpe \
        --vocab_size 64000

사용법 (JSONL):
    python -m mecab_bbpe_tokenizer.train_tokenizer \
        --input /path/to/corpus.jsonl \
        --text_key text \
        --model_prefix mecab_bbpe \
        --vocab_size 64000

지원 형식:
  - .txt: 한 줄에 한 문장
  - .jsonl: 각 줄이 JSON 객체, --text_key로 텍스트 필드 지정

MeCab을 사전 분절기로 사용하여 형태소 단위로 분리한 뒤
SentencePiece BBPE로 서브워드 학습을 진행한다.
"""
import argparse
import json
import os
import sys
import tempfile
from typing import Optional

import MeCab
import sentencepiece as spm


def _create_mecab_tagger() -> MeCab.Tagger:
    """MeCab Tagger 생성 — pip 사전 자동 감지 (mecab-ko-dic 우선)"""
    # 1순위: mecab-ko-dic (한국어 사전)
    try:
        import mecab_ko_dic
        dicdir = mecab_ko_dic.DICDIR
        return MeCab.Tagger(f"-O wakati -r /dev/null -d {dicdir}")
    except (ImportError, AttributeError):
        pass

    # 2순위: unidic-lite (일본어 사전, 펴백)
    try:
        import unidic_lite
        dicdir = unidic_lite.DICDIR
        return MeCab.Tagger(f"-O wakati -r /dev/null -d {dicdir}")
    except (ImportError, AttributeError):
        pass

    # 3순위: 시스템 MeCab 사전
    try:
        return MeCab.Tagger("-O wakati")
    except RuntimeError:
        pass

    # 4순위: mecabrc 없이 시도
    try:
        return MeCab.Tagger("-O wakati -r /dev/null")
    except RuntimeError as e:
        raise RuntimeError(
            "MeCab 사전을 찾을 수 없습니다. 다음 중 하나를 설치해 주세요:\n"
            "  pip install mecab-ko-dic  (한국어 사전)\n"
            "  pip install unidic-lite   (일본어 사전)"
        ) from e


def _detect_text_key(jsonl_path: str) -> str:
    """JSONL 파일의 첫 번째 줄에서 텍스트 필드 키를 자동 감지

    우선순위: text > content > sentence > document > body
    """
    candidates = ["text", "content", "sentence", "document", "body"]
    with open(jsonl_path, "r", encoding="utf-8") as f:
        first_line = f.readline().strip()
        if not first_line:
            raise ValueError(f"JSONL 파일이 비어 있습니다: {jsonl_path}")
        obj = json.loads(first_line)
        for key in candidates:
            if key in obj:
                return key
        # 후보에 없으면 문자열 값을 가진 첫 번째 키 사용
        for key, val in obj.items():
            if isinstance(val, str) and len(val) > 10:
                return key
        raise ValueError(
            f"텍스트 필드를 찾을 수 없습니다. "
            f"사용 가능한 키: {list(obj.keys())}. "
            f"--text_key 옵션으로 지정해 주세요."
        )


def jsonl_to_txt(
    jsonl_path: str,
    output_path: str,
    text_key: str,
    max_lines: Optional[int] = None,
) -> int:
    """JSONL 파일에서 텍스트를 추출하여 plain text 파일로 변환

    Args:
        jsonl_path: 입력 JSONL 파일 경로
        output_path: 출력 텍스트 파일 경로
        text_key: JSON 객체에서 텍스트를 담고 있는 키 이름
        max_lines: 최대 처리 줄 수 (None이면 전부)

    Returns:
        처리된 줄 수
    """
    count = 0
    errors = 0
    with open(jsonl_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:
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
                    # 줄바꿈을 공백으로 치환 (SentencePiece는 한 줄 = 한 문장)
                    text = text.replace("\n", " ").replace("\r", " ").strip()
                    fout.write(text + "\n")
                    count += 1
            except (json.JSONDecodeError, KeyError):
                errors += 1
                if errors <= 5:
                    print(f"[WARN] line {line_no}: JSON 파싱 실패, 건너뜀",
                          file=sys.stderr)

            if count % 5_000_000 == 0 and count > 0:
                print(f"[INFO] JSONL→TXT 변환 중... {count:,}줄 처리됨")

    if errors > 5:
        print(f"[WARN] 총 {errors}줄 파싱 실패", file=sys.stderr)

    return count


def mecab_pretokenize(input_path: str, output_path: str) -> None:
    """MeCab으로 형태소 분절하여 공백 구분 텍스트로 변환

    Args:
        input_path: 원본 코퍼스 파일 경로 (한 줄에 한 문장)
        output_path: MeCab 분절 결과 저장 경로
    """
    tagger = _create_mecab_tagger()
    count = 0

    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            # MeCab wakati: 형태소를 공백으로 분리
            segmented = tagger.parse(line).strip()
            if segmented:
                fout.write(segmented + "\n")
                count += 1

            if count % 5_000_000 == 0 and count > 0:
                print(f"[INFO] MeCab 분절 중... {count:,}줄 처리됨")

    print(f"[INFO] MeCab 분절 완료: {count:,}줄")


def train_mecab_bbpe_tokenizer(
    input_path: str,
    model_prefix: str,
    vocab_size: int = 64000,
    character_coverage: float = 0.9995,
    use_mecab: bool = True,
    num_threads: int = 4,
    text_key: Optional[str] = None,
    max_lines: Optional[int] = None,
) -> str:
    """SentencePiece BBPE 토크나이저 학습

    Args:
        input_path: 학습 코퍼스 파일 경로 (.txt 또는 .jsonl)
        model_prefix: 모델 저장 경로 prefix
        vocab_size: 어휘 크기
        character_coverage: 문자 커버리지
        use_mecab: MeCab 사전 분절 사용 여부
        num_threads: 학습 스레드 수
        text_key: JSONL에서 텍스트 필드 키 (None이면 자동 감지)
        max_lines: 최대 처리 줄 수

    Returns:
        학습된 모델 파일 경로 (.model)
    """
    tmp_files = []  # 정리할 임시 파일 목록
    actual_input = input_path

    # --- 1단계: JSONL → TXT 변환 (필요 시) ---
    is_jsonl = input_path.endswith(".jsonl") or input_path.endswith(".json")
    if is_jsonl:
        if text_key is None:
            text_key = _detect_text_key(input_path)
            print(f"[INFO] 자동 감지된 텍스트 키: '{text_key}'")

        print(f"[INFO] JSONL → TXT 변환 시작 (text_key='{text_key}')...")
        txt_tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8",
            dir=os.path.dirname(input_path) or ".",  # 같은 디렉토리에 임시 파일
        )
        txt_tmp.close()
        n_lines = jsonl_to_txt(input_path, txt_tmp.name, text_key, max_lines)
        actual_input = txt_tmp.name
        tmp_files.append(txt_tmp.name)
        print(f"[INFO] JSONL → TXT 변환 완료: {n_lines:,}줄 → {txt_tmp.name}")

    # --- 2단계: MeCab 사전 분절 (필요 시) ---
    if use_mecab:
        print("[INFO] MeCab 사전 분절 시작...")
        mecab_tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8",
            dir=os.path.dirname(actual_input) or ".",
        )
        mecab_tmp.close()
        mecab_pretokenize(actual_input, mecab_tmp.name)
        actual_input = mecab_tmp.name
        tmp_files.append(mecab_tmp.name)

    # --- 3단계: SentencePiece 학습 ---
    # 파일 크기로 대용량 코퍼스 여부 판별
    file_size_gb = os.path.getsize(actual_input) / (1024 ** 3)
    is_large = file_size_gb > 1.0
    print(f"[INFO] 학습 입력: {actual_input} ({file_size_gb:.2f} GB)")
    print(f"[INFO] SentencePiece BBPE 학습 시작 (vocab_size={vocab_size}, "
          f"large_corpus={is_large})...")

    spm.SentencePieceTrainer.train(
        input=actual_input,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type="bpe",
        character_coverage=character_coverage,
        byte_fallback=True,  # BBPE: 바이트 폴백 활성화
        num_threads=num_threads,
        # Special tokens
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        pad_piece="[PAD]",
        unk_piece="[UNK]",
        bos_piece="[BOS]",
        eos_piece="[EOS]",
        # 사전 분절된 입력이므로 SentencePiece의 자체 전처리 최소화
        normalization_rule_name="identity" if use_mecab else "nmt_nfkc",
        # 대용량 코퍼스 지원
        train_extremely_large_corpus=is_large,
        input_sentence_size=5_000_000 if is_large else 0,   # 샘플링
        shuffle_input_sentence=True,
        max_sentence_length=8192,
        # user_defined_symbols로 추가 special token 정의 가능
        user_defined_symbols=["[SEP]", "[MASK]"],
    )

    model_path = f"{model_prefix}.model"
    print(f"[INFO] 토크나이저 학습 완료: {model_path}")

    # --- 정리: 임시 파일 삭제 ---
    for tmp in tmp_files:
        try:
            os.unlink(tmp)
            print(f"[INFO] 임시 파일 삭제: {tmp}")
        except OSError:
            pass

    return model_path


def main():
    parser = argparse.ArgumentParser(
        description="SentencePiece BBPE + MeCab 토크나이저 학습",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  # plain text 코퍼스
  python -m mecab_bbpe_tokenizer.train_tokenizer -i corpus.txt -m mecab_bbpe

  # JSONL 코퍼스 (텍스트 키 자동 감지)
  python -m mecab_bbpe_tokenizer.train_tokenizer -i corpus.jsonl -m mecab_bbpe

  # JSONL 코퍼스 (텍스트 키 지정)
  python -m mecab_bbpe_tokenizer.train_tokenizer -i corpus.jsonl --text_key content -m mecab_bbpe
        """,
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="학습 코퍼스 파일 경로 (.txt 또는 .jsonl)",
    )
    parser.add_argument(
        "--model_prefix", "-m",
        default="mecab_bbpe",
        help="모델 저장 경로 prefix (default: mecab_bbpe)",
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
        "--character_coverage",
        type=float,
        default=0.9995,
        help="문자 커버리지 (default: 0.9995)",
    )
    parser.add_argument(
        "--no_mecab",
        action="store_true",
        help="MeCab 사전 분절 비활성화",
    )
    parser.add_argument(
        "--num_threads",
        type=int,
        default=4,
        help="학습 스레드 수 (default: 4)",
    )
    parser.add_argument(
        "--max_lines",
        type=int,
        default=None,
        help="JSONL에서 최대 처리 줄 수 (default: 전부)",
    )

    args = parser.parse_args()

    train_mecab_bbpe_tokenizer(
        input_path=args.input,
        model_prefix=args.model_prefix,
        vocab_size=args.vocab_size,
        character_coverage=args.character_coverage,
        use_mecab=not args.no_mecab,
        num_threads=args.num_threads,
        text_key=args.text_key,
        max_lines=args.max_lines,
    )


if __name__ == "__main__":
    main()
