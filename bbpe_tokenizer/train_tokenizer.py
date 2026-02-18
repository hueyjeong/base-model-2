"""SentencePiece BBPE + MeCab 토크나이저 학습 스크립트

사용법:
    python -m bbpe_tokenizer.train_tokenizer \
        --input /path/to/corpus.txt \
        --model_prefix bbpe_ko \
        --vocab_size 64000

MeCab을 사전 분절기로 사용하여 형태소 단위로 분리한 뒤
SentencePiece BBPE로 서브워드 학습을 진행한다.
"""
import argparse
import os
import sys
import tempfile
from typing import Optional

import MeCab
import sentencepiece as spm


def mecab_pretokenize(input_path: str, output_path: str) -> None:
    """MeCab으로 형태소 분절하여 공백 구분 텍스트로 변환

    Args:
        input_path: 원본 코퍼스 파일 경로 (한 줄에 한 문장)
        output_path: MeCab 분절 결과 저장 경로
    """
    tagger = MeCab.Tagger("-O wakati")  # 분리된 형태소를 공백으로 연결

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


def train_bbpe_tokenizer(
    input_path: str,
    model_prefix: str,
    vocab_size: int = 64000,
    character_coverage: float = 0.9995,
    use_mecab: bool = True,
    num_threads: int = 4,
) -> str:
    """SentencePiece BBPE 토크나이저 학습

    Args:
        input_path: 학습 코퍼스 파일 경로
        model_prefix: 모델 저장 경로 prefix
        vocab_size: 어휘 크기
        character_coverage: 문자 커버리지
        use_mecab: MeCab 사전 분절 사용 여부
        num_threads: 학습 스레드 수

    Returns:
        학습된 모델 파일 경로 (.model)
    """
    actual_input = input_path

    # MeCab 사전 분절
    if use_mecab:
        print("[INFO] MeCab 사전 분절 중...")
        mecab_output = tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        )
        mecab_output.close()
        mecab_pretokenize(input_path, mecab_output.name)
        actual_input = mecab_output.name
        print(f"[INFO] MeCab 분절 완료: {mecab_output.name}")

    print(f"[INFO] SentencePiece BBPE 학습 시작 (vocab_size={vocab_size})...")

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
        # 학습 안정성
        train_extremely_large_corpus=False,
        max_sentence_length=8192,
        shuffle_input_sentence=True,
        # user_defined_symbols로 추가 special token 정의 가능
        user_defined_symbols=["[SEP]", "[MASK]"],
    )

    model_path = f"{model_prefix}.model"
    print(f"[INFO] 토크나이저 학습 완료: {model_path}")

    # MeCab 임시 파일 정리
    if use_mecab:
        os.unlink(mecab_output.name)

    return model_path


def main():
    parser = argparse.ArgumentParser(
        description="SentencePiece BBPE + MeCab 토크나이저 학습"
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="학습 코퍼스 파일 경로 (한 줄에 한 문장)",
    )
    parser.add_argument(
        "--model_prefix", "-m",
        default="bbpe_ko",
        help="모델 저장 경로 prefix (default: bbpe_ko)",
    )
    parser.add_argument(
        "--vocab_size", "-v",
        type=int,
        default=64000,
        help="어휘 크기 (default: 64000)",
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

    args = parser.parse_args()

    train_bbpe_tokenizer(
        input_path=args.input,
        model_prefix=args.model_prefix,
        vocab_size=args.vocab_size,
        character_coverage=args.character_coverage,
        use_mecab=not args.no_mecab,
        num_threads=args.num_threads,
    )


if __name__ == "__main__":
    main()
