"""한글 글자 단위 토크나이저 래퍼 — BaseTokenizer 인터페이스 구현

char_tokenizer/make_tokenizer.py로 생성된 JSON 토크나이저를 로드하여
공통 인터페이스를 제공한다.
"""
import os
import sys
from typing import List

from tokenizers import Tokenizer

# 프로젝트 루트를 path에 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tokenizer_base import BaseTokenizer


class CharTokenizer(BaseTokenizer):
    """한글 글자 단위 토크나이저 래퍼 (ByteLevel BPE 기반)

    한글 완성형, 영문, 가나, CJK 한자를 글자 단위로 토큰화하며
    미등록 문자는 바이트 폴백으로 처리한다.
    """

    def __init__(self, model_path: str):
        """
        Args:
            model_path: make_tokenizer.py로 생성된 .json 토크나이저 파일
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"토크나이저 파일을 찾을 수 없습니다: {model_path}")

        self._tokenizer = Tokenizer.from_file(model_path)

        # Special token IDs
        self._pad_id = self._tokenizer.token_to_id("[PAD]")
        self._unk_id = self._tokenizer.token_to_id("[UNK]")
        self._bos_id = self._tokenizer.token_to_id("[BOS]")
        self._eos_id = self._tokenizer.token_to_id("[EOS]")

    @property
    def vocab_size(self) -> int:
        return self._tokenizer.get_vocab_size()

    @property
    def pad_id(self) -> int:
        return self._pad_id

    @property
    def bos_id(self) -> int:
        return self._bos_id

    @property
    def eos_id(self) -> int:
        return self._eos_id

    @property
    def unk_id(self) -> int:
        return self._unk_id

    def encode(self, text: str, add_special: bool = True) -> List[int]:
        """텍스트 → 토큰 ID 리스트

        Args:
            text: 입력 텍스트
            add_special: True면 BOS/EOS 추가
        Returns:
            토큰 ID 리스트
        """
        encoded = self._tokenizer.encode(text)
        ids = encoded.ids
        if add_special:
            ids = [self._bos_id] + ids + [self._eos_id]
        return ids

    def decode(self, ids: List[int], skip_special: bool = True) -> str:
        """토큰 ID 리스트 → 텍스트

        Args:
            ids: 토큰 ID 리스트
            skip_special: True면 special token 제거 후 디코딩
        Returns:
            디코딩된 텍스트
        """
        if skip_special:
            special = {self._pad_id, self._bos_id, self._eos_id, self._unk_id}
            ids = [i for i in ids if i not in special]
        return self._tokenizer.decode(ids, skip_special_tokens=skip_special)

    def encode_batch(self, texts: List[str], add_special: bool = True) -> List[List[int]]:
        """배치 인코딩"""
        encodings = self._tokenizer.encode_batch(texts)
        all_ids = [enc.ids for enc in encodings]
        if add_special:
            all_ids = [[self._bos_id] + ids + [self._eos_id] for ids in all_ids]
        return all_ids

    def id_to_piece(self, id: int) -> str:
        """토큰 ID → 토큰 문자열"""
        return self._tokenizer.id_to_token(id)

    def piece_to_id(self, piece: str) -> int:
        """토큰 문자열 → 토큰 ID"""
        return self._tokenizer.token_to_id(piece)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="한글 글자 단위 토크나이저 테스트")
    parser.add_argument("--model", "-m", required=True,
                        help="토크나이저 .json 파일 경로")
    parser.add_argument("--test", action="store_true", help="테스트 실행")
    args = parser.parse_args()

    tok = CharTokenizer(args.model)
    print(f"Vocab size: {tok.vocab_size}")
    print(f"PAD={tok.pad_id}, BOS={tok.bos_id}, "
          f"EOS={tok.eos_id}, UNK={tok.unk_id}")

    if args.test:
        test_sentences = [
            "맞춤법을 확인해 주세요.",
            "Hello, world!",
            "한글과 English 혼합 테스트",
            "こんにちは世界",
            "大韓民國",
        ]
        print("\n--- 인코드/디코드 라운드트립 ---")
        for sent in test_sentences:
            ids = tok.encode(sent, add_special=False)
            decoded = tok.decode(ids, skip_special=False)
            match = sent == decoded
            print(f"  원문: {sent}")
            print(f"  토큰수: {len(ids)}")
            print(f"  복원: {decoded}")
            print(f"  일치: {'✓' if match else '✗'}")
            print()
