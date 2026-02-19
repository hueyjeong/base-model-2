"""BBPE 토크나이저 래퍼 — BaseTokenizer 인터페이스 구현

HuggingFace tokenizers JSON을 로드하여 공통 인터페이스를 제공한다.
"""
import os
import sys
from typing import List

from tokenizers import Tokenizer

# 프로젝트 루트를 path에 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tokenizer_base import BaseTokenizer


class BBPETokenizer(BaseTokenizer):
    """ByteLevel BPE 토크나이저 래퍼

    HuggingFace tokenizers JSON을 로드하여 사용한다.
    다른 토크나이저(nfd, keyboard, char)와 동일한 형식.
    """

    def __init__(self, json_path: str):
        """
        Args:
            json_path: HuggingFace tokenizers JSON 파일 경로
        """
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"토크나이저 파일을 찾을 수 없습니다: {json_path}")

        self._tok = Tokenizer.from_file(json_path)

        # Special token IDs
        self._pad_id = self._tok.token_to_id("[PAD]")
        self._unk_id = self._tok.token_to_id("[UNK]")
        self._bos_id = self._tok.token_to_id("[BOS]")
        self._eos_id = self._tok.token_to_id("[EOS]")
        self._sep_id = self._tok.token_to_id("[SEP]")
        self._cls_id = self._tok.token_to_id("[CLS]")
        self._mask_id = self._tok.token_to_id("[MASK]")

    @property
    def vocab_size(self) -> int:
        return self._tok.get_vocab_size()

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

    @property
    def sep_id(self) -> int:
        return self._sep_id

    @property
    def cls_id(self) -> int:
        return self._cls_id

    @property
    def mask_id(self) -> int:
        return self._mask_id

    def encode(self, text: str, add_special: bool = True) -> List[int]:
        """텍스트 → 토큰 ID 리스트

        Args:
            text: 입력 텍스트
            add_special: True면 BOS/EOS 추가
        Returns:
            토큰 ID 리스트
        """
        encoded = self._tok.encode(text)
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
        return self._tok.decode(ids, skip_special_tokens=False)

    def encode_batch(self, texts: List[str], add_special: bool = True) -> List[List[int]]:
        """배치 인코딩"""
        encoded_list = self._tok.encode_batch(texts)
        all_ids = [enc.ids for enc in encoded_list]
        if add_special:
            all_ids = [[self._bos_id] + ids + [self._eos_id] for ids in all_ids]
        return all_ids

    def id_to_piece(self, id: int) -> str:
        """토큰 ID → 토큰 문자열"""
        return self._tok.id_to_token(id)

    def piece_to_id(self, piece: str) -> int:
        """토큰 문자열 → 토큰 ID"""
        return self._tok.token_to_id(piece)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="BBPE 토크나이저 테스트")
    parser.add_argument("--model", "-m", required=True, help="토크나이저 JSON 파일 경로")
    parser.add_argument("--test", action="store_true", help="테스트 실행")
    args = parser.parse_args()

    tokenizer = BBPETokenizer(args.model)
    print(f"Vocab size: {tokenizer.vocab_size}")
    print(f"PAD={tokenizer.pad_id}, BOS={tokenizer.bos_id}, "
          f"EOS={tokenizer.eos_id}, UNK={tokenizer.unk_id}")

    if args.test:
        test_sentences = [
            "맞춤법을 확인해 주세요.",
            "Hello, world!",
            "한글과 English 혼합 테스트",
            "こんにちは世界",
        ]
        print("\n--- 인코드/디코드 라운드트립 ---")
        for sent in test_sentences:
            ids = tokenizer.encode(sent, add_special=False)
            decoded = tokenizer.decode(ids, skip_special=False)
            match = sent == decoded
            print(f"  원문: {sent}")
            print(f"  토큰수: {len(ids)}")
            print(f"  복원: {decoded}")
            print(f"  일치: {match}")
            if not match:
                print("  !! 불일치 !!")
            print()
