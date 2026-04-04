"""ByteTokenizer — UTF-8 바이트 단위 토크나이저

텍스트를 UTF-8 바이트로 분해하여 토큰화.
한글 1음절 = 3바이트, ASCII = 1바이트.
외부 의존성 없음.
"""
import sys
import os
from typing import List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from tokenizer_base import BaseTokenizer

# 특수 토큰 ID (0~6), 바이트 값은 +7 offset
_PAD = 0
_UNK = 1
_BOS = 2
_EOS = 3
_MASK = 4
_SEP = 5
_CLS = 6
_BYTE_OFFSET = 7  # byte 0 → ID 7, byte 255 → ID 262


class ByteTokenizer(BaseTokenizer):
    """UTF-8 바이트 단위 토크나이저"""

    @property
    def vocab_size(self) -> int:
        return 263  # 7 special + 256 bytes

    @property
    def pad_id(self) -> int:
        return _PAD

    @property
    def bos_id(self) -> int:
        return _BOS

    @property
    def eos_id(self) -> int:
        return _EOS

    @property
    def unk_id(self) -> int:
        return _UNK

    @property
    def sep_id(self) -> int:
        return _SEP

    @property
    def cls_id(self) -> int:
        return _CLS

    @property
    def mask_id(self) -> int:
        return _MASK

    def encode(self, text: str, add_special: bool = True) -> List[int]:
        """텍스트 → UTF-8 바이트 ID 리스트"""
        ids = [b + _BYTE_OFFSET for b in text.encode("utf-8")]
        if add_special:
            ids = [_BOS] + ids + [_EOS]
        return ids

    def decode(self, ids: List[int], skip_special: bool = True) -> str:
        """바이트 ID 리스트 → 텍스트"""
        special = {_PAD, _UNK, _BOS, _EOS, _MASK, _SEP, _CLS}
        raw_bytes = []
        for i in ids:
            if skip_special and i in special:
                continue
            if _BYTE_OFFSET <= i < _BYTE_OFFSET + 256:
                raw_bytes.append(i - _BYTE_OFFSET)
        return bytes(raw_bytes).decode("utf-8", errors="replace")


if __name__ == "__main__":
    tok = ByteTokenizer()
    print(f"Vocab size: {tok.vocab_size}")
    print(f"PAD={tok.pad_id}, BOS={tok.bos_id}, EOS={tok.eos_id}, UNK={tok.unk_id}")

    tests = [
        "까마귀",
        "맞춤법을 확인해 주세요.",
        "Hello, world!",
        "한글English혼합",
        "읽다",
        "ㅋㅋㅋ아ㅋㅋ",
    ]

    print("\n--- 인코드/디코드 라운드트립 ---")
    all_pass = True
    for text in tests:
        ids = tok.encode(text, add_special=False)
        decoded = tok.decode(ids, skip_special=False)
        match = text == decoded
        if not match:
            all_pass = False
        print(f"  원문: {text}")
        print(f"  토큰수: {len(ids)} (바이트)")
        print(f"  복원: {decoded}")
        print(f"  일치: {'O' if match else 'X'}")
        print()

    print(f"전체: {'PASS' if all_pass else 'FAIL'}")
