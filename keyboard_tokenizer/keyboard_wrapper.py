"""키보드 시퀀스 토크나이저 래퍼 — BaseTokenizer 인터페이스 구현

텍스트 → 키보드 자모 분해 → 토큰 ID 변환
토큰 ID → 키보드 자모 시퀀스 → 한글 재합성
"""
import json
import os
import sys
from typing import List

from tokenizers import Tokenizer

# 프로젝트 루트를 path에 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tokenizer_base import BaseTokenizer
from keyboard_tokenizer.ko_keyboard import (
    preprocess, postprocess, SHIFT, BLANK,
    BASIC_CONSONANTS, BASIC_VOWELS, ALL_CONSONANTS, ALL_VOWELS,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


class KeyboardTokenizer(BaseTokenizer):
    """키보드 시퀀스 토크나이저 래퍼

    한글을 2벌식 키보드 자모 시퀀스로 분해한 뒤 토큰화.
    비한글 문자는 byte-level 폴백으로 처리.
    """

    def __init__(
        self,
        tokenizer_path: str | None = None,
        jamo_map_path: str | None = None,
    ):
        """
        Args:
            tokenizer_path: keyboard_tokenizer.json 경로
            jamo_map_path: jamo_token_map.json 경로
        """
        if tokenizer_path is None:
            tokenizer_path = os.path.join(SCRIPT_DIR, "keyboard_tokenizer.json")
        if jamo_map_path is None:
            jamo_map_path = os.path.join(SCRIPT_DIR, "jamo_token_map.json")

        if not os.path.exists(tokenizer_path):
            raise FileNotFoundError(
                f"토크나이저 파일 없음: {tokenizer_path}\n"
                f"먼저 make_tokenizer.py를 실행하세요."
            )

        self._tokenizer = Tokenizer.from_file(tokenizer_path)

        # 자모 → token ID 직접 매핑 (전처리된 토큰 → ID 빠른 변환용)
        self._jamo_map = {}
        if os.path.exists(jamo_map_path):
            with open(jamo_map_path, "r", encoding="utf-8") as f:
                self._jamo_map = json.load(f)

        # 역매핑: token ID → 자모 문자열
        self._id_to_jamo = {v: k for k, v in self._jamo_map.items()}

        # Special token IDs
        self._pad_id = self._tokenizer.token_to_id("[PAD]")
        self._unk_id = self._tokenizer.token_to_id("[UNK]")
        self._bos_id = self._tokenizer.token_to_id("[BOS]")
        self._eos_id = self._tokenizer.token_to_id("[EOS]")
        self._sep_id = self._tokenizer.token_to_id("[SEP]")
        self._cls_id = self._tokenizer.token_to_id("[CLS]")
        self._mask_id = self._tokenizer.token_to_id("[MASK]")

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
        """텍스트 → 키보드 시퀀스 → 토큰 ID 리스트"""
        # 1. 전처리: 텍스트 → 키보드 토큰 리스트
        keystroke_tokens = preprocess(text)

        # 2. 각 토큰을 ID로 변환
        ids = []
        non_jamo_buf = []  # 비자모 토큰 버퍼 (byte-level로 한번에 처리)

        def _flush_buf():
            if non_jamo_buf:
                chunk = "".join(non_jamo_buf)
                enc = self._tokenizer.encode(chunk)
                ids.extend(enc.ids)
                non_jamo_buf.clear()

        for tok in keystroke_tokens:
            if tok in self._jamo_map:
                _flush_buf()
                ids.append(self._jamo_map[tok])
            else:
                non_jamo_buf.append(tok)

        _flush_buf()

        if add_special:
            ids = [self._bos_id] + ids + [self._eos_id]
        return ids

    def decode(self, ids: List[int], skip_special: bool = True) -> str:
        """토큰 ID 리스트 → 키보드 시퀀스 → 텍스트"""
        if skip_special:
            special = {self._pad_id, self._bos_id, self._eos_id, self._unk_id}
            ids = [i for i in ids if i not in special]

        # 1. ID → 키보드 토큰 리스트
        keystroke_tokens = []
        non_jamo_ids = []

        def _flush_ids():
            if non_jamo_ids:
                decoded = self._tokenizer.decode(non_jamo_ids)
                for ch in decoded:
                    keystroke_tokens.append(ch)
                non_jamo_ids.clear()

        for tid in ids:
            if tid in self._id_to_jamo:
                _flush_ids()
                keystroke_tokens.append(self._id_to_jamo[tid])
            else:
                non_jamo_ids.append(tid)

        _flush_ids()

        # 2. 후처리: 키보드 토큰 → 한글 텍스트
        return postprocess(keystroke_tokens)

    def encode_batch(self, texts: List[str], add_special: bool = True) -> List[List[int]]:
        """배치 인코딩"""
        return [self.encode(t, add_special) for t in texts]

    def id_to_piece(self, id: int) -> str:
        """토큰 ID → 토큰 문자열"""
        if id in self._id_to_jamo:
            return self._id_to_jamo[id]
        return self._tokenizer.id_to_token(id)

    def piece_to_id(self, piece: str) -> int:
        """토큰 문자열 → 토큰 ID"""
        if piece in self._jamo_map:
            return self._jamo_map[piece]
        return self._tokenizer.token_to_id(piece)


if __name__ == "__main__":
    tok = KeyboardTokenizer()
    print(f"Vocab size: {tok.vocab_size}")
    print(f"PAD={tok.pad_id}, BOS={tok.bos_id}, "
          f"EOS={tok.eos_id}, UNK={tok.unk_id}")

    tests = [
        "까마귀",
        "맞춤법을 확인해 주세요.",
        "Hello, world!",
        "고ㄱㄱ",
        "곡ㄱ",
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
        print(f"  토큰수: {len(ids)}")
        print(f"  복원: {decoded}")
        print(f"  일치: {'✓' if match else '✗'}")
        print()

    print(f"전체: {'PASS ✓' if all_pass else 'FAIL ✗'}")
