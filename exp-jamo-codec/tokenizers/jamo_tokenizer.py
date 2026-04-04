"""JamoTokenizer — 한글 자모 분해 토크나이저

한글 음절 → 초성/중성/종성 분해, 비한글 → byte fallback.
SHIFT/BLANK 없음, 순수 유니코드 자모 단위.
"""
import sys
import os
from typing import List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from tokenizer_base import BaseTokenizer
from keyboard_tokenizer.ko_keyboard import (
    _decompose_syllable, _compose_syllable, _is_hangul_syllable, _is_compat_jamo,
    INITIALS, MEDIALS, FINALS,
    ALL_CONSONANTS, ALL_VOWELS,
)

# ── 특수 토큰 ──
_PAD = 0
_UNK = 1
_BOS = 2
_EOS = 3
_MASK = 4
_SEP = 5
_CLS = 6
_N_SPECIAL = 7

# ── 자모 영역 ──
# 초성 19개 (ID 7~25)
_INITIAL_OFFSET = _N_SPECIAL
# 중성 21개 (ID 26~46)
_MEDIAL_OFFSET = _INITIAL_OFFSET + len(INITIALS)
# 종성 27개 (ID 47~73, FINALS[0]=None 제외)
_FINAL_OFFSET = _MEDIAL_OFFSET + len(MEDIALS)
_N_FINALS = len(FINALS) - 1  # None 제외 = 27
# 바이트 256개 (ID 74~329)
_BYTE_OFFSET = _FINAL_OFFSET + _N_FINALS

# 호환 자모 → ID 매핑 (단독 자모가 텍스트에 나올 때)
_COMPAT_CONSONANT_TO_ID = {}
_COMPAT_VOWEL_TO_ID = {}

# 호환 자모 중 초성으로도 쓰이는 자음 매핑
for _i, _ini in enumerate(INITIALS):
    _COMPAT_CONSONANT_TO_ID[_ini] = _INITIAL_OFFSET + _i

# 호환 자모 모음 매핑
for _i, _med in enumerate(MEDIALS):
    _COMPAT_VOWEL_TO_ID[_med] = _MEDIAL_OFFSET + _i

# 역매핑
_ID_TO_INITIAL = {_INITIAL_OFFSET + i: c for i, c in enumerate(INITIALS)}
_ID_TO_MEDIAL = {_MEDIAL_OFFSET + i: c for i, c in enumerate(MEDIALS)}
_ID_TO_FINAL = {_FINAL_OFFSET + i: FINALS[i + 1] for i in range(_N_FINALS)}


class JamoTokenizer(BaseTokenizer):
    """한글 자모 분해 토크나이저

    한글 음절 → 초성 + 중성 + (종성) 분해
    호환 자모 → 해당 자모 ID
    비한글 → byte fallback
    """

    @property
    def vocab_size(self) -> int:
        return _BYTE_OFFSET + 256  # 330

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
        """텍스트 → 자모/바이트 ID 리스트"""
        ids = []
        for ch in text:
            if _is_hangul_syllable(ch):
                ini_idx, med_idx, fin_idx = _decompose_syllable(ch)
                ids.append(_INITIAL_OFFSET + ini_idx)
                ids.append(_MEDIAL_OFFSET + med_idx)
                if fin_idx != 0:
                    ids.append(_FINAL_OFFSET + (fin_idx - 1))
            elif ch in ALL_CONSONANTS and ch in _COMPAT_CONSONANT_TO_ID:
                ids.append(_COMPAT_CONSONANT_TO_ID[ch])
            elif ch in ALL_VOWELS and ch in _COMPAT_VOWEL_TO_ID:
                ids.append(_COMPAT_VOWEL_TO_ID[ch])
            else:
                # byte fallback
                for b in ch.encode("utf-8"):
                    ids.append(_BYTE_OFFSET + b)
        if add_special:
            ids = [_BOS] + ids + [_EOS]
        return ids

    def decode(self, ids: List[int], skip_special: bool = True) -> str:
        """자모/바이트 ID 리스트 → 텍스트"""
        special = {_PAD, _UNK, _BOS, _EOS, _MASK, _SEP, _CLS}
        result = []
        byte_buf = []

        # 자모 상태 머신: 초성 → 중성 → (종성) → flush
        ini_idx = None
        med_idx = None
        fin_idx = None

        def _flush_syllable():
            nonlocal ini_idx, med_idx, fin_idx
            if ini_idx is not None and med_idx is not None:
                result.append(_compose_syllable(ini_idx, med_idx, fin_idx or 0))
            elif ini_idx is not None:
                result.append(INITIALS[ini_idx])
            elif med_idx is not None:
                result.append(MEDIALS[med_idx])
            ini_idx = med_idx = fin_idx = None

        def _flush_bytes():
            if byte_buf:
                result.append(bytes(byte_buf).decode("utf-8", errors="replace"))
                byte_buf.clear()

        for i in ids:
            if skip_special and i in special:
                continue

            is_initial = i in _ID_TO_INITIAL
            is_medial = i in _ID_TO_MEDIAL
            is_final = i in _ID_TO_FINAL
            is_byte = _BYTE_OFFSET <= i < _BYTE_OFFSET + 256

            if is_initial or is_medial or is_final:
                _flush_bytes()

                if is_initial:
                    idx = i - _INITIAL_OFFSET
                    if ini_idx is not None:
                        _flush_syllable()
                    ini_idx = idx

                elif is_medial:
                    idx = i - _MEDIAL_OFFSET
                    if med_idx is not None:
                        # 이미 중성이 있는데 또 중성 → flush 후 새로 시작
                        _flush_syllable()
                    med_idx = idx

                elif is_final:
                    idx = i - _FINAL_OFFSET
                    if fin_idx is not None or med_idx is None:
                        _flush_syllable()
                    fin_idx = idx + 1  # FINALS[0]=None이므로 +1

            elif is_byte:
                _flush_syllable()
                byte_buf.append(i - _BYTE_OFFSET)

            else:
                _flush_syllable()
                _flush_bytes()

        _flush_syllable()
        _flush_bytes()
        return "".join(result)


if __name__ == "__main__":
    tok = JamoTokenizer()
    print(f"Vocab size: {tok.vocab_size}")
    print(f"PAD={tok.pad_id}, BOS={tok.bos_id}, EOS={tok.eos_id}, UNK={tok.unk_id}")
    print(f"자모 영역: 초성 {_INITIAL_OFFSET}~{_MEDIAL_OFFSET-1}, "
          f"중성 {_MEDIAL_OFFSET}~{_FINAL_OFFSET-1}, "
          f"종성 {_FINAL_OFFSET}~{_BYTE_OFFSET-1}")
    print(f"바이트 영역: {_BYTE_OFFSET}~{_BYTE_OFFSET+255}")

    tests = [
        ("까마귀", "쌍자음 + 복합모음"),
        ("맞춤법을 확인해 주세요.", "일반 문장"),
        ("Hello, world!", "영문"),
        ("한글English혼합", "한영 혼합"),
        ("읽다", "복합 종성"),
        ("ㅋㅋㅋ아ㅋㅋ", "호환 자모 + 음절"),
        ("고ㄱㄱ", "음절 + 단독 자음"),
        ("곡ㅏ", "음절 + 단독 모음"),
    ]

    print("\n--- 인코드/디코드 라운드트립 ---")
    all_pass = True
    for text, desc in tests:
        ids = tok.encode(text, add_special=False)
        decoded = tok.decode(ids, skip_special=False)
        match = text == decoded
        if not match:
            all_pass = False
        print(f"  [{desc}] 원문: {text}")
        print(f"  토큰수: {len(ids)} (자모+바이트)")
        print(f"  IDs: {ids[:20]}{'...' if len(ids) > 20 else ''}")
        print(f"  복원: {decoded}")
        print(f"  일치: {'O' if match else 'X'}")
        print()

    # 시퀀스 길이 비교
    print("--- 시퀀스 길이 비교 ---")
    sample = "맞춤법을 확인해 주세요."
    from tokenizers.byte_tokenizer import ByteTokenizer
    btok = ByteTokenizer()
    j_ids = tok.encode(sample, add_special=False)
    b_ids = btok.encode(sample, add_special=False)
    print(f"  텍스트: {sample}")
    print(f"  자모: {len(j_ids)} 토큰")
    print(f"  바이트: {len(b_ids)} 토큰")
    print(f"  원문 문자: {len(sample)} 글자")

    print(f"\n전체: {'PASS' if all_pass else 'FAIL'}")
