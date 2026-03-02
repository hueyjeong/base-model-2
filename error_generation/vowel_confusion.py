"""
모음 혼동 오류 — ㅐ/ㅔ, ㅗ/ㅜ 등 모음 혼동.

단어 수준 치환 + 자모 수준 랜덤 ㅐ↔ㅔ 치환 기능 제공.
"""

import random
import re
from typing import Optional


# ── ㅐ/ㅔ 혼동 단어 수준 ──
AE_E_WORD_MAP: dict[str, list[str]] = {
    "대개": ["대게"],              # 대부분 의미
    "냄새": ["냄세"],
    "요새": ["요세"],
    "거예요": ["거에요"],
    "것이에요": ["것이예요"],
    "배다": ["베다"],              # 아이를 배다
    "세다": ["쌔다", "쎄다"],      # 힘이 센
    "되게": ["되겡", "디게"],
    "대개는": ["대게는"],
    "네가": ["니가"],
    "내것": ["네것"],
}

# ── 기타 모음 혼동 ──
OTHER_VOWEL_MAP: dict[str, list[str]] = {
    "같아": ["같에", "같애"],
    "같아요": ["같에요", "같애요"],
    "같아서": ["같에서", "같애서"],
}


# ── 자모 분리/결합을 통한 ㅐ↔ㅔ 랜덤 치환 ──
# 한글 유니코드 연산 상수
_HANGUL_BASE = 0xAC00
_HANGUL_END = 0xD7A3
_CHO_COUNT = 19
_JUNG_COUNT = 21
_JONG_COUNT = 28

# 중성 인덱스: ㅐ=1, ㅔ=5, ㅒ=3, ㅖ=7
_AE_IDX = 1
_E_IDX = 5
_YAE_IDX = 3
_YE_IDX = 7


def _decompose(char: str) -> Optional[tuple[int, int, int]]:
    """한글 완성형 글자를 초성/중성/종성 인덱스로 분리."""
    code = ord(char)
    if code < _HANGUL_BASE or code > _HANGUL_END:
        return None
    offset = code - _HANGUL_BASE
    cho = offset // (_JUNG_COUNT * _JONG_COUNT)
    jung = (offset % (_JUNG_COUNT * _JONG_COUNT)) // _JONG_COUNT
    jong = offset % _JONG_COUNT
    return cho, jung, jong


def _compose(cho: int, jung: int, jong: int) -> str:
    """초성/중성/종성 인덱스를 한글 완성형 글자로 결합."""
    code = _HANGUL_BASE + cho * _JUNG_COUNT * _JONG_COUNT + jung * _JONG_COUNT + jong
    return chr(code)


def _swap_ae_e(char: str) -> Optional[str]:
    """ㅐ↔ㅔ 또는 ㅒ↔ㅖ 치환."""
    parts = _decompose(char)
    if parts is None:
        return None
    cho, jung, jong = parts
    if jung == _AE_IDX:
        return _compose(cho, _E_IDX, jong)
    elif jung == _E_IDX:
        return _compose(cho, _AE_IDX, jong)
    elif jung == _YAE_IDX:
        return _compose(cho, _YE_IDX, jong)
    elif jung == _YE_IDX:
        return _compose(cho, _YAE_IDX, jong)
    return None


def apply_vowel_confusion(text: str, rng: random.Random) -> Optional[str]:
    """
    텍스트에 모음 혼동 오류를 적용.

    두 가지 전략을 랜덤으로 선택:
    1. 단어 수준 치환
    2. 글자 수준 ㅐ↔ㅔ 랜덤 치환

    Args:
        text: 올바른 한국어 문장
        rng: 랜덤 시드 관리용 Random 인스턴스

    Returns:
        오류가 적용된 문장. 적용 가능한 패턴이 없으면 None.
    """
    all_maps = {**AE_E_WORD_MAP, **OTHER_VOWEL_MAP}

    # 단어 수준 후보 수집
    word_candidates = [(c, w) for c, ws in all_maps.items() if c in text for w in ws]

    # 글자 수준 ㅐ/ㅔ 포함 인덱스 수집
    char_candidates = [i for i, ch in enumerate(text) if _swap_ae_e(ch) is not None]

    if not word_candidates and not char_candidates:
        return None

    # 전략 선택
    strategies = []
    if word_candidates:
        strategies.append("word")
    if char_candidates:
        strategies.append("char")

    strategy = rng.choice(strategies)

    if strategy == "word":
        candidates = []
        for c, ws in all_maps.items():
            if c in text:
                for m in re.finditer(f"(?<![가-힣]){c}", text):
                    for w in ws:
                        candidates.append((m.start(), m.end(), w))
        if not candidates:
            return None
        start, end, wrong = rng.choice(candidates)
        return text[:start] + wrong + text[end:]
    else:
        idx = rng.choice(char_candidates)
        chars = list(text)
        swapped = _swap_ae_e(chars[idx])
        if swapped:
            chars[idx] = swapped
        return "".join(chars)


def get_error_count() -> int:
    """이 모듈이 가진 오류 패턴 수를 반환."""
    word_count = sum(len(v) for v in AE_E_WORD_MAP.values()) + sum(len(v) for v in OTHER_VOWEL_MAP.values())
    return word_count + 4  # +4 for ㅐ↔ㅔ, ㅒ↔ㅖ 자모 치환
