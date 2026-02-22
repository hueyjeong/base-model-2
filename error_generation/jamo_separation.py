import random
from typing import Optional

_HANGUL_BASE = 0xAC00
_HANGUL_END = 0xD7A3
_JONG_COUNT = 28
_JUNG_COUNT = 21

_CHO_LIST = [
    "ㄱ", "ㄲ", "ㄴ", "ㄷ", "ㄸ", "ㄹ", "ㅁ", "ㅂ", "ㅃ",
    "ㅅ", "ㅆ", "ㅇ", "ㅈ", "ㅉ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ",
]
_JUNG_LIST = [
    "ㅏ", "ㅐ", "ㅑ", "ㅒ", "ㅓ", "ㅔ", "ㅕ", "ㅖ", "ㅗ",
    "ㅘ", "ㅙ", "ㅚ", "ㅛ", "ㅜ", "ㅝ", "ㅞ", "ㅟ", "ㅠ",
    "ㅡ", "ㅢ", "ㅣ",
]
_JONG_LIST = [
    "", "ㄱ", "ㄲ", "ㄳ", "ㄴ", "ㄵ", "ㄶ", "ㄷ", "ㄹ",
    "ㄺ", "ㄻ", "ㄼ", "ㄽ", "ㄾ", "ㄿ", "ㅀ", "ㅁ", "ㅂ",
    "ㅄ", "ㅅ", "ㅆ", "ㅇ", "ㅈ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ",
]

def get_error_count() -> int:
    """오류 패턴의 총 개수를 반환."""
    return 5  # 초성 추가, 자모 완전 분리, 모음 추가, 종성 분리, 초성/모음 분리

def apply_jamo_separation(text: str, rng: random.Random) -> Optional[str]:
    """자음과 모음이 잘못 타이핑되어 분리되거나 반복되는 현상 모사."""
    hangul_indices = [
        i for i, ch in enumerate(text)
        if _HANGUL_BASE <= ord(ch) <= _HANGUL_END
    ]
    if not hangul_indices:
        return None
        
    idx = rng.choice(hangul_indices)
    ch = text[idx]
    
    code = ord(ch) - _HANGUL_BASE
    cho = code // (_JUNG_COUNT * _JONG_COUNT)
    jung = (code % (_JUNG_COUNT * _JONG_COUNT)) // _JONG_COUNT
    jong = code % _JONG_COUNT
    
    cho_char = _CHO_LIST[cho]
    jung_char = _JUNG_LIST[jung]
    jong_char = _JONG_LIST[jong]
    
    patterns = []
    # 1. 초성만 추가 (예: 안녕 -> 안ㄴ녕)
    patterns.append(cho_char + ch)
    # 2. 풀어서 표현 (예: 안 -> ㅇㅏㄴ)
    patterns.append(cho_char + jung_char + jong_char)
    # 3. 모음만 추가 (예: 안 -> 안ㅏ)
    patterns.append(ch + jung_char)
    # 4. 종성만 분리 (예: 안 -> 아ㄴ)
    if jong > 0:
        base_char = chr(_HANGUL_BASE + cho * _JUNG_COUNT * _JONG_COUNT + jung * _JONG_COUNT)
        patterns.append(base_char + jong_char)
    # 5. 초성 분리 및 모음 결합 해제 (예: 하 -> ㅎㅏ)
    if jong == 0:
        patterns.append(cho_char + jung_char)
        
    separated = rng.choice(patterns)
    return text[:idx] + separated + text[idx+1:]
