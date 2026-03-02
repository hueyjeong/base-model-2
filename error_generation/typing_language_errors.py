"""
타이핑 언어 오류 (Typing Language Errors) - K-NCT 9번 항목
한글 입력 상태에서 영문 QWERTY 자판을 치는 오작동 (예: 안녕 -> dkssud)
"""
import random
from typing import Optional

# 초/중/종성 분해용 베이스
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
    None, "ㄱ", "ㄲ", "ㄳ", "ㄴ", "ㄵ", "ㄶ", "ㄷ", "ㄹ",
    "ㄺ", "ㄻ", "ㄼ", "ㄽ", "ㄾ", "ㄿ", "ㅀ", "ㅁ", "ㅂ",
    "ㅄ", "ㅅ", "ㅆ", "ㅇ", "ㅈ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ",
]

# 한글 자자모 -> 영문 맵핑
KO_TO_EN = {
    'ㅂ': 'q', 'ㅃ': 'Q', 'ㅈ': 'w', 'ㅉ': 'W', 'ㄷ': 'e', 'ㄸ': 'E', 'ㄱ': 'r', 'ㄲ': 'R', 'ㅅ': 't', 'ㅆ': 'T',
    'ㅛ': 'y', 'ㅕ': 'u', 'ㅑ': 'i', 'ㅐ': 'o', 'ㅒ': 'O', 'ㅔ': 'p', 'ㅖ': 'P',
    'ㅁ': 'a', 'ㄴ': 's', 'ㅇ': 'd', 'ㄹ': 'f', 'ㅎ': 'g', 'ㅗ': 'h', 'ㅓ': 'j', 'ㅏ': 'k', 'ㅣ': 'l',
    'ㅋ': 'z', 'ㅌ': 'x', 'ㅊ': 'c', 'ㅍ': 'v', 'ㅠ': 'b', 'ㅜ': 'n', 'ㅡ': 'm',
    'ㅘ': 'hk', 'ㅙ': 'ho', 'ㅚ': 'hl', 'ㅝ': 'nj', 'ㅞ': 'np', 'ㅟ': 'nl', 'ㅢ': 'ml',
    'ㄳ': 'rt', 'ㄵ': 'sw', 'ㄶ': 'sg', 'ㄺ': 'fr', 'ㄻ': 'fa', 'ㄼ': 'fq', 'ㄽ': 'ft', 'ㄾ': 'fx', 'ㄿ': 'fv', 'ㅀ': 'fg', 'ㅄ': 'qt'
}

def decompose_to_qwerty(text: str) -> str:
    result = []
    for ch in text:
        code = ord(ch)
        if _HANGUL_BASE <= code <= _HANGUL_END:
            offset = code - _HANGUL_BASE
            cho = offset // (_JUNG_COUNT * _JONG_COUNT)
            jung = (offset % (_JUNG_COUNT * _JONG_COUNT)) // _JONG_COUNT
            jong = offset % _JONG_COUNT

            cho_str = _CHO_LIST[cho]
            jung_str = _JUNG_LIST[jung]
            jong_str = _JONG_LIST[jong] if jong > 0 else None

            # 치환
            result.append(KO_TO_EN.get(cho_str, ''))
            result.append(KO_TO_EN.get(jung_str, ''))
            if jong_str:
                result.append(KO_TO_EN.get(jong_str, ''))
        elif ch in KO_TO_EN:
            # 단독 자모 형태일 때
            result.append(KO_TO_EN[ch])
        else:
            result.append(ch)
    return "".join(result)

def apply_typing_language_error(text: str, rng: random.Random) -> Optional[str]:
    """임의의 어절을 선택하여 영문 QWERTY 타건으로 변경한다."""
    words = text.split(" ")
    if not words:
        return None
    
    # 두 글자 이상인 단어를 선호
    candidates = [(i, w) for i, w in enumerate(words) if len(w) > 1]
    if not candidates:
        candidates = [(i, w) for i, w in enumerate(words) if len(w) > 0]
        
    if not candidates:
        return None
        
    idx, target = rng.choice(candidates)
    
    words[idx] = decompose_to_qwerty(target)
    
    return " ".join(words)

def get_error_count() -> int:
    return 1
