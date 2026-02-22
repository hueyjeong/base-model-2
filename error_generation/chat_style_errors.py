import random
import re
from typing import Optional

# 통신체 / 신조어 치환 패턴
_CHAT_STYLE_PATTERNS = [
    (re.compile(r"안녕하세요"), ["안냐세여", "안녕하세욤", "ㅎㅇ", "하이요"]),
    (re.compile(r"감사합니다"), ["감사요", "ㄱㅅㄱㅅ", "감쟈합니다", "고맙습니당", "감삼다"]),
    (re.compile(r"수고하셨습니다"), ["수고해씀다", "수고여", "수고수고", "ㅅㄱㅅㄱ"]),
    (re.compile(r"죄송합니다"), ["ㅈㅅ", "죄송요", "뎨송합니다", "ㅈㅅㅈㅅ"]),
    (re.compile(r"\b명작\b"), ["띵작"]),
    (re.compile(r"\b명언\b"), ["띵언"]),
    (re.compile(r"확인했습니다"), ["확인염", "ㅇㅋㅇㅋ", "확인여", "ㅇㅋ"]),
    (re.compile(r"재미있어(요)?"), ["존잼", "꿀잼", "핵잼", "노잼"]),
    (re.compile(r"재미있게"), ["존잼으로", "재밌게"]),
    (re.compile(r"어떻게( |)해(요)?"), ["어케", "우째", "어뜨케"]),
    (re.compile(r"그러니까"), ["그니까", "긍까", "그니깐"]),
    (re.compile(r"나중에"), ["나중엔", "담에"]),
    (re.compile(r"너무"), ["넘", "넘무", "개", "존나"]),
    (re.compile(r"진짜"), ["찐", "레알", "렬루"]),
    (re.compile(r"엄청"), ["개", "존나", "완전"]),
    (re.compile(r"귀여워(요)?"), ["커여워", "졸귀", "귀욤"]),
]

def get_error_count() -> int:
    """오류 패턴의 총 개수를 반환."""
    return len(_CHAT_STYLE_PATTERNS)

def apply_chat_style(text: str, rng: random.Random) -> Optional[str]:
    """통신체나 신조어를 주입."""
    matches = []
    for pattern, replacements in _CHAT_STYLE_PATTERNS:
        for m in pattern.finditer(text):
            matches.append((m, replacements))
    
    if not matches:
        return None
    
    match, replacements = rng.choice(matches)
    chosen_repl = rng.choice(replacements)
    
    return text[:match.start()] + chosen_repl + text[match.end():]
