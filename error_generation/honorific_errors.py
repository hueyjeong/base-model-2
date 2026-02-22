import random
import re
from typing import Optional

# 주체/객체 높임 등의 호응을 깨뜨리는 패턴
_HONORIFIC_PATTERNS = [
    (re.compile(r"\b제(가|는|를|게)\b"), lambda m: f"내{m.group(1)}"),
    (re.compile(r"\b내(가|는|를|게)\b"), lambda m: f"제{m.group(1)}"),
    (re.compile(r"\b저(는|를|에게)\b"), lambda m: f"나{m.group(1)}"),
    (re.compile(r"\b나(는|를|에게)\b"), lambda m: f"저{m.group(1)}"),
    (re.compile(r"께서\b"), lambda m: "가"),
    (re.compile(r"께서는\b"), lambda m: "는"),
    (re.compile(r"께\b"), lambda m: "한테"),
    (re.compile(r"\b드(립|렸|릴|려|리)\b"), lambda m: f"주{m.group(1)}"),
    (re.compile(r"\b주(십|셨|실|셔|시)\b"), lambda m: f"드{m.group(1)}"),
    (re.compile(r"\b계(십|셨|실|셔|시)\b"), lambda m: f"있으{m.group(1)}"),
    (re.compile(r"\b있으(십|셨|실|셔|시)\b"), lambda m: f"계{m.group(1)}"),
    (re.compile(r"\b말씀\b"), lambda m: "말"),
    (re.compile(r"\b잡수시(고|어|다)\b"), lambda m: f"먹으{m.group(1)}"),
]

def get_error_count() -> int:
    """오류 패턴의 총 개수를 반환."""
    return len(_HONORIFIC_PATTERNS)

def apply_honorific_error(text: str, rng: random.Random) -> Optional[str]:
    """높임말과 낮춤말의 호응이 맞지 않는 오류 생성."""
    matches = []
    for pattern, replacer in _HONORIFIC_PATTERNS:
        for m in pattern.finditer(text):
            matches.append((m, replacer))
            
    if not matches:
        return None
        
    m, replacer = rng.choice(matches)
    replacement = replacer(m)
    return text[:m.start()] + replacement + text[m.end():]
