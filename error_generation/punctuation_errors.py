import random
import re
from typing import Optional

_PUNCTS = [".", ",", "!", "?", "~"]

def get_error_count() -> int:
    """오류 패턴의 총 개수를 반환."""
    return 3  # 삭제, 반복/교체, 무작위 삽입

def apply_punctuation_error(text: str, rng: random.Random) -> Optional[str]:
    """문장부호의 누락, 반복사용, 잘못된 위치 삽입 등의 오류 주입."""
    punct_matches = [(m.start(), m.end(), m.group()) for m in re.finditer(r"[.,!?~]+", text)]
    
    if punct_matches and rng.random() < 0.6:
        # 기존 문장부호 변형
        start, end, punct_str = rng.choice(punct_matches)
        
        action = rng.choice(["delete", "repeat", "swap"])
        if action == "delete":
            replacement = ""
        elif action == "repeat":
            replacement = punct_str[0] * rng.randint(2, 4)
        else: # swap
            replacement = rng.choice(_PUNCTS)
            
        return text[:start] + replacement + text[end:]
    else:
        # 엉뚱한 위치에 삽입
        spaces = [m.start() for m in re.finditer(r"\s+", text)]
        if not spaces:
            if len(text) < 2: 
                return None
            idx = rng.randint(1, len(text) - 1)
        else:
            idx = rng.choice(spaces)
            
        inserted = rng.choice(_PUNCTS)
        return text[:idx] + inserted + text[idx:]
