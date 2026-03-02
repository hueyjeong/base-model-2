"""
외래어 변환 오류 (Foreign Word Conversion Errors) - K-NCT 10번 항목
"""
import random
import re
from typing import Optional

import json
import os

_RESOURCE_PATH = os.path.join(os.path.dirname(__file__), "resources", "foreign_words.json")

def _load_foreign_words():
    if not os.path.exists(_RESOURCE_PATH):
        return {}
    with open(_RESOURCE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

FOREIGN_CONFUSION_MAP = _load_foreign_words()

def apply_foreign_word_error(text: str, rng: random.Random) -> Optional[str]:
    candidates = []
    
    # 단어 경계(\b 기능을 대체하기 위해 앞뒤 띄어쓰기 또는 문자열 시작/끝을 확인)
    # 정규식 치환을 사용하여 원치 않는 파생어 변경(예: '주스'가 '우주스파'에 있으면 안 바뀜) 방지
    for correct, wrongs in FOREIGN_CONFUSION_MAP.items():
        # 정규표현식: [시작 or 공백] + 단어 + [끝 or 기호 or 공백]
        # 조사 결합을 고려하여 뒤쪽에는 한글 조사가 붙거나 비어있을 수 있음을 감안
        # 간단히 단어가 text에 존재하는지 보고 세밀하게 교체
        if correct in text:
            # 안전하게 찾은 위치를 오프셋으로 기록 (다중 등장은 무시하고 하나만)
            for m in re.finditer(f"(?<![가-힣]){correct}", text):
                for wrong in wrongs:
                    candidates.append((m.start(), m.end(), wrong))
                
    if not candidates:
        return None
        
    start, end, replacement = rng.choice(candidates)
    return text[:start] + replacement + text[end:]

def get_error_count() -> int:
    return len(FOREIGN_CONFUSION_MAP)
