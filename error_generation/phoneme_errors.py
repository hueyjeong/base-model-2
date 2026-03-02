"""
발음 향 표기 및 자음 모음 변환 (Phoneme & Consonant Vowel Errors) - K-NCT 11, 12번 항목
"""
import random
import re
from typing import Optional

# 대표적인 소리나는대로 쓴 오타 패턴 (추가 가능)
PHONEME_MAP = {
    "해돋이": "해도지",
    "같이": "가치",
    "굳이": "구지",
    "맞추다": "마추다",
    "꽃이": "꼬치",
    "무릎이": "무르피",
    "잎이": "이피",
    "맑은": "말근",
    "밝은": "발근",
    "읽어": "일거",
    "앉아": "안자",
    "넓은": "널븐",
    "없어": "업서",
    "어떻게": "어떠케",
    "좋아": "조아",
    "설거지": "설겆이", # 이건 역방향
    "며칠": "몇일",
}

# 의미없는 자음이 끼어드는 패턴
NON_SPEAKING_JAMO = ["ㅋ", "ㅎ", "ㅇ", "ㄴ"]

def apply_phoneme_error(text: str, rng: random.Random) -> Optional[str]:
    """소리나는 대로 적는(연음/구개음화) 파괴적 치환 에러"""
    candidates = []
    
    # 1. Phoneme map 변환
    for correct, wrong in PHONEME_MAP.items():
        if correct in text:
            for m in re.finditer(f"{correct}", text):
                candidates.append(("replace", m.start(), m.end(), wrong))
                
    # 2. 임의 자음 중간 삽입 (예: 이제 곧 -> 이제콘)
    # 어절 사이 공백 부근에 의미없는 자음을 슬쩍 삽입
    spaces = [m.start() for m in re.finditer(r"\s+", text)]
    if spaces:
        for s in spaces:
            candidates.append(("insert", s, s, rng.choice(NON_SPEAKING_JAMO)))

    if not candidates:
        return None
        
    choice = rng.choice(candidates)
    
    if choice[0] == "replace":
        _, start, end, wrong = choice
        return text[:start] + wrong + text[end:]
    else:
        # insert
        _, start, end, jamo = choice
        # 공백 앞글자의 받침으로 들어가는 것이 더 악랄하겠으나 일단 문자열 삽입
        return text[:start] + jamo + text[end:]

def get_error_count() -> int:
    return len(PHONEME_MAP)
