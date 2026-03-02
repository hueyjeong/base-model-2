"""
수치 오류 (Number Errors) - K-NCT 3번 항목
수량(기수)과 순서(서수) 표현을 혼동하게 만듦.
"""
import random
from typing import Optional
from error_generation.utils import get_mecab_offsets, replace_by_offset, TokenInfo

# 기수(수량) <-> 서수(순서/한자어 수사) 서로 잘못 쓰는 매핑
NUMBER_CONFUSION_MAP = {
    "한": "일",
    "하나": "일",
    "두": "이",
    "둘": "이",
    "세": "삼",
    "셋": "삼",
    "네": "사",
    "넷": "사",
    
    "일": "한",
    "이": "두",
    "삼": "세",
    "사": "네",
    "오": "다섯",
    "육": "여섯",
    "칠": "일곱",
    "팔": "여덟",
    "구": "아홉",
    "십": "열",
    
    "첫": "일",
    "첫째": "일째",
}

def apply_number_error(text: str, rng: random.Random) -> Optional[str]:
    """
    텍스트 내 수사(NR) 토큰을 찾아 확률적으로 엉뚱한 수사로 치환합니다.
    """
    tokens = get_mecab_offsets(text)
    
    candidate_tokens = []
    for token in tokens:
        # MeCab에서 NR은 수사
        # SN(숫자)는 1, 2, 3이므로 텍스트 길이에 따라 변경할 수도 있지만, 일단 한글 수사(NR)에 집중
        if token.pos.startswith('NR') and token.surface in NUMBER_CONFUSION_MAP:
            candidate_tokens.append(token)
            
    if not candidate_tokens:
        return None
        
    target_token: TokenInfo = rng.choice(candidate_tokens)
    replacement = NUMBER_CONFUSION_MAP[target_token.surface]
    
    return replace_by_offset(text, target_token.start, target_token.end, replacement)

def get_error_count() -> int:
    """오류 패턴의 총 개수를 반환 (단어 매핑 수)."""
    return len(NUMBER_CONFUSION_MAP)
