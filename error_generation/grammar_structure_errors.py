"""
문법 구조 오류 (Grammar Structure Errors) - K-NCT 4번, 5번 항목
조사(Josa)나 어미(Eomi)의 논리적 삭제 및 부적절한 첨가를 수행.
"""
import random
from typing import Optional
from error_generation.utils import get_mecab_offsets, replace_by_offset, TokenInfo

# 임의로 첨가할 만한 흔한 조사/어미 목록
RANDOM_JOSA = ["이", "가", "은", "는", "을", "를", "의", "에", "에서", "으로"]
RANDOM_EOMI = ["고", "며", "면", "니", "다", "요"]

def apply_remove_error(text: str, rng: random.Random) -> Optional[str]:
    """조사(J)나 어미(E) 중 하나를 무작위로 삭제한다."""
    tokens = get_mecab_offsets(text)
    
    candidates = [
        t for t in tokens 
        # J: 조사, E: 어미
        if t.pos.startswith('J') or t.pos.startswith('E')
    ]
    
    if not candidates:
        return None
        
    target = rng.choice(candidates)
    
    # 해당 형태소를 빈 문자열로 치환 (삭제)
    return replace_by_offset(text, target.start, target.end, "")

def apply_addition_error(text: str, rng: random.Random) -> Optional[str]:
    """조사(J)나 명사(N), 동사(V) 뒤에 무작위 조사/어미를 강제로 첨가한다."""
    tokens = get_mecab_offsets(text)
    
    candidates = [
        t for t in tokens 
        # 명사, 동사, 조사 뒤에 무언갈 덧붙이는 게 자연스러운 형태소 왜곡
        if t.pos.startswith('N') or t.pos.startswith('V') or t.pos.startswith('J')
    ]
    
    if not candidates:
        return None
        
    target = rng.choice(candidates)
    
    # 어떤 형태의 첨가를 할 것인가
    addition = rng.choice(RANDOM_JOSA) if rng.random() > 0.5 else rng.choice(RANDOM_EOMI)
    
    # 대상 형태소 뒤(end)에 addition을 삽입한다.
    return replace_by_offset(text, target.end, target.end, addition)

def get_error_count() -> int:
    """오류 패턴의 총 개수 (추정) 반환."""
    return 2 # 삭제, 첩가
