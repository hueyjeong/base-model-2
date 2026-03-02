"""
의미론적 오류 (Semantic/Behavioral Errors) - K-NCT 17번 항목
문맥적으로 말이 안되는 엉뚱한 행동: 같은 문장 내 명사 2개를 무작위 추출하여 오프셋 스왑.
"""
import random
from typing import Optional
from error_generation.utils import get_mecab_offsets, swap_by_offset, TokenInfo

def find_nominals(tokens: list[TokenInfo]) -> list[TokenInfo]:
    """일반명사(NNG), 고유명사(NNP), 대명사(NP) 토큰 리스트 추출"""
    return [
        t for t in tokens 
        if t.pos in ('NNG', 'NNP', 'NP')
        # 한 글자짜리는 스왑 시 식별 효용이 떨어지거나 부작용이 크므로(예: 이, 것 등) 2글자 이상 선호
        and len(t.surface) >= 2
    ]

def apply_semantic_error(text: str, rng: random.Random) -> Optional[str]:
    """텍스트에서 2개 이상의 유효한 명사를 뽑아, 그 둘의 오프셋 블록을 바꾼다."""
    tokens = get_mecab_offsets(text)
    nominals = find_nominals(tokens)
    
    if len(nominals) < 2:
        # 2글자 명사가 부족한 경우 모든 명사 포함
        nominals = [t for t in tokens if t.pos in ('NNG', 'NNP', 'NP')]
        if len(nominals) < 2:
            return None
            
    # 중복 토큰(같은 단어 교환) 방지
    candidates = []
    for t in nominals:
        # 서로 겹치지 않는 확실한 명사만
        candidates.append(t)
        
    if len(candidates) < 2:
        return None
        
    t1, t2 = rng.sample(candidates, k=2)
    s1, e1 = t1.start, t1.end
    s2, e2 = t2.start, t2.end
    
    return swap_by_offset(text, s1, e1, s2, e2)

def get_error_count() -> int:
    return 1 # 전체 문장 내 의미 어위 스왑이므로 패턴 수 1
