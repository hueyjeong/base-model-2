"""
문장 요소 순서 오류 (Element Errors) - K-NCT 16번 항목
주어와 목적어, 혹은 서술어의 위치(어순)를 강제로 비정상으로 바꿈.
"""
import random
from typing import Optional
from error_generation.utils import get_mecab_offsets, swap_by_offset, TokenInfo

def find_phrases(tokens: list[TokenInfo]) -> list[tuple[int, int]]:
    """명사구 덩어리의 (start_offset, end_offset) 리스트를 추출.
    명사(NN) 뒤에 조사(J)가 붙어있는 덩어리를 하나의 어절로 간주.
    """
    phrases = []
    i = 0
    n_tokens = len(tokens)
    
    while i < n_tokens:
        tok = tokens[i]
        
        # 명사류(NNG, NNP, NP 등) 발견
        if tok.pos.startswith('N'):
            phrase_start = tok.start
            phrase_end = tok.end
            
            # 이어지는 조사가 있는지 확인
            j = i + 1
            while j < n_tokens and tokens[j].pos.startswith('J'):
                phrase_end = tokens[j].end
                j += 1
                
            phrases.append((phrase_start, phrase_end))
            i = j
        else:
            i += 1
            
    return phrases

def apply_word_order_error(text: str, rng: random.Random) -> Optional[str]:
    """문장 내 존재하는 명사구 덩어리 2개의 위치를 맞바꾼다."""
    tokens = get_mecab_offsets(text)
    phrases = find_phrases(tokens)
    
    # 교환할 두 개의 명사구가 없다면 포기
    if len(phrases) < 2:
        return None
        
    p1, p2 = rng.sample(phrases, k=2)
    s1, e1 = p1
    s2, e2 = p2
    
    # 두 프레이즈(텍스트 슬라이스) 스왑
    return swap_by_offset(text, s1, e1, s2, e2)

def get_error_count() -> int:
    return 1 # 전체 문장 어위 스왑이므로 패턴 수 1
