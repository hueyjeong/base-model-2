"""
시제 오류 (Tense Errors) - K-NCT 14번 항목
선어말어미(EP)를 엉뚱한 시제 형태로 변형시켜 시제 불일치를 유도.
"""
import random
from typing import Optional
from error_generation.utils import get_mecab_offsets, replace_by_offset

# 한국어 대표 선어말어미 혼동 쌍
TENSE_CONFUSION_MAP = {
    "았": "겠",  
    "었": "겠",
    "겠": "었",
    "였": "겠",
    "시": "셨", # 주체높임, 엄밀힌 시제는 아니나 같이 혼동
    "셨": "시",
}

def apply_tense_error(text: str, rng: random.Random) -> Optional[str]:
    """문장 내 선어말어미(EP)를 찾아서 그 표면형이 교체 가능하면 교체한다."""
    tokens = get_mecab_offsets(text)
    
    # 선어말어미(EP) 토큰 추려내기
    ep_tokens = [t for t in tokens if t.pos == 'EP' and t.surface in TENSE_CONFUSION_MAP]
    
    if not ep_tokens:
        return None
        
    target = rng.choice(ep_tokens)
    wrong_surface = TENSE_CONFUSION_MAP[target.surface]
    
    # 오프셋 단위 치환
    return replace_by_offset(text, target.start, target.end, wrong_surface)

def get_error_count() -> int:
    return len(TENSE_CONFUSION_MAP)
