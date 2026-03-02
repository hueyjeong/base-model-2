"""
접미사 오류 — 부사화 접미사, 접미사 혼동.

-이/-히 혼동, -스러운/-스런, -장이/-쟁이, 하 탈락 관련 등.
"""

import random
from typing import Optional


# ── -이/-히 혼동 ──
I_HI_MAP: dict[str, list[str]] = {
    "깨끗이": ["깨끗히"],
    "따뜻이": ["따뜻히"],
    "곰곰이": ["곰곰히"],
    "일일이": ["일일히"],
    "틈틈이": ["틈틈히"],
    "간간이": ["간간히"],      # 의미가 다른 단어이지만 혼동
    "번번이": ["번번히"],
    "깊숙이": ["깊숙히"],
    # 반대로 '-히'가 맞는데 '-이'로 잘못
    "솔직히": ["솔직이"],
    "똑똑히": ["똑똑이"],
    "묵묵히": ["묵묵이"],
}

# ── -스러운/-스런 혼동 ──
SUREOUN_MAP: dict[str, list[str]] = {
    "자랑스러운": ["자랑스런"],
    "사랑스러운": ["사랑스런"],
    "아름다운": ["아름다운"],  # 정상
    "명예스러운": ["명예스런"],
    "영광스러운": ["영광스런"],
    "수치스러운": ["수치스런"],
}

# ── -장이/-쟁이 혼동 ──
JANGI_JAENGI_MAP: dict[str, list[str]] = {
    # 이들은 구별이 필요하지만 혼용이 많음
    "점쟁이": ["점장이"],
    "심술쟁이": ["심술장이"],
    "멋쟁이": ["멋장이"],
}

# ── 하 탈락 관련 ──
HA_DROP_MAP: dict[str, list[str]] = {
    "넉넉지 않다": ["넉넉치 않다"],
    "탐탁지 않다": ["탐탁치 않다"],
    "녹록지 않다": ["녹록치 않다"],
    "익숙지 않다": ["익숙치 않다"],
    "섭섭지 않다": ["섭섭치 않다"],
    "서슴지 않다": ["서슴치 않다"],
    "생각건대": ["생각컨대"],
}

# ── -ㄹ런지/-ㄹ는지 혼동 ──
RYEONJI_MAP: dict[str, list[str]] = {
    "올는지": ["올런지", "올련지"],
    "있을는지": ["있을런지"],
    "갈는지": ["갈런지"],
}

# ── -ㄹ려고/-려고 혼동 ──
RYEOGO_MAP: dict[str, list[str]] = {
    "하려고": ["할려고"],
    "먹으려고": ["먹을려고"],
    "가려고": ["갈려고"],
    "보려고": ["볼려고"],
}

# ── -ㄹ걸/-ㄹ껄, -ㄹ게/-ㄹ께 혼동 ──
GEOL_GE_MAP: dict[str, list[str]] = {
    "후회할걸": ["후회할껄"],
    "해 줄게": ["해 줄께"],
    "할게": ["할께"],
    "먹을걸": ["먹을껄"],
    "할 거야": ["할 꺼야"],
    "먹을 거야": ["먹을 꺼야"],
}

# ── -뜨리다/-트리다 (복수표준어지만 혼동) ──
TTURIDA_MAP: dict[str, list[str]] = {
    "깨뜨리다": ["깨트리다"],
    "떨어뜨리다": ["떨어트리다"],
    "빠뜨리다": ["빠트리다"],
}


import re

def apply_suffix_error(text: str, rng: random.Random) -> Optional[str]:
    """
    텍스트에 접미사 오류를 적용.

    Args:
        text: 올바른 한국어 문장
        rng: 랜덤 시드 관리용 Random 인스턴스

    Returns:
        오류가 적용된 문장. 적용 가능한 패턴이 없으면 None.
    """
    all_maps = [
        I_HI_MAP, SUREOUN_MAP, JANGI_JAENGI_MAP,
        HA_DROP_MAP, RYEONJI_MAP, RYEOGO_MAP,
        GEOL_GE_MAP, TTURIDA_MAP,
    ]

    candidates = []
    for m in all_maps:
        for correct, wrongs in m.items():
            if correct in text:
                for match in re.finditer(f"(?<![가-힣]){correct}", text):
                    for wrong in wrongs:
                        candidates.append((match.start(), match.end(), wrong))

    if not candidates:
        return None

    start, end, wrong = rng.choice(candidates)
    return text[:start] + wrong + text[end:]


def get_error_count() -> int:
    """이 모듈이 가진 오류 패턴 수를 반환."""
    all_maps = [
        I_HI_MAP, SUREOUN_MAP, JANGI_JAENGI_MAP,
        HA_DROP_MAP, RYEONJI_MAP, RYEOGO_MAP,
        GEOL_GE_MAP, TTURIDA_MAP,
    ]
    return sum(len(v) for m in all_maps for v in m.values())
