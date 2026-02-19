"""
겹말/이중표현 오류 — 의미가 중복되는 표현과 이중 피동.
"""

import random
from typing import Optional


# ── 겹말 (의미 중복) ──
# 올바른 표현 → [겹말이 포함된 잘못된 표현들]
REDUNDANCY_MAP: dict[str, list[str]] = {
    "역 앞": ["역전 앞"],
    "역전에서": ["역전 앞에서"],
    "검은색": ["검정색"],
    "검정": ["검정색"],
    "빨간색": ["빨강색"],
    "과반수": ["과반수 이상"],
    "낙엽이 지다": ["낙엽이 떨어지다"],
    "나뭇잎이 떨어지다": ["낙엽이 떨어지다"],
    "전장": ["전장터"],
}

# ── 이중 피동 ──
DOUBLE_PASSIVE_MAP: dict[str, list[str]] = {
    "잊히다": ["잊혀지다"],
    "잊어지다": ["잊혀지다"],
    "믿기다": ["믿겨지다"],
    "믿어지다": ["믿겨지다"],
    "짜이다": ["짜여지다"],
    "짜지다": ["짜여지다"],
    "불리다": ["불리우다"],
    "쓰이다": ["씌이다"],
}

# ── 이중 사동 / 과잉 피동 ──
OVER_PASSIVE_MAP: dict[str, list[str]] = {
    "소개해 줘": ["소개시켜 줘"],
    "환기하다": ["환기시키다"],
    "소개합니다": ["소개시킵니다"],
}


def apply_double_expression(text: str, rng: random.Random) -> Optional[str]:
    """
    텍스트에 겹말/이중표현 오류를 적용.

    Args:
        text: 올바른 한국어 문장
        rng: 랜덤 시드 관리용 Random 인스턴스

    Returns:
        오류가 적용된 문장. 적용 가능한 패턴이 없으면 None.
    """
    all_maps = [REDUNDANCY_MAP, DOUBLE_PASSIVE_MAP, OVER_PASSIVE_MAP]

    candidates: list[tuple[str, str]] = []
    for m in all_maps:
        for correct, wrongs in m.items():
            if correct in text:
                for wrong in wrongs:
                    candidates.append((correct, wrong))

    if not candidates:
        return None

    correct, wrong = rng.choice(candidates)
    return text.replace(correct, wrong, 1)


def get_error_count() -> int:
    """이 모듈이 가진 오류 패턴 수를 반환."""
    all_maps = [REDUNDANCY_MAP, DOUBLE_PASSIVE_MAP, OVER_PASSIVE_MAP]
    return sum(len(v) for m in all_maps for v in m.values())
