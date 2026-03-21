"""
조사 오류 — 격조사 혼동.

~로서/~로써, ~에/~의, 사물존칭 등.
"""

import random
from typing import Optional


# ── ~로서/~로써 혼동 ──
ROSEO_ROSSEO_MAP: dict[str, list[str]] = {
    "학생으로서": ["학생으로써"],
    "사람으로서": ["사람으로써"],
    "국민으로서": ["국민으로써"],
    "대표로서": ["대표로써"],
    "교사로서": ["교사로써"],
    "도구로써": ["도구로서"],     # 역방향도 혼동
    "수단으로써": ["수단으로서"],
}

# ── ~에/~의 혼동 ──
E_UI_MAP: dict[str, list[str]] = {
    "문맥상의": ["문맥상"],   # '의' 누락
    "최소한의": ["최소한"],
    "나름의": ["나름"],
}

# ── 기타 조사 오류 ──
OTHER_PARTICLE_MAP: dict[str, list[str]] = {
    # 인용 조사
    "라고": ["다고"],          # '이라고'를 '이다고'로
    "이라고": ["이다고"],

    # 커녕 조사 (띄어쓰기 → spacing_errors 에도 있지만 형태 혼동)
    "마저": ["마져"],

    # 사물존칭 — 사물에 '~시' 붙이기
    "나오셨습니다": ["나왔습니다"],  # 역방향
    "결제됩니다": ["결제되십니다"],

    # ~으므로/~음으로
    "이므로": ["임으로"],     # 이유일 때
    "하므로": ["함으로"],
}


import re

# ── 동적 조사 교체 규칙 ──
# (조사, 대체 조사) — MeCab 조사(J*) 토큰에서 매칭
_DYNAMIC_JOSA_RULES: list[tuple[str, str]] = [
    # 주격 조사 자음/모음 뒤 혼동
    ("이", "가"),
    ("가", "이"),
    # 목적격 조사 자음/모음 뒤 혼동
    ("을", "를"),
    ("를", "을"),
    # 보조사 혼동
    ("은", "는"),
    ("는", "은"),
    # 장소 조사 혼동
    ("에서", "에"),
    ("에", "에서"),
    # 방향 조사 혼동
    ("으로", "로"),
    ("로", "으로"),
    # 인용 조사 혼동
    ("라고", "다고"),
    ("이라고", "이다고"),
    # 도구/자격 혼동
    ("로서", "로써"),
    ("로써", "로서"),
]


def _dynamic_particle_error(text: str, rng: random.Random) -> Optional[str]:
    """MeCab 기반 동적 조사 교체."""
    try:
        from error_generation.utils import get_mecab_offsets
        tokens = get_mecab_offsets(text)
    except Exception:
        return None

    # 조사 토큰 찾기 (J로 시작하는 POS)
    josa_tokens = [t for t in tokens if t.pos.startswith('J')]
    if not josa_tokens:
        return None

    # 교체 가능한 후보 수집
    candidates = []
    for t in josa_tokens:
        for pattern, replacement in _DYNAMIC_JOSA_RULES:
            if t.surface == pattern:
                candidates.append((t.start, t.end, replacement))

    if not candidates:
        return None

    start, end, replacement = rng.choice(candidates)
    return text[:start] + replacement + text[end:]


def apply_particle_error(text: str, rng: random.Random) -> Optional[str]:
    """
    텍스트에 조사 오류를 적용.

    1단계: 패턴 기반 매칭 시도.
    2단계: 실패 시 MeCab 기반 동적 조사 교체.

    Args:
        text: 올바른 한국어 문장
        rng: 랜덤 시드 관리용 Random 인스턴스

    Returns:
        오류가 적용된 문장. 적용 가능한 패턴이 없으면 None.
    """
    all_maps = [ROSEO_ROSSEO_MAP, E_UI_MAP, OTHER_PARTICLE_MAP]

    # 1. 패턴 기반
    candidates = []
    for m in all_maps:
        for correct, wrongs in m.items():
            if correct in text:
                for match in re.finditer(f"(?<![가-힣]){correct}", text):
                    for wrong in wrongs:
                        candidates.append((match.start(), match.end(), wrong))

    if candidates:
        start, end, wrong = rng.choice(candidates)
        return text[:start] + wrong + text[end:]

    # 2. 동적 생성
    return _dynamic_particle_error(text, rng)


def get_error_count() -> int:
    """이 모듈이 가진 오류 패턴 수를 반환."""
    all_maps = [ROSEO_ROSSEO_MAP, E_UI_MAP, OTHER_PARTICLE_MAP]
    return sum(len(v) for m in all_maps for v in m.values())
