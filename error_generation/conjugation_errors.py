"""
활용형 오류 — 용언 활용에서 발생하는 오류들.

-든/-던, -되/-돼, ㅡ탈락, 모음조화, ㅂ불규칙, 관형사형 등.
"""

import random
from typing import Optional


# ── -든/-던 혼동 ──
DEUN_DEON_MAP: dict[str, list[str]] = {
    "든지": ["던지"],          # 선택의 뜻일 때
    "든가": ["던가"],
    "든지 간에": ["던지 간에"],
    "하든": ["하던"],          # 선택의 뜻
    "먹든": ["먹던"],
    "가든": ["가던"],
    "오든": ["오던"],
}

# ── -되/-돼 혼동 ──
DOE_DWAE_MAP: dict[str, list[str]] = {
    "안 돼": ["안 되"],
    "돼요": ["되요"],
    "돼서": ["되서"],
    "됐다": ["됬다"],
    "되어": ["돼어"],         # 이미 줄인 건데 다시 줄이려는 오류
    "안 됩니다": ["안 됩니다"],
    "하면 돼": ["하면 되"],
    "가능해요": ["가능하세요"],    # 사물존칭 문제
    "하시면 돼요": ["하시면 되세요"],
}

# ── ㅡ 탈락 미적용 ──
EU_DROP_MAP: dict[str, list[str]] = {
    "치르다": ["치루다"],
    "치렀다": ["치뤘다"],
    "치러": ["치뤄"],
    "담그다": ["담구다"],
    "담갔다": ["담궜다"],
    "담가": ["담궈"],
    "잠그다": ["잠구다"],
    "잠갔다": ["잠궜다"],
}

# ── 모음조화 오류 (양성→음성 방향) ──
VOWEL_HARMONY_MAP: dict[str, list[str]] = {
    "가까워": ["가까와"],
    "아름다워요": ["아름다와요"],
    "아름다워서": ["아름다와서"],
    "무서워": ["무서와"],
    "고마워": ["고마와"],
    "기뻐서": ["기뻐서"],    # 정상
    # ㅂ 불규칙 활용 — 도와/고와만 '-와' (예외)
    "도와": ["도워"],  # 역방향 오류: 도와가 맞음
    "고와": ["고워"],  # 역방향 오류: 고와가 맞음
}

# ── 관형사형 오류 ──
GWANHYEONG_MAP: dict[str, list[str]] = {
    "걸맞은": ["걸맞는"],
    "알맞은": ["알맞는"],
    "모르는 척": ["모른 척"],   # 현재를 과거형으로
}

# ── 사동/피동 혼동 ──
SADONG_MAP: dict[str, list[str]] = {
    "맞히다": ["맞추다"],     # 정답을 맞히다
    "늘이다": ["늘리다"],     # 길이를 늘이다
    "줄이다": ["줄리다"],
    "높이다": ["높히다"],
}

# ── 놀라다/놀래다 혼동 ──
NOLLADA_MAP: dict[str, list[str]] = {
    "놀랐다": ["놀랬다"],
    "놀랐어": ["놀랬어"],
    "놀라다": ["놀래다"],     # 자동사를 사동사로
}

# ── 기타 활용형 오류 ──
OTHER_CONJUGATION: dict[str, list[str]] = {
    "바라": ["바래"],          # 희망의 뜻일 때
    "바랍니다": ["바랩니다"],
    "바라요": ["바래요"],
    "놓다": ["놓다"],
    "웃기는": ["웃긴"],        # 사동사인데 형용사처럼
    "빼앗다": ["빼았다"],
    "설렘": ["설레임"],
}


def apply_conjugation_error(text: str, rng: random.Random) -> Optional[str]:
    """
    텍스트에 활용형 오류를 적용.

    Args:
        text: 올바른 한국어 문장
        rng: 랜덤 시드 관리용 Random 인스턴스

    Returns:
        오류가 적용된 문장. 적용 가능한 패턴이 없으면 None.
    """
    all_maps = [
        DEUN_DEON_MAP, DOE_DWAE_MAP, EU_DROP_MAP,
        VOWEL_HARMONY_MAP, GWANHYEONG_MAP, SADONG_MAP,
        NOLLADA_MAP, OTHER_CONJUGATION,
    ]

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
    all_maps = [
        DEUN_DEON_MAP, DOE_DWAE_MAP, EU_DROP_MAP,
        VOWEL_HARMONY_MAP, GWANHYEONG_MAP, SADONG_MAP,
        NOLLADA_MAP, OTHER_CONJUGATION,
    ]
    return sum(len(v) for m in all_maps for v in m.values())
