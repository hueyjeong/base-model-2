"""
활용형 오류 — 용언 활용에서 발생하는 오류들.

두 가지 모드:
1. 패턴 기반: -든/-던, -되/-돼, ㅡ탈락, 모음조화 등 고정 패턴
2. 동적 생성: MeCab 형태소 분석 기반 어미 교체

동적 규칙: -았/-었 혼동, -는/-은 관형사형 혼동, -아/-어 혼동 등.
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


import re

# ── 동적 어미 교체 규칙 ──

# 어미(E*) 토큰 전체 매칭 규칙 — surface 전체를 교체
_EOMI_SWAP_RULES: dict[str, list[str]] = {
    # 시제/상 혼동
    "었": ["았"],           # 과거 모음조화 혼동
    "았": ["었"],
    # 관형사형 혼동
    "는": ["은"],           # 현재/과거 관형사형
    "은": ["는"],
    "던": ["든"],           # 회상/선택 혼동
    "든": ["던"],
    # 연결어미 혼동
    "어": ["아"],           # 모음조화
    "아": ["어"],
    "며": ["고"],           # 나열 연결어미
    "고": ["며"],
    "지만": ["는데"],       # 대조 연결어미
    "면": ["며"],           # 조건 → 나열 혼동
    "면서": ["며"],         # 동시 → 나열 혼동
    "게": ["도록"],         # 결과/목적 혼동
    "도록": ["게"],
    # 종결어미 혼동
    "다": ["다고"],         # 평서 → 인용 (한 글자 추가)
    "다고": ["다"],         # 인용 → 평서 (한 글자 제거)
}

# V*/XS* 토큰 접미 매칭 규칙 — surface 끝에서 매칭 (surface > pattern 길이)
_VERB_SUFFIX_RULES: list[tuple[str, str]] = [
    ("았다", "었다"),
    ("었다", "았다"),
    ("았어", "었어"),
    ("었어", "았어"),
    ("았는데", "었는데"),
    ("었는데", "았는데"),
    ("아서", "어서"),
    ("어서", "아서"),
    ("아요", "어요"),
    ("어요", "아요"),
]


def _dynamic_conjugation_error(text: str, rng: random.Random) -> Optional[str]:
    """MeCab 기반 동적 활용형 오류 생성.

    두 가지 전략:
    1. 어미(E*) 토큰 직접 교체 — 독립 어미 토큰의 surface 전체를 스왑
    2. 동사(V*/XS*) 토큰 접미 교체 — 어절 끝의 활용형 변환
    """
    try:
        from error_generation.utils import get_mecab_offsets
        tokens = get_mecab_offsets(text)
    except Exception:
        return None

    candidates = []

    for t in tokens:
        # 전략 1: 어미(E*) 토큰 전체 매칭
        if t.pos.startswith('E') and t.surface in _EOMI_SWAP_RULES:
            for replacement in _EOMI_SWAP_RULES[t.surface]:
                candidates.append((t.start, t.end, replacement))

        # 전략 2: 동사/접미사 토큰 접미 매칭
        if t.pos.startswith(('V', 'XS')):
            for pattern, replacement in _VERB_SUFFIX_RULES:
                if t.surface.endswith(pattern) and len(t.surface) > len(pattern):
                    new_surface = t.surface[:-len(pattern)] + replacement
                    if new_surface != t.surface:
                        candidates.append((t.start, t.end, new_surface))

    if not candidates:
        return None

    start, end, new_surface = rng.choice(candidates)
    return text[:start] + new_surface + text[end:]


def apply_conjugation_error(text: str, rng: random.Random) -> Optional[str]:
    """
    텍스트에 활용형 오류를 적용.

    1단계: 패턴 기반 매칭 시도.
    2단계: 실패 시 동적 생성 (MeCab 기반 어미 교체).

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
    return _dynamic_conjugation_error(text, rng)


def get_error_count() -> int:
    """이 모듈이 가진 오류 패턴 수를 반환."""
    all_maps = [
        DEUN_DEON_MAP, DOE_DWAE_MAP, EU_DROP_MAP,
        VOWEL_HARMONY_MAP, GWANHYEONG_MAP, SADONG_MAP,
        NOLLADA_MAP, OTHER_CONJUGATION,
    ]
    return sum(len(v) for m in all_maps for v in m.values())
