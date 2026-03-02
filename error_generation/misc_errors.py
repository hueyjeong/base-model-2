"""
기타 오류 — 문장부호, 숫자 표현, 유행어, 기타 혼동 등.
"""

import random
from typing import Optional


# ── 숫자/시간 관련 혼동 ──
NUMBER_MAP: dict[str, list[str]] = {
    "사흘": ["3일"],            # 사흘은 3일이 맞지만 4일로 혼동
    "나흘": ["4일"],            # 위와 동일
    "하나도": ["1도"],          # 유행어 혼동
    "이튿날": ["이틀날"],       # 다음날 의미
    "금일": ["금요일"],         # 今日을 금요일로 혼동
    "내달": ["이번 달"],        # 來달을 내(內)달로 혼동
    "열흘": ["십흘", "10흘"],
}

# ── 문장부호 오류 ──
PUNCTUATION_MAP: dict[str, list[str]] = {
    # 큰따옴표 + 인용 조사
    '"라고 말했다': ['"고 말했다'],
    '"이라고 했다': ['"이다고 했다'],
}

# ── 한글/한국어 혼동 ──
HANGUL_HANGUGEO_MAP: dict[str, list[str]] = {
    "한국어": ["한글"],          # 한국어(언어)를 한글(글자)로
}

# ── 감탄 표현 오류 ──
EXCLAMATION_MAP: dict[str, list[str]] = {
    "재미있는 거라니": ["재미있는 거였다니"],
    "약하다니": ["약했다니"],
    "있다니": ["있었다니"],
}

# ── 역대급 등 유행어 ──
SLANG_MAP: dict[str, list[str]] = {
    "역대 최고": ["역대급"],
    "기록적인": ["역대급"],
}

# ── 존칭/높임 오류 ──
HONORIFIC_MAP: dict[str, list[str]] = {
    # 사물존칭
    "가능해요": ["가능하세요"],
    "나왔습니다": ["나오셨습니다"],
    # 잘못된 '-셔서' 표기
    "보셔서": ["보셨어"],
    "하셔서": ["하셨어"],
    "드셔서": ["드셨어"],
}

# ── 완전, 정말 등 부사 오용 ──
ADVERB_MISUSE_MAP: dict[str, list[str]] = {
    "정말 좋다": ["완전 좋다"],    # '완전'은 명사
}

# ── 기타 혼동 ──
MISC_MAP: dict[str, list[str]] = {
    "가능한 한": ["가능한"],       # '가능한 한 빨리'에서 '한' 누락
    "어떡해": ["어떻해"],
    "애걔": ["에게"],
    "에계": ["에계"],
    "죔죔": ["잼잼"],
    "맞는다": ["맞다"],          # 동사 현재형 - 이제 둘 다 가능
    "모르는": ["모른"],          # 현재형을 과거형으로
    "위하는": ["위한"],          # 현재형을 과거형으로
    "오래다": ["오래이다"],
}


import re

def apply_misc_error(text: str, rng: random.Random) -> Optional[str]:
    """
    텍스트에 기타 오류를 적용.

    Args:
        text: 올바른 한국어 문장
        rng: 랜덤 시드 관리용 Random 인스턴스

    Returns:
        오류가 적용된 문장. 적용 가능한 패턴이 없으면 None.
    """
    all_maps = [
        NUMBER_MAP, PUNCTUATION_MAP, HANGUL_HANGUGEO_MAP,
        EXCLAMATION_MAP, SLANG_MAP, HONORIFIC_MAP,
        ADVERB_MISUSE_MAP, MISC_MAP,
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
        NUMBER_MAP, PUNCTUATION_MAP, HANGUL_HANGUGEO_MAP,
        EXCLAMATION_MAP, SLANG_MAP, HONORIFIC_MAP,
        ADVERB_MISUSE_MAP, MISC_MAP,
    ]
    return sum(len(v) for m in all_maps for v in m.values())
