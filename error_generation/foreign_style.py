"""
번역체/외래어 오류 — 영어/일본어 번역투 표현.
"""

import random
from typing import Optional


# ── 영어 번역체 ──
ENGLISH_STYLE_MAP: dict[str, list[str]] = {
    # 현재진행형 번역투
    "곧 도착합니다": ["도착하고 있습니다"],
    "곧 도착해요": ["도착하고 있어요"],

    # 'have' 직역
    "시간을 보내자": ["시간을 갖자"],
    "대화를 하자": ["대화를 갖자"],

    # 'one of the most' 직역
    "아주 유명한": ["가장 유명한 것 중 하나인"],
    "매우 아름다운": ["가장 아름다운 것 중 하나인"],

    # 'because' 직역
    "이므로": ["이기 때문에"],
    "하므로": ["하기 때문에"],

    # 특성의 'have' 직역
    "특징이 있다": ["특징을 갖고 있다"],
}

# ── 일본어 번역체 ──
JAPANESE_STYLE_MAP: dict[str, list[str]] = {
    # ~に他ならない
    "다름없다": ["에 다름 아니다"],
    "다르지 않다": ["에 다름 아니다"],

    # ~という것だ
    "인 것이다": ["라는 것이다"],
    "는 것이다": ["다는 것이다"],
}

# ── 외래어 표기 오류 ──
FOREIGN_WORD_MAP: dict[str, list[str]] = {
    "가라테": ["공수"],         # 일본 무술
    "랍스터": ["로브스터"],
}


import re

def apply_foreign_style(text: str, rng: random.Random) -> Optional[str]:
    """
    텍스트에 번역체/외래어 오류를 적용.

    Args:
        text: 올바른 한국어 문장
        rng: 랜덤 시드 관리용 Random 인스턴스

    Returns:
        오류가 적용된 문장. 적용 가능한 패턴이 없으면 None.
    """
    all_maps = [ENGLISH_STYLE_MAP, JAPANESE_STYLE_MAP, FOREIGN_WORD_MAP]

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
    all_maps = [ENGLISH_STYLE_MAP, JAPANESE_STYLE_MAP, FOREIGN_WORD_MAP]
    return sum(len(v) for m in all_maps for v in m.values())
