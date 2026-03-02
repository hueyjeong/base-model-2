"""
사이시옷 오류 — 사이시옷을 잘못 넣거나 빼는 경우.

한자어+한자어, 순우리말+한자어 등 사이시옷 규칙 혼동.
"""

import random
from typing import Optional


# ── 사이시옷 불필요 (넣으면 안 되는데 넣는 경우) ──
SAISIOT_UNNECESSARY: dict[str, list[str]] = {
    "개수": ["갯수"],
    "대가": ["댓가"],
    "대구": ["댓구"],          # 대답 의미
    "대꾸": ["댓구"],          # 대답 의미
    "뒤태": ["뒷태"],
    "마구간": ["마굿간"],
    "반대말": ["반댓말"],
    "시가": ["싯가"],          # 시장 가격
    "초점": ["촛점"],
    "해님": ["햇님"],          # 님은 접사라 사이시옷 불가
    "오랜만": ["오랫만"],
}

# ── 사이시옷 필요 (빼면 안 되는데 빼는 경우) ──
SAISIOT_REQUIRED: dict[str, list[str]] = {
    "최솟값": ["최소값"],
    "최댓값": ["최대값"],
    "절댓값": ["절대값"],
    "변숫값": ["변수값"],
    "대푯값": ["대표값"],
    "송홧가루": ["송화가루"],
    "막냇동생": ["막내동생"],
    "봇물": ["보물"],          # 한자어+고유어
}

# ── 한자어+한자어 예외 6개 (사이시옷을 쓰는 한자어 합성어) ──
# 곳간, 셋방, 숫자, 찻간, 툇간, 횟수
# 이들을 사이시옷 없이 쓰는 오류
HANJA_EXCEPTION: dict[str, list[str]] = {
    "곳간": ["고간"],
    "셋방": ["세방"],
    "숫자": ["수자"],
    "찻간": ["차간"],
    "횟수": ["회수"],          # '회수(回收)'와도 혼동
}


import re

def apply_saisiot_error(text: str, rng: random.Random) -> Optional[str]:
    """
    텍스트에 사이시옷 오류를 적용.

    Args:
        text: 올바른 한국어 문장
        rng: 랜덤 시드 관리용 Random 인스턴스

    Returns:
        오류가 적용된 문장. 적용 가능한 패턴이 없으면 None.
    """
    all_maps = [SAISIOT_UNNECESSARY, SAISIOT_REQUIRED, HANJA_EXCEPTION]

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
    all_maps = [SAISIOT_UNNECESSARY, SAISIOT_REQUIRED, HANJA_EXCEPTION]
    return sum(len(v) for m in all_maps for v in m.values())
