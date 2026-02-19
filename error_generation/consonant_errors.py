"""
자음/받침 혼동 오류 — 받침 ㄷ/ㅌ/ㅅ/ㅈ/ㅊ/ㅎ 혼동, 겹받침, 구개음화 등.

받침이 같은 소리로 발음되어 혼동하는 경우를 다룸.
"""

import random
from typing import Optional


# ── 받침 혼동 (같은 발음, 다른 표기) ──
BATCHIM_CONFUSION: dict[str, list[str]] = {
    # ㅅ/ㅌ/ㅈ/ㅊ 받침 → [ㄷ] 발음
    "윷놀이": ["윳놀이"],
    "숯": ["숫"],            # 나무 연료
    "젓갈": ["젖갈"],
    "빛": ["빗"],            # 광선
    "낱알": ["낟알"],        # 하나하나의 알 의미일 때
    "붓다": ["붇다"],        # 액체를 따르다
    "핥다": ["햝다"],
    "밟다": ["밥다"],

    # ㄱ/ㄲ/ㅋ 혼동
    "부엌": ["부억"],
    "깎다": ["깍다"],
    "꺾다": ["꺽다"],

    # ㄹ 탈락 관련 과잉적용
    "거친": ["거칠은"],
    "나는": ["날으는"],       # '날다'의 활용
    "가는": ["갈으는"],       # '갈다'(교체)의 활용

    # 겹받침 오류
    "밝히다": ["발키다"],
    "읽다": ["익다"],
    "없다": ["업다"],
    "흙": ["흑"],

    # 구개음화 관련 (겉표기 혼동)
    "같이": ["가치"],         # 역방향은 드물지만 발음 영향
    "굳이": ["구지"],         # 대표적 구개음화 혼동
    "맏이": ["마지"],
    "해돋이": ["해도지"],
    "굳히다": ["구치다"],

    # ㅂ/ㅍ 혼동
    "숲": ["숩"],
    "입다": ["잎다"],

    # 기타 자음 관련
    "꽃": ["꼳"],
    "빛나다": ["빗나다"],
    "갖다": ["갓다"],
}


def apply_consonant_error(text: str, rng: random.Random) -> Optional[str]:
    """
    텍스트에 자음/받침 혼동 오류를 적용.

    Args:
        text: 올바른 한국어 문장
        rng: 랜덤 시드 관리용 Random 인스턴스

    Returns:
        오류가 적용된 문장. 적용 가능한 패턴이 없으면 None.
    """
    candidates: list[tuple[str, str]] = []
    for correct, wrongs in BATCHIM_CONFUSION.items():
        if correct in text:
            for wrong in wrongs:
                candidates.append((correct, wrong))

    if not candidates:
        return None

    correct, wrong = rng.choice(candidates)
    return text.replace(correct, wrong, 1)


def get_error_count() -> int:
    """이 모듈이 가진 오류 패턴 수를 반환."""
    return sum(len(v) for v in BATCHIM_CONFUSION.values())
