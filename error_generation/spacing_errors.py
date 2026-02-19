"""
띄어쓰기 오류 — 붙여 써야 할 것을 띄어 쓰거나, 그 반대.

보조사, 접미사, 의존명사 등의 띄어쓰기 혼동을 다룸.
"""

import random
from typing import Optional


# (올바른 표현, 잘못된 표현) 쌍
# 붙여 써야 하는데 띄어 쓰는 경우 (JOIN→SPLIT)
JOIN_TO_SPLIT: list[tuple[str, str]] = [
    ("것뿐이다", "것 뿐이다"),
    ("것뿐이야", "것 뿐이야"),
    ("것뿐", "것 뿐"),
    ("것투성이다", "것 투성이다"),
    ("것투성이야", "것 투성이야"),
    ("것만으로도", "것 만으로도"),
    ("것만으로", "것 만으로"),
    ("수밖에", "수 밖에"),
    ("듯하다", "듯 하다"),
    ("만하다", "만 하다"),
    ("뻔하다", "뻔 하다"),
    ("척하다", "척 하다"),
    ("체하다", "체 하다"),
    ("듯싶다", "듯 싶다"),
    ("듯하지만", "듯 하지만"),
    ("듯한데", "듯 한데"),
    ("는커녕", "는 커녕"),
    ("은커녕", "은 커녕"),
]

# 띄어 써야 하는데 붙여 쓰는 경우 (SPLIT→JOIN)
SPLIT_TO_JOIN: list[tuple[str, str]] = [
    ("할 수 있다", "할수있다"),
    ("할 수 없다", "할수없다"),
    ("할 수 있는", "할수있는"),
    ("될 수 있다", "될수있다"),
    ("먹을 거야", "먹을거야"),
    ("할 거야", "할거야"),
    ("할 것이다", "할것이다"),
    ("한 것이다", "한것이다"),
    ("할 때", "할때"),
    ("할 수록", "할수록"),
    ("할 줄", "할줄"),
    ("할 만하다", "할만하다"),
    ("할 리가", "할리가"),
    ("할 법하다", "할법하다"),
    ("할 테니", "할테니"),
    ("할 텐데", "할텐데"),
    ("나올 즈음", "나올즈음"),
]

# 의존명사 '-ㄹ 걸' 관련 (띄어야 vs 붙여야)
DEPENDENT_NOUN_SPACING: list[tuple[str, str]] = [
    ("할걸", "할 걸"),      # 추측·후회의 종결어미일 때는 붙여야
    ("먹을걸", "먹을 걸"),
    ("갈걸", "갈 걸"),
]

# 날짜 마침표 띄어쓰기
DATE_SPACING: list[tuple[str, str]] = [
    ("2000. 1. 1.", "2000.1.1"),
    ("2000. 1. 1.", "2000.1.1."),
    ("2000. 12.", "2000.12"),
    ("12. 10.", "12.10"),
]


def apply_spacing_error(text: str, rng: random.Random) -> Optional[str]:
    """
    텍스트에 띄어쓰기 오류를 적용.

    Args:
        text: 올바른 한국어 문장
        rng: 랜덤 시드 관리용 Random 인스턴스

    Returns:
        오류가 적용된 문장. 적용 가능한 패턴이 없으면 None.
    """
    all_patterns = JOIN_TO_SPLIT + SPLIT_TO_JOIN + DEPENDENT_NOUN_SPACING + DATE_SPACING
    candidates = [(correct, wrong) for correct, wrong in all_patterns if correct in text]

    if not candidates:
        return None

    correct, wrong = rng.choice(candidates)
    return text.replace(correct, wrong, 1)


def get_error_count() -> int:
    """이 모듈이 가진 오류 패턴 수를 반환."""
    return (len(JOIN_TO_SPLIT) + len(SPLIT_TO_JOIN) +
            len(DEPENDENT_NOUN_SPACING) + len(DATE_SPACING))
