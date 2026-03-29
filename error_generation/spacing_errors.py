"""
띄어쓰기 오류 — 붙여 써야 할 것을 띄어 쓰거나, 그 반대.

두 가지 모드:
1. 패턴 기반: 보조사, 접미사, 의존명사 등의 고정 패턴 매칭
2. 동적 생성: MeCab 형태소 분석 기반 어절 합치기/쪼개기/의존명사 결합
"""

import random
import re
from typing import Optional


# (올바른 표현, 잘못된 표현) 쌍
# 붙여 써야 하는데 띄어 쓰는 경우 (JOIN→SPLIT)
JOIN_TO_SPLIT: list[tuple[str, str]] = [
    # ── 의존명사 뒤 보조사/접미사 (붙여야 함) ──
    ("것뿐이다", "것 뿐이다"),
    ("것뿐이야", "것 뿐이야"),
    ("것뿐", "것 뿐"),
    ("것투성이다", "것 투성이다"),
    ("것투성이야", "것 투성이야"),
    ("것만으로도", "것 만으로도"),
    ("것만으로", "것 만으로"),
    ("수밖에", "수 밖에"),

    # ── 보조 형용사/동사 (붙여 써야 하는 관용 표현) ──
    ("듯하다", "듯 하다"),
    ("듯한", "듯 한"),
    ("듯이", "듯 이"),
    ("듯싶다", "듯 싶다"),
    ("듯하지만", "듯 하지만"),
    ("듯한데", "듯 한데"),
    ("만하다", "만 하다"),
    ("만한", "만 한"),
    ("만큼", "만 큼"),
    ("뻔하다", "뻔 하다"),
    ("뻔했다", "뻔 했다"),
    ("척하다", "척 하다"),
    ("체하다", "체 하다"),

    # ── 접미사 (붙여야 함) ──
    ("는커녕", "는 커녕"),
    ("은커녕", "은 커녕"),
    ("조차도", "조차 도"),
    ("마저도", "마저 도"),
    ("밖에는", "밖에 는"),
    ("만큼은", "만큼 은"),
    ("대로는", "대로 는"),

    # ── 복합어 (붙여야 함) ──
    ("그러니까", "그러 니까"),
    ("그런데도", "그런데 도"),
    ("그렇지만", "그렇 지만"),
    ("때문에", "때문 에"),
    ("하지만", "하지 만"),
    ("그래서", "그래 서"),
    ("그리고", "그리 고"),
    ("그러나", "그러 나"),
    ("그런데", "그런 데"),
    ("하지만", "하지 만"),
    ("그래도", "그래 도"),

    # ── 합성어 (붙여야 함) ──
    ("되돌리다", "되 돌리다"),
    ("되찾다", "되 찾다"),
    ("재미있다", "재미 있다"),
    ("재미없다", "재미 없다"),
    ("맛있다", "맛 있다"),
    ("맛없다", "맛 없다"),
    ("멋있다", "멋 있다"),

    # ── 어미 (붙여야 함) ──
    ("하기는", "하기 는"),
    ("하기도", "하기 도"),
    ("하고는", "하고 는"),
    ("하면서", "하면 서"),
    ("하더라", "하더 라"),
    ("하더니", "하더 니"),
    ("했는데", "했는 데"),
    ("하는데", "하는 데"),     # 어미 '-는데'는 붙여야 (의존명사 '데'와 구분)
    ("있는데", "있는 데"),
    ("없는데", "없는 데"),
    ("인데요", "인데 요"),
    ("거든요", "거든 요"),
    ("잖아요", "잖아 요"),
    ("네요", "네 요"),
    ("군요", "군 요"),
    ("은요", "은 요"),
    ("는요", "는 요"),
]

# 띄어 써야 하는데 붙여 쓰는 경우 (SPLIT→JOIN)
SPLIT_TO_JOIN: list[tuple[str, str]] = [
    # ── 의존명사 '수' ──
    ("할 수 있다", "할수있다"),
    ("할 수 없다", "할수없다"),
    ("할 수 있는", "할수있는"),
    ("될 수 있다", "될수있다"),
    ("할 수 있어", "할수있어"),
    ("할 수 없어", "할수없어"),
    ("볼 수 있다", "볼수있다"),
    ("갈 수 있다", "갈수있다"),
    ("먹을 수 있다", "먹을수있다"),
    ("알 수 있다", "알수있다"),
    ("올 수 있다", "올수있다"),
    ("살 수 있다", "살수있다"),

    # ── 의존명사 '것/거/건' ──
    ("먹을 거야", "먹을거야"),
    ("할 거야", "할거야"),
    ("갈 거야", "갈거야"),
    ("할 것이다", "할것이다"),
    ("한 것이다", "한것이다"),
    ("할 건데", "할건데"),
    ("한 건데", "한건데"),
    ("먹을 건데", "먹을건데"),
    ("하는 건", "하는건"),
    ("먹는 건", "먹는건"),
    ("하는 거", "하는거"),
    ("먹는 거", "먹는거"),
    ("좋은 거", "좋은거"),
    ("싶은 거", "싶은거"),
    ("있는 거", "있는거"),
    ("하는 것", "하는것"),
    ("있는 것", "있는것"),

    # ── 의존명사 '때/데/지' ──
    ("할 때", "할때"),
    ("갈 때", "갈때"),
    ("먹을 때", "먹을때"),
    ("있을 때", "있을때"),
    ("없을 때", "없을때"),
    ("하는 데", "하는데"),
    ("먹는 데", "먹는데"),
    ("있는 데", "있는데"),
    ("된 지", "된지"),
    ("한 지", "한지"),

    # ── 보조용언 (띄어 쓰기 원칙) ──
    ("해 주다", "해주다"),
    ("해 줘", "해줘"),
    ("해 줄게", "해줄게"),
    ("해 줄까", "해줄까"),
    ("해 주세요", "해주세요"),
    ("해 드리다", "해드리다"),
    ("해 보다", "해보다"),
    ("해 봐", "해봐"),
    ("해 볼게", "해볼게"),
    ("가 보다", "가보다"),
    ("먹어 보다", "먹어보다"),
    ("사 주다", "사주다"),
    ("사 줘", "사줘"),
    ("고 싶다", "고싶다"),
    ("고 싶어", "고싶어"),
    ("고 싶은", "고싶은"),
    ("어 버리다", "어버리다"),
    ("아 버리다", "아버리다"),
    ("해 버리다", "해버리다"),

    # ── 수 관형사 + 단위 ──
    ("몇 년", "몇년"),
    ("몇 번", "몇번"),
    ("몇 개", "몇개"),
    ("몇 명", "몇명"),
    ("몇 시", "몇시"),
    ("몇 월", "몇월"),
    ("한 번", "한번"),
    ("두 번", "두번"),
    ("한 개", "한개"),
    ("한 명", "한명"),

    # ── 기타 빈출 ──
    ("할 수록", "할수록"),
    ("할 줄", "할줄"),
    ("할 만하다", "할만하다"),
    ("할 리가", "할리가"),
    ("할 법하다", "할법하다"),
    ("할 테니", "할테니"),
    ("할 텐데", "할텐데"),
    ("나올 즈음", "나올즈음"),
    ("안 되다", "안되다"),
    ("안 하다", "안하다"),
    ("못 하다", "못하다"),
    ("잘 하다", "잘하다"),
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


def _dynamic_spacing_error(text: str, rng: random.Random) -> Optional[str]:
    """동적 띄어쓰기 오류 생성 — 패턴 매칭 실패 시 fallback.

    세 가지 전략을 랜덤 선택:
    a) 어절 합치기: 인접 어절 2개를 붙임 (가장 흔한 띄어쓰기 오류)
    b) 어절 내부 쪼개기: MeCab 형태소 경계에서 공백 삽입
    c) 의존명사 결합: 의존명사 앞 공백 제거 ("할 수" → "할수")
    """
    words = text.split()
    if len(words) < 2:
        return None

    strategy = rng.choice(["join", "join", "split", "dep_noun"])  # join 빈도 2배

    if strategy == "join":
        # 어절 합치기: 랜덤 위치에서 인접 어절 결합
        idx = rng.randrange(len(words) - 1)
        merged = list(words)
        merged[idx] = merged[idx] + merged.pop(idx + 1)
        return " ".join(merged)

    elif strategy == "split":
        # 어절 내부 쪼개기: MeCab 형태소 경계에서 공백 삽입
        try:
            from error_generation.utils import get_mecab_offsets
            tokens = get_mecab_offsets(text)
        except Exception:
            return None

        # 같은 어절 내 인접 형태소 경계 찾기
        split_candidates = []
        for i in range(len(tokens) - 1):
            t1, t2 = tokens[i], tokens[i + 1]
            # 공백 없이 붙어있는 형태소 쌍 (같은 어절)
            if t1.end == t2.start and len(t1.surface) > 0 and len(t2.surface) > 0:
                # 명사+조사 또는 어간+어미 경계
                if (t1.pos.startswith('N') and t2.pos.startswith('J')) or \
                   (t1.pos.startswith('V') and t2.pos.startswith('E')):
                    split_candidates.append(t1.end)

        if not split_candidates:
            return None

        pos = rng.choice(split_candidates)
        return text[:pos] + " " + text[pos:]

    elif strategy == "dep_noun":
        # 의존명사(NNB) 앞 공백 제거: "할 수" → "할수"
        try:
            from error_generation.utils import get_mecab_offsets
            tokens = get_mecab_offsets(text)
        except Exception:
            return None

        dep_noun_candidates = []
        for i, t in enumerate(tokens):
            if t.pos == 'NNB' and t.start > 0 and text[t.start - 1] == ' ':
                dep_noun_candidates.append(t.start - 1)

        if not dep_noun_candidates:
            return None

        pos = rng.choice(dep_noun_candidates)
        return text[:pos] + text[pos + 1:]

    return None


def apply_spacing_error(text: str, rng: random.Random) -> Optional[str]:
    """
    텍스트에 띄어쓰기 오류를 적용.

    1단계: 패턴 기반 매칭 시도.
    2단계: 실패 시 동적 생성 (어절 합치기/쪼개기/의존명사 결합).

    Args:
        text: 올바른 한국어 문장
        rng: 랜덤 시드 관리용 Random 인스턴스

    Returns:
        오류가 적용된 문장. 적용 가능한 패턴이 없으면 None.
    """
    # 1. 패턴 기반 (30% 확률로 우선 시도)
    all_patterns = JOIN_TO_SPLIT + SPLIT_TO_JOIN + DEPENDENT_NOUN_SPACING + DATE_SPACING
    candidates = [(correct, wrong) for correct, wrong in all_patterns if correct in text]

    if candidates and rng.random() < 0.3:
        correct, wrong = rng.choice(candidates)
        return text.replace(correct, wrong, 1)

    # 2. 동적 생성
    result = _dynamic_spacing_error(text, rng)
    if result is not None:
        return result

    # 3. 패턴 기반 fallback
    if candidates:
        correct, wrong = rng.choice(candidates)
        return text.replace(correct, wrong, 1)

    return None


def get_error_count() -> int:
    """이 모듈이 가진 오류 패턴 수를 반환."""
    return (len(JOIN_TO_SPLIT) + len(SPLIT_TO_JOIN) +
            len(DEPENDENT_NOUN_SPACING) + len(DATE_SPACING))
