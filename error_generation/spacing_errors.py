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

    # MeCab 기반 전략: 공백 제거(join)와 공백 삽입(split) 후보를 먼저 수집
    try:
        from error_generation.utils import get_mecab_offsets
        tokens = get_mecab_offsets(text)
    except Exception:
        tokens = None

    # ── 공백 제거 후보 (띄어 쓴 곳을 붙이기) ──
    join_candidates = []
    if tokens:
        for i, t in enumerate(tokens):
            if t.start > 0 and text[t.start - 1] == ' ':
                pos_tag = t.pos.split('+')[0]
                prev_pos = tokens[i - 1].pos.split('+')[-1] if i > 0 else ''
                # 의존명사 앞: "먹을 거야" → "먹을거야"
                if pos_tag == 'NNB':
                    join_candidates.append((t.start - 1, 4))
                # 보조용언 앞: "해 주세요" → "해주세요"
                elif pos_tag == 'VX':
                    join_candidates.append((t.start - 1, 4))
                # 단위명사 앞: "몇 년" → "몇년"
                elif pos_tag == 'NNBC':
                    join_candidates.append((t.start - 1, 3))
                # 접미사 앞: 붙여야 함
                elif pos_tag in ('XSN', 'XSV', 'XSA'):
                    join_candidates.append((t.start - 1, 3))
                # 명사+명사 복합어: "외부 액정" → "외부액정" (1위 빈출)
                elif prev_pos.startswith('N') and pos_tag in ('NNG', 'NNP'):
                    join_candidates.append((t.start - 1, 4))
                # 관형어+명사: "예쁜 집" → "예쁜집"
                elif prev_pos == 'ETM' and pos_tag.startswith('N'):
                    join_candidates.append((t.start - 1, 3))
                # 부사+동사/형용사: "빨리 해" → "빨리해"
                elif prev_pos == 'MAG' and pos_tag in ('VV', 'VA'):
                    join_candidates.append((t.start - 1, 3))
                # 명사+동사: "추천 해" → "추천해"
                elif prev_pos.startswith('N') and pos_tag in ('VV', 'VA', 'XSV'):
                    join_candidates.append((t.start - 1, 3))
                # 어미+동사: "먹고 싶다" → "먹고싶다"
                elif prev_pos.startswith('E') and pos_tag in ('VV', 'VA', 'VX'):
                    join_candidates.append((t.start - 1, 2))
                # 조사+명사/부사/동사: 일반적인 어절 경계
                elif prev_pos.startswith('J') and pos_tag in ('NNG', 'NNP', 'MAG', 'VV', 'VA'):
                    join_candidates.append((t.start - 1, 2))
                # 부사+부사: "너무 많이" → "너무많이"
                elif prev_pos == 'MAG' and pos_tag == 'MAG':
                    join_candidates.append((t.start - 1, 1))
                # 일반 어절 합치기 (낮은 가중치)
                else:
                    join_candidates.append((t.start - 1, 1))

    # ── 공백 삽입 후보 (붙여 쓴 곳을 띄우기) ──
    split_candidates = []
    if tokens:
        for i in range(len(tokens) - 1):
            t1, t2 = tokens[i], tokens[i + 1]
            if t1.end == t2.start and len(t1.surface) > 0 and len(t2.surface) > 0:
                p1 = t1.pos.split('+')[-1]  # 마지막 복합 태그
                p2 = t2.pos.split('+')[0]   # 첫 복합 태그
                weight = 1
                # 명사+조사: "아이들이" → "아이들 이"
                if p1.startswith('N') and p2.startswith('J'):
                    weight = 2
                # 어간+어미: "먹었다" → "먹 었다"
                elif p1.startswith('V') and p2.startswith('E'):
                    weight = 2
                # 어미+보조용언: "먹고있다" → "먹고 있다"
                elif p1.startswith('E') and p2.startswith('V'):
                    weight = 2
                # 명사+접미사: "아이들" → "아이 들"
                elif p1.startswith('N') and p2.startswith('X'):
                    weight = 3
                # 명사+동사화접미사: "공부하다" → "공부 하다"
                elif p1.startswith('N') and p2 == 'XSV':
                    weight = 3
                # 용언+선어말어미: "먹었" → "먹 었"
                elif p1.startswith('V') and p2 == 'EP':
                    weight = 1
                # 종결어미 앞: "걸려요" → "걸려 요"
                elif p2 == 'EF':
                    weight = 2
                else:
                    continue
                split_candidates.append((t1.end, weight))

    # 전략 선택: join 60%, split 40% (공백 제거가 더 흔한 오류)
    has_join = bool(join_candidates)
    has_split = bool(split_candidates)

    if not has_join and not has_split:
        # fallback: 단순 어절 합치기
        if len(words) >= 2:
            idx = rng.randrange(len(words) - 1)
            merged = list(words)
            merged[idx] = merged[idx] + merged.pop(idx + 1)
            return " ".join(merged)
        return None

    if has_join and has_split:
        strategy = "join" if rng.random() < 0.6 else "split"
    elif has_join:
        strategy = "join"
    else:
        strategy = "split"

    if strategy == "join":
        positions = [c[0] for c in join_candidates]
        weights = [c[1] for c in join_candidates]
        [pos] = rng.choices(positions, weights=weights, k=1)
        return text[:pos] + text[pos + 1:]
    else:
        positions = [c[0] for c in split_candidates]
        weights = [c[1] for c in split_candidates]
        [pos] = rng.choices(positions, weights=weights, k=1)
        return text[:pos] + " " + text[pos:]


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
