"""
자모 규칙 기반 구어체 오류 생성.

NIKL PARA 전수 분석에서 발견된 대량 패턴을 규칙으로 처리:
1. ㅅ→ㅆ 받침 탈락: 있→잇, 했→햇, 봤→봣, 겠→겟 (620+ 변형)
2. -요→-용/-여 종결: 좋아요→좋아용/좋아여 (986+ 변형)
3. -다→-당/-앙 종결: 좋다→좋당 (다수 변형)
4. -고→-구 연결: 먹고→먹구 (420+ 변형)
"""

import random
import re
from typing import Optional


# ── 한글 유니코드 분해/조합 ──
_CHO = list("ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ")
_JUNG = list("ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ")
_JONG = [''] + list("ㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎ")

def _decompose(ch: str):
    """한글 1음절 → (초성, 중성, 종성) 인덱스. 한글 아니면 None."""
    code = ord(ch) - 0xAC00
    if code < 0 or code > 11171:
        return None
    cho = code // (21 * 28)
    jung = (code % (21 * 28)) // 28
    jong = code % 28
    return cho, jung, jong

def _compose(cho: int, jung: int, jong: int) -> str:
    """(초성, 중성, 종성) 인덱스 → 한글 1음절."""
    return chr(0xAC00 + cho * 21 * 28 + jung * 28 + jong)


def _apply_ssang_to_single(text: str, rng: random.Random) -> Optional[str]:
    """ㅆ→ㅅ 받침 오류: 있→잇, 했→햇, 봤→봣, 겠→겟, 맛있→맛잇"""
    # ㅆ 받침이 있는 음절 위치 수집
    candidates = []
    for i, ch in enumerate(text):
        d = _decompose(ch)
        if d is None:
            continue
        cho, jung, jong = d
        # 종성 ㅆ (인덱스 20)
        if _JONG[jong] == 'ㅆ':
            candidates.append(i)

    if not candidates:
        return None

    idx = rng.choice(candidates)
    cho, jung, jong = _decompose(text[idx])
    # ㅆ → ㅅ (인덱스 19→18... 아니 JONG 리스트에서 찾기)
    new_jong = _JONG.index('ㅅ')
    new_ch = _compose(cho, jung, new_jong)
    return text[:idx] + new_ch + text[idx + 1:]


def _apply_yo_to_yong(text: str, rng: random.Random) -> Optional[str]:
    """-요→-용/-여 종결 오류: 좋아요→좋아용, 같아요→같아여"""
    # '요'로 끝나는 어절 찾기
    words = text.split()
    candidates = []
    for i, w in enumerate(words):
        w_clean = re.sub(r'[.,!?~…;:]$', '', w)
        if w_clean.endswith('요') and len(w_clean) >= 2:
            candidates.append(i)

    if not candidates:
        return None

    idx = rng.choice(candidates)
    word = words[idx]
    # 구두점 분리
    m = re.match(r'^(.+?)(([.,!?~…;:])+)?$', word)
    body = m.group(1)
    punct = m.group(2) or ''

    replacement = rng.choice(['용', '여'])
    new_body = body[:-1] + replacement
    words[idx] = new_body + punct
    return ' '.join(words)


def _apply_da_to_dang(text: str, rng: random.Random) -> Optional[str]:
    """-다→-당 종결 오류: 좋다→좋당, 싶다→싶당"""
    words = text.split()
    candidates = []
    for i, w in enumerate(words):
        w_clean = re.sub(r'[.,!?~…;:]$', '', w)
        if w_clean.endswith('다') and len(w_clean) >= 2:
            candidates.append(i)

    if not candidates:
        return None

    idx = rng.choice(candidates)
    word = words[idx]
    m = re.match(r'^(.+?)(([.,!?~…;:])+)?$', word)
    body = m.group(1)
    punct = m.group(2) or ''

    new_body = body[:-1] + '당'
    words[idx] = new_body + punct
    return ' '.join(words)


def _apply_go_to_gu(text: str, rng: random.Random) -> Optional[str]:
    """-고→-구 연결 오류: 먹고→먹구, 있고→있구"""
    # '고'가 포함된 어절 (어절 끝 또는 '고요', '고는' 등)
    words = text.split()
    candidates = []
    for i, w in enumerate(words):
        w_clean = re.sub(r'[.,!?~…;:]$', '', w)
        if '고' in w_clean and len(w_clean) >= 2:
            # 어절 끝이 '고' 또는 '고요', '고는' 등
            if w_clean.endswith('고') or w_clean.endswith('고요') or \
               w_clean.endswith('고는') or w_clean.endswith('고서'):
                candidates.append(i)

    if not candidates:
        return None

    idx = rng.choice(candidates)
    word = words[idx]
    m = re.match(r'^(.+?)(([.,!?~…;:])+)?$', word)
    body = m.group(1)
    punct = m.group(2) or ''

    new_body = body.replace('고', '구', 1)  # 첫 번째 '고'만
    if new_body == body:
        return None
    words[idx] = new_body + punct
    return ' '.join(words)


def _apply_suffix_swap(text: str, rng: random.Random) -> Optional[str]:
    """빈출 접미사 변환 — NIKL PARA 전수 분석에서 발견된 어미 변형 규칙.

    각 규칙은 어절 끝(구두점 제외)에서 매칭:
    - 잖아→자나 (89회), 데→디 (68회), 지→징 (48회)
    - 더라→드라 (46회), 어→엉 (43회), 데→뎅 (58회)
    - 게→께 (31회), 죠→져 (31회)
    """
    # (정답 접미사, 오류 접미사, 가중치)
    _SUFFIX_RULES = [
        ("잖아", "자나", 3),
        ("잖아요", "자나요", 3),
        ("는데", "는디", 2),
        ("은데", "은디", 2),
        ("인데", "인디", 2),
        ("는데", "는뎅", 2),
        ("은데", "은뎅", 2),
        ("인데", "인뎅", 2),
        ("지", "징", 2),
        ("더라", "드라", 2),
        ("죠", "져", 2),
        ("게", "께", 1),
        ("게요", "께요", 1),
    ]

    words = text.split()
    candidates = []
    for i, w in enumerate(words):
        m = re.match(r'^(.+?)(([.,!?~…;:])+)?$', w)
        body = m.group(1)
        for correct_suf, wrong_suf, weight in _SUFFIX_RULES:
            if body.endswith(correct_suf) and len(body) > len(correct_suf):
                candidates.append((i, body, correct_suf, wrong_suf, m.group(2) or '', weight))

    if not candidates:
        return None

    indices = list(range(len(candidates)))
    weights = [c[5] for c in candidates]
    [chosen_idx] = rng.choices(indices, weights=weights, k=1)
    i, body, correct_suf, wrong_suf, punct, _ = candidates[chosen_idx]

    new_body = body[:-len(correct_suf)] + wrong_suf
    words[i] = new_body + punct
    return ' '.join(words)


# ── 통합 인터페이스 ──

_RULES = [
    (_apply_ssang_to_single, 3),  # ㅆ→ㅅ (가장 빈출)
    (_apply_yo_to_yong,      3),  # -요→-용/-여
    (_apply_da_to_dang,      1),  # -다→-당
    (_apply_go_to_gu,        2),  # -고→-구
    (_apply_suffix_swap,     3),  # 접미사 변환 (잖아→자나, 데→디 등)
]


def apply_jamo_rule(text: str, rng: random.Random) -> Optional[str]:
    """자모 규칙 기반 구어체 오류 생성."""
    fns = [fn for fn, _ in _RULES]
    weights = [w for _, w in _RULES]

    for _ in range(5):
        [chosen] = rng.choices(fns, weights=weights, k=1)
        result = chosen(text, rng)
        if result is not None and result != text:
            return result

    return None


def get_error_count() -> int:
    return len(_RULES)
