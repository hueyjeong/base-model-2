"""
이형문자(Heterograph) 노이즈 생성 모듈.

유사 발음이지만 다른 철자인 음절 치환.
종성(받침) 혼동과 모음 혼동을 체계적으로 처리.

종성 혼동:
- ㄱ/ㅋ/ㄲ  → 같은 [k] 발음 (예: "부엌"→"부억")
- ㄷ/ㅌ/ㅅ/ㅆ/ㅈ/ㅊ → 같은 [t] 발음 (예: "빛"→"빗")
- ㅂ/ㅍ → 같은 [p] 발음 (예: "앞"→"압")

모음 혼동:
- ㅐ/ㅔ → [e] 발음 합류 (예: "게"→"개")
- ㅒ/ㅖ → [je] 발음 합류 (예: "예상"→"얘상")
- ㅚ/ㅙ/ㅞ → [we] 발음 합류 (예: "회"→"훼")

사용 예시:
    from error_generation.heterograph import apply_heterograph_error
    import random
    result = apply_heterograph_error("빛이 났다", random.Random(42))
    # → "빗이 났다"
"""

import random
from typing import Optional

# ── 한글 유니코드 상수 ──
_HANGUL_BASE = 0xAC00
_HANGUL_END = 0xD7A3
_JUNG_COUNT = 21
_JONG_COUNT = 28

# 초성 목록 (19개)
CHOSEONG = [
    "ㄱ", "ㄲ", "ㄴ", "ㄷ", "ㄸ", "ㄹ", "ㅁ", "ㅂ", "ㅃ", "ㅅ",
    "ㅆ", "ㅇ", "ㅈ", "ㅉ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ",
]

# 중성 목록 (21개)
JUNGSEONG = [
    "ㅏ", "ㅐ", "ㅑ", "ㅒ", "ㅓ", "ㅔ", "ㅕ", "ㅖ", "ㅗ", "ㅘ",
    "ㅙ", "ㅚ", "ㅛ", "ㅜ", "ㅝ", "ㅞ", "ㅟ", "ㅠ", "ㅡ", "ㅢ",
    "ㅣ",
]

# 종성 목록 (28개, 0번은 종성 없음)
JONGSEONG = [
    "", "ㄱ", "ㄲ", "ㄳ", "ㄴ", "ㄵ", "ㄶ", "ㄷ", "ㄹ", "ㄺ",
    "ㄻ", "ㄼ", "ㄽ", "ㄾ", "ㄿ", "ㅀ", "ㅁ", "ㅂ", "ㅄ", "ㅅ",
    "ㅆ", "ㅇ", "ㅈ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ",
]

# 인덱스 매핑
_JUNG_IDX = {j: i for i, j in enumerate(JUNGSEONG)}
_JONG_IDX = {j: i for i, j in enumerate(JONGSEONG)}

# ── 혼동 그룹 정의 ──

# 종성 혼동: 같은 발음의 받침끼리 (단일 종성만)
FINAL_CONFUSION_GROUPS: list[list[str]] = [
    ["ㄱ", "ㅋ", "ㄲ"],                      # [k] 발음
    ["ㄷ", "ㅌ", "ㅅ", "ㅆ", "ㅈ", "ㅊ", "ㅎ"],  # [t] 발음
    ["ㅂ", "ㅍ"],                             # [p] 발음
]

# 종성 → 혼동 그룹 (빠른 조회용)
_FINAL_TO_GROUP: dict[str, list[str]] = {}
for group in FINAL_CONFUSION_GROUPS:
    for jamo in group:
        _FINAL_TO_GROUP[jamo] = group

# 모음 혼동: 같은 발음으로 합류되는 모음끼리
VOWEL_CONFUSION_GROUPS: list[list[str]] = [
    ["ㅐ", "ㅔ"],           # [e] 발음 합류
    ["ㅒ", "ㅖ"],           # [je] 발음 합류
    ["ㅚ", "ㅙ", "ㅞ"],     # [we] 발음 합류
]

# 모음 → 혼동 그룹 (빠른 조회용)
_VOWEL_TO_GROUP: dict[str, list[str]] = {}
for group in VOWEL_CONFUSION_GROUPS:
    for jamo in group:
        _VOWEL_TO_GROUP[jamo] = group


def _decompose(char: str) -> tuple[int, int, int] | None:
    """한글 완성형 글자를 (초성, 중성, 종성) 인덱스로 분해."""
    code = ord(char)
    if code < _HANGUL_BASE or code > _HANGUL_END:
        return None
    offset = code - _HANGUL_BASE
    cho = offset // (_JUNG_COUNT * _JONG_COUNT)
    jung = (offset % (_JUNG_COUNT * _JONG_COUNT)) // _JONG_COUNT
    jong = offset % _JONG_COUNT
    return cho, jung, jong


def _compose(cho: int, jung: int, jong: int) -> str:
    """(초성, 중성, 종성) 인덱스로 한글 완성형 글자 조합."""
    return chr(_HANGUL_BASE + cho * _JUNG_COUNT * _JONG_COUNT + jung * _JONG_COUNT + jong)


def apply_heterograph_error(text: str, rng: random.Random) -> Optional[str]:
    """이형문자 치환 — 유사 발음 음절 내 자모 교체.

    텍스트에서 혼동 가능한 음절을 찾아 1개를 치환.
    종성 혼동과 모음 혼동 모두 후보에 포함.

    Args:
        text: 올바른 한국어 문장
        rng: random.Random 인스턴스

    Returns:
        오류가 적용된 문장, 또는 None (적용 불가 시)
    """
    # 후보 수집: (위치, 변환 함수) 튜플
    candidates: list[tuple[int, int, int, int]] = []
    # (char_idx, new_cho, new_jung, new_jong) 형태

    for i, ch in enumerate(text):
        decomposed = _decompose(ch)
        if decomposed is None:
            continue

        cho, jung, jong = decomposed
        jung_name = JUNGSEONG[jung]
        jong_name = JONGSEONG[jong]

        # 종성 혼동 후보
        if jong_name and jong_name in _FINAL_TO_GROUP:
            group = _FINAL_TO_GROUP[jong_name]
            for alt in group:
                alt_idx = _JONG_IDX[alt]
                if alt_idx != jong:
                    candidates.append((i, cho, jung, alt_idx))

        # 모음 혼동 후보
        if jung_name in _VOWEL_TO_GROUP:
            group = _VOWEL_TO_GROUP[jung_name]
            for alt in group:
                alt_idx = _JUNG_IDX[alt]
                if alt_idx != jung:
                    candidates.append((i, cho, alt_idx, jong))

    if not candidates:
        return None

    # 랜덤 선택
    char_idx, new_cho, new_jung, new_jong = rng.choice(candidates)
    new_char = _compose(new_cho, new_jung, new_jong)

    return text[:char_idx] + new_char + text[char_idx + 1:]


def get_error_count() -> int:
    """혼동 그룹 내 치환 쌍 수 (통계용)."""
    count = 0
    # 종성: 각 그룹의 순열 수
    for group in FINAL_CONFUSION_GROUPS:
        n = len(group)
        count += n * (n - 1)  # 자기 자신 제외
    # 모음: 각 그룹의 순열 수
    for group in VOWEL_CONFUSION_GROUPS:
        n = len(group)
        count += n * (n - 1)
    return count


if __name__ == "__main__":
    # 단독 테스트
    test_texts = [
        "예상하지 못했다",
        "부엌에서 요리했다",
        "빛이 났다",
        "앞에서 기다렸다",
        "세게 밀었다",
        "돼지고기를 먹었다",
    ]

    rng = random.Random(42)
    print("=== 이형문자(Heterograph) 노이즈 테스트 ===")
    for t in test_texts:
        result = apply_heterograph_error(t, rng)
        print(f"  {t}")
        print(f"  → {result}")
        print()
