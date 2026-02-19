"""한글 2벌식 키보드 시퀀스 전처리/후처리 모듈

전처리: 한글 텍스트 → 자모 키보드 스트로크 시퀀스
후처리: 자모 키보드 스트로크 시퀀스 → 한글 텍스트

예) "까마귀" → ["[SHIFT]", "ㄱ", "ㅏ", "ㅁ", "ㅏ", "ㄱ", "ㅜ", "ㅣ"]
    "고ㄱㄱ" → ["ㄱ", "ㅗ", "[BLANK]", "ㄱ", "ㄱ"]

제어 토큰:
  [SHIFT]  — 쌍자음/ㅒㅖ 표시
  [BLANK]  — 음절-자모 경계 구분 (합성 방지)
"""

# ── 2벌식 키보드 매핑 테이블 ─────────────────────────────────────────

# 제어 토큰 (대괄호로 감싸서 일반 문자와 충돌 방지)
SHIFT = "[SHIFT]"
BLANK = "[BLANK]"

# 초성 (19개) — 인덱스 순서
INITIALS = [
    "ㄱ", "ㄲ", "ㄴ", "ㄷ", "ㄸ", "ㄹ", "ㅁ", "ㅂ", "ㅃ",
    "ㅅ", "ㅆ", "ㅇ", "ㅈ", "ㅉ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ",
]

# 중성 (21개) — 인덱스 순서
MEDIALS = [
    "ㅏ", "ㅐ", "ㅑ", "ㅒ", "ㅓ", "ㅔ", "ㅕ", "ㅖ", "ㅗ",
    "ㅘ", "ㅙ", "ㅚ", "ㅛ", "ㅜ", "ㅝ", "ㅞ", "ㅟ", "ㅠ",
    "ㅡ", "ㅢ", "ㅣ",
]

# 종성 (28개, 0=없음) — 인덱스 순서
FINALS = [
    None, "ㄱ", "ㄲ", "ㄳ", "ㄴ", "ㄵ", "ㄶ", "ㄷ", "ㄹ",
    "ㄺ", "ㄻ", "ㄼ", "ㄽ", "ㄾ", "ㄿ", "ㅀ", "ㅁ", "ㅂ",
    "ㅄ", "ㅅ", "ㅆ", "ㅇ", "ㅈ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ",
]

# 쌍자음 → SHIFT + 기본 자음
DOUBLE_CONSONANT_MAP = {
    "ㄲ": "ㄱ", "ㄸ": "ㄷ", "ㅃ": "ㅂ", "ㅆ": "ㅅ", "ㅉ": "ㅈ",
}

# 복합 모음 → 키보드 스트로크 분해 (2벌식 기준)
COMPOUND_VOWEL_MAP = {
    "ㅘ": ["ㅗ", "ㅏ"],
    "ㅙ": ["ㅗ", "ㅐ"],
    "ㅚ": ["ㅗ", "ㅣ"],
    "ㅝ": ["ㅜ", "ㅓ"],
    "ㅞ": ["ㅜ", "ㅔ"],
    "ㅟ": ["ㅜ", "ㅣ"],
    "ㅢ": ["ㅡ", "ㅣ"],
}

# 시프트 모음
SHIFT_VOWEL_MAP = {
    "ㅒ": "ㅐ",
    "ㅖ": "ㅔ",
}

# 복합 종성 → 자음 분해
COMPOUND_FINAL_MAP = {
    "ㄳ": ["ㄱ", "ㅅ"],
    "ㄵ": ["ㄴ", "ㅈ"],
    "ㄶ": ["ㄴ", "ㅎ"],
    "ㄺ": ["ㄹ", "ㄱ"],
    "ㄻ": ["ㄹ", "ㅁ"],
    "ㄼ": ["ㄹ", "ㅂ"],
    "ㄽ": ["ㄹ", "ㅅ"],
    "ㄾ": ["ㄹ", "ㅌ"],
    "ㄿ": ["ㄹ", "ㅍ"],
    "ㅀ": ["ㄹ", "ㅎ"],
    "ㅄ": ["ㅂ", "ㅅ"],
}

# 기본 자음 (키보드에 단일 키로 존재하는 것들)
BASIC_CONSONANTS = set("ㄱㄴㄷㄹㅁㅂㅅㅇㅈㅊㅋㅌㅍㅎ")

# 기본 모음 (키보드에 단일 키로 존재하는 것들)
BASIC_VOWELS = set("ㅏㅑㅓㅕㅗㅛㅜㅠㅡㅣㅐㅔ")

# 자음 집합 (호환자모)
ALL_CONSONANTS = set("ㄱㄲㄳㄴㄵㄶㄷㄸㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅃㅄㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ")

# 모음 집합 (호환자모)
ALL_VOWELS = set("ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ")


# ── 음절 분해/조합 ──────────────────────────────────────────────────

def _is_hangul_syllable(ch: str) -> bool:
    return 0xAC00 <= ord(ch) <= 0xD7A3


def _is_compat_jamo(ch: str) -> bool:
    return ch in ALL_CONSONANTS or ch in ALL_VOWELS


def _decompose_syllable(ch: str):
    """한글 완성형 → (초성idx, 중성idx, 종성idx)"""
    code = ord(ch) - 0xAC00
    initial = code // 588
    medial = (code % 588) // 28
    final = code % 28
    return initial, medial, final


def _compose_syllable(ini_idx: int, med_idx: int, fin_idx: int = 0) -> str:
    """(초성idx, 중성idx, 종성idx) → 한글 완성형"""
    return chr(0xAC00 + ini_idx * 588 + med_idx * 28 + fin_idx)


def _consonant_to_keystrokes(jamo: str):
    """자음 → 키보드 스트로크 리스트"""
    if jamo in DOUBLE_CONSONANT_MAP:
        return [SHIFT, DOUBLE_CONSONANT_MAP[jamo]]
    if jamo in COMPOUND_FINAL_MAP:
        result = []
        for part in COMPOUND_FINAL_MAP[jamo]:
            result.extend(_consonant_to_keystrokes(part))
        return result
    return [jamo]


def _vowel_to_keystrokes(jamo: str):
    """모음 → 키보드 스트로크 리스트"""
    if jamo in SHIFT_VOWEL_MAP:
        return [SHIFT, SHIFT_VOWEL_MAP[jamo]]
    if jamo in COMPOUND_VOWEL_MAP:
        result = []
        for part in COMPOUND_VOWEL_MAP[jamo]:
            result.extend(_vowel_to_keystrokes(part))
        return result
    return [jamo]


# ── 전처리: 텍스트 → 키보드 토큰 리스트 ─────────────────────────────

def _needs_blank_between(prev_char: str, next_char: str) -> bool:
    """두 원본 문자 사이에 BLANK가 필요한지 판단

    BLANK가 필요한 경우 (IME가 잘못 합성하는 경우):
      1. 음절(종성X) → 단독 자음: 자음이 종성으로 흡수됨
      2. 음절(종성X) → 단독 모음: 복합 모음으로 합쳐질 수 있음
      3. 음절(종성O) → 단독 모음: 종성이 다음 초성으로 빼앗김
      4. 음절(종성O) → 단독 자음: 복합 종성으로 합쳐질 수 있음
      5. 단독 자음 → 단독 모음: 음절로 합성됨
      6. 단독 모음 → 단독 모음: 복합 모음으로 합쳐질 수 있음
    """
    prev_syl = _is_hangul_syllable(prev_char)
    prev_jamo = _is_compat_jamo(prev_char)
    next_jamo = _is_compat_jamo(next_char)

    # 다음 문자가 단독 자모가 아니면 BLANK 불필요
    if not next_jamo:
        return False
    # 이전 문자가 한글(음절/자모)이 아니면 BLANK 불필요
    if not (prev_syl or prev_jamo):
        return False

    next_is_consonant = next_char in ALL_CONSONANTS
    next_is_vowel = next_char in ALL_VOWELS

    if prev_syl:
        _, med_idx, fin_idx = _decompose_syllable(prev_char)
        has_final = fin_idx != 0

        if not has_final:
            # Case 1: 음절(종성X) + 단독 자음
            if next_is_consonant:
                return True
            # Case 2: 음절(종성X) + 단독 모음 → 복합 모음 가능?
            if next_is_vowel:
                med = MEDIALS[med_idx]
                next_keys = _vowel_to_keystrokes(next_char)
                return (med, next_keys[0]) in _COMPOUND_VOWEL_REVERSE
        else:
            fin = FINALS[fin_idx]
            # Case 3: 음절(종성O) + 단독 모음 → 종성 도둑
            if next_is_vowel:
                return True
            # Case 4: 음절(종성O) + 단독 자음 → 복합 종성 가능?
            if next_is_consonant:
                fin_keys = _consonant_to_keystrokes(fin)
                last_fin = fin_keys[-1]  # 종성의 마지막 기본 자음
                next_keys = _consonant_to_keystrokes(next_char)
                first_next = next_keys[0]
                if first_next == SHIFT:
                    return False  # 쌍자음은 복합 종성 안됨
                return (last_fin, first_next) in _COMPOUND_FINAL_REVERSE

    elif prev_jamo:
        # Case 5: 단독 자음 + 단독 모음
        if prev_char in ALL_CONSONANTS and next_is_vowel:
            return True
        # Case 6: 단독 모음 + 단독 모음 → 복합 모음 가능?
        if prev_char in ALL_VOWELS and next_is_vowel:
            prev_keys = _vowel_to_keystrokes(prev_char)
            last_prev = prev_keys[-1]
            next_keys = _vowel_to_keystrokes(next_char)
            first_next = next_keys[0]
            return (last_prev, first_next) in _COMPOUND_VOWEL_REVERSE

    return False


def preprocess(text: str) -> list[str]:
    """텍스트를 2벌식 키보드 스트로크 토큰 리스트로 변환

    한글 음절 → 초성+중성(+종성) 키스트로크로 분해
    호환자모 → 그대로 (앞에 BLANK 필요 시 삽입)
    비한글 → 각 문자를 그대로 토큰으로

    Returns:
        키보드 스트로크 토큰 리스트
    """
    tokens = []

    for i, ch in enumerate(text):
        # BLANK 삽입 판단: 이전 문자와 현재 문자 사이
        if i > 0 and _needs_blank_between(text[i - 1], ch):
            tokens.append(BLANK)

        if _is_hangul_syllable(ch):
            ini_idx, med_idx, fin_idx = _decompose_syllable(ch)
            ini = INITIALS[ini_idx]
            med = MEDIALS[med_idx]
            fin = FINALS[fin_idx]

            tokens.extend(_consonant_to_keystrokes(ini))
            tokens.extend(_vowel_to_keystrokes(med))
            if fin is not None:
                tokens.extend(_consonant_to_keystrokes(fin))

        elif _is_compat_jamo(ch):
            if ch in ALL_CONSONANTS:
                tokens.extend(_consonant_to_keystrokes(ch))
            else:
                tokens.extend(_vowel_to_keystrokes(ch))

        else:
            tokens.append(ch)

    return tokens


# ── 후처리: 키보드 토큰 리스트 → 텍스트 ─────────────────────────────

# 역매핑 테이블: 키보드 자음 → 초성 인덱스
_INITIAL_TO_IDX = {}
for _i, _ini in enumerate(INITIALS):
    if _ini not in _INITIAL_TO_IDX:
        _INITIAL_TO_IDX[_ini] = _i

# 역매핑: 키보드 모음 시퀀스 → 중성 인덱스
_MEDIAL_TO_IDX = {}
for _i, _med in enumerate(MEDIALS):
    _MEDIAL_TO_IDX[_med] = _i

# 역매핑: 종성 → 종성 인덱스
_FINAL_TO_IDX = {}
for _i, _fin in enumerate(FINALS):
    if _fin is not None:
        _FINAL_TO_IDX[_fin] = _i

# Shift로 만드는 쌍자음 역매핑
_SHIFT_CONSONANT_REVERSE = {v: k for k, v in DOUBLE_CONSONANT_MAP.items()}

# Shift로 만드는 모음 역매핑
_SHIFT_VOWEL_REVERSE = {v: k for k, v in SHIFT_VOWEL_MAP.items()}

# 복합 모음 역매핑: (첫모음, 둘째모음) → 복합모음
_COMPOUND_VOWEL_REVERSE = {}
for _cv, _parts in COMPOUND_VOWEL_MAP.items():
    _COMPOUND_VOWEL_REVERSE[tuple(_parts)] = _cv

# 복합 종성 역매핑: (첫자음, 둘째자음) → 복합종성
_COMPOUND_FINAL_REVERSE = {}
for _cf, _parts in COMPOUND_FINAL_MAP.items():
    _COMPOUND_FINAL_REVERSE[tuple(_parts)] = _cf


def postprocess(tokens: list[str]) -> str:
    """키보드 스트로크 토큰 리스트를 텍스트로 재합성

    2벌식 키보드 IME 로직을 시뮬레이션하여 자모를 음절로 합성한다.
    """
    result = []

    # 1단계: SHIFT 토큰 해석 — <⇧> + ㄱ → ㄲ, <⇧> + ㅐ → ㅒ
    expanded = []
    i = 0
    while i < len(tokens):
        if tokens[i] == SHIFT and i + 1 < len(tokens):
            next_tok = tokens[i + 1]
            if next_tok in _SHIFT_CONSONANT_REVERSE:
                expanded.append(_SHIFT_CONSONANT_REVERSE[next_tok])
            elif next_tok in _SHIFT_VOWEL_REVERSE:
                expanded.append(_SHIFT_VOWEL_REVERSE[next_tok])
            else:
                expanded.append(tokens[i])
                expanded.append(next_tok)
                i += 2
                continue
            i += 2
        else:
            expanded.append(tokens[i])
            i += 1

    # 2단계: IME 시뮬레이션으로 음절 합성
    ini_idx = None
    med_idx = None
    fin = None
    fin_idx = None

    def _flush():
        nonlocal ini_idx, med_idx, fin, fin_idx
        if ini_idx is not None and med_idx is not None:
            result.append(_compose_syllable(ini_idx, med_idx, fin_idx or 0))
        elif ini_idx is not None:
            result.append(INITIALS[ini_idx])
        elif med_idx is not None:
            result.append(MEDIALS[med_idx])
        ini_idx = med_idx = fin = fin_idx = None

    def _try_compound_vowel(v1_idx, v2):
        v1 = MEDIALS[v1_idx]
        return _COMPOUND_VOWEL_REVERSE.get((v1, v2))

    def _try_compound_final(f1, f2):
        return _COMPOUND_FINAL_REVERSE.get((f1, f2))

    for tok in expanded:
        if tok == BLANK:
            _flush()
            continue

        is_consonant = tok in ALL_CONSONANTS and tok in _INITIAL_TO_IDX
        is_vowel = tok in ALL_VOWELS and tok in _MEDIAL_TO_IDX

        if is_consonant:
            if ini_idx is None and med_idx is None:
                ini_idx = _INITIAL_TO_IDX[tok]
            elif ini_idx is not None and med_idx is None:
                _flush()
                ini_idx = _INITIAL_TO_IDX[tok]
            elif ini_idx is not None and med_idx is not None and fin is None:
                if tok in _FINAL_TO_IDX:
                    fin = tok
                    fin_idx = _FINAL_TO_IDX[tok]
                else:
                    _flush()
                    ini_idx = _INITIAL_TO_IDX[tok]
            elif fin is not None:
                compound = _try_compound_final(fin, tok)
                if compound and compound in _FINAL_TO_IDX:
                    fin = compound
                    fin_idx = _FINAL_TO_IDX[compound]
                else:
                    _flush()
                    ini_idx = _INITIAL_TO_IDX[tok]
            else:
                _flush()
                ini_idx = _INITIAL_TO_IDX[tok]

        elif is_vowel:
            if ini_idx is not None and med_idx is not None and fin is not None:
                if fin in COMPOUND_FINAL_MAP:
                    parts = COMPOUND_FINAL_MAP[fin]
                    fin = parts[0]
                    fin_idx = _FINAL_TO_IDX[fin]
                    _flush()
                    ini_idx = _INITIAL_TO_IDX[parts[1]]
                else:
                    carry = fin
                    fin = None
                    fin_idx = None
                    _flush()
                    ini_idx = _INITIAL_TO_IDX[carry]
                med_idx = _MEDIAL_TO_IDX[tok]
            elif ini_idx is not None and med_idx is not None and fin is None:
                compound = _try_compound_vowel(med_idx, tok)
                if compound:
                    med_idx = _MEDIAL_TO_IDX[compound]
                else:
                    _flush()
                    med_idx = _MEDIAL_TO_IDX[tok]
            elif ini_idx is not None and med_idx is None:
                med_idx = _MEDIAL_TO_IDX[tok]
            elif med_idx is not None:
                compound = _try_compound_vowel(med_idx, tok)
                if compound:
                    med_idx = _MEDIAL_TO_IDX[compound]
                else:
                    _flush()
                    med_idx = _MEDIAL_TO_IDX[tok]
            else:
                med_idx = _MEDIAL_TO_IDX[tok]

        else:
            _flush()
            result.append(tok)

    _flush()
    return "".join(result)

# ── 테스트 ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        ("까마귀", "쌍자음 초성 + 복합 모음"),
        ("행동", "기본 초성+중성+종성"),
        ("대장이", "기본 음절들"),
        ("움직였다", "종성 ㅆ 분해"),
        ("다들", "기본"),
        ("고ㄱㄱ", "음절(종X) + 단독 자음 → BLANK"),
        ("곡ㄱ", "음절(종O) + 단독 자음 (복합 안됨)"),
        ("곡ㅅ", "음절(종O) + 단독 자음 (복합 가능ㄳ) → BLANK"),
        ("곡ㅏ", "음절(종O) + 단독 모음 → BLANK (종성 도둑 방지)"),
        ("오ㅏ", "음절(종X) + 단독 모음 (ㅗ+ㅏ=ㅘ) → BLANK"),
        ("아ㅗ", "음절(종X) + 단독 모음 (ㅏ+ㅗ 복합X) → BLANK 불필요"),
        ("ㅇㅇ오ㅗ", "혼합: 단독자모 + 음절 + 단독모음"),
        ("ㅇㅇㅇㅗㅗ", "단독 자모 연속"),
        ("ㄱㅏ", "단독 자음 + 단독 모음 → BLANK"),
        ("ㅗㅏ", "단독 모음 + 단독 모음 (복합 가능) → BLANK"),
        ("ㅗㅜ", "단독 모음 + 단독 모음 (복합 불가)"),
        ("맞춤법을 확인해 주세요.", "전체 문장"),
        ("Hello, world!", "비한글 패스스루"),
        ("한글English혼합", "한/영 혼합"),
        ("ㄱㄴㄷㄹ ㅏㅓㅗㅜ", "단독 자모"),
        ("ㅋㅋㅋ아ㅋㅋ", "음절 뒤 자모"),
        ("쏟아지다", "쌍자음 초성 + 복합 종성"),
        ("읽다", "복합 종성 ㄺ"),
        ("긁다", "복합 종성 ㄺ + 다"),
    ]

    all_pass = True
    for text, desc in tests:
        tokens = preprocess(text)
        restored = postprocess(tokens)
        match = text == restored
        if not match:
            all_pass = False
        print(f"[{'✓' if match else '✗'}] {desc}")
        print(f"  원문:   {text}")
        print(f"  토큰:   {' '.join(tokens)}")
        print(f"  복원:   {restored}")
        if not match:
            print(f"  !! 불일치 !!")
        print()

    print(f"전체: {'PASS ✓' if all_pass else 'FAIL ✗'}")

