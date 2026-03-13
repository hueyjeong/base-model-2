"""BART 스타일 디노이징 노이즈 모듈

텍스트 레벨 노이즈 (토크나이징 전):
    1. Korean Error Rules  — error_generation.KoreanErrorGenerator
    2. Spacing Noise        — 공백 전체/부분 제거, 랜덤 삽입
    3. Keyboard Typo         — 자간 거리 기반 인접 키 치환 (KO/EN/JA)
    4. N-gram Shuffle        — 연속 N-gram 순서 뒤섞기
    5. Word Reorder         — 단어/어절 순서 swap
    6. Whitespace Strip     — 전체/부분 공백 제거

토큰 레벨 노이즈 (토크나이징 후):
    7. Token Masking        — 15% 토큰 → [MASK]
    8. Token Deletion       — 5% 토큰 삭제
    9. Text Infilling       — Poisson(λ=3) span → 단일 [MASK]
"""
import math
import random
from dataclasses import dataclass, field
from typing import Optional

from tokenizer_base import BaseTokenizer

# ── 키보드 레이아웃 (좌표 기반, row/col) ──────────────────────────────

# 영문 QWERTY — (row, col) 좌표
_EN_LAYOUT: dict[str, tuple[float, float]] = {}
_EN_ROWS = [
    "`1234567890-=",
    "qwertyuiop[]\\",
    "asdfghjkl;'",
    "zxcvbnm,./",
]
for _r, _row in enumerate(_EN_ROWS):
    _offset = [0.0, 0.5, 0.75, 1.25][_r]  # QWERTY standard staggering
    for _c, _ch in enumerate(_row):
        _EN_LAYOUT[_ch] = (float(_r), _c + _offset)

# 특수문자 Shift 레이아웃 
_SHIFT_MAP_EN = {
    "`": "~", "1": "!", "2": "@", "3": "#", "4": "$", "5": "%",
    "6": "^", "7": "&", "8": "*", "9": "(", "0": ")", "-": "_", "=": "+",
    "[": "{", "]": "}", "\\": "|", ";": ":", "'": "\"", ",": "<", ".": ">", "/": "?"
}
_SHIFT_MAP_KO = {
    "ㄱ": "ㄲ", "ㄷ": "ㄸ", "ㅂ": "ㅃ", "ㅅ": "ㅆ", "ㅈ": "ㅉ",
    "ㅐ": "ㅒ", "ㅔ": "ㅖ",
}
_SHIFT_MAP_REVERSE = {v: k for k, v in {**_SHIFT_MAP_EN, **_SHIFT_MAP_KO}.items()}

# 특수문자(Shift 입력)의 좌표는 원래 키와 동일하게 설정
for _base, _shift in _SHIFT_MAP_EN.items():
    if _base in _EN_LAYOUT:
        _EN_LAYOUT[_shift] = _EN_LAYOUT[_base]

# Numpad 좌표 (row는 QWERTY 배열 기준 0.0=숫자열 수준)
_NUMPAD_LAYOUT: dict[str, tuple[float, float]] = {
    "/": (0.0, 15.0), "*": (0.0, 16.0), "-": (0.0, 17.0),
    "7": (1.0, 14.5), "8": (1.0, 15.5), "9": (1.0, 16.5), "+": (1.0, 17.5),
    "4": (2.0, 14.5), "5": (2.0, 15.5), "6": (2.0, 16.5), # + continues down
    "1": (3.0, 14.5), "2": (3.0, 15.5), "3": (3.0, 16.5), "Enter": (3.0, 17.5),
    "0": (4.0, 15.0), ".": (4.0, 16.5)
}

# 한글 2벌식 — QWERTY 키에 대응하는 자모
_KO_QWERTY_MAP = {
    "q": "ㅂ", "w": "ㅈ", "e": "ㄷ", "r": "ㄱ", "t": "ㅅ",
    "y": "ㅛ", "u": "ㅕ", "i": "ㅑ", "o": "ㅐ", "p": "ㅔ",
    "a": "ㅁ", "s": "ㄴ", "d": "ㅇ", "f": "ㄹ", "g": "ㅎ",
    "h": "ㅗ", "j": "ㅓ", "k": "ㅏ", "l": "ㅣ",
    "z": "ㅋ", "x": "ㅌ", "c": "ㅊ", "v": "ㅍ",
    "b": "ㅠ", "n": "ㅜ", "m": "ㅡ",
}
_KO_KEY_TO_EN: dict[str, str] = {v: k for k, v in _KO_QWERTY_MAP.items()}

# 한글 기본 자모 좌표
_KO_LAYOUT: dict[str, tuple[float, float]] = {
    jamo: _EN_LAYOUT[en_key]
    for jamo, en_key in _KO_KEY_TO_EN.items()
    if en_key in _EN_LAYOUT
}
# 한글 쌍자모(Shift)의 좌표
for _base, _shift in _SHIFT_MAP_KO.items():
    if _base in _KO_LAYOUT:
        _KO_LAYOUT[_shift] = _KO_LAYOUT[_base]

# 일본어 로마자 — 영문 QWERTY와 동일 좌표 사용
_JA_LAYOUT = _EN_LAYOUT  # 로마자 입력이므로 동일


def _keyboard_distance(
    a: str, b: str, layout: dict[str, tuple[float, float]]
) -> float:
    """두 키 사이의 유클리드 거리."""
    if a not in layout or b not in layout:
        return float("inf")
    ra, ca = layout[a]
    rb, cb = layout[b]
    return math.sqrt((ra - rb) ** 2 + (ca - cb) ** 2)


def _get_neighbors(
    key: str, layout: dict[str, tuple[float, float]], max_dist: float = 1.6
) -> list[str]:
    """주어진 키의 인접 키 목록 (거리 기준)."""
    return [
        k for k in layout
        if k != key and _keyboard_distance(key, k, layout) <= max_dist
    ]

# 인접 키 캐시 사전 생성 (Shift + Numpad 포함 통합 거리)
_ALL_LAYOUT = {**_EN_LAYOUT, **_NUMPAD_LAYOUT}
_EN_NEIGHBORS: dict[str, list[str]] = {k: _get_neighbors(k, _ALL_LAYOUT) for k in _ALL_LAYOUT}
_KO_NEIGHBORS: dict[str, list[str]] = {k: _get_neighbors(k, _KO_LAYOUT) for k in _KO_LAYOUT}
_JA_NEIGHBORS: dict[str, list[str]] = _EN_NEIGHBORS  # 동일

# 대소문자 변환 (Shift 오입력)
def _get_shift_typo(ch: str) -> str | None:
    if ch in _SHIFT_MAP_EN: return _SHIFT_MAP_EN[ch]
    if ch in _SHIFT_MAP_KO: return _SHIFT_MAP_KO[ch]
    if ch in _SHIFT_MAP_REVERSE: return _SHIFT_MAP_REVERSE[ch]
    if ch.islower(): return ch.upper()
    if ch.isupper(): return ch.lower()
    return None


# ── 한글 자모 분해/조합 유틸리티 ──────────────────────────────────────

_HANGUL_BASE = 0xAC00
_HANGUL_END = 0xD7A3
_JONG_COUNT = 28
_JUNG_COUNT = 21

_CHO_LIST = [
    "ㄱ", "ㄲ", "ㄴ", "ㄷ", "ㄸ", "ㄹ", "ㅁ", "ㅂ", "ㅃ",
    "ㅅ", "ㅆ", "ㅇ", "ㅈ", "ㅉ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ",
]
_JUNG_LIST = [
    "ㅏ", "ㅐ", "ㅑ", "ㅒ", "ㅓ", "ㅔ", "ㅕ", "ㅖ", "ㅗ",
    "ㅘ", "ㅙ", "ㅚ", "ㅛ", "ㅜ", "ㅝ", "ㅞ", "ㅟ", "ㅠ",
    "ㅡ", "ㅢ", "ㅣ",
]
_JONG_LIST = [
    None, "ㄱ", "ㄲ", "ㄳ", "ㄴ", "ㄵ", "ㄶ", "ㄷ", "ㄹ",
    "ㄺ", "ㄻ", "ㄼ", "ㄽ", "ㄾ", "ㄿ", "ㅀ", "ㅁ", "ㅂ",
    "ㅄ", "ㅅ", "ㅆ", "ㅇ", "ㅈ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ",
]

# 초성에서 사용 가능한 기본 자모만 (쌍자음 제외)
_BASIC_CHO = {"ㄱ", "ㄴ", "ㄷ", "ㄹ", "ㅁ", "ㅂ", "ㅅ", "ㅇ", "ㅈ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ"}
_BASIC_JUNG = {"ㅏ", "ㅐ", "ㅑ", "ㅓ", "ㅔ", "ㅕ", "ㅗ", "ㅛ", "ㅜ", "ㅠ", "ㅡ", "ㅣ"}


def _decompose_hangul(ch: str) -> Optional[tuple[int, int, int]]:
    code = ord(ch)
    if code < _HANGUL_BASE or code > _HANGUL_END:
        return None
    offset = code - _HANGUL_BASE
    cho = offset // (_JUNG_COUNT * _JONG_COUNT)
    jung = (offset % (_JUNG_COUNT * _JONG_COUNT)) // _JONG_COUNT
    jong = offset % _JONG_COUNT
    return cho, jung, jong


def _compose_hangul(cho: int, jung: int, jong: int = 0) -> str:
    return chr(_HANGUL_BASE + cho * _JUNG_COUNT * _JONG_COUNT + jung * _JONG_COUNT + jong)


# ── 노이즈 설정 ──────────────────────────────────────────────────────

@dataclass
class NoiseConfig:
    """노이즈 하이퍼파라미터"""
    # 텍스트 레벨
    korean_error_prob: float = 0.3       # 한국어 오류 룰 적용 확률
    korean_error_count: int = 2          # 적용 시 오류 수
    spacing_noise_prob: float = 0.3      # 공백 노이즈 확률
    spacing_remove_ratio: float = 0.3    # 공백 제거 비율 (전체 공백 중)
    spacing_insert_prob: float = 0.05    # 문자 사이 공백 삽입 확률
    spacing_full_remove_prob: float = 0.1  # 전체 공백 제거 확률
    keyboard_typo_prob: float = 0.15     # 키보드 타이포 확률
    keyboard_typo_ratio: float = 0.05    # 타이포 적용 문자 비율
    ngram_shuffle_prob: float = 0.1      # N-gram 셔플 확률
    ngram_n: int = 3                     # N-gram 크기
    word_reorder_prob: float = 0.1       # 어절 순서 변경 확률
    word_reorder_swaps: int = 2          # swap 횟수

    # 토큰 레벨
    token_mask_ratio: float = 0.15       # 토큰 마스킹 비율
    token_delete_ratio: float = 0.05     # 토큰 삭제 비율
    text_infill_ratio: float = 0.15      # 텍스트 인필링 비율
    infill_poisson_lambda: float = 3.0   # span 길이 Poisson λ
    token_noise_mask_weight: float = 0.4     # 토큰 노이즈 타입 선택 가중치(mask)
    token_noise_delete_weight: float = 0.2   # 토큰 노이즈 타입 선택 가중치(delete)
    token_noise_infill_weight: float = 0.4   # 토큰 노이즈 타입 선택 가중치(infill)

    # 키보드 Shift 오입력 확률
    keyboard_shift_typo_prob_alpha: float = 0.3     # 영문/일본어/숫자/기호 Shift 오입력 확률
    keyboard_shift_typo_prob_ko_alpha: float = 0.3  # 한글 경로 내 alpha(숫자/기호 포함) Shift 오입력 확률
    keyboard_shift_typo_prob_ko_jamo: float = 0.3   # 한글 자모 Shift 오입력 확률

    # 한국어 error_generation 모듈별 가중치 (0 = 비활성화, 기본값 = 코드 하드코딩 값)
    # 빈 dict({}) 이면 ERROR_GENERATORS의 기본 가중치를 그대로 사용
    korean_error_weights: dict = None  # type: ignore

    def __post_init__(self):
        if self.korean_error_weights is None:
            self.korean_error_weights = {}


# ── 텍스트 레벨 노이즈 함수 ───────────────────────────────────────────

def _apply_spacing_noise(
    text: str, rng: random.Random, cfg: NoiseConfig
) -> str:
    """공백 노이즈: 부분 제거, 전체 제거, 또는 랜덤 삽입"""
    if rng.random() < cfg.spacing_full_remove_prob:
        # 전체 공백 제거
        return text.replace(" ", "")

    chars = list(text)
    result = []
    for ch in chars:
        if ch == " ":
            # 부분 공백 제거
            if rng.random() < cfg.spacing_remove_ratio:
                continue
            result.append(ch)
        else:
            # 랜덤 공백 삽입
            if rng.random() < cfg.spacing_insert_prob:
                result.append(" ")
            result.append(ch)
    return "".join(result)


def _apply_keyboard_typo(
    text: str, rng: random.Random, cfg: NoiseConfig, lang: str
) -> str:
    """키보드 자간 거리 기반 타이포 생성"""
    if lang == "ko":
        return _apply_keyboard_typo_ko(text, rng, cfg)
    elif lang == "ja":
        return _apply_keyboard_typo_alpha(text, rng, cfg, _JA_NEIGHBORS)
    else:  # en
        return _apply_keyboard_typo_alpha(text, rng, cfg, _EN_NEIGHBORS)


def _apply_keyboard_typo_alpha(
    text: str, rng: random.Random, cfg: NoiseConfig,
    neighbors: dict[str, list[str]]
) -> str:
    """영문/일본어(로마자) 및 숫자, 특수문자 키보드 타이포"""
    chars = list(text)
    n_typo = max(1, int(len(chars) * cfg.keyboard_typo_ratio))
    
    # Numpad/기호/알파벳 인접키 캐시 활용
    indices = [i for i, ch in enumerate(chars) if ch in neighbors or ch.lower() in neighbors or ch.upper() in neighbors]
    if not indices:
        return text
        
    chosen = rng.sample(indices, min(n_typo, len(indices)))
    for i in chosen:
        ch = chars[i]
        
        # 30% 확률로 Shift 오입력 실수 발생 (Numpad는 Shift가 안 먹으므로 제외)
        shift_typo = _get_shift_typo(ch)
        if shift_typo and rng.random() < cfg.keyboard_shift_typo_prob_alpha:
            chars[i] = shift_typo
            continue
            
        lower = ch.lower()
        if ch in neighbors:
            nbr = neighbors.get(ch, [])
        else:
            nbr = neighbors.get(lower, [])
            
        if nbr:
            replacement = rng.choice(nbr)
            # 원래 문자가 영문 대문자였다면 대문자로 바꿔줌 
            # (기호나 숫자의 이웃인 경우 isupper/lower가 무시됨)
            if ch.isupper() and replacement.isalpha():
                chars[i] = replacement.upper()
            else:
                chars[i] = replacement
    return "".join(chars)


def _apply_keyboard_typo_ko(
    text: str, rng: random.Random, cfg: NoiseConfig
) -> str:
    """한글 키보드 타이포 — 자모 단위로 인접 키 치환 및 Shift 치환"""
    chars = list(text)
    
    # 한글 및 기호, 숫자가 섞여있을 수 있으므로 알파벳/기호 타이포 로직도 혼합 처리
    # 한글은 자모 분해하여 치환, 그 외(숫자, 기호)는 알파벳 로직으로
    indices = []
    for i, ch in enumerate(chars):
        if _decompose_hangul(ch) is not None:
            indices.append((i, "ko"))
        elif ch in _EN_NEIGHBORS or ch.lower() in _EN_NEIGHBORS or ch.upper() in _EN_NEIGHBORS:
            indices.append((i, "alpha"))
            
    if not indices:
        return text

    n_typo = max(1, int(len(indices) * cfg.keyboard_typo_ratio))
    chosen = rng.sample(indices, min(n_typo, len(indices)))

    for i, lang_type in chosen:
        ch = chars[i]
        
        if lang_type == "alpha":
            # 숫자, 알파벳, 특수문자에 대한 Shift / 인접 오타
            shift_typo = _get_shift_typo(ch)
            if shift_typo and rng.random() < cfg.keyboard_shift_typo_prob_ko_alpha:
                chars[i] = shift_typo
                continue
            
            lower = ch.lower()
            nbr = _EN_NEIGHBORS.get(ch, _EN_NEIGHBORS.get(lower, []))
            if nbr:
                replacement = rng.choice(nbr)
                if ch.isupper() and replacement.isalpha():
                    chars[i] = replacement.upper()
                else:
                    chars[i] = replacement
        else:
            # 한글 오타
            decomposed = _decompose_hangul(ch)
            if decomposed is None:
                continue
            cho, jung, jong = decomposed
            cho_jamo = _CHO_LIST[cho]
            jung_jamo = _JUNG_LIST[jung]

            targets = ["cho", "jung"]
            if jong > 0:
                targets.append("jong")
            target = rng.choice(targets)
            
            # Shift 오타 적용 (30% 확률)
            if rng.random() < cfg.keyboard_shift_typo_prob_ko_jamo:
                if target == "cho":
                    shift_cho = _get_shift_typo(cho_jamo)
                    if shift_cho and shift_cho in _CHO_LIST:
                        new_cho = _CHO_LIST.index(shift_cho)
                        chars[i] = _compose_hangul(new_cho, jung, jong)
                        continue
                elif target == "jung":
                    shift_jung = _get_shift_typo(jung_jamo)
                    if shift_jung and shift_jung in _JUNG_LIST:
                        new_jung = _JUNG_LIST.index(shift_jung)
                        chars[i] = _compose_hangul(cho, new_jung, jong)
                        continue

            if target == "cho" and cho_jamo in _KO_NEIGHBORS:
                nbr = _KO_NEIGHBORS[cho_jamo]
                nbr_cho = [j for j in nbr if j in _BASIC_CHO and j in _CHO_LIST]
                if nbr_cho:
                    new_cho = _CHO_LIST.index(rng.choice(nbr_cho))
                    chars[i] = _compose_hangul(new_cho, jung, jong)
            elif target == "jung" and jung_jamo in _KO_NEIGHBORS:
                nbr = _KO_NEIGHBORS[jung_jamo]
                nbr_jung = [j for j in nbr if j in _BASIC_JUNG and j in _JUNG_LIST]
                if nbr_jung:
                    new_jung = _JUNG_LIST.index(rng.choice(nbr_jung))
                    chars[i] = _compose_hangul(cho, new_jung, jong)
            elif target == "jong" and jong > 0:
                jong_jamo = _JONG_LIST[jong]
                if jong_jamo and jong_jamo in _KO_NEIGHBORS:
                    nbr = _KO_NEIGHBORS[jong_jamo]
                    nbr_jong = [j for j in nbr if j in _JONG_LIST[1:]]
                    if nbr_jong:
                        new_jong = _JONG_LIST.index(rng.choice(nbr_jong))
                        chars[i] = _compose_hangul(cho, jung, new_jong)

    return "".join(chars)


def _apply_ngram_shuffle(
    text: str, rng: random.Random, cfg: NoiseConfig
) -> str:
    """N-gram 단위 순서 뒤섞기 (어절 내)"""
    words = text.split()
    if len(words) < 2:
        return text

    # 단어 중 긴 것 하나를 골라 글자 단위 N-gram 셔플
    long_words = [(i, w) for i, w in enumerate(words) if len(w) >= cfg.ngram_n + 1]
    if not long_words:
        # 단어가 모두 짧으면 단어 순서 자체를 셔플
        rng.shuffle(words)
        return " ".join(words)

    idx, word = rng.choice(long_words)
    chars = list(word)
    # N-gram 청크로 분할 후 셔플
    n = cfg.ngram_n
    chunks = [chars[i:i + n] for i in range(0, len(chars), n)]
    if len(chunks) > 1:
        rng.shuffle(chunks)
    words[idx] = "".join(c for chunk in chunks for c in chunk)
    return " ".join(words)


def _apply_word_reorder(
    text: str, rng: random.Random, cfg: NoiseConfig
) -> str:
    """어절 순서 일부 swap"""
    words = text.split()
    if len(words) < 3:
        return text
    for _ in range(cfg.word_reorder_swaps):
        i = rng.randint(0, len(words) - 2)
        j = rng.randint(i + 1, len(words) - 1)
        words[i], words[j] = words[j], words[i]
    return " ".join(words)


# ── 토큰 레벨 노이즈 함수 ────────────────────────────────────────────

def _apply_token_masking(
    ids: list[int], weights: list[float], mask_id: int, rng: random.Random, ratio: float
) -> tuple[list[int], list[float]]:
    """토큰 마스킹 — 원본 토큰(weight=1.0) 중에서만 마스킹"""
    result_ids = list(ids)
    result_weights = list(weights)
    
    valid_indices = [i for i, w in enumerate(weights) if w == 1.0]
    n_mask = max(1, int(len(result_ids) * ratio))
    
    rng.shuffle(valid_indices)
    for i in valid_indices[:n_mask]:
        result_ids[i] = mask_id
        result_weights[i] = 0.5  # 마스킹된 토큰도 노이즈이므로 0.5
    return result_ids, result_weights


def _apply_token_deletion(
    ids: list[int], weights: list[float], rng: random.Random, ratio: float
) -> tuple[list[int], list[float]]:
    """토큰 삭제 — 원본 토큰(weight=1.0) 중에서만 랜덤 제거"""
    result_ids = []
    result_weights = []
    
    for tok_id, w in zip(ids, weights):
        # 원본 토큰이 아니면 삭제하지 않음
        if w == 1.0 and rng.random() < ratio:
            continue
        result_ids.append(tok_id)
        result_weights.append(w)
        
    if not result_ids:  # 최소 1토큰 유지
        return ids[:1], weights[:1]
    return result_ids, result_weights


def _apply_text_infilling(
    ids: list[int], weights: list[float], mask_id: int, rng: random.Random,
    ratio: float, poisson_lambda: float
) -> tuple[list[int], list[float]]:
    """텍스트 인필링 — 원본 토큰 구간(weight=1.0)에서 span을 단일 [MASK]로 교체"""
    n = len(ids)
    if n == 0 or ratio <= 0.0 or poisson_lambda <= 0.0:
        return ids, weights
    n_to_mask = max(1, int(n * ratio))
    masked = 0
    
    result_ids = list(ids)
    result_weights = list(weights)
    visited = [False] * n

    max_attempts = n * 2
    attempts = 0
    while masked < n_to_mask and attempts < max_attempts:
        attempts += 1
        span_len = min(max(1, int(rng.expovariate(1.0 / poisson_lambda))), n - masked)
        start = rng.randint(0, n - 1)
        
        if visited[start]: continue
        end = min(start + span_len, n)
        
        # 이미 방문했거나, 노이즈가 섞여있으면(weight != 1.0) 건너뛰기
        can_mask = True
        for j in range(start, end):
            if visited[j] or result_weights[j] != 1.0:
                can_mask = False
                break
        if not can_mask:
            continue

        for j in range(start, end):
            visited[j] = True
        result_ids[start] = mask_id
        result_weights[start] = 0.5  # 인필링 마스크 토큰 가중치
        
        for j in range(start + 1, end):
            result_ids[j] = None  # 삭제 마크
            result_weights[j] = None
            
        masked += (end - start)

    final_ids = [tok for tok in result_ids if tok is not None]
    final_weights = [w for w in result_weights if w is not None]
    return final_ids, final_weights


# ── 메인 DenoisingNoiser 클래스 ──────────────────────────────────────

class DenoisingNoiser:
    """BART 스타일 디노이징 노이저

    텍스트 레벨 노이즈를 먼저 적용하고, 토크나이징 후 토큰 레벨 노이즈를 적용.
    원본 텍스트는 별도로 토크나이징하여 target_ids로 사용.

    Args:
        tokenizer: BaseTokenizer 구현체
        config: NoiseConfig 인스턴스
        seed: 랜덤 시드
        use_korean_errors: error_generation 모듈 사용 여부
    """

    def __init__(
        self,
        tokenizer: BaseTokenizer,
        config: NoiseConfig | None = None,
        seed: int = 42,
        use_korean_errors: bool = True,
    ):
        self.tokenizer = tokenizer
        self.cfg = config or NoiseConfig()
        self.rng = random.Random(seed)
        self._error_gen = None
        if use_korean_errors:
            try:
                from error_generation import KoreanErrorGenerator
                self._error_gen = KoreanErrorGenerator(
                    seed=seed,
                    weights_override=self.cfg.korean_error_weights or {},
                )
            except ImportError:
                pass

    def set_seed(self, seed: int) -> None:
        """랜덤 시드 재설정"""
        self.rng = random.Random(seed)
        if self._error_gen is not None:
            self._error_gen.set_seed(seed)

    def state_dict(self) -> dict:
        """RNG 내부 상태를 직렬화 가능한 dict로 반환"""
        state = {"rng_state": self.rng.getstate()}
        if self._error_gen is not None:
            state["error_gen_state"] = self._error_gen.state_dict()
        return state

    def load_state_dict(self, state: dict) -> None:
        """저장된 RNG 상태를 복원"""
        self.rng.setstate(state["rng_state"])
        if self._error_gen is not None and "error_gen_state" in state:
            self._error_gen.load_state_dict(state["error_gen_state"])

    def _detect_lang(self, text: str) -> str:
        """간단한 언어 감지 (한글/일본어/영어)"""
        ko_count = 0
        ja_count = 0
        en_count = 0
        for ch in text:
            code = ord(ch)
            if 0xAC00 <= code <= 0xD7A3 or 0x1100 <= code <= 0x11FF:
                ko_count += 1
            elif (0x3040 <= code <= 0x309F or 0x30A0 <= code <= 0x30FF
                  or 0x4E00 <= code <= 0x9FFF):
                ja_count += 1
            elif code < 0x0080 and ch.isalpha():
                en_count += 1

        if ko_count >= ja_count and ko_count >= en_count:
            return "ko"
        elif ja_count >= en_count:
            return "ja"
        return "en"

    def _apply_text_noise(self, text: str, lang: str) -> str:
        """텍스트 레벨 노이즈 적용 (토크나이징 전)"""
        cfg = self.cfg
        rng = self.rng

        # 1. 한국어 오류 룰 (한국어만)
        if lang == "ko" and self._error_gen is not None:
            if rng.random() < cfg.korean_error_prob:
                text = self._error_gen.apply_random_errors(
                    text, n_errors=cfg.korean_error_count
                )

        # 2. 공백 노이즈
        if rng.random() < cfg.spacing_noise_prob:
            text = _apply_spacing_noise(text, rng, cfg)

        # 3. 키보드 타이포
        if rng.random() < cfg.keyboard_typo_prob:
            text = _apply_keyboard_typo(text, rng, cfg, lang)

        # 4. N-gram 셔플
        if rng.random() < cfg.ngram_shuffle_prob:
            text = _apply_ngram_shuffle(text, rng, cfg)

        # 5. 어절 순서 변경
        if rng.random() < cfg.word_reorder_prob:
            text = _apply_word_reorder(text, rng, cfg)

        return text

    def _apply_token_noise(self, ids: list[int], weights: list[float]) -> tuple[list[int], list[float]]:
        """토큰 레벨 노이즈 적용 (토크나이징 후)

        BART 논문: 3가지 중 하나를 랜덤 선택하여 적용
        """
        cfg = self.cfg
        rng = self.rng
        mask_id = self.tokenizer.mask_id

        # 3가지 토큰 노이즈 중 랜덤 선택
        noise_weights = [
            max(0.0, float(cfg.token_noise_mask_weight)),
            max(0.0, float(cfg.token_noise_delete_weight)),
            max(0.0, float(cfg.token_noise_infill_weight)),
        ]
        if sum(noise_weights) <= 0.0:
            noise_weights = [1.0, 1.0, 1.0]

        noise_type = rng.choices(
            ["mask", "delete", "infill"],
            weights=noise_weights,
            k=1
        )[0]

        if noise_type == "mask":
            return _apply_token_masking(ids, weights, mask_id, rng, cfg.token_mask_ratio)
        elif noise_type == "delete":
            return _apply_token_deletion(ids, weights, rng, cfg.token_delete_ratio)
        else:
            return _apply_text_infilling(
                ids, weights, mask_id, rng,
                cfg.text_infill_ratio, cfg.infill_poisson_lambda
            )

    def __call__(
        self, text: str, lang: str | None = None
    ) -> tuple[list[int], list[int], list[float]]:
        """원본 텍스트 → (noised_ids, target_ids, src_weights)

        Args:
            text: 원본 텍스트
            lang: 언어 코드 ("ko", "en", "ja"). None이면 자동 감지.

        Returns:
            noised_ids: 노이즈가 적용된 토큰 목록 (특수 토큰 포함)
            target_ids: 원본 텍스트 토큰 목록 (특수 토큰 포함)
            src_weights: noised_ids 각 위치의 노이즈 여부 (0.5 = 변경됨/노이즈, 1.0 = 유지됨)
        """
        if lang is None:
            lang = self._detect_lang(text)

        # 타겟: 원본을 토크나이징 (BOS/EOS 미포함 후 따로 추가)
        target_ids_base = self.tokenizer.encode(text, add_special=False)
        target_ids = [self.tokenizer.bos_id] + target_ids_base + [self.tokenizer.eos_id]

        # 소스: 텍스트 노이즈 → 토크나이징
        noised_text = self._apply_text_noise(text, lang)
        noised_ids_base = self.tokenizer.encode(noised_text, add_special=False)

        # rapidfuzz C++ opcodes로 텍스트 노이즈 결과와 원본 간의 일치하는 블록 찾기
        # (difflib.SequenceMatcher O(N²) → rapidfuzz C++ backend O(N×M) with early exit)
        from rapidfuzz.distance import Opcodes as _Opcodes
        from rapidfuzz.distance.Indel import opcodes as _indel_opcodes
        _ops = _indel_opcodes(target_ids_base, noised_ids_base)
        src_weights_base = [0.5] * len(noised_ids_base)
        
        for tag, i1, i2, j1, j2 in _ops:
            if tag == 'equal':
                for j in range(j1, j2):
                    src_weights_base[j] = 1.0

        # 토큰 레벨 노이즈 (특수 토큰 제외 상태에서, 가중치 함께 전달)
        noised_ids_base, src_weights_base = self._apply_token_noise(noised_ids_base, src_weights_base)

        # BOS/EOS 추가 (BOS/EOS는 노이즈가 아니므로 weight 1.0)
        bos = self.tokenizer.bos_id
        eos = self.tokenizer.eos_id
        noised_ids = [bos] + noised_ids_base + [eos]
        src_weights = [1.0] + src_weights_base + [1.0]

        return noised_ids, target_ids, src_weights


# ── 테스트 ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

    from bbpe_tokenizer.bbpe_wrapper import BBPETokenizer

    bbpe_path = os.path.join(
        os.path.dirname(__file__), "..", "bbpe_tokenizer", "bbpe.json"
    )
    tok = BBPETokenizer(bbpe_path)

    cfg = NoiseConfig()
    noiser = DenoisingNoiser(tok, cfg, seed=42)

    test_cases = [
        ("맞춤법을 확인해 주세요. 올바른 문장을 만들어 봅시다.", "ko"),
        ("The quick brown fox jumps over the lazy dog.", "en"),
        ("東京は日本の首都です。とても大きな都市です。", "ja"),
        ("한글과 English가 혼합된 문장입니다.", None),
    ]

    for text, lang in test_cases:
        detected = lang or noiser._detect_lang(text)
        print(f"[{detected.upper()}] 원문: {text}")

        noised_ids, target_ids, src_weights = noiser(text, lang)
        noised_text = tok.decode(noised_ids, skip_special=True)

        print(f"  노이즈: {noised_text}")
        print(f"  src 토큰수: {len(noised_ids)}, tgt 토큰수: {len(target_ids)}")
        
        # 가중치 확인 시각화
        weighted_tokens = []
        for i, weight in zip(noised_ids, src_weights):
            ch = tok.id_to_piece(i)
            if weight == 0.5:
                weighted_tokens.append(f"*{ch}*")
            else:
                weighted_tokens.append(ch)
        print(f"  Weights: {' '.join(weighted_tokens)}")

        mask_count = sum(1 for i in noised_ids if i == tok.mask_id)
        print(f"  [MASK] 수: {mask_count}")
        print()

    # 반복 호출 시 다양성 확인
    print("--- 다양성 테스트 (같은 문장 5회) ---")
    for i in range(5):
        noiser.set_seed(i * 100)
        noised_ids, _, _ = noiser("맞춤법을 확인해 주세요.")
        print(f"  시드 {i * 100}: {tok.decode(noised_ids, skip_special=True)}")
