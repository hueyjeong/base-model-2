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
    "qwertyuiop",
    "asdfghjkl",
    "zxcvbnm",
]
for _r, _row in enumerate(_EN_ROWS):
    _offset = [0.0, 0.25, 0.75][_r]
    for _c, _ch in enumerate(_row):
        _EN_LAYOUT[_ch] = (float(_r), _c + _offset)

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
_KO_LAYOUT: dict[str, tuple[float, float]] = {
    jamo: _EN_LAYOUT[en_key]
    for jamo, en_key in _KO_KEY_TO_EN.items()
    if en_key in _EN_LAYOUT
}

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


# 인접 키 캐시 사전 생성
_EN_NEIGHBORS: dict[str, list[str]] = {k: _get_neighbors(k, _EN_LAYOUT) for k in _EN_LAYOUT}
_KO_NEIGHBORS: dict[str, list[str]] = {k: _get_neighbors(k, _KO_LAYOUT) for k in _KO_LAYOUT}
_JA_NEIGHBORS: dict[str, list[str]] = _EN_NEIGHBORS  # 동일


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
    """영문/일본어(로마자) 키보드 타이포"""
    chars = list(text)
    n_typo = max(1, int(len(chars) * cfg.keyboard_typo_ratio))
    indices = [i for i, ch in enumerate(chars) if ch.lower() in neighbors]
    if not indices:
        return text
    chosen = rng.sample(indices, min(n_typo, len(indices)))
    for i in chosen:
        ch = chars[i]
        lower = ch.lower()
        nbr = neighbors.get(lower, [])
        if nbr:
            replacement = rng.choice(nbr)
            chars[i] = replacement.upper() if ch.isupper() else replacement
    return "".join(chars)


def _apply_keyboard_typo_ko(
    text: str, rng: random.Random, cfg: NoiseConfig
) -> str:
    """한글 키보드 타이포 — 자모 단위로 인접 키 치환"""
    chars = list(text)
    hangul_indices = [i for i, ch in enumerate(chars) if _decompose_hangul(ch) is not None]
    if not hangul_indices:
        return text

    n_typo = max(1, int(len(hangul_indices) * cfg.keyboard_typo_ratio))
    chosen = rng.sample(hangul_indices, min(n_typo, len(hangul_indices)))

    for i in chosen:
        decomposed = _decompose_hangul(chars[i])
        if decomposed is None:
            continue
        cho, jung, jong = decomposed
        cho_jamo = _CHO_LIST[cho]
        jung_jamo = _JUNG_LIST[jung]

        # 랜덤으로 초성, 중성, 종성 중 하나를 타이포
        targets = ["cho", "jung"]
        if jong > 0:
            targets.append("jong")
        target = rng.choice(targets)

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
    ids: list[int], mask_id: int, rng: random.Random, ratio: float
) -> list[int]:
    """토큰 마스킹 — 랜덤 토큰을 [MASK]로 교체"""
    result = list(ids)
    n_mask = max(1, int(len(result) * ratio))
    indices = list(range(len(result)))
    rng.shuffle(indices)
    for i in indices[:n_mask]:
        result[i] = mask_id
    return result


def _apply_token_deletion(
    ids: list[int], rng: random.Random, ratio: float
) -> list[int]:
    """토큰 삭제 — 랜덤 토큰 제거"""
    result = []
    for tok_id in ids:
        if rng.random() >= ratio:
            result.append(tok_id)
    return result if result else ids[:1]  # 최소 1토큰 유지


def _apply_text_infilling(
    ids: list[int], mask_id: int, rng: random.Random,
    ratio: float, poisson_lambda: float
) -> list[int]:
    """텍스트 인필링 — Poisson(λ) 길이 span을 단일 [MASK]로 교체"""
    n = len(ids)
    n_to_mask = max(1, int(n * ratio))
    masked = 0
    result = list(ids)
    visited = [False] * n

    max_attempts = n * 2
    attempts = 0
    while masked < n_to_mask and attempts < max_attempts:
        attempts += 1
        # span 길이 샘플
        span_len = min(max(1, int(rng.expovariate(1.0 / poisson_lambda))), n - masked)
        # 시작 위치 랜덤
        start = rng.randint(0, n - 1)
        if visited[start]:
            continue
        # span 범위 결정
        end = min(start + span_len, n)
        # 이미 방문한 위치 건너뛰기
        if any(visited[j] for j in range(start, end)):
            continue
        # span을 [MASK] 하나로 교체 (나머지는 None으로 마크)
        for j in range(start, end):
            visited[j] = True
        result[start] = mask_id
        for j in range(start + 1, end):
            result[j] = None  # 삭제 마크
        masked += (end - start)

    return [tok for tok in result if tok is not None]


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
                self._error_gen = KoreanErrorGenerator(seed=seed)
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

    def _apply_token_noise(self, ids: list[int]) -> list[int]:
        """토큰 레벨 노이즈 적용 (토크나이징 후)

        BART 논문: 3가지 중 하나를 랜덤 선택하여 적용
        """
        cfg = self.cfg
        rng = self.rng
        mask_id = self.tokenizer.mask_id

        # 3가지 토큰 노이즈 중 랜덤 선택
        noise_type = rng.choices(
            ["mask", "delete", "infill"],
            weights=[0.4, 0.2, 0.4],
            k=1
        )[0]

        if noise_type == "mask":
            return _apply_token_masking(ids, mask_id, rng, cfg.token_mask_ratio)
        elif noise_type == "delete":
            return _apply_token_deletion(ids, rng, cfg.token_delete_ratio)
        else:
            return _apply_text_infilling(
                ids, mask_id, rng,
                cfg.text_infill_ratio, cfg.infill_poisson_lambda
            )

    def __call__(
        self, text: str, lang: str | None = None
    ) -> tuple[list[int], list[int]]:
        """원본 텍스트 → (noised_ids, target_ids)

        Args:
            text: 원본 텍스트
            lang: 언어 코드 ("ko", "en", "ja"). None이면 자동 감지.

        Returns:
            (noised_ids, target_ids) — 둘 다 special token (BOS/EOS) 포함
        """
        if lang is None:
            lang = self._detect_lang(text)

        # 타겟: 원본을 토크나이징
        target_ids = self.tokenizer.encode(text, add_special=True)

        # 소스: 텍스트 노이즈 → 토크나이징 → 토큰 노이즈
        noised_text = self._apply_text_noise(text, lang)
        noised_ids = self.tokenizer.encode(noised_text, add_special=False)

        # 토큰 레벨 노이즈 (special token 제외 상태에서)
        noised_ids = self._apply_token_noise(noised_ids)

        # BOS/EOS 추가
        bos = self.tokenizer.bos_id
        eos = self.tokenizer.eos_id
        noised_ids = [bos] + noised_ids + [eos]

        return noised_ids, target_ids


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

        noised_ids, target_ids = noiser(text, lang)
        noised_text = tok.decode(noised_ids, skip_special=True)

        print(f"  노이즈: {noised_text}")
        print(f"  src 토큰수: {len(noised_ids)}, tgt 토큰수: {len(target_ids)}")

        mask_count = sum(1 for i in noised_ids if i == tok.mask_id)
        print(f"  [MASK] 수: {mask_count}")
        print()

    # 반복 호출 시 다양성 확인
    print("--- 다양성 테스트 (같은 문장 5회) ---")
    for i in range(5):
        noiser.set_seed(i * 100)
        noised_ids, _ = noiser("맞춤법을 확인해 주세요.")
        print(f"  시드 {i * 100}: {tok.decode(noised_ids, skip_special=True)}")
