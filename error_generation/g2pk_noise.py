"""
G2PK 기반 발음→철자 노이즈 생성 모듈.

G2PK 라이브러리로 텍스트의 표준 발음을 얻고,
발음을 그대로 표기하여 "발음대로 잘못 쓴" 오류를 생성.

변환 대상 음운 규칙:
- 연음 (liaison): "밥을" → "바블"
- 경음화 (fortis): "학교" → "학꾜"
- 비음화 (nasalization): "독립" → "동닙"
- 구개음화 (palatalization): "굳이" → "구지"
- 격음화 (aspiration): "놓고" → "노코"
- ㅎ탈락: "좋은" → "조은"

사용 예시:
    from error_generation.g2pk_noise import apply_g2pk_error
    import random
    result = apply_g2pk_error("독립문에서 밥을 먹었다", random.Random(42))
    # → "동님문에서 밥을 먹었다"  (어절 1개만 변환)
"""

import random
import re
from typing import Optional

# G2p 인스턴스 lazy singleton
_g2p = None


def _get_g2p():
    """G2p 인스턴스를 lazy 초기화하여 반환."""
    global _g2p
    if _g2p is None:
        from g2pk import G2p
        _g2p = G2p()
    return _g2p


# 한글 음절 범위
_HANGUL_RE = re.compile(r"[가-힣]")


def _has_hangul(text: str) -> bool:
    """문자열에 한글이 포함되어 있는지 확인."""
    return bool(_HANGUL_RE.search(text))


def _pronounce_word(word: str) -> str:
    """어절의 발음을 G2PK로 변환.

    G2PK는 문장 단위로 동작하지만, 단일 어절도 처리 가능.
    반환값은 발음 표기 문자열 (한글 음절).
    """
    g2p = _get_g2p()
    # G2PK는 descriptive=False (표준 발음)
    # g2pk numerals.py 버그: 특정 숫자에서 UnboundLocalError 발생
    try:
        pronounced = g2p(word)
    except (UnboundLocalError, Exception):
        return word
    return pronounced


def apply_g2pk_error(text: str, rng: random.Random) -> Optional[str]:
    """G2PK 발음 변환 기반 철자 오류 생성.

    어절 단위로 분리 후 랜덤 어절 1개를 발음 표기로 교체.
    발음이 원본과 동일하면 다른 어절 시도 (최대 5회).

    Args:
        text: 올바른 한국어 문장
        rng: random.Random 인스턴스

    Returns:
        오류가 적용된 문장, 또는 None (적용 불가 시)
    """
    words = text.split()
    if not words:
        return None

    # 한글 포함 어절만 후보로
    hangul_indices = [i for i, w in enumerate(words) if _has_hangul(w)]
    if not hangul_indices:
        return None

    # 랜덤 셔플 후 최대 5개 시도
    candidates = list(hangul_indices)
    rng.shuffle(candidates)

    for idx in candidates[:5]:
        original = words[idx]
        pronounced = _pronounce_word(original)

        # 발음이 원본과 다른 경우에만 적용
        if pronounced != original and _has_hangul(pronounced):
            result_words = list(words)
            result_words[idx] = pronounced
            return " ".join(result_words)

    return None


def get_error_count() -> int:
    """음운 규칙 수 반환 (통계용)."""
    # 연음, 경음화, 비음화, 구개음화, 격음화, ㅎ탈락
    return 6


if __name__ == "__main__":
    # 단독 테스트
    test_texts = [
        "독립문에서 밥을 먹었다",
        "놓고 간 물건을 찾았다",
        "좋은 사람이 되자",
        "굳이 가야 할 필요가 있나",
        "학교에 갔다 왔다",
        "맞히다와 맞추다는 다르다",
    ]

    rng = random.Random(42)
    print("=== G2PK 발음→철자 노이즈 테스트 ===")
    for t in test_texts:
        result = apply_g2pk_error(t, rng)
        print(f"  {t}")
        print(f"  → {result}")
        print()
