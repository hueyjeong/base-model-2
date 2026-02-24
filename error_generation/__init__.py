"""
한국어 오류 생성 모듈 — Korean Error Generation Module.

GEC 학습 데이터 증강을 위한 오류 주입 시스템.
매 에포크마다 다른 시드를 사용하여 다양한 오류를 생성할 수 있음.

사용 예시:
    from error_generation import KoreanErrorGenerator

    gen = KoreanErrorGenerator(seed=42)
    erroneous = gen.apply_random_errors("굳이 그럴 필요가 없다.", n_errors=1)
    # → "구지 그럴 필요가 없다."

    # 에포크마다 다른 시드 사용
    gen.set_seed(epoch * 1000 + step)
"""

import random
from typing import Optional, Callable

from error_generation.common_misspellings import apply_misspelling
from error_generation.spacing_errors import apply_spacing_error
from error_generation.vowel_confusion import apply_vowel_confusion
from error_generation.consonant_errors import apply_consonant_error
from error_generation.conjugation_errors import apply_conjugation_error
from error_generation.suffix_errors import apply_suffix_error
from error_generation.particle_errors import apply_particle_error
from error_generation.word_substitution import apply_word_substitution
from error_generation.saisiot_errors import apply_saisiot_error
from error_generation.double_expression import apply_double_expression
from error_generation.foreign_style import apply_foreign_style
from error_generation.misc_errors import apply_misc_error
from error_generation.chat_style_errors import apply_chat_style
from error_generation.jamo_separation import apply_jamo_separation
from error_generation.punctuation_errors import apply_punctuation_error
from error_generation.honorific_errors import apply_honorific_error

from error_generation import common_misspellings
from error_generation import spacing_errors
from error_generation import vowel_confusion
from error_generation import consonant_errors
from error_generation import conjugation_errors
from error_generation import suffix_errors
from error_generation import particle_errors
from error_generation import word_substitution
from error_generation import saisiot_errors
from error_generation import double_expression
from error_generation import foreign_style
from error_generation import misc_errors
from error_generation import chat_style_errors
from error_generation import jamo_separation
from error_generation import punctuation_errors
from error_generation import honorific_errors


# 오류 생성 함수 목록과 가중치
# 가중치는 실제 사용 빈도/중요도를 반영한 상대적 확률
ErrorFn = Callable[[str, random.Random], Optional[str]]

ERROR_GENERATORS: list[tuple[str, ErrorFn, float]] = [
    ("common_misspellings",  apply_misspelling,        3.0),
    ("spacing_errors",       apply_spacing_error,      2.0),
    ("vowel_confusion",      apply_vowel_confusion,    2.0),
    ("consonant_errors",     apply_consonant_error,    1.5),
    ("conjugation_errors",   apply_conjugation_error,  2.5),
    ("suffix_errors",        apply_suffix_error,       2.0),
    ("particle_errors",      apply_particle_error,     1.5),
    ("word_substitution",    apply_word_substitution,  1.5),
    ("saisiot_errors",       apply_saisiot_error,      1.0),
    ("double_expression",    apply_double_expression,  1.0),
    ("foreign_style",        apply_foreign_style,      0.5),
    ("misc_errors",          apply_misc_error,         1.0),
    ("chat_style_errors",    apply_chat_style,         1.5),
    ("jamo_separation",      apply_jamo_separation,    1.0),
    ("punctuation_errors",   apply_punctuation_error,  1.0),
    ("honorific_errors",     apply_honorific_error,    1.0),
]

# 모든 모듈 목록 (패턴 수 집계용)
ALL_MODULES = [
    common_misspellings, spacing_errors, vowel_confusion,
    consonant_errors, conjugation_errors, suffix_errors,
    particle_errors, word_substitution, saisiot_errors,
    double_expression, foreign_style, misc_errors,
    chat_style_errors, jamo_separation, punctuation_errors,
    honorific_errors,
]


class KoreanErrorGenerator:
    """
    한국어 오류 생성기.

    여러 유형의 한국어 오류를 랜덤으로 주입하여
    GEC 모델의 학습 데이터를 다양하게 만드는 역할.

    Args:
        seed: 랜덤 시드 (재현성 보장)
    """

    def __init__(self, seed: int = 42):
        self._rng = random.Random(seed)
        self._names = [name for name, _, _ in ERROR_GENERATORS]
        self._fns = [fn for _, fn, _ in ERROR_GENERATORS]
        self._weights = [w for _, _, w in ERROR_GENERATORS]

    def set_seed(self, seed: int) -> None:
        """랜덤 시드를 재설정. 에포크마다 호출 권장."""
        self._rng = random.Random(seed)

    def state_dict(self) -> dict:
        return {"rng_state": self._rng.getstate()}

    def load_state_dict(self, state: dict) -> None:
        self._rng.setstate(state["rng_state"])

    def apply_single_error(self, text: str,
                           error_type: Optional[str] = None) -> Optional[str]:
        """
        텍스트에 단일 오류를 적용.

        Args:
            text: 올바른 한국어 문장
            error_type: 특정 오류 유형 이름 (None이면 가중치 기반 랜덤 선택)

        Returns:
            오류가 적용된 문장. 적용 가능한 패턴이 없으면 None.
        """
        if error_type is not None:
            # 지정된 유형 사용
            try:
                idx = self._names.index(error_type)
            except ValueError:
                raise ValueError(
                    f"알 수 없는 오류 유형: {error_type}. "
                    f"사용 가능: {self._names}"
                )
            return self._fns[idx](text, self._rng)

        # 가중치 기반 랜덤 선택, 최대 10번 시도
        for _ in range(10):
            [chosen_fn] = self._rng.choices(self._fns, weights=self._weights, k=1)
            result = chosen_fn(text, self._rng)
            if result is not None:
                return result

        return None

    def apply_random_errors(self, text: str, n_errors: int = 1) -> str:
        """
        텍스트에 n개의 독립적인 오류를 순차 적용.

        Args:
            text: 올바른 한국어 문장
            n_errors: 적용할 오류 수

        Returns:
            오류가 적용된 문장 (적용 가능한 패턴이 없으면 원문 반환)
        """
        result = text
        for _ in range(n_errors):
            errored = self.apply_single_error(result)
            if errored is not None:
                result = errored
        return result

    @staticmethod
    def get_total_pattern_count() -> int:
        """모든 모듈의 총 오류 패턴 수를 반환."""
        return sum(m.get_error_count() for m in ALL_MODULES)

    @staticmethod
    def get_module_stats() -> dict[str, int]:
        """각 모듈별 오류 패턴 수를 반환."""
        return {
            m.__name__.split(".")[-1]: m.get_error_count()
            for m in ALL_MODULES
        }

    @property
    def error_types(self) -> list[str]:
        """사용 가능한 오류 유형 이름 목록."""
        return list(self._names)
