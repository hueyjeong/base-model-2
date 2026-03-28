"""
KAGAS 11-type 오류 분류 매핑 테이블.

Yoon et al. 2023 (ACL)의 KAGAS 체계에 기반하여
우리 error_generation 모듈 24+α 타입을 11개 표준 유형으로 분류.

Phase 1/2 평가 시 오류 유형별 P/R/F0.5 분석에 사용.

참고:
- KAGAS 원본: 14 → 11 타입 (통합)
- KoGEC 2025: 모델 교정 패턴 기준 분포 데이터
"""

# KAGAS 유형 → 우리 오류 모듈 이름 매핑
KAGAS_MAP: dict[str, list[str]] = {
    "WS":          ["spacing_errors"],
    "PUNCT":       ["punctuation_errors", "emoticon_spacing", "char_repeat"],
    "SPELL":       ["common_misspellings", "vowel_confusion", "consonant_errors",
                    "g2pk_pronunciation", "heterograph_errors", "jamo_separation",
                    "phoneme_errors", "typing_language_errors"],
    "PART":        ["particle_errors"],
    "VERB_ADJ":    ["conjugation_errors", "tense_errors"],
    "PRO_NOUN":    ["word_substitution", "semantic_errors", "number_errors"],
    "END":         ["suffix_errors"],
    "MODIFIER":    ["foreign_style"],
    "INS":         ["grammar_addition"],
    "DEL":         ["grammar_remove"],
    "SP_RELATION": ["word_order_errors", "double_expression"],
}

# 역방향 매핑: 우리 모듈 이름 → KAGAS 유형
REVERSE_KAGAS_MAP: dict[str, str] = {
    our_type: kagas_type
    for kagas_type, our_types in KAGAS_MAP.items()
    for our_type in our_types
}

# KoGEC 2025 기준 실제 오류 분포 (참고용)
KOGEC_DISTRIBUTION: dict[str, float] = {
    "WS":          0.213,
    "PUNCT":       0.298,
    "VERB_ADJ":    0.106,
    "PRO_NOUN":    0.106,
    "DEL":         0.106,
    "INS":         0.064,
    "END":         0.043,
    "SPELL":       0.021,
    "PART":        0.021,
    "MODIFIER":    0.011,
    "SP_RELATION": 0.011,
}

# 매핑되지 않는 보조 모듈 (KAGAS에 직접 대응 없음)
UNMAPPED_MODULES = [
    "saisiot_errors",      # 사이시옷 → SPELL에 가까움
    "foreign_word_errors",  # 외래어 → SPELL에 가까움
    "misc_errors",          # 기타
    "chat_style_errors",    # 구어체
    "honorific_errors",     # 존칭
]


def get_kagas_type(module_name: str) -> str:
    """모듈 이름으로 KAGAS 유형 조회. 매핑 없으면 'OTHER' 반환."""
    return REVERSE_KAGAS_MAP.get(module_name, "OTHER")


if __name__ == "__main__":
    print("=== KAGAS 11-type 매핑 ===")
    for kagas, modules in KAGAS_MAP.items():
        pct = KOGEC_DISTRIBUTION.get(kagas, 0) * 100
        print(f"  {kagas:12s} ({pct:5.1f}%) → {', '.join(modules)}")

    print()
    print(f"총 매핑 모듈 수: {len(REVERSE_KAGAS_MAP)}")
    print(f"미매핑 모듈: {UNMAPPED_MODULES}")
