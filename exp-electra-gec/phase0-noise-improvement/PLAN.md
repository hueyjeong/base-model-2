# Phase 0: 노이즈 엔진 개선 — 실제 오류 분포 기반 체계화

## 배경

한국어 GEC 연구 조사 결과 (Lee et al. 2021, Yoon et al. 2023 ACL, KoGEC 2025),
현재 노이즈 엔진에 두 가지 핵심 갭 발견:

1. **G2PK(발음→철자) 노이즈 부재**: "밥을"→"바블"(연음), "독립"→"동닙"(비음화) 등
   한국어 특유의 발음 기반 오류를 동적으로 생성하는 기능 없음.
   현재 `phoneme_errors`는 27개 하드코딩 패턴만 존재.

2. **가중치와 실제 오류 분포 불일치**: KoGEC 데이터 기준 WS 21%, PUNCT 30%인데,
   현재 가중치(`spacing:4.0, punctuation:2.0`)는 이 비율을 정확히 반영하지 못함.

이 개선은 Phase 1/2에 선행하며, 기존 BiMamba-2 학습에도 바로 적용 가능.

## 실제 오류 분포 참고

### KoGEC 2025 — 모델 교정 패턴 기준

| 오류 유형 | GPT-4o | HCX-3 | KoGEC |
|---------|--------|-------|-------|
| WS (띄어쓰기) | 14.1% | 13.6% | 21.3% |
| PUNCT (구두점) | 43.8% | 52.3% | 29.8% |
| VERB_ADJ (용언) | 4.7% | 2.3% | 10.6% |
| PRO_NOUN (체언) | 1.6% | 4.5% | 10.6% |
| DEL (삭제) | 6.3% | 5.7% | 10.6% |
| INS (삽입) | 6.3% | 3.4% | 6.4% |
| END (어미) | 10.9% | 10.2% | 4.3% |
| SPELL (철자) | 4.7% | 5.7% | 2.1% |
| PART (조사) | 1.6% | 2.3% | 2.1% |

### 대학생 글쓰기 연구 (이경미 2018) — 띄어쓰기 세부

| 세부 유형 | 비율 |
|---------|------|
| 용언+보조용언 ("해 주다/해주다") | 40.5% |
| 명사+명사 (복합명사) | 21.8% |
| 체언+용언 결합 | 14.9% |
| 의존명사 ("할 수/할수") | 6.4% |

---

## 개선 1: G2PK 기반 발음→철자 노이즈

`g2pk` 패키지로 텍스트의 표준 발음을 얻고, 발음을 다시 한글로 역변환하여
"발음대로 잘못 쓴" 오류를 생성.

```
원문:  "독립문"  →  발음: "동님문"  →  오류: "동님문"
원문:  "밥을"    →  발음: "바블"    →  오류: "바블"
```

- 신규 파일: `error_generation/g2pk_noise.py`
- 의존성: `pip install g2pk`
- 등록 가중치: 2.0

변환 대상 음운 규칙:
- 연음, 경음화, 비음화, 구개음화, 격음화, ㅎ탈락

## 개선 2: 이형문자(Heterograph) 노이즈

유사 발음 음절 내 자모 교체 (음절 단위).

```
원문:  "예상하다"  →  오류: "얘상하다"  (ㅖ ↔ ㅒ)
원문:  "먹었다"    →  오류: "먺었다"    (받침 ㄱ ↔ ㄲ)
```

- 신규 파일: `error_generation/heterograph.py`
- 종성 혼동: ㄱ/ㅋ/ㄲ, ㄷ/ㅌ/ㅅ/ㅆ/ㅈ/ㅊ, ㅂ/ㅍ
- 모음 혼동: ㅐ/ㅔ, ㅒ/ㅖ, ㅚ/ㅙ/ㅞ
- 기존 `vowel_confusion.py`의 ㅐ↔ㅔ와 중복되나, 종성까지 포함한 체계적 확장
- 등록 가중치: 1.5

## 개선 3: 가중치 리밸런싱

실제 오류 분포를 반영한 `"realistic"` 프리셋 추가. 기존 가중치는 `"default"`로 유지.

| KAGAS 유형 | 실제 비율 | 우리 모듈 | default | realistic |
|-----------|---------|---------|---------|-----------|
| WS | ~25% | spacing_errors | 4.0 | 5.0 |
| PUNCT | ~25% | punctuation_errors | 2.0 | 4.0 |
| SPELL | ~10% | misspelling + vowel + consonant + G2PK + heterograph | 3.0 | 3.5 (합산) |
| VERB_ADJ | ~10% | conjugation + tense | 1.5 | 2.5 |
| PRO_NOUN | ~8% | word_sub + semantic | 2.5 | 2.0 |
| DEL/INS | ~8% | grammar_remove/add | 2.0 | 2.0 |
| END | ~5% | suffix_errors | 0.5 | 1.5 |
| PART | ~3% | particle_errors | 0.5 | 1.0 |

- `NoiseConfig`에 `weight_preset: str = "default"` 필드 추가
- `training/noising.py`에 `WEIGHT_PRESETS` 딕셔너리
- CLI: `--noise_preset realistic`

## 개선 4: KAGAS 11-type 매핑 테이블

우리 24 타입 → KAGAS 11 타입 매핑. Phase 1 평가 시 표준 비교에 사용.

- 신규 파일: `error_generation/kagas_mapping.py`

```python
KAGAS_MAP = {
    "WS":          ["spacing_errors"],
    "PUNCT":       ["punctuation_errors"],
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
```

---

## 수정/생성 파일 요약

| 파일 | 작업 | 분량 |
|------|------|------|
| `error_generation/g2pk_noise.py` | 신규 | ~80줄 |
| `error_generation/heterograph.py` | 신규 | ~100줄 |
| `error_generation/kagas_mapping.py` | 신규 | ~30줄 |
| `error_generation/__init__.py` | 수정 | +10줄 |
| `training/noising.py` | 수정 | +20줄 |
| `training/noise_config.example.json` | 수정 | 프리셋 추가 |

기존 동작 변경 없음 — `weight_preset="default"` 기본값으로 하위호환.

## 참고 문헌

- Lee et al. 2021 — Korean GEC with Noise Implantation (G2PK, Heterograph, Heuristic, Spacing)
- Yoon et al. 2023 — KAGAS 14→11 오류 유형 체계 (ACL)
- KoGEC 2025 — 오류 유형별 분포 데이터
- 이경미 2018 — 대학생 띄어쓰기 오류 유형 통계
