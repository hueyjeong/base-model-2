"""
오류 생성 모듈 테스트 스크립트.

실행:
    python -m error_generation.test_errors
"""

import sys
import os

# 프로젝트 루트를 sys.path에 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from error_generation import KoreanErrorGenerator


def test_module_stats():
    """모든 모듈이 최소 1개 이상의 오류 패턴을 가지는지 확인."""
    stats = KoreanErrorGenerator.get_module_stats()
    print("=== 모듈별 오류 패턴 수 ===")
    total = 0
    for name, count in stats.items():
        print(f"  {name}: {count}개")
        total += count
        assert count > 0, f"{name} 모듈에 오류 패턴이 없습니다!"
    print(f"  총 패턴 수: {total}개")
    print()
    return total


def test_individual_modules():
    """각 모듈의 치환이 실제로 동작하는지 확인."""
    gen = KoreanErrorGenerator(seed=42)

    test_cases = [
        ("common_misspellings",  "굳이 그럴 필요가 없다."),
        ("spacing_errors",       "이것뿐이다."),
        ("vowel_confusion",      "대개 그런 편이다."),
        ("consonant_errors",     "윷놀이를 하자."),
        ("conjugation_errors",   "먹든 말든 알아서 해라."),
        ("suffix_errors",        "깨끗이 닦아라."),
        ("particle_errors",      "학생으로서 해야 할 일이다."),
        ("word_substitution",    "그것은 다른 이야기다."),
        ("saisiot_errors",       "개수를 세어보자."),
        ("double_expression",    "잊히다."),
        ("foreign_style",        "곧 도착합니다."),
        ("misc_errors",          "사흘 뒤에 만나자."),
    ]

    print("=== 개별 모듈 테스트 ===")
    passed = 0
    for error_type, sentence in test_cases:
        result = gen.apply_single_error(sentence, error_type=error_type)
        if result is not None and result != sentence:
            print(f"  ✓ {error_type}: \"{sentence}\" → \"{result}\"")
            passed += 1
        else:
            print(f"  ✗ {error_type}: 오류 적용 실패 (원문: \"{sentence}\")")

    print(f"  통과: {passed}/{len(test_cases)}")
    print()
    return passed == len(test_cases)


def test_reproducibility():
    """같은 시드에서 같은 결과가 나오는지 확인 (재현성)."""
    sentence = "굳이 그럴 필요가 없다."

    gen1 = KoreanErrorGenerator(seed=123)
    result1 = gen1.apply_random_errors(sentence, n_errors=1)

    gen2 = KoreanErrorGenerator(seed=123)
    result2 = gen2.apply_random_errors(sentence, n_errors=1)

    print("=== 재현성 테스트 ===")
    if result1 == result2:
        print(f"  ✓ 동일 시드 → 동일 결과: \"{result1}\"")
    else:
        print(f"  ✗ 결과 불일치: \"{result1}\" vs \"{result2}\"")
    print()
    return result1 == result2


def test_diversity():
    """다른 시드에서 다른 결과가 나오는지 확인 (다양성)."""
    sentence = "굳이 그럴 필요가 없다."
    results = set()

    for seed in range(100):
        gen = KoreanErrorGenerator(seed=seed)
        result = gen.apply_random_errors(sentence, n_errors=1)
        results.add(result)

    print("=== 다양성 테스트 ===")
    print(f"  100개 시드에서 {len(results)}개의 고유 결과 생성")
    samples = list(results)[:5]
    for s in samples:
        print(f"    예시: \"{s}\"")
    print()
    return len(results) > 1


def test_multiple_errors():
    """여러 오류를 연속 적용할 수 있는지 확인."""
    gen = KoreanErrorGenerator(seed=42)
    sentence = "굳이 깨끗이 닦을 필요가 없다."

    result = gen.apply_random_errors(sentence, n_errors=3)

    print("=== 다중 오류 적용 테스트 ===")
    print(f"  원문: \"{sentence}\"")
    print(f"  결과: \"{result}\"")
    changed = sentence != result
    if changed:
        print("  ✓ 오류 적용 성공")
    else:
        print("  ✗ 오류 적용 실패")
    print()
    return changed


def test_no_crash_on_empty():
    """빈 문자열이나 오류 적용이 불가능한 문장에서 크래시하지 않는지 확인."""
    gen = KoreanErrorGenerator(seed=42)

    tests = [
        "",
        "Hello World",
        "12345",
        "abc def",
    ]

    print("=== 안전성 테스트 ===")
    all_safe = True
    for t in tests:
        try:
            result = gen.apply_random_errors(t, n_errors=1)
            print(f"  ✓ 입력: \"{t}\" → \"{result}\"")
        except Exception as e:
            print(f"  ✗ 입력: \"{t}\" → 예외 발생: {e}")
            all_safe = False
    print()
    return all_safe


def test_error_types_list():
    """사용 가능한 오류 유형 목록 확인."""
    gen = KoreanErrorGenerator(seed=42)
    types = gen.error_types

    print("=== 오류 유형 목록 ===")
    for t in types:
        print(f"  - {t}")
    print(f"  총 {len(types)}개 유형")
    print()
    return len(types) == 12


def main():
    print("=" * 60)
    print("  한국어 오류 생성 모듈 테스트")
    print("=" * 60)
    print()

    results = []
    results.append(("모듈 패턴 수", test_module_stats() > 0))
    results.append(("개별 모듈", test_individual_modules()))
    results.append(("재현성", test_reproducibility()))
    results.append(("다양성", test_diversity()))
    results.append(("다중 오류", test_multiple_errors()))
    results.append(("안전성", test_no_crash_on_empty()))
    results.append(("유형 목록", test_error_types_list()))

    print("=" * 60)
    print("  테스트 결과 요약")
    print("=" * 60)
    all_passed = True
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {name}")
        if not passed:
            all_passed = False

    print()
    if all_passed:
        print("  🎉 모든 테스트 통과!")
    else:
        print("  ⚠️  일부 테스트 실패")
        sys.exit(1)


if __name__ == "__main__":
    main()
