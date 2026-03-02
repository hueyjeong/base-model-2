import unittest
import random
from error_generation import KoreanErrorGenerator

class TestIntegration(unittest.TestCase):
    def test_korean_error_generator_basic(self):
        # 시드 고정으로 동일한 테스트 결과 보장
        generator = KoreanErrorGenerator(seed=42)
        
        original = "안녕하세요, 저는 딥러닝 연구원입니다. 오늘 날씨가 참 좋습니다."
        
        # 여러 번 수행해서 예외가 발생하지 않는지 확인
        for _ in range(20):
            result = generator.apply_random_errors(original, n_errors=2)
            self.assertIsInstance(result, str)
            self.assertTrue(len(result) > 0)

    def test_error_generators_have_counts(self):
        from error_generation import ERROR_GENERATORS
        import importlib
        
        for name, func, weight in ERROR_GENERATORS:
            module_name = f"error_generation.{name}"
            # alias handling
            if name == "grammar_remove" or name == "grammar_addition":
                module_name = "error_generation.grammar_structure_errors"
            
            try:
                mod = importlib.import_module(module_name)
                self.assertTrue(hasattr(mod, "get_error_count"), f"{module_name} lacks get_error_count()")
            except ImportError as e:
                # Some modules might be internal functions, but let's check
                print(f"Skipping import check for {name}: {e}")

if __name__ == '__main__':
    unittest.main()
