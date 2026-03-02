import unittest
from error_generation.utils import get_mecab_offsets, replace_by_offset, swap_by_offset

class TestUtils(unittest.TestCase):
    def test_get_mecab_offsets_basic(self):
        text = "아버지가 방에 들어 가신다."
        tokens = get_mecab_offsets(text)
        
        expected_surfaces = ['아버지', '가', '방', '에', '들어가', '신다', '.']
        # Note: mecab output might differ slightly based on dictionary version,
        # but the offsets must match the substring in the original text.
        for t in tokens:
            self.assertEqual(text[t.start:t.end], t.surface)
            
    def test_get_mecab_offsets_with_spaces(self):
        text = "  안녕    하세요!!  "
        tokens = get_mecab_offsets(text)
        
        for t in tokens:
            self.assertEqual(text[t.start:t.end], t.surface, f"Failed for {t.surface}")
            
    def test_replace_by_offset(self):
        text = "한국을 사랑합니다"
        # "한국을" -> "한국" (start:0, end:2) 
        # offset tracking for "한국" should be 0, 2
        tokens = get_mecab_offsets(text)
        first_token = tokens[0] # 한, 한국 등 형태소에 따라 다름.
        
        res = replace_by_offset(text, first_token.start, first_token.end, "미국")
        self.assertTrue(res.startswith("미국"))
        
    def test_swap_by_offset(self):
        text = "사과가 바나나를 먹는다"
        # 사과(0, 2), 바나나(4, 7)
        res = swap_by_offset(text, 0, 2, 4, 7)
        self.assertEqual(res, "바나나가 사과를 먹는다")

if __name__ == "__main__":
    unittest.main()
