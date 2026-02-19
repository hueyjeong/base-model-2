"""토크나이저 래퍼 — 한자 전처리 + NFD + ByteLevelBPE 통합

encode: 한자전처리 → NFD → BPE토큰화 → BOS/EOS 추가
decode: BPE디코드 → NFC복원 → 한자후처리
"""
import os
import re
import unicodedata
from typing import List

from tokenizers import Tokenizer

from hanja_preprocessor import preprocess as hanja_preprocess
from hanja_preprocessor import postprocess as hanja_postprocess

_HANGUL_RE = re.compile(r'[\uAC00-\uD7A3\u1100-\u1112\u1161-\u1175\u11A8-\u11C2\u3131-\u3163]')

_DEFAULT_TOKENIZER_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "custom_gec_tokenizer_manual.json",
)


def _nfd(text: str) -> str:
    if _HANGUL_RE.search(text):
        return unicodedata.normalize("NFD", text)
    return text


def _nfc(text: str) -> str:
    if _HANGUL_RE.search(text):
        return unicodedata.normalize("NFC", text)
    return text


class GECTokenizer:
    """한자 전처리 + NFD + ByteLevelBPE 통합 토크나이저"""

    def __init__(self, path: str = _DEFAULT_TOKENIZER_PATH):
        self.tokenizer = Tokenizer.from_file(path)
        self.bos_id = self.tokenizer.token_to_id("[BOS]")
        self.eos_id = self.tokenizer.token_to_id("[EOS]")
        self.pad_id = self.tokenizer.token_to_id("[PAD]")
        self.sep_id = self.tokenizer.token_to_id("[SEP]")
        self.unk_id = self.tokenizer.token_to_id("[UNK]")
        self.mask_id = self.tokenizer.token_to_id("[MASK]")
        self.vocab_size = self.tokenizer.get_vocab_size()

    def encode(self, text: str, add_special: bool = True) -> List[int]:
        """텍스트 → 토큰 ID 리스트

        파이프라인: 한자전처리 → NFD분해 → BPE토큰화
        """
        text = hanja_preprocess(text)
        text = _nfd(text)
        ids = self.tokenizer.encode(text).ids
        if add_special:
            ids = [self.bos_id] + ids + [self.eos_id]
        return ids

    def decode(self, ids: List[int], skip_special: bool = True) -> str:
        """토큰 ID 리스트 → 텍스트

        파이프라인: BPE디코드 → NFC복원 → 한자후처리
        """
        text = self.tokenizer.decode(ids, skip_special_tokens=skip_special)
        text = _nfc(text)
        text = hanja_postprocess(text)
        return text

    def encode_batch(self, texts: List[str], add_special: bool = True) -> List[List[int]]:
        return [self.encode(t, add_special) for t in texts]

    def __len__(self) -> int:
        return self.vocab_size
