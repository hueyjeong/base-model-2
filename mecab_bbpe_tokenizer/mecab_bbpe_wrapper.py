"""MeCab + BBPE 토크나이저 래퍼 — BaseTokenizer 인터페이스 구현

HuggingFace tokenizers JSON을 로드하여 공통 인터페이스를 제공한다.
학습 시 MeCab 사전 분절을 사용했다면, 추론 시에도 동일하게 적용한다.
"""
import os
import sys
from typing import List

from tokenizers import Tokenizer

# 프로젝트 루트를 path에 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tokenizer_base import BaseTokenizer


class MeCabBBPETokenizer(BaseTokenizer):
    """MeCab + ByteLevel BPE 토크나이저 래퍼

    학습 시 MeCab 사전 분절을 사용한 경우,
    encode 시에도 동일하게 MeCab 분절을 적용하여
    학습-추론 간 토큰화 일관성을 유지한다.
    """

    def __init__(self, json_path: str, use_mecab: bool = True):
        """
        Args:
            json_path: HuggingFace tokenizers JSON 파일 경로
            use_mecab: MeCab 사전 분절 사용 여부 (학습 시 사용했다면 True)
        """
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"토크나이저 파일을 찾을 수 없습니다: {json_path}")

        self._tok = Tokenizer.from_file(json_path)

        # MeCab 설정
        self._use_mecab = use_mecab
        self._mecab = None
        if use_mecab:
            import MeCab
            tagger_created = False
            for dic_module in ["mecab_ko_dic", "unidic_lite"]:
                try:
                    mod = __import__(dic_module)
                    dicdir = mod.DICDIR
                    self._mecab = MeCab.Tagger(f"-O wakati -r /dev/null -d {dicdir}")
                    tagger_created = True
                    break
                except (ImportError, AttributeError, RuntimeError):
                    continue
            if not tagger_created:
                try:
                    self._mecab = MeCab.Tagger("-O wakati")
                except RuntimeError:
                    self._mecab = MeCab.Tagger("-O wakati -r /dev/null")

        # Special token IDs
        self._pad_id = self._tok.token_to_id("[PAD]")
        self._unk_id = self._tok.token_to_id("[UNK]")
        self._bos_id = self._tok.token_to_id("[BOS]")
        self._eos_id = self._tok.token_to_id("[EOS]")
        self._sep_id = self._tok.token_to_id("[SEP]")
        self._cls_id = self._tok.token_to_id("[CLS]")
        self._mask_id = self._tok.token_to_id("[MASK]")

    @property
    def vocab_size(self) -> int:
        return self._tok.get_vocab_size()

    @property
    def pad_id(self) -> int:
        return self._pad_id

    @property
    def bos_id(self) -> int:
        return self._bos_id

    @property
    def eos_id(self) -> int:
        return self._eos_id

    @property
    def unk_id(self) -> int:
        return self._unk_id

    @property
    def sep_id(self) -> int:
        return self._sep_id

    @property
    def cls_id(self) -> int:
        return self._cls_id

    @property
    def mask_id(self) -> int:
        return self._mask_id

    def _mecab_segment(self, text: str) -> str:
        """MeCab wakati 분절: 형태소 단위로 공백 구분"""
        if self._mecab is None:
            return text
        result = self._mecab.parse(text)
        return result.strip() if result else text

    def encode(self, text: str, add_special: bool = True) -> List[int]:
        """텍스트 → 토큰 ID 리스트

        파이프라인: [MeCab 분절 →] BPE 인코딩 → [BOS/EOS 추가]

        Args:
            text: 입력 텍스트
            add_special: True면 BOS/EOS 추가
        Returns:
            토큰 ID 리스트
        """
        if self._use_mecab:
            text = self._mecab_segment(text)
        encoded = self._tok.encode(text)
        ids = encoded.ids
        if add_special:
            ids = [self._bos_id] + ids + [self._eos_id]
        return ids

    def decode(self, ids: List[int], skip_special: bool = True) -> str:
        """토큰 ID 리스트 → 텍스트

        Args:
            ids: 토큰 ID 리스트
            skip_special: True면 special token 제거 후 디코딩
        Returns:
            디코딩된 텍스트
        """
        if skip_special:
            special = {self._pad_id, self._bos_id, self._eos_id, self._unk_id}
            ids = [i for i in ids if i not in special]
        return self._tok.decode(ids, skip_special_tokens=False)

    def encode_batch(self, texts: List[str], add_special: bool = True) -> List[List[int]]:
        """배치 인코딩 (MeCab 분절 포함)"""
        if self._use_mecab:
            texts = [self._mecab_segment(t) for t in texts]
        encoded_list = self._tok.encode_batch(texts)
        all_ids = [enc.ids for enc in encoded_list]
        if add_special:
            all_ids = [[self._bos_id] + ids + [self._eos_id] for ids in all_ids]
        return all_ids

    def id_to_piece(self, id: int) -> str:
        """토큰 ID → 토큰 문자열"""
        return self._tok.id_to_token(id)

    def piece_to_id(self, piece: str) -> int:
        """토큰 문자열 → 토큰 ID"""
        return self._tok.token_to_id(piece)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MeCab BBPE 토크나이저 테스트")
    parser.add_argument("--model", "-m", required=True, help="토크나이저 JSON 파일 경로")
    parser.add_argument("--test", action="store_true", help="테스트 실행")
    args = parser.parse_args()

    tokenizer = MeCabBBPETokenizer(args.model)
    print(f"Vocab size: {tokenizer.vocab_size}")
    print(f"PAD={tokenizer.pad_id}, BOS={tokenizer.bos_id}, "
          f"EOS={tokenizer.eos_id}, UNK={tokenizer.unk_id}")

    if args.test:
        test_sentences = [
            "맞춤법을 확인해 주세요.",
            "Hello, world!",
            "한글과 English 혼합 테스트",
            "こんにちは世界",
        ]
        print("\n--- 인코드/디코드 라운드트립 ---")
        for sent in test_sentences:
            ids = tokenizer.encode(sent, add_special=False)
            decoded = tokenizer.decode(ids, skip_special=False)
            match = sent == decoded
            print(f"  원문: {sent}")
            print(f"  토큰수: {len(ids)}")
            print(f"  복원: {decoded}")
            print(f"  일치: {match}")
            if not match:
                print("  !! 불일치 !!")
            print()
