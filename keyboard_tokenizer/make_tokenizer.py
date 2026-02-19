"""키보드 시퀀스 토크나이저 생성

vocab 구성:
  1. Special tokens: [PAD], [UNK], [BOS], [EOS], [SEP], [CLS], [MASK], [UNUSED0]~[UNUSED9]
  2. 제어 토큰: [SHIFT], [BLANK] — added token으로 등록
  3. 기본 자음 14개 (ㄱ~ㅎ)
  4. 기본 모음 12개 (ㅏ~ㅣ, ㅐ, ㅔ)
  5. 256 byte-level tokens (비한글 폴백)

ByteLevel BPE + merge rule로 비한글 문자를 바이트 폴백 처리.
한글 자모는 개별 토큰으로 직접 등록.
"""
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, AddedToken
import os
import sys
import json

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, ".."))

from keyboard_tokenizer.ko_keyboard import (
    SHIFT, BLANK, BASIC_CONSONANTS, BASIC_VOWELS,
)


def _build_byte_level_table():
    """ByteLevel 바이트↔문자 매핑 테이블 (GPT-2 스타일)"""
    bs = (list(range(ord("!"), ord("~") + 1)) +
          list(range(ord("¡"), ord("¬") + 1)) +
          list(range(ord("®"), ord("ÿ") + 1)))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


BYTE_TO_CHAR = _build_byte_level_table()


def build_vocab():
    """keyboard tokenizer vocab 구성"""
    vocab = {}
    tid = 0

    # 1. Special tokens
    special_tokens = [
        "[PAD]", "[UNK]", "[BOS]", "[EOS]", "[SEP]", "[CLS]", "[MASK]",
    ]
    # 여분 특수 토큰 (미래 사용 예약)
    special_tokens += [f"[UNUSED{i}]" for i in range(10)]
    for tok in special_tokens:
        vocab[tok] = tid
        tid += 1

    # 2. 제어 토큰
    for tok in [SHIFT, BLANK]:
        vocab[tok] = tid
        tid += 1

    # 3. 기본 자음 (14개)
    for ch in sorted(BASIC_CONSONANTS):
        byte_repr = "".join(BYTE_TO_CHAR[b] for b in ch.encode("utf-8"))
        if byte_repr not in vocab:
            vocab[byte_repr] = tid
            tid += 1

    # 4. 기본 모음 (12개)
    for ch in sorted(BASIC_VOWELS):
        byte_repr = "".join(BYTE_TO_CHAR[b] for b in ch.encode("utf-8"))
        if byte_repr not in vocab:
            vocab[byte_repr] = tid
            tid += 1

    # 5. 256 byte-level tokens
    for b in range(256):
        ch = BYTE_TO_CHAR[b]
        if ch not in vocab:
            vocab[ch] = tid
            tid += 1

    # 6. 자모 문자에 대한 merge rules (3바이트 UTF-8)
    merges = []
    merge_set = set()
    # 자모만 merge (제어 토큰은 added token이므로 merge 불필요)
    jamo_chars = sorted(BASIC_CONSONANTS | BASIC_VOWELS)

    for ch in jamo_chars:
        byte_tokens = [BYTE_TO_CHAR[b] for b in ch.encode("utf-8")]
        if len(byte_tokens) <= 1:
            continue
        current = byte_tokens[0]
        for i in range(1, len(byte_tokens)):
            next_byte = byte_tokens[i]
            merged = current + next_byte
            if merged not in vocab:
                vocab[merged] = tid
                tid += 1
            pair = (current, next_byte)
            if pair not in merge_set:
                merge_set.add(pair)
                merges.append(pair)
            current = merged

    return vocab, merges


def main():
    vocab, merges = build_vocab()

    print(f"Vocab size: {len(vocab)}")
    print(f"Merge rules: {len(merges)}")

    # 토크나이저 구성
    tokenizer = Tokenizer(models.BPE(
        vocab=vocab,
        merges=merges,
        unk_token="[UNK]",
    ))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    tokenizer.add_special_tokens([
        AddedToken("[PAD]", special=True),
        AddedToken("[UNK]", special=True),
        AddedToken("[BOS]", special=True),
        AddedToken("[EOS]", special=True),
        AddedToken("[SEP]", special=True),
        AddedToken("[CLS]", special=True),
        AddedToken("[MASK]", special=True),
        AddedToken(SHIFT, special=True),
        AddedToken(BLANK, special=True),
    ])

    # 저장
    output_path = os.path.join(SCRIPT_DIR, "keyboard_tokenizer.json")
    tokenizer.save(output_path)

    # vocab ID 매핑도 별도 저장 (래퍼에서 전처리된 토큰→ID 직접 매핑용)
    token_map = {}
    for ch in sorted(BASIC_CONSONANTS | BASIC_VOWELS):
        byte_repr = "".join(BYTE_TO_CHAR[b] for b in ch.encode("utf-8"))
        token_map[ch] = vocab[byte_repr]
    # 제어 토큰은 added token ID 사용
    token_map[SHIFT] = tokenizer.token_to_id(SHIFT)
    token_map[BLANK] = tokenizer.token_to_id(BLANK)
    token_map_path = os.path.join(SCRIPT_DIR, "jamo_token_map.json")
    with open(token_map_path, "w", encoding="utf-8") as f:
        json.dump(token_map, f, ensure_ascii=False, indent=2)

    print(f"\n토크나이저 저장: {output_path}")
    print(f"자모 토큰 맵 저장: {token_map_path}")
    print(f"  자모/제어 토큰: {len(token_map)}개")

    # 테스트
    print("\n--- 토큰화 테스트 (byte-level) ---")
    test_strs = ["ㄱ", "ㅏ", ".", "Hello", " "]
    for s in test_strs:
        enc = tokenizer.encode(s)
        dec = tokenizer.decode(enc.ids)
        print(f"  '{s}' → ids={enc.ids} → '{dec}' (일치: {s == dec})")


if __name__ == "__main__":
    main()
