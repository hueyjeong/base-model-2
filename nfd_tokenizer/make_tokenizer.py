from tokenizers import Tokenizer, models, pre_tokenizers, decoders, AddedToken
import unicodedata
import os
import re


_HANGUL_RE = re.compile(r'[\uAC00-\uD7A3\u1100-\u1112\u1161-\u1175\u11A8-\u11C2\u3131-\u3163]')


def nfd_decompose(text):
    if _HANGUL_RE.search(text):
        return unicodedata.normalize('NFD', text)
    return text


def recover_nfd(text):
    if _HANGUL_RE.search(text):
        return unicodedata.normalize('NFC', text)
    return text


def _build_byte_level_table():
    """ByteLevel 바이트↔문자 매핑 테이블"""
    bs = list(range(ord("!"), ord("~") + 1)) + \
         list(range(ord("¡"), ord("¬") + 1)) + \
         list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


BYTE_TO_CHAR = _build_byte_level_table()


def char_to_byte_level(char):
    """유니코드 문자 1개 → ByteLevel 토큰 문자열"""
    return "".join(BYTE_TO_CHAR[b] for b in char.encode("utf-8"))


def build_vocab_and_merges():
    """vocab + merge 구성: 멀티바이트 문자에 대해 바이트→문자 merge를 자동 생성"""
    vocab = {}
    merges = []
    tid = 0

    # 1. special tokens
    for tok in ["[UNK]", "[BOS]", "[EOS]", "[PAD]", "[SEP]", "[MASK]"]:
        vocab[tok] = tid
        tid += 1

    # 2. ByteLevel 기본 256 바이트 (single-byte tokens)
    for b in range(256):
        ch = BYTE_TO_CHAR[b]
        if ch not in vocab:
            vocab[ch] = tid
            tid += 1

    # 3. 멀티바이트 문자 등록 대상
    multibyte_chars = []
    # 한글 결합용 초성
    multibyte_chars += [chr(cp) for cp in range(0x1100, 0x1113)]
    # 한글 결합용 중성
    multibyte_chars += [chr(cp) for cp in range(0x1161, 0x1176)]
    # 한글 결합용 종성
    multibyte_chars += [chr(cp) for cp in range(0x11A8, 0x11C3)]
    # 한글 호환용 자모
    multibyte_chars += [chr(cp) for cp in range(0x3131, 0x3164)]
    # 히라가나
    multibyte_chars += [chr(cp) for cp in range(0x3041, 0x3097)]
    # 가타카나
    multibyte_chars += [chr(cp) for cp in range(0x30A1, 0x30FB)]
    # 4. 각 문자에 대해 merge 체인 생성
    #    예: 3바이트 문자 [b0, b1, b2]
    #    → merge(b0, b1) → b0b1 (중간토큰, vocab 등록)
    #    → merge(b0b1, b2) → b0b1b2 (최종토큰, vocab 등록)
    for ch in multibyte_chars:
        byte_tokens = [BYTE_TO_CHAR[b] for b in ch.encode("utf-8")]
        # 순차적으로 merge
        current = byte_tokens[0]
        for i in range(1, len(byte_tokens)):
            next_byte = byte_tokens[i]
            merged = current + next_byte
            # 중간/최종 토큰을 vocab에 등록
            if merged not in vocab:
                vocab[merged] = tid
                tid += 1
            merge_pair = (current, next_byte)
            if merge_pair not in merges:
                merges.append(merge_pair)
            current = merged

    return vocab, merges


vocab, merges = build_vocab_and_merges()
print(f"vocab size: {len(vocab)}")
print(f"merge rules: {len(merges)}")

# 토크나이저 구성
tokenizer = Tokenizer(models.BPE(vocab=vocab, merges=merges, unk_token="[UNK]"))
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
tokenizer.decoder = decoders.ByteLevel()

# special tokens를 단일 토큰으로 등록
tokenizer.add_special_tokens([
    AddedToken("<|BOHJ|>", special=True),
    AddedToken("<|EOHJ|>", special=True),
    AddedToken("[MASK]", special=True),
])

# 테스트
print("\n--- 토큰화 테스트 ---")
test_sentences = [
    "꼰대희: 빕묵자",
    "맞춤법을 확인해 주세요.",
    "Hello, world!",
    "こんにちは世界",
    "カタカナテスト",
    "<|BOHJ|>국+21<|EOHJ|>と<|BOHJ|>국+23<|EOHJ|>",
]
for test in test_sentences:
    decomposed = nfd_decompose(test)
    encoded = tokenizer.encode(decomposed)
    decoded = recover_nfd(tokenizer.decode(encoded.ids, skip_special_tokens=False))
    match = test == decoded
    print(f"원문: {test}")
    print(f"토큰수: {len(encoded.ids)}")
    print(f"복원: {decoded}")
    print(f"일치: {match}")
    if not match:
        print("  !! 불일치 !!")
    print()

output_path = os.path.join(os.path.dirname(__file__), "custom_gec_tokenizer_manual.json")
tokenizer.save(output_path)
print(f"토크나이저 생성됨: {output_path}")
print(f"총 vocab: {len(vocab)} (special 7 + bytes 256 + 멀티바이트 토큰)")
