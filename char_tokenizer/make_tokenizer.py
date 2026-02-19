"""한글 글자 단위 토크나이저 생성 (Character-level + Byte fallback)

vocabs 구성:
  1. Special tokens: [PAD], [UNK], [BOS], [EOS], [SEP], [MASK]
  2. 256 Byte-level tokens (BBPE 호환 바이트 폴백)
  3. 한글 완성형 11,172자 (가~힣, U+AC00~U+D7A3)
  4. 한글 호환 자모 (ㄱ~ㅎ, ㅏ~ㅣ)
  5. 영문 대소문자 + 숫자
  6. 일본어 히라가나 + 가타카나
  7. CJK 한자 (hanja + hanzi_cn + hanzi_tw + kanji 합집합)

각 멀티바이트 문자에 대해 바이트 → 문자 merge rule을 자동 생성하여
ByteLevel BPE와 호환되도록 한다.
"""
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, AddedToken
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
HANJA_DIR = os.path.join(PROJECT_ROOT, "hanja")


# ── ByteLevel 바이트↔문자 매핑 (GPT-2 스타일) ──────────────────────────

def _build_byte_level_table():
    """ByteLevel 바이트↔문자 매핑 테이블 (256 바이트 전체)"""
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


def char_to_byte_level(char: str) -> str:
    """유니코드 문자 1개 → ByteLevel 토큰 문자열"""
    return "".join(BYTE_TO_CHAR[b] for b in char.encode("utf-8"))


# ── 문자 집합 수집 ──────────────────────────────────────────────────────

def collect_hangul_syllables():
    """한글 완성형 11,172자 (가~힣)"""
    return [chr(cp) for cp in range(0xAC00, 0xD7A4)]


def collect_hangul_jamo_compat():
    """한글 호환 자모 (ㄱ~ㅎ, ㅏ~ㅣ)"""
    return [chr(cp) for cp in range(0x3131, 0x3164)]


def collect_english_and_digits():
    """영문 대소문자 + 숫자 (1바이트라 merge 불필요, 이미 byte-level에 포함)"""
    chars = []
    chars += [chr(c) for c in range(ord('A'), ord('Z') + 1)]
    chars += [chr(c) for c in range(ord('a'), ord('z') + 1)]
    chars += [chr(c) for c in range(ord('0'), ord('9') + 1)]
    return chars


def collect_japanese_kana():
    """히라가나 (U+3041~U+3096) + 가타카나 (U+30A1~U+30FA)"""
    hiragana = [chr(cp) for cp in range(0x3041, 0x3097)]
    katakana = [chr(cp) for cp in range(0x30A1, 0x30FB)]
    return hiragana + katakana


def collect_cjk_from_files():
    """hanja, hanzi_cn, hanzi_tw, kanji 파일에서 한자 합집합 수집"""
    cjk_set = set()
    filenames = ["hanja.txt", "hanzi_cn.txt", "hanzi_tw.txt", "kanji.txt"]
    for fname in filenames:
        fpath = os.path.join(HANJA_DIR, fname)
        if not os.path.exists(fpath):
            print(f"[WARN] {fpath} 없음, 건너뜀")
            continue
        with open(fpath, "r", encoding="utf-8") as f:
            text = f.read().strip()
            for ch in text:
                if ch.strip():
                    cjk_set.add(ch)
    return sorted(cjk_set)


# ── Vocab + Merge 빌드 ──────────────────────────────────────────────────

def build_vocab_and_merges():
    """vocab + merge 구성: 모든 멀티바이트 문자에 대해 바이트→문자 merge 자동 생성"""
    vocab = {}
    merges = []
    tid = 0

    # 1. Special tokens
    for tok in ["[PAD]", "[UNK]", "[BOS]", "[EOS]", "[SEP]", "[MASK]"]:
        vocab[tok] = tid
        tid += 1

    print(f"[1] Special tokens: {tid}개")

    # 2. ByteLevel 기본 256 바이트
    for b in range(256):
        ch = BYTE_TO_CHAR[b]
        if ch not in vocab:
            vocab[ch] = tid
            tid += 1

    print(f"[2] Byte-level tokens: {tid}개 (누적)")

    # 3. 멀티바이트 문자 수집
    multibyte_chars = []

    # 한글 완성형 11,172자
    hangul = collect_hangul_syllables()
    multibyte_chars += hangul
    print(f"[3] 한글 완성형: {len(hangul)}자")

    # 한글 호환 자모
    jamo = collect_hangul_jamo_compat()
    multibyte_chars += jamo
    print(f"[4] 한글 호환 자모: {len(jamo)}자")

    # 일본어 가나
    kana = collect_japanese_kana()
    multibyte_chars += kana
    print(f"[5] 일본어 가나: {len(kana)}자")

    # CJK 한자 (합집합)
    cjk = collect_cjk_from_files()
    multibyte_chars += cjk
    print(f"[6] CJK 한자: {len(cjk)}자")

    # 영문/숫자는 1바이트이므로 이미 byte-level에 포함 → merge 불필요
    english = collect_english_and_digits()
    print(f"[7] 영문/숫자 (byte-level에 이미 포함): {len(english)}자")

    # 4. 각 멀티바이트 문자에 대해 merge chain 생성
    #    예: 3바이트 문자 [b0, b1, b2]
    #    → merge(b0, b1) → b0b1 (중간토큰)
    #    → merge(b0b1, b2) → b0b1b2 (최종토큰 = 문자)
    seen_chars = set()
    merge_set = set()
    for ch in multibyte_chars:
        if ch in seen_chars:
            continue
        seen_chars.add(ch)

        byte_tokens = [BYTE_TO_CHAR[b] for b in ch.encode("utf-8")]
        if len(byte_tokens) <= 1:
            continue  # 1바이트는 merge 불필요

        current = byte_tokens[0]
        for i in range(1, len(byte_tokens)):
            next_byte = byte_tokens[i]
            merged = current + next_byte

            if merged not in vocab:
                vocab[merged] = tid
                tid += 1

            merge_pair = (current, next_byte)
            if merge_pair not in merge_set:
                merge_set.add(merge_pair)
                merges.append(merge_pair)

            current = merged

    print(f"\n[총계]")
    print(f"  Vocab size: {len(vocab)}")
    print(f"  Merge rules: {len(merges)}")
    print(f"  고유 멀티바이트 문자: {len(seen_chars)}")

    return vocab, merges


# ── 메인 ────────────────────────────────────────────────────────────────

def main():
    vocab, merges = build_vocab_and_merges()

    # 토크나이저 구성
    tokenizer = Tokenizer(models.BPE(
        vocab=vocab,
        merges=merges,
        unk_token="[UNK]",
    ))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    # HuggingFace Tokenizers 형식으로 special token 등록
    tokenizer.add_special_tokens([
        AddedToken("[PAD]", special=True),
        AddedToken("[UNK]", special=True),
        AddedToken("[BOS]", special=True),
        AddedToken("[EOS]", special=True),
        AddedToken("[SEP]", special=True),
        AddedToken("[MASK]", special=True),
    ])

    # 테스트
    print("\n--- 토큰화 테스트 ---")
    test_sentences = [
        "안녕하세요, 세계!",
        "맞춤법을 확인해 주세요.",
        "Hello, world!",
        "こんにちは世界",
        "カタカナテスト",
        "한자 테스트: 大韓民國",
        "中文测试简体",
        "mixed 한글English混合テスト",
        "ㄱㄴㄷㄹ ㅏㅓㅗㅜ",
    ]

    all_pass = True
    for test in test_sentences:
        encoded = tokenizer.encode(test)
        decoded = tokenizer.decode(encoded.ids, skip_special_tokens=False)
        match = test == decoded
        if not match:
            all_pass = False
        print(f"  원문: {test}")
        print(f"  토큰수: {len(encoded.ids)}")
        print(f"  복원: {decoded}")
        print(f"  일치: {'✓' if match else '✗ 불일치!'}")
        print()

    # 저장
    output_path = os.path.join(SCRIPT_DIR, "char_level_tokenizer.json")
    tokenizer.save(output_path)

    print(f"{'='*60}")
    print(f"토크나이저 생성됨: {output_path}")
    print(f"총 vocab: {len(vocab)}")
    print(f"  special: 6")
    print(f"  byte-level: 256")
    print(f"  멀티바이트 문자 + 중간 merge 토큰: {len(vocab) - 262}")
    print(f"전체 테스트: {'PASS ✓' if all_pass else 'FAIL ✗'}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
