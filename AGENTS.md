# AGENTS.md — base-model-2

> Context document for AI coding assistants working on this project.

## Project Overview

**base-model-2** is a Korean Grammatical Error Correction (GEC) project built around a **BitNet-Mamba encoder-decoder (Seq2Seq)** architecture. The project includes:

- Multiple **tokenizer variants** for Korean text, all conforming to a shared abstract interface
- A custom **BitNet-Mamba Seq2Seq** model (≈128M non-embedding parameters)
- **Hanja preprocessing** utilities
- **Korean error generation** for synthetic training data
- Training corpus management

The primary language of in-code comments and docstrings is **Korean (한국어)**.

---

## Directory Structure

```
base-model-2/
├── tokenizer_base.py          # Abstract base class (BaseTokenizer) for all tokenizers
├── verify_model.py            # Model verification script (forward/backward pass tests)
├── requirements.txt           # Python dependencies (pip freeze)
│
├── model/                     # BitNet-Mamba Seq2Seq model package
│   ├── __init__.py
│   ├── config.py              # BitMambaSeq2SeqConfig dataclass
│   ├── seq2seq.py             # Top-level Seq2Seq model
│   ├── encoder.py             # Mamba-based encoder
│   ├── decoder.py             # Mamba-based decoder
│   ├── mamba_block.py         # Mamba SSM block
│   ├── bitlinear.py           # BitLinear (1-bit weight) layer
│   └── cross_attention.py     # Cross-attention between encoder/decoder
│
├── nfd_tokenizer/             # NFD-decomposition + ByteLevel BPE tokenizer
│   ├── make_tokenizer.py      # Builds tokenizer vocabulary & merges
│   ├── tokenizer_wrapper.py   # GECTokenizer wrapper (implements BaseTokenizer)
│   ├── hanja_preprocessor.py  # Hanja → Hangul conversion
│   ├── hangul_hanja_map.json  # Hanja mapping data
│   └── custom_gec_tokenizer_manual.json  # Generated tokenizer JSON
│
├── keyboard_tokenizer/        # Korean 2-beolsik keyboard sequence tokenizer
│   ├── ko_keyboard.py         # Preprocess/postprocess (keystroke ↔ text)
│   ├── make_tokenizer.py      # Builds keyboard tokenizer vocab
│   ├── keyboard_wrapper.py    # KeyboardTokenizer wrapper (implements BaseTokenizer)
│   ├── keyboard_tokenizer.json
│   └── jamo_token_map.json    # Jamo/control token → ID mapping
│
├── char_tokenizer/            # Character-level tokenizer
│   ├── make_tokenizer.py      # Builds character-level vocabulary
│   ├── char_wrapper.py        # CharTokenizer wrapper (implements BaseTokenizer)
│   └── char_level_tokenizer.json
│
├── bbpe_tokenizer/            # Plain Byte-level BPE tokenizer (HuggingFace tokenizers)
│   ├── train_tokenizer.py     # Trains BPE tokenizer on corpus (outputs JSON)
│   ├── bbpe_wrapper.py        # BBPETokenizer wrapper (implements BaseTokenizer)
│   └── bbpe.json              # Generated tokenizer JSON (after training)
│
├── mecab_bbpe_tokenizer/      # MeCab-segmented + Byte-level BPE tokenizer
│   ├── train_tokenizer.py     # Trains MeCab-aware BPE tokenizer (outputs JSON)
│   ├── mecab_bbpe_wrapper.py  # MecabBBPETokenizer wrapper (implements BaseTokenizer)
│   └── mecab_bbpe.json        # Generated tokenizer JSON (after training)
│
├── error_generation/          # Korean error generation for synthetic training data
│   ├── common_misspellings.py # 맞춤법 오류 생성
│   ├── spacing_errors.py      # 띄어쓰기 오류 생성
│   ├── vowel_confusion.py     # 모음 혼동 오류 생성
│   └── test_errors.py         # 오류 생성 테스트
│
├── hanja/                     # CJK character reference data
│   ├── hanja.txt              # Korean Hanja
│   ├── kanji.txt              # Japanese Kanji
│   ├── hanzi_cn.txt           # Simplified Chinese
│   └── hanzi_tw.txt           # Traditional Chinese
│
├── corpus/                    # Training corpus (*.jsonl — gitignored)
├── .gitignore
└── ref.txt
```

---

## Key Abstractions

### BaseTokenizer (`tokenizer_base.py`)

All tokenizer implementations inherit from `BaseTokenizer` (ABC). This interface guarantees:

**Required properties:** `vocab_size`, `pad_id`, `bos_id`, `eos_id`, `unk_id`

**Required methods:** `encode(text, add_special) → List[int]`, `decode(ids, skip_special) → str`

**Default implementations:** `encode_batch`, `decode_batch`, `__len__`

Model code depends **only** on `BaseTokenizer`, making tokenizers freely interchangeable.

### Special Tokens Convention

All tokenizers use `[NAME]` format for special tokens. Standard tokens (IDs may differ by tokenizer):

| Token          | Purpose                              |
|----------------|--------------------------------------|
| `[PAD]`        | Padding (always ID 0)                |
| `[UNK]`        | Unknown token                        |
| `[BOS]`        | Beginning of sequence                |
| `[EOS]`        | End of sequence                      |
| `[SEP]`        | Separator                            |
| `[CLS]`        | Classification token                 |
| `[MASK]`       | Masked token (for MLM-style tasks)   |
| `[UNUSED0-9]`  | Reserved for future use              |

Tokenizer-specific tokens:
- **NFD**: `[BOHJ]` / `[EOHJ]` — Hanja boundary markers
- **Keyboard**: `[SHIFT]` / `[BLANK]` — keystroke control tokens

---

## Model Architecture

The model is a **BitNet-Mamba Seq2Seq** (encoder-decoder):

- **Encoder**: Stacked Mamba blocks (SSM-based, no attention)
- **Decoder**: Stacked Mamba blocks + cross-attention to encoder outputs
- **BitLinear**: 1-bit quantized linear layers (weights are ternary: {-1, 0, +1})
- **Embedding**: Optionally shared between encoder/decoder, and optionally tied with the LM head

Default config (`BitMambaSeq2SeqConfig`):
- `d_model=768`, `d_inner=1536`, `d_state=16`, `d_conv=4`
- `n_encoder_layers=6`, `n_decoder_layers=10`
- `n_heads=12`, `d_ff=1280`
- `vocab_size=64000`, `max_seq_len=512`
- Target: ~128M parameters (excluding embeddings)

---

## Development Conventions

### Language
- Source code is Python 3 with type hints
- Comments and docstrings are in **Korean**, though class/function/variable names are in English
- File and directory names are in English (lowercase + underscores)

### Running Code
- The project uses a Python virtual environment at `.venv/`
- Activate with: `source .venv/bin/activate`
- Dependencies are in `requirements.txt` (generated via `pip freeze`)
- Key dependencies: `torch`, `tokenizers` (HuggingFace), `mecab-python3`, `mecab-ko-dic`

### Tokenizer Workflow
All tokenizers output **HuggingFace tokenizers JSON** format and follow a consistent pattern:
1. **`make_tokenizer.py`** (or `train_tokenizer.py`): Script to build/train the tokenizer and output a JSON file
2. **`*_wrapper.py`**: A `BaseTokenizer` subclass that loads the JSON via `Tokenizer.from_file()` and implements `encode`/`decode`
3. **Generated artifacts**: JSON tokenizer files, token maps, etc. (checked into the repo)

nfd/keyboard/char tokenizers construct vocab deterministically. bbpe/mecab_bbpe learn from corpus:
```bash
# Deterministic tokenizers
python nfd_tokenizer/make_tokenizer.py
python keyboard_tokenizer/make_tokenizer.py
python char_tokenizer/make_tokenizer.py

# Corpus-trained tokenizers (supports .txt and .jsonl input)
python -m bbpe_tokenizer.train_tokenizer -i corpus/sample_10g.jsonl --text_key text
python -m mecab_bbpe_tokenizer.train_tokenizer -i corpus/sample_10g.jsonl --text_key text
```

### Model Verification
Run `python verify_model.py` from the project root to:
1. Instantiate the model with default config
2. Count parameters (should be ~128M excluding embeddings)
3. Run a forward pass with random inputs
4. Run a backward pass and check for NaN/Inf gradients
5. Test config JSON serialization roundtrip

### Testing
- Most scripts include inline `if __name__ == "__main__"` test blocks
- There is no separate test framework (pytest, etc.) — tests are run directly:
  ```bash
  python keyboard_tokenizer/ko_keyboard.py      # Runs 24 preprocess/postprocess tests
  python keyboard_tokenizer/keyboard_wrapper.py  # Runs 8 roundtrip tests
  python nfd_tokenizer/make_tokenizer.py         # Runs NFD tokenizer tests
  ```

### Git
- Large corpus files (`corpus/*.jsonl`) are gitignored
- Model checkpoints (`*.pt`, `*.pth`, `*.ckpt`, `*.safetensors`, `*.bin`) are gitignored
- SentencePiece model files (`*.model`, `*.vocab`) are gitignored
- Generated tokenizer JSON files **are** checked in

---

## Important Notes for AI Agents

1. **Korean text handling is central** — always consider Unicode normalization (NFC vs NFD), Hanja preprocessing, and jamo decomposition when modifying tokenizer code.

2. **Roundtrip correctness is critical** — tokenizer changes must preserve `decode(encode(text)) ≈ text`. The keyboard tokenizer's `<BLANK>` insertion logic is especially sensitive.

3. **`sys.path` manipulation** — some scripts insert the project root into `sys.path` to resolve imports. Always run scripts from the project root (`/workspace/base-model-2/`).

4. **The `[PAD]` token is always ID 0** — this is assumed by `BitMambaSeq2SeqConfig.pad_id` default and should remain consistent across tokenizers.
