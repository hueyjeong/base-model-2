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
├── bench_cuda_ablation.py     # INT8 CUDA 최적화 A/B 벤치 스크립트
├── requirements.txt           # Python dependencies (pip freeze)
│
├── model/                     # BitNet-Mamba Seq2Seq model package
│   ├── __init__.py
│   ├── config.py              # BitMambaSeq2SeqConfig dataclass
│   ├── seq2seq.py             # Top-level Seq2Seq model
│   ├── encoder.py             # Mamba-based encoder
│   ├── decoder.py             # Mamba-based decoder
│   ├── mamba_block.py         # Mamba SSM block
│   ├── linear_attention.py    # Grouped-Query Linear Attention (O(N) cross-attention)
│   ├── bitlinear.py           # BitLinear (1.58b weight, 8-bit activation) layer
│   ├── triton_bitlinear.py    # Triton-based INT8 BitLinear backend
│   ├── cuda_bitlinear.py      # CUDA INT8 BitLinear backend (custom autograd)
│   ├── cuda_bitlinear_ext.cpp # CUDA extension C++ binding
│   ├── cuda_bitlinear_kernel.cu # CUDA kernels (__dp4a, quantize, grad weight)
│   └── cross_attention.py     # (Deprecated) Original standard cross-attention
│
├── training/                  # Pre-training scripts
│   ├── pretrain.py            # Main pre-training script (supports AMP, Fused CE, JIT)
│   ├── dataset.py             # Streaming packed dataset logic
│   └── noising.py             # BART-style noise generation
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
│   ├── chat_style_errors.py   # 채팅/통신체 변형 오류 생성
│   ├── honorific_errors.py    # 높임말/반말 혼동 오류 생성
│   ├── jamo_separation.py     # 자모 분리 오류 (ㅋㅋ, ㅎㅎ 등 연관) 생성
│   ├── punctuation_errors.py  # 온점, 쉼표 등 문장부호 오류 생성
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
- **Decoder**: Stacked Mamba blocks with **Linear Cross-Attention**.
  - 기존의 단순 concat 방식을 대체하여, 각 Decoder 레이어에서 Encoder 문맥을 $O(N)$ 복잡도의 Linear Cross-Attention 매커니즘으로 전달합니다.
  - 리니어 어텐션을 위한 양수 보장 특징 매핑으로 `phi(x) = elu(x) + 1`을 사용.
- **BitLinear**: 1.58-bit ternary quantized linear layers (weights are ternary: {-1, 0, +1}, activations are 8-bit)
- **Embedding**: Optionally shared between encoder/decoder, and optionally tied with the LM head. Additonal experimental modes include:
  - **Logit Space Copy Gate (Trial B)**: 디코더 출력 로짓과 원문(source) unigram 분포를 logit space에서 `gate`를 활용해 혼합하여 복사(Copy) 능력을 향상시킵니다.
  - **Source-Aware Logit Bias (Trial A)**: 타겟 로짓에 노이즈 가중치(`src_weights`, 예: 원본 토큰은 1.0, 노이즈 토큰은 0.5)를 근거로 bias를 추가해 원문 단어가 유지되도록 유도합니다.

Default config (`BitMambaSeq2SeqConfig`):
- `d_model=768`, `d_inner=1536`, `d_state=16`, `d_conv=4`
- `n_encoder_layers=6`, `n_decoder_layers=10`
- `n_heads=12`, `d_ff=1280`
- `vocab_size=64000`, `max_seq_len=512`
- Target: Ranging from ~8M up to **~1B** parameters (including 256M, 512M, 1B presets, managed via `MODEL_CONFIGS` in `pretrain.py`)

---

## Noising Algorithm (BART-style)

The dataset introduces robust perturbations (via `training/noising.py` and `error_generation/`) through two phases before input into the model. Weights are tracked simultaneously, yielding 1.0 for original unnoised tokens and 0.5 for altered/masked tokens:

### 1. Text-Level Noise (Pre-Tokenization)
- **Korean Error Injection**: Utilizes `error_generation` modules (spacing, spelling, chat style, honorifics, etc.).
- **Spacing Noise**: Randomly removes, partially removes, or inserts whitespaces.
- **Keyboard Typos**: Euclidean 2D coordinate-based distance replacements (handles Korean Jamo, QWERTY Shift typos, and Numpad layouts).
- **N-gram Shuffle & Word Reorder**: Scrambles sequential word components.

### 2. Token-Level Noise (Post-Tokenization)
- `SequenceMatcher` calculates diff blocks mapping the original sequence against the text-perturbed sequence.
- **Token Masking**: Replaces ~15% of selected un-modified tokens with `[MASK]`.
- **Token Deletion**: Removes ~5% of unmodified tokens.
- **Text Infilling**: Replaces Poisson-distributed spans ($\lambda=3$) with a single `[MASK]`.

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

### Pre-training
The model uses `training/pretrain.py` for training the Seq2Seq BitMamba architecture across a custom corpus.
- **Metrics**: 
  - Uses `Bits Per Character (BPC)` for metric stability, measuring loss against `--max_chars` to compare disparate tokenizer performances fairly.
  - Evaluation logs native unnoised target mappings measuring the `Character Error Rate (CER)` as standard validation.
- **Optimization Strategy**: 
  - Required PyTorch `bfloat16` (`--bf16`) for mixed-precision to avoid BitLinear scaler overflow.
  - Uses `liger-kernel` for Fused Cross-Entropy (`--fused_ce`) to circumvent VRAM materialization.
  - Supports `torch.compile` (`--compile`) for non-INT8 eager path speedups.
  - Cosine Decay scheduling is implemented for learning rate convergence over steps.
  - Supports INT8 backend selection: `--int8 --int8_backend {triton,cuda}`.

#### INT8 CUDA practical guidance (latest)
- Recommended stack phrase: **non-graph + `gradw_lt` + `fused_quant`**.
- Recommended default on current codebase:
  - `BITLINEAR_CUDA_BACKWARD=bf16_tc`
  - `BITLINEAR_CUDA_GRADW_LT=1`
  - `BITLINEAR_CUDA_FUSED_ACT=1`
  - `BITLINEAR_CUDA_FUSED_WEIGHT=1`
- `--int8` with `--compile` is currently skipped by design (custom autograd path).
- `grad_ckpt` has strong memory impact and notable speed cost (memory ↓, tok/s ↓).
- `fused_ce` impact is usually smaller than `grad_ckpt` for memory/throughput trade-off.
- Current low/mid VRAM reliability target is batch `1~2`; batch `4+` requires stronger GPU or tighter settings.

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
- Generated tokenizer JSON files **are** checked in

---

## Important Notes for AI Agents

1. **Korean text handling is central** — always consider Unicode normalization (NFC vs NFD), Hanja preprocessing, and jamo decomposition when modifying tokenizer code.

2. **Roundtrip correctness is critical** — tokenizer changes must preserve `decode(encode(text)) ≈ text`. The keyboard tokenizer's `<BLANK>` insertion logic and extended QWERTY/Shift layout bounds are especially sensitive.

3. **`sys.path` manipulation** — some scripts insert the project root into `sys.path` to resolve imports. Always run scripts from the project root (`/workspace/base-model-2-enchance-error/`).

4. **The `[PAD]` token is always ID 0** — this is assumed by `BitMambaSeq2SeqConfig.pad_id` default and should remain consistent across tokenizers.
