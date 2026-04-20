# BLT (Byte Latent Transformer) — 아키텍처 정리

Meta 2024.12, arXiv [2412.09871](https://arxiv.org/abs/2412.09871).
Repo: [facebookresearch/blt](https://github.com/facebookresearch/blt).

## 핵심 아이디어

- 토크나이저 제거. 입력은 raw byte (V=256).
- 고정 크기 tokenization 대신 **엔트로피 기반 dynamic patching** 로 byte → patch 묶기.
- 3 모듈 구조:
  1. **Local Encoder** (가벼움) — byte 시퀀스 → patch representation
  2. **Latent Transformer** (무거움) — patch 단위로 메인 처리 (block-causal)
  3. **Local Decoder** (가벼움) — patch → 다음 byte

## Patching — 엔트로피 기반 경계 결정

Entropy model (작은 byte LM) 이 각 위치의 H(x_t) = -Σ p log p 계산. 새 patch 시작 조건:

- **Global**: `H(x_t) > θ_g` (전역 threshold)
- **Approximate Monotonic**: `H(x_t) - H(x_{t-1}) > θ_r` (엔트로피 spike)

θ 는 사전학습 코퍼스에서 target 평균 patch size (예: 4.5, 6, 8 bytes) 에 맞춰 추정.

**핵심 제약** — Incremental property: `fp(x_<i) = fp(x)_<i`. 생성 시 과거 patching 이 바뀌지 않음. (BPE 는 이게 깨짐, 그래서 BLT 가 유리.)

**논문 entropy model**: 100M params, 14L, hidden=512, sliding window=512 bytes.

## Local Encoder

구성: `h_ℰ = 512` 고정, **1~5 layer** (ablation 결과 n-gram embedding 있으면 1L 도 충분).

각 layer = `self-attn (512 sliding window, block-causal) → cross-attention pool`.

### Hash n-gram byte embedding

byte embedding 에 과거 n-gram 정보 주입:

```
e_i = x_i + Σ_{n=3..8} E_n^hash( Hash(g_{i,n}) )
e_i ← e_i / (6+1)   # 정규화 (n-gram 개수 + 1)
```

- `g_{i,n}` = position i 에서 끝나는 길이 n byte subsequence
- `Hash = RollPolyHash % |E_n^hash|` — rolling polynomial hash
- Total hash buckets: **500K** (n 별로 분산)
- n < i 일 때는 해당 항 생략

결과: 각 byte 는 자기 자신 + 선행 3~8 gram 의 hashed context 를 합쳐서 표현. BPE-like semantic unit 을 lookup 1회로 근사.

### Cross-attention pooling (encoder)

각 transformer layer 뒤에 patch 단위 pooling:

- **Query**: 각 patch j 의 초기값 = 해당 patch 내 byte representations 의 **max-pool**
- **Key/Value**: byte representation `h_{ℓ-1, i}`
- **Mask**: query Q_j 는 patch j 에 속하는 byte 들만 attend
- Position encoding **없음** (cross-attention 에는 붙이지 않음)

Layer 반복하며 patch query 가 점점 refinement.

## Latent Transformer (메인)

Patch representation 시퀀스 위에서 block-causal attention. Llama 계열 표준 구성:

- SwiGLU FFN, RMSNorm, RoPE (θ=500,000), Flash Attention
- Block-causal mask = document 내에서만 과거 patch attend
- 공개 config (Appendix Table 10):
  - 400M: 16L, h_𝒢=1024
  - 1B:   24L, h_𝒢=1536
  - 8B:   32L, h_𝒢=4096

## Local Decoder

`h_𝒟 = 512`, **7~9 layer** (ablation — decoder-heavy 가 optimal).

각 layer = `cross-attention (byte ← patch) → self-attn`.

### Cross-attention (decoder)

encoder 와 query/KV 가 반대로 뒤집힘:

- **Query**: byte representation `d_{ℓ-1, i}`
- **Key/Value**: patch representation `o_j` (linear projection `𝒟_C`)
- **Mask**: byte i 는 자기가 속한 patch (혹은 그 이전 patch) 만 참조
- Position encoding 없음

**초기 byte representation**: 마지막 encoder layer 의 byte embedding 을 그대로 residual 연결.
→ decoder 가 local context (byte 단위) + global context (patch 단위) 둘 다 접근.

### 출력

byte vocab V=256 분류. Cross-entropy → **Bits-Per-Byte (BPB)** 리포트.

## 학습 세팅 (논문 기본값)

- LR: 4e-4
- Optimizer: AdamW (β1=0.9, β2=0.95, ε=1e-8)
- Warmup: 2000 step linear → cosine to 0
- Weight decay: 0.1, grad clip: 1.0
- Loss: next byte cross-entropy
- Context: Llama2 8K byte, BLT-1T 16K byte
- Batch: 16M byte 평균

## 왜 patch 가 token 보다 scale 이 좋은가

FLOP 동일 조건에서 token LLM 대비:

- Patch size 를 늘리면 **inference FLOP 50% 절감** 까지
- Small scale (1B) 에서는 BPE 에 밀리지만 7B 근처에서 역전
- Robust: byte noise, low-resource lang, OOV 에 강함
- 같은 FLOP 으로 **patch 와 model 을 동시에 키울 수 있다** 는 게 패러다임 차이

## 우리 프로젝트 적용 메모

### 한국어 byte stream

UTF-8 한글 = 3 byte. 영어/공백 = 1 byte. 평균 patch 4.5 byte ≈ 한글 1~2 음절.

- 엔트로피 model 을 한국어 corpus 로 학습하면 한글 경계 부근에서 H 급증 패턴 형성 예상
- 자모 (NFD) 경로 vs raw byte 경로 둘 다 후보
  - NFD: 자모 seq 길어짐 (3~5 byte → 6~9 자모 byte), 음절/형태소 경계 명확
  - raw UTF-8: 자연스러운 byte stream, 기존 BLT 설정 그대로

### 현재 exp-jamo-codec/codec 에 있는 것

```
conv_codec.py     — Conv1d 기반 encode/decode
sa_codec.py       — Self-attention
simple_codec.py   — 현행 per-token encoder + slot decoder
xattn_codec.py    — Cross-attention (BLT 스타일에 가장 가까움)
entropy_codec.py  — 엔트로피 기반 가변 patching (작업 중이었음)
head_codec.py     — inter-slot interaction 없는 단순 head (plateau 확인)
```

### 축소 entropy model (1M params)

BLT 논문 100M → 1M 축소. 목표:

- 한글 + 영어 mix corpus 로 byte LM 학습
- 구성 예시: 6L, d=128, h=4, d_ff=512, sliding window=256 byte → ~1M
- 학습: `corpus/jamo-codec-v3/train.parquet` byte 단위 cross-entropy
- output: 각 위치 byte logit → softmax → H(x_t)
- patch 경계 = global threshold θ_g (target avg patch 4.5 byte 기준 calibration)

### Hash n-gram embedding 설계

한국어는 UTF-8 3 byte 단위 덩어리가 많음 → n=3,6,9 을 포함시키면 글자/어절 단위 lookup 이 자연 발생.

- n ∈ {3, 4, 5, 6, 7, 8} (논문 그대로) 또는 {3, 6, 9} (한글 정렬)
- |E_n^hash| = 500K / 6 ≈ 83K per n, d_byte 에 맞춰 table 만듦
- RollPolyHash: `h_i = (h_{i-1} * B + byte_i) mod M`, B=257 등 소수 기반

### Phase 0 실행 계획

1. docs 정리 (이 문서) ✅
2. `codec/entropy_lm.py` — 1M byte LM 설계 + 학습 스크립트
3. `codec/hash_ngram.py` — RollPolyHash + E_n lookup
4. `codec/blt_encoder.py` / `blt_decoder.py` — cross-attention pool/unpool
5. 통합 학습 (entropy frozen → encoder+latent+decoder 전체)
6. ablation — patch size {4.5, 6, 8}, encoder layers {1, 3, 5}

## 참고 문헌

- [arXiv 2412.09871 (HTML)](https://arxiv.org/html/2412.09871v1)
- [facebookresearch/blt (GitHub)](https://github.com/facebookresearch/blt)
- [AI Papers Academy 설명글](https://aipapersacademy.com/byte-latent-transformer/)
- [HuggingFace `transformers` BLT 문서](https://huggingface.co/docs/transformers/en/model_doc/blt)
- [ACL 2025 proceedings PDF](https://aclanthology.org/2025.acl-long.453.pdf)
