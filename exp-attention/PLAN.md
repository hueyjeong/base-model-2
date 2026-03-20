# Attention 기반 GEC 편집 태거 실험 계획

## 목표
- 타자 0.5초 멈추면 현재 문장 교정 추천
- 모델 크기 128~200MB, CPU 4코어에서 <500ms
- 크로스 플랫폼: ONNX RT (Win/Mac/Linux/iOS/Android/WASM)

## 유지하는 것
- keyboard 토크나이저 (vocab=303, n_tags=608)
- 편집 태그 체계 (Levenshtein 기반 KEEP/DELETE/INSERT)
- 노이즈 엔진 (DenoisingNoiser)
- 데이터 파이프라인 (EditorDataset, 패킹, DDP)
- Gumbel consensus (4s/4agree, T=0.3)
- bench_quality.py (2k-step 품질 벤치마크)

## 모델 크기
- 8M: 빠른 실험용 (d=256, ~6L)
- 32M: 중간 검증 (d=384, ~10L)
- 128M: 최종 목표 (d=640, ~15L)

---

## Baseline: Full Attention + GQA + SwiGLU + INT8 QAT

```
Embedding (303, d_model) × √d_model
├── Layer × N
│   ├── RMSNorm → Full Attention (GQA) → Dropout → (+residual)
│   └── RMSNorm → SwiGLU FFN → Dropout → (+residual)
├── Final RMSNorm
└── Tag Head (d_model → 608)
```

### Attention 설정
- Full bidirectional self-attention (causal mask 없음)
- GQA: n_heads=20, n_kv_groups=4 (5:1 공유) — d=640 기준
- RoPE positional encoding
- `F.scaled_dot_product_attention` (Flash Attention 2 자동)
- pad_mask로 패딩 토큰 마스킹

### FFN 설정
- SwiGLU: d_ff = d_model × 8/3
- `nn.Linear` (BitLinear 아님)

### 양자화
- BF16 학습 → INT8 QAT (fake_quantize in forward)
- 추론: ONNX RT INT8 자동 최적화

### 학습 설정 (기존과 동일)
- AdamW, betas=(0.9, 0.98), wd=0.01
- WSD schedule, label_smoothing=0.1, edit_loss_weight=2.0
- 노이즈: error_prob=0.5, error_count=3
- 패킹: BOS/EOS 단위, max_seq_len=2048

### 문서 격리 (패킹 시)
- Attention mask에 BOS 경계 반영
- 같은 문서 내 토큰끼리만 attention 허용
- BiMamba의 reset_mask/seq_idx 대신 attention mask로 처리

---

## 실험 1: Baseline 품질 검증

### 목적
Full attention이 BiMamba-2와 동등 이상의 GEC 품질을 내는지 확인

### 방법
- `bench_quality.py --mixing_type attention --d_model 640 --max_steps 2000`
- BiMamba-2 (현재 최고: loss 0.107, R 71.0%) 대비 비교

### 예상
- Attention이 직접 참조 가능 → GEC 태스크에서 유리할 가능성
- 단, 아키텍처 벤치마크에서 RetNet이 실패했으므로 확신은 없음
  (RetNet ≠ full attention, linear attention의 한계였을 수 있음)

---

## 실험 2: 커스텀 Mixing 조합

### 목적
Full attention 외에 더 효율적인 mixing 조합 탐색

### 후보 (레이어 내 병렬 또는 레이어별 교대)

#### A. Conv1d + Full Attention
```
x → conv1d(k=4~8) → attention → add
```
- conv1d: 자모 패턴 (음절 내 인접 자모 관계)
- attention: 어절/문장 수준 문맥

#### B. Window Attention + FFT
```
x → window_attn(w=32~64) + FFT_mixing → add
```
- window: 날카로운 로컬 참조 (O(n×w))
- FFT: 전체 문장 주파수 특성 (O(n log n))
- 학습 패킹(2048)에서 문서 격리가 자연스러움

#### C. Conv1d + Window Attention + FFT (풀 조합)
```
x → conv1d → window_attn + FFT → add
```

#### D. Depth-wise 조합
```
Layer 0-4:  Conv1d + Window Attention (저수준 자모 패턴)
Layer 5-9:  Full Attention (중수준 어절 문맥)
Layer 10-14: Full Attention (고수준 문장 구조)
```

### 평가 기준
- 2k-step val_loss + edit_recall 기준으로 빠르게 스크리닝
- 상위 2-3개를 10k-step까지 확장

---

## 실험 3: 모델 효율화 기법

### GQA sweep
- groups: 1 (MQA), 2, 4, 10, 20 (MHA)
- 8M 모델에서 빠르게 비교

### INT8 QAT vs BF16 only
- QAT 적용 유무에 따른 품질 차이
- 8M에서 검증 후 32M/128M으로 확장

### Conv1d kernel size
- k = 3, 4, 5, 8
- 자모 토크나이저 기준 최적 커널 탐색

---

## 실험 순서

1. **mixing_type=attention 등록** + 8M 모델 forward 검증
2. **8M baseline** 2k-step 품질 (attention vs mamba2)
3. **8M 커스텀** conv1d+attention, window+FFT 등 2k-step 비교
4. **32M baseline** 유망 조합 10k-step
5. **128M 최종** 선택된 아키텍처 full training

---

## 파라미터 예산 (d_model 기준)

| d_model | layers | attention params/L | FFN params/L | total (approx) |
|---|---|---|---|---|
| 256 | 6 | Q,K,V,O = 4×256² = 262K | gate_up+down = 256×684×2+684×256 = 524K | ~5M + embed |
| 384 | 10 | 4×384² = 590K | 384×1024×2+1024×384 = 1.2M | ~18M + embed |
| 640 | 15 | 4×640² = 1.6M (GQA→1.0M) | 640×1707×2+1707×640 = 3.3M | ~65M + embed |

(GQA groups=4 적용 시 K,V가 1/5로 줄어 attention params 40% 절감)
