# Consensus 기반 Stochastic 추론 실험 결과

## 개요

DenseEditor (BiMamba-2, 128M, d_model=640) 모델의 CPU 추론 시,
Gumbel noise 기반 stochastic sampling + consensus vote로
**추가 비용 없이 Precision을 올릴 수 있는지** 검증한 실험.

- **체크포인트**: `dense_mamba2_d640_step_26000.pt` (val_loss 1.83, R 80.9%)
- **Stochasticity**: Gumbel-max trick (logit/T + Gumbel(0,1) → argmax)
- **핵심 발견**: forward 1회 → logits → N회 sampling은 **사실상 무료** (오버헤드 <1%)

---

## 1. 실험 아키텍처

### 기존 V3 (2-sample consensus)
```
input → forward(비싼) → logits → sample_A + sample_B → consensus → output
                                  ~~~~~~~~~~~~~~~~~~~~~~
                                  이 부분이 거의 무료
```

### V5 (N-sample majority vote, 일반화)
```
input → forward(비싼) → logits → sample_1..N → majority_vote(min_agree) → output
                                  ~~~~~~~~~~~~
                                  N=8이어도 11ms (forward 502ms 대비 2%)
```

---

## 2. GPU 품질 실험

### 2.1 V1-V4 비교 (GPU, Gumbel T=0.3)

#### 500문장 x 2반복

| Variation | Precision | Recall | **F0.5** | F1 | edits/sent |
|---|---|---|---|---|---|
| V1 single-pass | 0.761 | 0.644 | 0.734 | 0.698 | 10.76 |
| V2 2-pass | 0.713 | 0.631 | 0.695 | 0.670 | 11.26 |
| **V3 consensus-2** | **0.817** | 0.627 | **0.770** | **0.709** | 9.75 |
| V4 2-stage cons. | 0.775 | 0.624 | 0.739 | 0.691 | 10.23 |

#### 2000문장 x 3반복 (가장 신뢰)

| Variation | Precision | Recall | **F0.5** | std | edits/sent |
|---|---|---|---|---|---|
| V1 single-pass | 0.735 | 0.623 | 0.710 | ±0.017 | 10.43 |
| **V3 consensus-2** | **0.773** | 0.584 | **0.726** | ±0.001 | 9.30 |
| V4 2-stage cons. | 0.730 | 0.588 | 0.697 | ±0.001 | 9.91 |

**결론**:
- **V3 > V1 > V4 > V2** (F0.5 기준)
- V2 (단순 반복 교정): 오히려 **유해** — 반복이 degradation 유발
- V4 (2-stage consensus): recall 회복 실패, V1보다 나쁨
- **V3가 분산을 극적으로 줄임** (std 0.017 → 0.001)

### 2.2 Gumbel vs MC Dropout (GPU, 500문장 x 2반복)

| 방식 | V1 F0.5 | V3 F0.5 | V3-V1 ΔF0.5 |
|---|---|---|---|
| MC Dropout (p=0.1) | 0.749 | 0.768 | +0.019 |
| Gumbel (T=0.3) | 0.734 | 0.770 | +0.036 |

- Gumbel consensus 이득이 더 큼 (+3.6pp vs +1.9pp)
- 최종 V3 F0.5는 거의 동일 (0.770 vs 0.768)
- **Gumbel이 CPU에서 사용 가능하므로 선호**

### 2.3 Temperature sweep (GPU, 500문장 x 1반복)

| T | V1 F0.5 | V3 F0.5 | ΔF0.5 | V3 P | V3 R |
|---|---|---|---|---|---|
| 0.1 | 0.747 | 0.757 | +0.010 | 0.792 | 0.643 |
| 0.2 | 0.740 | 0.766 | +0.026 | 0.807 | 0.637 |
| **0.3** | 0.734 | **0.770** | +0.036 | 0.817 | 0.627 |
| **0.5** | 0.711 | **0.775** | +0.065 | 0.833 | 0.606 |
| 1.0 | 0.190 | 0.743 | +0.553 | 0.857 | 0.485 |

- T=0.3~0.5이 최적 범위
- T가 올라갈수록 P 상승, R 하락, consensus 이득 증가
- T=1.0: V1은 망가지지만 consensus로 극적 복구 (P 0.16→0.86)

---

## 3. N-sample Majority Vote (CPU, 500문장, T=0.3)

forward pass 1회 후 N번 Gumbel sampling → min_agree개 이상 동의 시 채택.

| Config | P | R | **F0.5** | edits/sent | Wall-clock |
|---|---|---|---|---|---|
| V1 argmax | 0.627 | 0.251 | 0.482 | 5.1 | 259s |
| 2s/2agree | 0.654 | 0.241 | 0.487 | 4.7 | 259s |
| 3s/3agree | 0.671 | 0.236 | 0.490 | 4.5 | 255s |
| 4s/3agree | 0.644 | 0.245 | 0.486 | 4.9 | 257s |
| **4s/4agree** | **0.689** | 0.234 | **0.496** | 4.3 | 260s |
| 5s/4agree | 0.656 | 0.242 | 0.489 | 4.7 | 258s |
| 8s/6agree | 0.655 | 0.244 | 0.489 | 4.8 | 260s |

**핵심 발견**:
1. **만장일치(N/N)가 항상 majority(K/N)보다 우수** — 관대한 기준이 FP를 다시 허용
2. **4s/4agree가 F0.5 최고** (0.496) — V1 대비 P +0.063, F0.5 +0.014
3. **시간 차이 없음**: 8회 sampling도 11ms (forward 502ms의 2%)
4. N을 더 올려도 R 하락이 커져서 F0.5가 떨어짐 → 4가 sweet spot

---

## 4. CPU 비용 분석

### 32스레드 Rust ternary (500문장, avg 214 tok/sent)

| 모드 | Wall-clock | ms/문장 | 오버헤드 |
|---|---|---|---|
| Single-pass (argmax) | 4분 19초 | 528ms | — |
| Consensus 2s/2agree | 4분 19초 | 540ms | +2.3% |
| Consensus 4s/4agree | 4분 20초 | 540ms | +2.3% |
| Consensus 8s/8agree | 4분 22초 | 541ms | +2.5% |

- **forward: 502ms/문장** (전체 비용의 99%+)
- **Gumbel N회 sampling: 3~11ms** (무시 가능)
- **Consensus = Precision 향상을 사실상 무료로 얻음**

### Sampling 오버헤드 상세

| N samples | sampling ms/문장 | forward 대비 |
|---|---|---|
| 2 | 3.3ms | 0.66% |
| 4 | 6.0ms | 1.2% |
| 8 | 11.0ms | 2.2% |

---

## 5. 가설 검증

EXPERIMENT_OUTLINE.md 가설 대조:

| 가설 | 결과 |
|---|---|
| H1: consensus가 P를 올린다 | **확인** — P +3.8~6.3pp |
| H2: 2단계가 recall을 회복한다 | **미확인** — V4가 V1보다 나쁨 |
| H3: 2-stage > single-pass | **미확인** — V3 > V4 |
| H4: CPU cost < pass수 × 선형 | **확인 이상** — cost ≈ 1x (sampling 무료!) |

---

## 6. 최종 추천

### Production 배포 설정

```bash
OMP_NUM_THREADS=32 ./dense-editor-inference \
    --consensus \
    --config config.json --model model.bmmq \
    --temperature 0.3 --n-samples 4 --min-agree 4
```

- **4s/4agree, T=0.3**: 최적 F0.5 (P +6.3pp, 비용 +2%)
- F0.5 중심이 아닌 recall 우선이면: 2s/2agree 또는 T=0.2로 낮추기
- **비용**: single-pass와 사실상 동일 (~540ms/문장, 32스레드)

### 요약

| | Single-pass | Consensus 2/2 | Consensus 4/4 |
|---|---|---|---|
| Precision | 0.627 | 0.654 (+0.027) | 0.689 (+0.063) |
| Recall | 0.251 | 0.241 (-0.010) | 0.234 (-0.017) |
| F0.5 | 0.482 | 0.487 (+0.005) | 0.496 (+0.014) |
| ms/문장 | 528ms | 540ms | 540ms |
| 추가 비용 | — | +2.3% | +2.3% |

---

## 7. 구현 파일

| 파일 | 내용 |
|---|---|
| `inference_dense/src/infer.rs` | Rust 엔진: `forward_logits()`, `majority_tags()`, `run_consensus()` |
| `inference_dense/src/main.rs` | CLI: `--consensus`, `--n-samples`, `--min-agree`, `--temperature` |
| `run_experiment.py` | GPU 실험 스크립트 (Gumbel + MC Dropout) |
| `run_experiment_cpu.py` | CPU Rust wrapper (프로세스별 — deprecated) |
| `exported_step26000/` | step_26000 ternary BMMQ export (32.5MB) |
| `results_gpu_step26k/` | GPU 500문장 V1-V4 결과 |
| `results_gpu_2k_step26k/` | GPU 2000문장 V1/V3/V4 결과 |
| `results_gpu_dropout_step26k/` | GPU MC Dropout 비교 결과 |
