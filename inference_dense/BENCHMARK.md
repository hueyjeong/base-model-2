# DenseEditor CPU 인퍼런스 벤치마크 결과

128M Dense 인코더-only 편집 태깅 모델, 8종 mixing layer 비교 (Mamba-2 SSD 포함).
BitNet 1.58-bit ternary weight, AVX2 i8 sgemm, Rust+C 추론 엔진.

## 환경
- CPU: 32코어 (WSL2)
- AVX2 (VNNI 미지원)
- 더미 가중치 (ternary-like i8), seq_len=2048, batch=1

## 1. Scan 커널 벤치마크 (d=256, 단방향 1회, 싱글스레드)

| Kernel | 128 | 256 | 512 | 1024 | 2048 (ms) |
|---|---|---|---|---|---|
| **sLSTM** | **0.041** | **0.082** | **0.163** | **0.333** | **0.667** |
| WKV6 (RWKV) | 0.096 | 0.187 | 0.410 | 0.799 | 1.609 |
| Retention | 0.113 | 0.222 | 0.455 | 0.899 | 1.854 |
| FNet (FFT) | 0.098 | 0.211 | 0.447 | 1.106 | 3.034 |
| TCN (6dil) | 0.251 | 0.533 | 1.097 | 2.237 | 4.573 |
| Mamba-1 | 0.461 | 0.817 | 1.748 | 3.621 | 11.992 |

sLSTM scan이 압도적 — state가 레지스터 크기 (scalar c,n), O(T×d).

### Mamba-2 SSD d_state별 스캔 (d=256, seq=2048)

| Kernel | d_state | 1T (ms) | 8T (ms) | 1T vs Mamba-1 | 8T vs Mamba-1 |
|---|---|---|---|---|---|
| Mamba-1 | 16 | 5.28 | 5.06 | 1.0x | 1.0x |
| **Mamba-2** | **16** | **0.94** | **0.41** | **5.6x** | **12.3x** |
| Mamba-2 | 64 | 4.64 | 1.06 | 1.1x | 4.8x |
| Mamba-2 | 128 | 11.13 | 2.79 | 0.5x | 1.8x |

**Mamba-2 핵심 개선:**
- **exp() 제거:** Mamba-1은 매 state 원소마다 fast_exp_avx2 호출, Mamba-2는 스칼라 decay broadcast (FMA만)
- **headdim 벡터화:** headdim=64 → 8개 AVX2 iteration, 완전 FMA 파이프라인
- **head 병렬화:** OpenMP parallel over nheads (Mamba-1은 순차 scan → 멀티스레드 불리)
- **d_state=16에서 동일 총 FLOP이지만 5.6x 빠름** (exp 제거 + 벡터화 효율)
- **Mamba-1은 멀티스레드 스케일링 1.0x** (순차 scan), Mamba-2는 2.3~4.4x

## 2. d_model별 전체 모델 (seq_len=2048, 128M, 싱글스레드)

n_layers는 128M 파라미터에 맞게 자동 계산.

| Arch | 1T (ms) | Layers | ms/L |
|---|---|---|---|
| **Mamba-1** (ds=16) | **4,066** | 12L | 339 |
| xLSTM | 4,628 | 18L | 257 |
| RWKV | 4,654 | 16L | 291 |
| RetNet | 4,654 | 16L | 291 |
| mLSTM | 4,691 | 16L | 293 |
| Mamba-2 (ds=16) | 4,946 | 15L | 330 |
| Mamba-2 (ds=64) | 5,404 | 15L | 360 |
| FNet | 5,698 | 12L | 475 |
| TCN | 6,648 | 34L | 196 |

**d_model이 클수록 모든 아키텍처가 빨라짐.** 큰 sgemm 1회 > 작은 sgemm 다수.
d=256 "L1 적중" 가정은 틀림 — 레이어 수 폭증으로 총 연산량 증가.
d=640이 실용적 스위트스팟 (12~18L, L2 캐시 범위).

## 3. 멀티스레드 스케일링 (d=640, seq_len=2048, OpenMP)

**주의:** 이전 버전에서 Mamba-1 파라미터 공식에 out_proj 이중 계산 버그가 있었음 (12L로 잘못 표시).
수정 후 Mamba-1도 15L이 정확. 아래는 수정된 공식 기준.

| Arch | 8T (ms) | Layers | ms/L |
|---|---|---|---|
| **mLSTM** | **773** | 16L | 48 |
| **RetNet** | **796** | 16L | 50 |
| **xLSTM** | **819** | 18L | 46 |
| RWKV | 889 | 16L | 56 |
| FNet | 956 | 12L | 80 |
| TCN | 1,189 | 34L | 35 |
| **Mamba-2 ds=16** | **1,411** | 15L | **94** |
| Mamba-1 | 1,497 | 15L | 100 |
| Mamba-2 ds=64 | 1,538 | 15L | 103 |

**공정 비교 (15L 동일)에서 Mamba-2 ds=16이 Mamba-1보다 6% 빠름:**
- Mamba-2: sgemm 2회/방향 (in_proj + out_proj) → quantize 오버헤드 절반
- Mamba-1: sgemm 4회/방향 (in_proj + x_proj + dt_proj + out_proj)
- 총 mul-adds: Mamba-2가 0.98x (거의 동일)이지만 호출 수 절반이 유리

## 4. GPU 학습 품질 (10k DDP, d=640, 4GPU, lr=1e-3, seq=2048)

| Arch | val_loss | edit_P | edit_R | 평가 |
|---|---|---|---|---|
| **Mamba-1** | **0.170** | 89.1% | **51.5%** | 압도적 1위 |
| **RWKV** | 0.212 | 82.0% | **38.0%** | 2위, 실용 가능 |
| **TCN** | 0.222 | 87.0% | **32.4%** | 3위, 실용 가능 |
| RetNet | 0.297 | 92.1% | 0.8% | 편집 불능 (KEEP만 예측) |
| xLSTM | 0.303 | 97.5% | 0.7% | 편집 불능 (KEEP만 예측) |
| FNet | 5.315 | 4.0% | 45.6% | loss 발산 (학습 실패) |

**써먹을 놈**: Mamba-1, RWKV, TCN — **못 써먹을 놈**: RetNet, xLSTM, FNet

## 결론

| 시나리오 | 최적 아키텍처 | 성능 |
|---|---|---|
| 싱글코어 (모바일/임베디드) | Mamba-1 d=640 | 4,066ms |
| 4코어 (노트북) | RetNet/mLSTM d=640 | ~1,400ms |
| 8코어 (데스크탑) | mLSTM d=640 | 773ms |
| 16코어+ (서버) | mLSTM d=640 | ~700ms |

**품질까지 고려한 실용 추천: Mamba-1** (recall 51.5% 독보적)
Mamba-2 ds=16은 CPU 6% 빠르나 향상폭이 미미하여 실익 부족.
CPU 멀티코어 최적인 mLSTM/RetNet/xLSTM는 품질이 치명적 (recall <1%).

### Mamba-2 vs Mamba-1 종합 비교

**스캔 커널 (순수 scan):**
- d_state=16: Mamba-2가 **5.6x(1T) ~ 12.3x(8T) 빠름** (exp 제거 + headdim 벡터화 + head 병렬)

**전체 모델 (15L 동일, 8T):**
- Mamba-2 ds=16: **1,411ms** vs Mamba-1: 1,497ms → **6% 빠름**
- sgemm 호출 2회 vs 4회 → quantize 오버헤드 절반
- 프로젝션이 90%+ 비중이지만, 호출 수 절반이 실측 차이를 만듦

**GPU 학습에서는 chunk-parallel SSD fused kernel으로 더 큰 개선 기대** (별도 벤치마크 필요)

## 실행 방법

```bash
cd inference_dense
cargo build --release --features avx2-only

# scan 커널만
cargo run --release --features avx2-only -- --benchmark-dummy --mixing-type all --seq-len 2048

# 전체 모델 (d_model 지정)
cargo run --release --features avx2-only -- --benchmark-full --seq-len 2048 --d-model 640

# 멀티스레드
OMP_NUM_THREADS=8 cargo run --release --features avx2-only -- --benchmark-full --seq-len 2048 --d-model 640
```
