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

| Arch | 1T | 8T | 스케일(1→8) |
|---|---|---|---|
| **mLSTM** | 4,691 | **773** | 6.1x |
| **RetNet** | 4,654 | **796** | 5.9x |
| **xLSTM** | 4,628 | **819** | 5.7x |
| RWKV | 4,654 | 889 | 5.2x |
| FNet | 5,698 | 956 | 6.0x |
| Mamba-1 (ds=16) | 4,066 | 1,191 | 3.4x |
| TCN | 6,648 | 1,189 | 5.6x |
| Mamba-2 (ds=16) | 4,946 | 1,360 | 3.6x |
| Mamba-2 (ds=64) | 5,404 | 1,480 | 3.7x |

**멀티스레드에서 순위 변화:**
- 싱글: Mamba-1 1위 → 멀티(8T): mLSTM 1위, Mamba-1 6위
- mLSTM/RetNet/xLSTM: scan이 가볍고 sgemm 병렬 스케일링 우수 (5.7~6.1x)
- Mamba-1: 순차 scan(d_inner=1280) 병렬화 불리 (3.4x)
- Mamba-2 (ds=16): 레이어당 91ms로 Mamba-1(99ms)보다 8% 빠르나, 15L vs 12L로 총 시간 14% 느림
- Mamba-2: head 병렬화로 스케일링 개선 (3.6~3.7x vs 3.4x)

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

**품질까지 고려한 실용 추천: Mamba-1** (recall 51.5% 독보적, CPU 싱글 1위)
CPU 멀티코어 최적인 mLSTM/RetNet/xLSTM는 품질이 치명적 (recall <1%).

### Mamba-2 vs Mamba-1 종합 비교

**스캔 커널 (순수 scan):**
- d_state=16: Mamba-2가 **5.6x(1T) ~ 12.3x(8T) 빠름** (exp 제거 + headdim 벡터화 + head 병렬)

**전체 모델 (프로젝션 포함):**
- 프로젝션(sgemm)이 90%+ 비중 → scan 12x 개선이 레이어당 8% 수준으로 축소
- Mamba-2(ds=16)는 15L, Mamba-1은 12L → 레이어 수 차이로 총 시간은 Mamba-2가 14~22% 느림
- **레이어당 기준: Mamba-2(91ms/L) < Mamba-1(99ms/L)** — 8% 효율 개선

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
