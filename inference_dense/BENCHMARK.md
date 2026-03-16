# DenseEditor CPU 인퍼런스 벤치마크 결과

128M Dense 인코더-only 편집 태깅 모델, 7종 mixing layer 비교.
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
| Mamba | 0.461 | 0.817 | 1.748 | 3.621 | 11.992 |

sLSTM scan이 압도적 — state가 레지스터 크기 (scalar c,n), O(T×d).

## 2. d_model별 전체 모델 (seq_len=2048, 128M, 싱글스레드)

n_layers는 128M 파라미터에 맞게 자동 계산.

| Arch | d=256 | d=384 | d=512 | d=640 | d=768 | d=1024 |
|---|---|---|---|---|---|---|
| Mamba | 5722 | 4717 | 4340 | **4018** | **3740** | **3174** |
| xLSTM | 5460 | 4993 | 4730 | 4565 | 4403 | 4451 |
| RWKV | 5460 | 5044 | 4699 | 4556 | 4447 | 4239 |
| RetNet | 5672 | 5078 | 4787 | 4559 | 4507 | 4243 |
| mLSTM | 5803 | 5152 | 4851 | 4622 | 4533 | 4229 |
| FNet | 6733 | 6158 | 5878 | 5617 | 5349 | 4897 |
| TCN | 7921 | 7268 | 6727 | 6485 | 6236 | 6019 |

**d_model이 클수록 모든 아키텍처가 빨라짐.** 큰 sgemm 1회 > 작은 sgemm 다수.
d=256 "L1 적중" 가정은 틀림 — 레이어 수 폭증으로 총 연산량 증가.
d=640이 실용적 스위트스팟 (12~18L, L2 캐시 범위).

## 3. 멀티스레드 스케일링 (d=640, seq_len=2048, OpenMP)

| Arch | 1T | 2T | 4T | 8T | 16T | 스케일(1→8) |
|---|---|---|---|---|---|---|
| **mLSTM** | 4667 | 2437 | 1401 | **775** | **697** | 6.0x |
| **RetNet** | 4719 | 2431 | 1392 | **798** | **730** | 5.9x |
| **xLSTM** | 4685 | 2442 | 1396 | **820** | **717** | 5.7x |
| RWKV | 4655 | 2500 | 1481 | 912 | 799 | 5.1x |
| FNet | 5685 | 2923 | 1600 | 956 | 804 | 5.9x |
| Mamba | 4143 | 2374 | 1646 | 1223 | 1071 | 3.4x |
| TCN | 6683 | 3510 | 2027 | 1216 | 1049 | 5.5x |

**멀티스레드에서 순위 완전히 뒤바뀜:**
- 싱글: Mamba 1위 → 멀티(8T): mLSTM 1위, Mamba 5위
- mLSTM/RetNet/xLSTM: scan이 가볍고 sgemm 병렬 스케일링 우수 (5.7~6.0x)
- Mamba: 순차 scan(d_inner=1280)이 병렬화 불리 (3.4x)

## 결론

| 시나리오 | 최적 아키텍처 | 성능 |
|---|---|---|
| 싱글코어 (모바일/임베디드) | Mamba d=640 | 4018ms |
| 4코어 (노트북) | RetNet/mLSTM d=640 | ~1400ms |
| 8코어 (데스크탑) | mLSTM d=640 | 775ms |
| 16코어 (서버) | mLSTM d=640 | 697ms |

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
