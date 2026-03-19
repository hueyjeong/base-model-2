# Attention GEC 실험 — 배경 컨텍스트

새 세션에서 이 파일을 읽고 시작하세요.

## 프로젝트 목표

한국어 문법 오류 교정(GEC) 편집 태깅 모델. 사용자가 글을 타이핑하다 0.5초 멈추면 현재 문장의 오탈자를 교정 추천하는 프로그램.

- 모델 크기: 128~200MB
- 추론 속도: CPU 4코어에서 <500ms
- 크로스 플랫폼: ONNX Runtime (Win/Mac/Linux/iOS/Android/WASM)

## 왜 Attention으로 전환하는가

### 기존 아키텍처: BiMamba-2 + BitNet (DenseEditor)
- 128M params, d_model=640, 15L, BiMamba-2 SSD + BitNet 1.58-bit
- 10k DDP 결과: val_loss=0.107, P=93.1%, R=71.0%
- CPU 8T: 1,159ms → 최적화 후 32T: ~500ms

### 문제점
1. **SSD scan 순차 의존성**: h[t] = f(h[t-1], x[t]) → CPU 병렬화 불가
2. **양방향 = 2× 비용**: forward scan + backward scan, attention은 자연 양방향
3. **거대한 projection**: in_proj(2708×640) — attention의 Q,K,V(1920×640)보다 큼. 양방향이면 2×2708 vs 1×1920
4. **상태 압축 한계**: GEC는 입력≈출력이라 직접 참조(attention)가 압축된 state보다 유리
5. **BitNet ternary**: d=640에서 F32 대비 속도 이점 없음. i8→f32 변환 오버헤드
6. **이식성**: Mamba scan은 ONNX 미지원, 플랫폼마다 Rust+C 직접 구현 필요

### Attention의 이점
- **O(n²) 무시 가능**: seq=50(배포)에서 50²=2.5K ops, matmul 50×640²=20M의 0.01%
- **완전 병렬 matmul**: BLAS 한 방. CPU/GPU/NPU 어디서든 최적화됨
- **ONNX 기본 지원**: `torch.onnx.export` → 끝. 별도 추론 엔진 불필요
- **INT8×INT8 정수 경로**: 현재 i8→f32 변환 대비 2-4x 빠른 matmul 가능
- **Flash Attention 2**: 학습 시 자동 적용, 메모리 O(n)

### 예상 추론 성능 (128M INT8, 4코어 CPU, seq=50)
- Attention matmul (Q,K,V,O): ~8ms
- FFN matmul (gate_up, down): ~10ms
- Attention score + softmax: ~1ms
- Norms + activation: ~0.5ms
- 메모리 대역폭 오버헤드: ~3ms
- Tag head: ~0.5ms
- **합계: ~23ms** (목표 500ms 대비 여유 넘침)

## 재사용하는 자산 (이 리포에 이미 있음)

| 자산 | 위치 | 설명 |
|---|---|---|
| keyboard 토크나이저 | `keyboard_tokenizer/` | vocab=303, 자모 단위 |
| 편집 태그 체계 | `model/dense_editor.py` | 608 tags (KEEP/DELETE/INSERT) |
| 노이즈 엔진 | `training/noiser.py` | DenoisingNoiser, error_prob/error_count |
| 데이터 파이프라인 | `training/editor_dataset.py` | JSONL 스트리밍, 패킹, DDP |
| Levenshtein C++ | `training/levenshtein_c/` | 편집 태그 생성 |
| 품질 벤치마크 | `bench_quality.py` | 2k-step 빠른 비교 |
| Gumbel consensus | `exp-2-pass-consensus/` | 4s/4agree, T=0.3, CPU 무료 |
| 학습 코드 | `training/pretrain_dense_editor.py` | WSD, label smoothing, DDP |

## 핵심 개념 정리

### SwiGLU FFN
- `x → gate_up_proj → SiLU(gate) × up → down_proj → out`
- d_ff = d_model × 8/3 (ReLU의 4x 대비 gate 때문에 축소)
- Attention 모델에서 FFN이 전체 비용의 ~55% (최대 병목)

### GQA (Grouped Query Attention)
- K,V를 여러 Q head가 공유. groups=4 추천 (5:1)
- 파라미터 40% 절감, 품질 거의 동일 (Llama/Mistral에서 검증)

### INT8 QAT (Quantization-Aware Training)
- 학습 중 fake_quantize로 INT8 시뮬레이션
- 추론 시 진짜 INT8 → INT8×INT8→INT32 정수 경로 사용
- PTQ 대비 품질 우수 (특히 작은 모델)

### 문서 격리 (패킹 시)
- 기존: BiMamba reset_mask → seq_idx (BOS에서 state 리셋)
- 새로: Attention mask에 BOS 경계 반영, 같은 문서 내만 attention 허용

### Gumbel Consensus (아키텍처 무관, 그대로 사용)
- forward 1회 → logits에서 N회 Gumbel sampling → majority vote
- 4 samples, 4/4 만장일치 (T=0.3): P +6.3pp, 오버헤드 <1%
- CPU에서 사실상 무료 (sampling만 반복, forward 재실행 안 함)

## 실험 아이디어 — 커스텀 mixing 조합

Full attention baseline 외에 시도할 조합:

1. **Conv1d + Full Attention**: 자모 패턴(conv) + 문맥(attention)
2. **Window Attention + FFT**: 로컬 참조(window) + 전체 분위기(FFT)
3. **Conv1d + Window Attention + FFT**: 풀 조합
4. **Depth-wise 조합**: 저층=conv+window, 고층=full attention
5. **자모 기반 MoE 라우터**: 초성/중성/종성별 expert 분기, 라우터 비용 0

<!-- Claude 의견: seq=50 배포에서는 full attention만으로 충분 (<0.5ms).
     window+FFT는 학습 패킹(2048)에서만 이점. 단순한 게 최적일 가능성 높음.
     하지만 실험해볼 가치는 있고, 결과가 나와야 판단 가능. -->

## 실험 순서

1. 8M (d=256) baseline: attention vs mamba2 품질 비교
2. 8M 커스텀 mixing 조합 스크리닝 (2k-step)
3. 32M (d=384) 유망 조합 검증 (10k-step)
4. 128M (d=640) 최종 아키텍처 full training

## 학습 환경
- 개발: RTX 5060 Ti 16GB
- 본학습: RTX 5090 ×4 DDP
- INT8 QAT, BF16 AMP
- AdamW betas=(0.9, 0.98), wd=0.01
- WSD schedule (warmup 2% → stable 80% → decay 20%)

## 참고 파일
- `PLAN.md`: 상세 실험 계획
- `/workspace/base-model-2/CLAUDE.md`: 전체 프로젝트 구조
- `/workspace/base-model-2/model/mixing/__init__.py`: mixing layer 레지스트리
- `/workspace/base-model-2/model/dense_editor_config.py`: `make_config()` 모델 설정
