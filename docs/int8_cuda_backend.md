# INT8 CUDA BitLinear 사용 가이드

이 문서는 `model/cuda_bitlinear.py` 기반 INT8 CUDA 백엔드 사용법을 정리합니다.

## 1) 환경 준비

```bash
cd /workspace/base-model-2
source .venv/bin/activate
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

- `torch.cuda.is_available()`가 `True`여야 합니다.
- CUDA extension JIT 빌드에 `ninja`가 필요합니다.

```bash
pip install ninja
```

## 2) 학습 실행 (CUDA 백엔드)

```bash
python -m training.pretrain \
  --size 8M \
  --tokenizer bbpe \
  --corpus corpus/sample_10g.jsonl \
  --text_key text \
  --int8 \
  --int8_backend cuda
```

- `--int8_backend cuda`를 주면 `BitLinear`를 CUDA 구현으로 교체합니다.
- CUDA 확장 로딩 실패 시 자동으로 `triton` 백엔드로 fallback 됩니다.
- 현재 권장 기본 조합은 **non-graph + `gradw_lt` + `fused_quant`** 입니다.

CUDA Graph 실험 모드(단일 GPU, non-fused CE, 고정 shape 구간) 예시:

```bash
python -m training.pretrain \
  --size 8M \
  --tokenizer bbpe \
  --corpus corpus/sample_10g.jsonl \
  --int8 \
  --int8_backend cuda \
  --bf16 \
  --cuda_graph
```

## 3) 빠른 스모크 테스트

```bash
python - <<'PY'
import torch
from model.cuda_bitlinear import BitLinearCuda

layer = BitLinearCuda(256, 256, bias=False).cuda()
x = torch.randn(8, 32, 256, device='cuda', requires_grad=True)
y = layer(x)
loss = y.pow(2).mean()
loss.backward()
print('ok', y.shape, torch.isfinite(x.grad).all().item())
PY
```

## 4) 현재 구현 특성

- Forward:
  - 내부에서 `M,K,N`을 자동 0-padding하여 8의 배수로 맞춘 뒤 `torch._int_mm` Tensor Core fast-path 시도
  - 활성화 양자화(`absmax + round + clamp + scale`)는 CUDA fused 커널 우선 시도
  - `torch._int_mm` 실패 시 C++/CUDA 커널 (타일드 shared-memory INT8 matmul + `__dp4a` 누적 + dequant) fallback
- Backward:
  - 기본값(권장): `fp32_tf32` (FP32 유지 + TF32 matmul 가속)
  - 선택 가능: `fp16_tc`, `bf16_tc`, `int8`
  - 환경변수로 제어: `BITLINEAR_CUDA_BACKWARD`
- 목적: 기존 Python custom autograd 경로 대비 더 일관된 커널 실행

## 5) 권장 튜닝 포인트

- BF16 권장: `--bf16`
- 긴 시퀀스/큰 배치에서 이득이 더 큼
- `--int8`와 `--compile` 동시 사용은 현재 비권장/제한 가능
- 기본 권장 조합:
  - `BITLINEAR_CUDA_GRADW_LT=1`
  - `BITLINEAR_CUDA_FUSED_ACT=1`
  - `BITLINEAR_CUDA_FUSED_WEIGHT=1`
  - `BITLINEAR_CUDA_BACKWARD=fp32_tf32`
- `--cuda_graph`는 실험용으로만 사용 권장 (현재 다수 케이스에서 손해 가능)

예시:

```bash
# 기본값 (명시 안 해도 동일)
export BITLINEAR_CUDA_BACKWARD=fp32_tf32

# 권장 최적화 조합 (non-graph)
export BITLINEAR_CUDA_GRADW_LT=1
export BITLINEAR_CUDA_FUSED_ACT=1
export BITLINEAR_CUDA_FUSED_WEIGHT=1

# 대안 모드
# export BITLINEAR_CUDA_BACKWARD=fp16_tc
# export BITLINEAR_CUDA_BACKWARD=bf16_tc
# export BITLINEAR_CUDA_BACKWARD=int8
```

## 6) 문제 해결

- `RuntimeError: Ninja is required...`
  - `.venv` 활성화 후 `pip install ninja`
- `ModuleNotFoundError: torch`
  - 시스템 python이 아니라 `.venv/bin/python`으로 실행
- VS Code `.cu` includePath 경고
  - 에디터 인덱서 경고일 수 있으며, 런타임 JIT 빌드 성공 여부를 우선 확인
