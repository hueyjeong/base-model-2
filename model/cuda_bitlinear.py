"""C++/CUDA 기반 BitLinear forward 커널 래퍼

목표:
- 순전파 INT8 matmul + dequantization을 CUDA extension으로 실행
- 역전파는 기본적으로 Tensor Core GEMM(혼합 정밀) 경로를 사용해 속도 최적화
- 필요 시 환경변수로 INT8 backward 경로를 강제 가능
"""
from __future__ import annotations

import math
import os

import torch
import torch.nn as nn
from torch.utils.cpp_extension import load


_CUDA_EXT = None
_BW_MODE_CACHE: dict[tuple[int, int, int, int], str] = {}


def _env_flag(name: str, default: str = "1") -> bool:
    value = os.environ.get(name, default).strip().lower()
    return value not in {"0", "false", "off", "no"}


def _should_use_gradw_lt(go_2d: torch.Tensor, x_2d: torch.Tensor) -> bool:
    if not _env_flag("BITLINEAR_CUDA_GRADW_LT", "1"):
        return False
    if torch.cuda.is_current_stream_capturing():
        return False
    if go_2d.dtype != torch.float32 or x_2d.dtype != torch.float32:
        return False

    M, N = go_2d.shape
    _, K = x_2d.shape
    # 작은 shape에서는 cublasLt descriptor/heuristic 오버헤드가 더 클 수 있음
    return (M * N * K) >= 32_000_000


def _load_cuda_ext():
    global _CUDA_EXT
    if _CUDA_EXT is not None:
        return _CUDA_EXT

    this_dir = os.path.dirname(__file__)
    cpp_src = os.path.join(this_dir, "cuda_bitlinear_ext.cpp")
    cu_src = os.path.join(this_dir, "cuda_bitlinear_kernel.cu")

    _CUDA_EXT = load(
        name="bitlinear_cuda_ext",
        sources=[cpp_src, cu_src],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        verbose=False,
    )
    return _CUDA_EXT


def _quantize_activations(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if (
        x.is_cuda
        and _env_flag("BITLINEAR_CUDA_FUSED_ACT", "1")
        and (not torch.cuda.is_current_stream_capturing())
    ):
        try:
            ext = _load_cuda_ext()
            x_2d = x.reshape(-1, x.shape[-1]).float().contiguous()
            x_q_2d, x_scale_2d = ext.quantize_activations_int8(x_2d)
            x_q = x_q_2d.reshape_as(x)
            scale = x_scale_2d.reshape(*x.shape[:-1], 1)
            return x_q, scale
        except Exception:
            pass

    eta = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-5)
    x_q = (x * 127.0 / eta).round().clamp(-128, 127)
    scale = eta / 127.0
    return x_q, scale


def _quantize_weights(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if (
        w.is_cuda
        and _env_flag("BITLINEAR_CUDA_FUSED_WEIGHT", "1")
        and (not torch.cuda.is_current_stream_capturing())
    ):
        try:
            ext = _load_cuda_ext()
            w_2d = w.float().contiguous()
            w_q_i8, gamma_1 = ext.quantize_weights_ternary(w_2d)
            w_q = w_q_i8.float()
            gamma = gamma_1.reshape([])
            return w_q, gamma
        except Exception:
            pass

    gamma = w.abs().mean().clamp(min=1e-5)
    w_q = (w / gamma).clamp(-1, 1).round()
    return w_q, gamma


def _get_backward_mode() -> str:
    """backward 모드 선택

    - fp32_tf32: FP32 유지 + TF32 matmul 가속 (기본, 정확도/속도 균형)
    - fp16_tc: FP16 Tensor Core matmul (속도 우선)
    - bf16_tc: BF16 Tensor Core matmul (속도/안정성 균형)
    - auto: shape별 후보 모드 마이크로벤치 후 최적 모드 캐시
    - int8: int8 quant + _int_mm/cuda fallback
    """
    mode = os.environ.get("BITLINEAR_CUDA_BACKWARD", "fp32_tf32").strip().lower()
    if mode not in {"fp32_tf32", "fp16_tc", "bf16_tc", "auto", "int8"}:
        return "fp32_tf32"
    return mode


def _preferred_amp_dtype() -> torch.dtype:
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def _enable_tf32_fast_matmul() -> None:
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")


def _select_best_backward_mode(
    go_2d: torch.Tensor,
    x_saved: torch.Tensor,
    w_deq_fp32: torch.Tensor,
) -> str:
    """현재 shape/device에서 가장 빠른 backward 모드를 선택해 캐시.

    캐시 키: (device_index, M, N, K)
    """
    device_index = go_2d.device.index or 0
    M = int(go_2d.shape[0])
    N = int(go_2d.shape[1])
    K = int(w_deq_fp32.shape[1])
    cache_key = (device_index, M, N, K)
    cached = _BW_MODE_CACHE.get(cache_key)
    if cached is not None:
        return cached

    candidates = ["fp32_tf32", "fp16_tc"]
    if torch.cuda.is_bf16_supported():
        candidates.append("bf16_tc")

    _enable_tf32_fast_matmul()
    best_mode = "fp32_tf32"
    best_ms = float("inf")

    repeats = 2
    for mode in candidates:
        elapsed_ms = 0.0
        for _ in range(repeats):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            start.record()
            if mode == "fp32_tf32":
                _ = go_2d @ w_deq_fp32
                _ = go_2d.t() @ x_saved
            elif mode == "fp16_tc":
                go_h = go_2d.half()
                x_h = x_saved.half()
                w_h = w_deq_fp32.half()
                _ = go_h @ w_h
                _ = go_h.t() @ x_h
            else:
                go_b = go_2d.to(torch.bfloat16)
                x_b = x_saved.to(torch.bfloat16)
                w_b = w_deq_fp32.to(torch.bfloat16)
                _ = go_b @ w_b
                _ = go_b.t() @ x_b
            end.record()

            torch.cuda.synchronize()
            elapsed_ms += start.elapsed_time(end)

        elapsed_ms /= repeats
        if elapsed_ms < best_ms:
            best_ms = elapsed_ms
            best_mode = mode

    _BW_MODE_CACHE[cache_key] = best_mode
    return best_mode


def _ceil_to_multiple_of_8(v: int) -> int:
    return ((v + 7) // 8) * 8


def _pad_int8_for_int_mm(x_int8: torch.Tensor, w_int8: torch.Tensor):
    """torch._int_mm 요구조건(모든 차원 8배수)을 만족하도록 0-padding.

    x_int8: [M, K], w_int8: [N, K]
    returns: x_pad [M8,K8], w_pad [N8,K8], (M,N)
    """
    M, K = x_int8.shape
    N = w_int8.shape[0]

    M8 = _ceil_to_multiple_of_8(M)
    K8 = _ceil_to_multiple_of_8(K)
    N8 = _ceil_to_multiple_of_8(N)

    if M8 == M and K8 == K and N8 == N:
        return x_int8, w_int8, M, N

    x_pad = torch.zeros((M8, K8), device=x_int8.device, dtype=torch.int8)
    w_pad = torch.zeros((N8, K8), device=w_int8.device, dtype=torch.int8)
    x_pad[:M, :K] = x_int8
    w_pad[:N, :K] = w_int8
    return x_pad, w_pad, M, N


def _pad_int8_mm_ab(a_int8: torch.Tensor, b_int8: torch.Tensor):
    """a[M,N] @ b[N,K] 용 8배수 패딩.

    returns:
        a_pad[M8,N8], b_pad[N8,K8], M_orig, K_orig
    """
    M, N = a_int8.shape
    N2, K = b_int8.shape
    if N != N2:
        raise ValueError(f"matmul dim mismatch: {N} vs {N2}")

    M8 = _ceil_to_multiple_of_8(M)
    N8 = _ceil_to_multiple_of_8(N)
    K8 = _ceil_to_multiple_of_8(K)

    if M8 == M and N8 == N and K8 == K:
        return a_int8, b_int8, M, K

    a_pad = torch.zeros((M8, N8), device=a_int8.device, dtype=torch.int8)
    b_pad = torch.zeros((N8, K8), device=b_int8.device, dtype=torch.int8)
    a_pad[:M, :N] = a_int8
    b_pad[:N, :K] = b_int8
    return a_pad, b_pad, M, K


@torch.compiler.allow_in_graph
class _BitLinearCudaFn(torch.autograd.Function):
    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda", cast_inputs=torch.float32)
    def forward(ctx, x_norm: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None):
        if not x_norm.is_cuda:
            raise RuntimeError("BitLinearCuda는 CUDA 텐서에서만 동작합니다.")

        x_q, x_scale = _quantize_activations(x_norm)
        w_q, w_scale = _quantize_weights(weight)

        batch_shape = x_norm.shape[:-1]
        M = x_q.reshape(-1, x_q.shape[-1]).shape[0]

        x_int8 = x_q.reshape(M, -1).to(torch.int8).contiguous()
        w_int8 = w_q.to(torch.int8).contiguous()

        x_scale_2d = x_scale.reshape(M, 1).contiguous().float()
        w_scale_1 = w_scale.reshape(1).contiguous().float()
        x_int8_pad, w_int8_pad, M_orig, N_orig = _pad_int8_for_int_mm(x_int8, w_int8)

        try:
            out_i32 = torch._int_mm(x_int8_pad, w_int8_pad.t().contiguous())
            out_2d = out_i32[:M_orig, :N_orig].float() * (x_scale_2d * w_scale_1)
            if bias is not None:
                out_2d = out_2d + bias.float()
        except RuntimeError:
            ext = _load_cuda_ext()
            bias_1d = bias.contiguous().float() if bias is not None else None
            out_2d = ext.bitlinear_int8_forward(x_int8, w_int8, x_scale_2d, w_scale_1, bias_1d)

        out = out_2d.reshape(*batch_shape, -1)

        backward_mode = _get_backward_mode()
        x_flat = x_norm.reshape(M, -1).contiguous()

        if backward_mode == "fp16_tc":
            x_saved = x_flat.half()
        elif backward_mode == "bf16_tc":
            x_saved = x_flat.to(torch.bfloat16)
        else:
            x_saved = x_flat.float()

        ctx.save_for_backward(x_saved, w_int8, w_scale_1)
        ctx.batch_shape = batch_shape
        ctx.has_bias = bias is not None
        ctx.backward_mode = backward_mode
        return out

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, grad_output: torch.Tensor):
        x_saved, w_int8, w_scale_1 = ctx.saved_tensors
        batch_shape = ctx.batch_shape
        M = x_saved.shape[0]

        go_2d = grad_output.reshape(M, -1).float()
        backward_mode = getattr(ctx, "backward_mode", "fp32_tf32")
        _enable_tf32_fast_matmul()

        w_deq_fp32 = (w_int8.float() * w_scale_1).contiguous()

        if backward_mode == "auto":
            backward_mode = _select_best_backward_mode(go_2d, x_saved, w_deq_fp32)

        if backward_mode == "fp32_tf32":
            w_fp32 = w_deq_fp32
            x_fp32 = x_saved.float()
            grad_x_2d = go_2d @ w_fp32
            if not _should_use_gradw_lt(go_2d, x_fp32):
                grad_w = go_2d.t() @ x_fp32
            else:
                try:
                    ext = _load_cuda_ext()
                    grad_w = ext.bitlinear_grad_weight(go_2d, x_fp32)
                except Exception:
                    grad_w = go_2d.t() @ x_fp32
        elif backward_mode == "fp16_tc":
            go_h = go_2d.half()
            grad_x_2d = (go_h @ w_deq_fp32.half()).float()
            grad_w = (go_h.t() @ x_saved.half()).float()
        elif backward_mode == "bf16_tc":
            go_b = go_2d.to(torch.bfloat16)
            grad_x_2d = (go_b @ w_deq_fp32.to(torch.bfloat16)).float()
            grad_w = (go_b.t() @ x_saved.to(torch.bfloat16)).float()
        else:
            go_abs = go_2d.abs().amax(dim=-1, keepdim=True).clamp(min=1e-5)
            go_int8 = (go_2d * 127.0 / go_abs).round().clamp(-128, 127).to(torch.int8)
            go_scale = (go_abs / 127.0).contiguous().float()

            go_pad, w_pad, M_orig, K_orig = _pad_int8_mm_ab(go_int8, w_int8)

            try:
                grad_x_i32 = torch._int_mm(go_pad, w_pad)
                grad_x_2d = grad_x_i32[:M_orig, :K_orig].float() * (go_scale * w_scale_1)
            except RuntimeError:
                ext = _load_cuda_ext()
                grad_x_2d = ext.bitlinear_int8_backward_input(go_int8, w_int8, go_scale, w_scale_1)

            x_fp32 = x_saved.float()
            if not _should_use_gradw_lt(go_2d, x_fp32):
                grad_w = go_2d.t() @ x_saved.float()
            else:
                try:
                    ext = _load_cuda_ext()
                    grad_w = ext.bitlinear_grad_weight(go_2d, x_fp32)
                except Exception:
                    grad_w = go_2d.t() @ x_saved.float()

        grad_x = grad_x_2d.reshape(*batch_shape, -1)
        grad_bias = go_2d.sum(dim=0) if ctx.has_bias else None

        return grad_x, grad_w, grad_bias


class BitLinearCuda(nn.Module):
    """CUDA extension 기반 BitLinear 대체 모듈"""

    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

        self.norm = nn.LayerNorm(in_features, elementwise_affine=False)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self.norm(x)

        weight = self.weight
        bias = self.bias
        if weight.device != x_norm.device:
            weight = weight.to(x_norm.device)
        if bias is not None and bias.device != x_norm.device:
            bias = bias.to(x_norm.device)

        return _BitLinearCudaFn.apply(x_norm, weight, bias)

    def extra_repr(self) -> str:
        return (
            f"in={self.in_features}, out={self.out_features}, "
            f"bias={self.bias is not None}, quant=1.58bit+cuda-int8"
        )


def replace_bitlinear_with_cuda(model: nn.Module) -> nn.Module:
    from model.bitlinear import BitLinear

    replacements = []
    for name, module in model.named_modules():
        if isinstance(module, BitLinear):
            replacements.append(name)

    for name in replacements:
        parts = name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)

        old = getattr(parent, parts[-1])
        old_device = old.weight.device
        new = BitLinearCuda(old.in_features, old.out_features, bias=old.bias is not None)
        new.weight.data.copy_(old.weight.data)
        if old.bias is not None:
            new.bias.data.copy_(old.bias.data)
        new.norm.load_state_dict(old.norm.state_dict())
        new = new.to(old_device)
        setattr(parent, parts[-1], new)

    print(f"[BitLinearCuda] Replaced {len(replacements)} BitLinear layers")
    return model
