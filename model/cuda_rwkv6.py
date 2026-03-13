"""RWKV-6 CUDA 커널 래퍼 — JIT 빌드 + custom autograd Function

fla 라이브러리 대체: BF16 직접 입력, dtype 변환/transpose 불필요.
"""
import os
import torch
from torch.autograd import Function

# JIT 빌드
_CUDA_RWKV6 = None

try:
    from torch.utils.cpp_extension import load as _cpp_load
    _kernel_dir = os.path.dirname(__file__)
    _CUDA_RWKV6 = _cpp_load(
        name="cuda_rwkv6_ext",
        sources=[
            os.path.join(_kernel_dir, "cuda_rwkv6_ext.cpp"),
            os.path.join(_kernel_dir, "cuda_rwkv6_kernel.cu"),
        ],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        verbose=False,
    )
except Exception as e:
    import warnings
    warnings.warn(f"CUDA RWKV-6 커널 빌드 실패: {e}")


class CudaRWKV6Fn(Function):
    """RWKV-6 recurrent scan custom autograd (CUDA)"""

    @staticmethod
    def forward(ctx, r, k, v, w, u, scale):
        # r, k, v, w: (B, T, H, D) — any dtype
        # u: (H, D) — float32
        out, state_saved = _CUDA_RWKV6.forward(r, k, v, w, u, True)

        if scale != 1.0:
            out = out * scale

        ctx.save_for_backward(r, k, v, w, u, state_saved)
        ctx.scale = scale

        return out

    @staticmethod
    def backward(ctx, grad_out):
        r, k, v, w, u, state_saved = ctx.saved_tensors

        if ctx.scale != 1.0:
            grad_out = grad_out * ctx.scale

        grad_r, grad_k, grad_v, grad_w, grad_u = _CUDA_RWKV6.backward(
            r, k, v, w, u, grad_out, state_saved
        )

        return grad_r, grad_k, grad_v, grad_w, grad_u, None


def cuda_rwkv6_recurrent(r, k, v, w, u, scale=1.0, output_final_state=False):
    """fla fused_recurrent_rwkv6 호환 API

    Args:
        r, k, v, w: (B, T, H, D)
        u: (H, D)
        scale: 출력 스케일 (기본 1.0)
        output_final_state: 무시 (호환성)

    Returns:
        (output, None) — fla와 동일한 반환 형식
    """
    out = CudaRWKV6Fn.apply(r, k, v, w, u, scale)
    return out, None


def is_available():
    """CUDA RWKV-6 커널 사용 가능 여부"""
    return _CUDA_RWKV6 is not None
