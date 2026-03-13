"""MoE Fused Dispatch CUDA 커널 래퍼

GPU-CPU sync 없이 expert별 토큰 분배/수집.
고정 capacity 버퍼 + atomic counter 방식.
"""
import os
import torch

_MOE_EXT = None
try:
    from torch.utils.cpp_extension import load as _cpp_load
    _kdir = os.path.dirname(__file__)
    _MOE_EXT = _cpp_load(
        name="cuda_moe_dispatch_ext",
        sources=[
            os.path.join(_kdir, "cuda_moe_dispatch_ext.cpp"),
            os.path.join(_kdir, "cuda_moe_dispatch_kernel.cu"),
        ],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3"],
        verbose=False,
    )
except Exception as e:
    import warnings
    warnings.warn(f"MoE Dispatch CUDA 빌드 실패: {e}")


def is_available():
    return _MOE_EXT is not None


@torch.compiler.disable
def moe_dispatch_forward(x_flat, expert_idx, expert_w, experts, n_experts, capacity_factor=2.0):
    """GPU-only MoE dispatch: scatter → expert forward → gather

    Args:
        x_flat: (N, D) 입력 토큰
        expert_idx: (N,) 각 토큰의 expert 인덱스
        expert_w: (N,) routing weight
        experts: BatchedBitNetFFN — batched expert 모듈
        n_experts: expert 수
        capacity_factor: capacity = ceil(N/n_experts) * factor

    Returns:
        (N, D) weighted expert 출력
    """
    N, D = x_flat.shape
    capacity = int((N / n_experts + 1) * capacity_factor)

    # Scatter (CUDA, no CPU sync)
    buffers, weights, counts, token_pos = _MOE_EXT.scatter(
        expert_idx, x_flat.float(), expert_w.float(), n_experts, capacity
    )

    # Expert forward — batched (1회 bmm으로 전체 expert 처리)
    # zero-padded 입력 → LayerNorm(0)=0 → quant(0)=0 → bmm(0)=0 → 출력 0
    # gather 커널이 token_pos 기반 유효 위치만 읽으므로 padding 출력 무시됨
    # experts: BatchedBitNetFFN — (E, capacity, D) → (E, capacity, D)
    expert_outs = experts(buffers.to(x_flat.dtype)).float()

    # Gather (CUDA, no CPU sync)
    out = _MOE_EXT.gather(expert_outs, weights, expert_idx, token_pos, N, D)

    return out.to(x_flat.dtype)
