"""Triton Fused Kernels — RMSNorm, SwiGLU Activation

학습/추론 시 커널 launch 횟수 및 메모리 대역폭 절감을 위한 fused 커널.
Triton 미설치 시 순수 PyTorch 폴백.
"""
import torch
import triton
import triton.language as tl


# ──────────────────────────────────────────────
# 1. Fused RMSNorm
# ──────────────────────────────────────────────

@triton.jit
def _rms_norm_fwd_kernel(
    X, W, Y,
    stride_x: tl.constexpr,
    N: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """RMSNorm forward: y = x * rsqrt(mean(x²) + eps) * w

    각 row(program)가 하나의 토큰을 처리.
    """
    row = tl.program_id(0)
    X += row * stride_x
    Y += row * stride_x

    # x² 합산
    _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
        _var += x * x
    var = tl.sum(_var) / N
    rstd = 1.0 / tl.sqrt(var + eps)

    # 정규화 + 스케일
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
        y = x * rstd * w
        tl.store(Y + cols, y, mask=mask)


@triton.jit
def _rms_norm_bwd_kernel(
    DY, X, W, DX, DW_partial,
    stride_x: tl.constexpr,
    N: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """RMSNorm backward: dx, dw 계산"""
    row = tl.program_id(0)
    DY += row * stride_x
    X += row * stride_x
    DX += row * stride_x
    DW_partial += row * N  # (n_rows, N) 레이아웃

    # rstd 재계산
    _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
        _var += x * x
    var = tl.sum(_var) / N
    rstd = 1.0 / tl.sqrt(var + eps)

    # sum(dy * x * w) for dx
    _sum = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        dy = tl.load(DY + cols, mask=mask, other=0.0).to(tl.float32)
        x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
        _sum += dy * x * w
    c = tl.sum(_sum) * rstd * rstd / N  # rstd³ / N = rstd² * rstd / N

    # dx = (dy * w - x * c) * rstd
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        dy = tl.load(DY + cols, mask=mask, other=0.0).to(tl.float32)
        x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
        dx = (dy * w - x * c) * rstd
        tl.store(DX + cols, dx, mask=mask)
        # dw partial (per-row)
        dw = dy * x * rstd
        tl.store(DW_partial + cols, dw, mask=mask)


class _FusedRMSNormFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps):
        orig_shape = x.shape
        x_2d = x.reshape(-1, orig_shape[-1])
        n_rows, N = x_2d.shape
        y = torch.empty_like(x_2d)

        BLOCK_SIZE = triton.next_power_of_2(N)
        BLOCK_SIZE = min(BLOCK_SIZE, 4096)

        _rms_norm_fwd_kernel[(n_rows,)](
            x_2d, weight, y,
            stride_x=N, N=N, eps=eps,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        ctx.save_for_backward(x_2d, weight)
        ctx.eps = eps
        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.n_rows = n_rows
        ctx.N = N
        return y.reshape(orig_shape)

    @staticmethod
    def backward(ctx, dy):
        x_2d, weight = ctx.saved_tensors
        dy_2d = dy.reshape(ctx.n_rows, ctx.N)
        dx = torch.empty_like(x_2d)
        dw_partial = torch.empty(ctx.n_rows, ctx.N, dtype=torch.float32, device=x_2d.device)

        _rms_norm_bwd_kernel[(ctx.n_rows,)](
            dy_2d, x_2d, weight, dx, dw_partial,
            stride_x=ctx.N, N=ctx.N, eps=ctx.eps,
            BLOCK_SIZE=ctx.BLOCK_SIZE,
        )
        dw = dw_partial.sum(0).to(weight.dtype)
        return dx.reshape(dy.shape), dw, None


def fused_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Fused RMSNorm (Triton)"""
    return _FusedRMSNormFn.apply(x, weight, eps)


# ──────────────────────────────────────────────
# 2. Fused SwiGLU Activation (sigmoid * mul)
# ──────────────────────────────────────────────

@triton.jit
def _fused_sigmoid_mul_fwd_kernel(
    GATE, UP, OUT,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """y = sigmoid(gate) * up — 단일 커널로 중간 텐서 제거"""
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < N

    gate = tl.load(GATE + offset, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(UP + offset, mask=mask, other=0.0).to(tl.float32)

    out = tl.sigmoid(gate) * up
    tl.store(OUT + offset, out, mask=mask)


@triton.jit
def _fused_sigmoid_mul_bwd_kernel(
    DY, GATE, UP, DGATE, DUP,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """backward: d_gate = dy * up * sigmoid(gate) * (1 - sigmoid(gate))
                 d_up = dy * sigmoid(gate)
    """
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < N

    dy = tl.load(DY + offset, mask=mask, other=0.0).to(tl.float32)
    gate = tl.load(GATE + offset, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(UP + offset, mask=mask, other=0.0).to(tl.float32)

    sig = tl.sigmoid(gate)
    d_up = dy * sig
    d_gate = dy * up * sig * (1.0 - sig)

    tl.store(DGATE + offset, d_gate, mask=mask)
    tl.store(DUP + offset, d_up, mask=mask)


class _FusedSigmoidMulFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gate, up):
        assert gate.shape == up.shape
        out = torch.empty_like(gate)
        N = gate.numel()
        BLOCK_SIZE = 1024
        grid = ((N + BLOCK_SIZE - 1) // BLOCK_SIZE,)

        _fused_sigmoid_mul_fwd_kernel[grid](
            gate, up, out, N=N, BLOCK_SIZE=BLOCK_SIZE,
        )
        ctx.save_for_backward(gate, up)
        ctx.N = N
        return out

    @staticmethod
    def backward(ctx, dy):
        gate, up = ctx.saved_tensors
        d_gate = torch.empty_like(gate)
        d_up = torch.empty_like(up)
        N = ctx.N
        BLOCK_SIZE = 1024
        grid = ((N + BLOCK_SIZE - 1) // BLOCK_SIZE,)

        _fused_sigmoid_mul_bwd_kernel[grid](
            dy, gate, up, d_gate, d_up, N=N, BLOCK_SIZE=BLOCK_SIZE,
        )
        return d_gate, d_up


def fused_sigmoid_mul(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Fused sigmoid(gate) * up (Triton)"""
    return _FusedSigmoidMulFn.apply(gate, up)
