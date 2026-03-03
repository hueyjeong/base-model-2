"""Linear Cross-Attention module

Grouped-Query Linear Attention (GQA) for efficient O(N) cross-attention.
Designed for integrating encoder context into a decoder without the O(N^2)
memory/compute bottleneck of standard Softmax attention.

Features:
- Feature map phi(x) = elu(x) + 1 implementation for positive keys/queries
- Custom CUDA fused kernel for K^T V + Q(K^T V) + normalization
- PyTorch exact fallback when CUDA kernel unavailable
"""
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# =========================================================================
# CUDA fused kernel (JIT compile on first use)
# =========================================================================
_CUDA_LINEAR_ATTN = None


def _load_cuda_linear_attn():
    global _CUDA_LINEAR_ATTN
    if _CUDA_LINEAR_ATTN is not None:
        return _CUDA_LINEAR_ATTN

    from torch.utils.cpp_extension import load

    this_dir = os.path.dirname(__file__)
    cpp_src = os.path.join(this_dir, "cuda_linear_attention_ext.cpp")
    cu_src = os.path.join(this_dir, "cuda_linear_attention_kernel.cu")

    _CUDA_LINEAR_ATTN = load(
        name="linear_attn_cuda_ext",
        sources=[cpp_src, cu_src],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        verbose=False,
    )
    return _CUDA_LINEAR_ATTN


def _env_flag(name: str, default: str = "1") -> bool:
    value = os.environ.get(name, default).strip().lower()
    return value not in {"0", "false", "off", "no"}


HAS_CUDA_LINEAR_ATTN = torch.cuda.is_available()


# =========================================================================
# Feature Map
# =========================================================================

def elu_feature_map(x: torch.Tensor) -> torch.Tensor:
    """리니어 어텐션을 위한 양수 보장 특징 매핑: phi(x) = elu(x) + 1"""
    return F.elu(x) + 1.0


# =========================================================================
# Autograd Function with CUDA forward + PyTorch backward
# =========================================================================

class _LinearCrossAttnCudaFn(torch.autograd.Function):
    """CUDA fused forward + PyTorch-based backward for linear cross-attention."""

    @staticmethod
    def forward(ctx, q, k, v, mask, eps):
        """
        q: (B, n_heads, tgt_len, d_head)  — phi(Q) 적용 후
        k: (B, n_heads, src_len, d_head)  — phi(K) 적용 + mask 적용 후
        v: (B, n_heads, src_len, d_head)  — mask 적용 후
        mask: (B, src_len) bool or empty tensor
        eps: float
        """
        ext = _load_cuda_linear_attn()
        out = ext.linear_cross_attn_fwd(q, k, v, mask, eps)

        # Save for backward (PyTorch matmul 기반)
        ctx.save_for_backward(q, k, v, mask)
        ctx.eps = eps
        return out

    @staticmethod
    def backward(ctx, grad_out):
        q, k, v, mask = ctx.saved_tensors
        eps = ctx.eps

        # Recompute context and z (cheap: d_head is small)
        # context = K^T V: (B, h, d, d)
        context = torch.matmul(k.transpose(-1, -2), v)
        # z = sum(K): (B, h, d)
        z = k.sum(dim=-2)

        # Forward values needed for gradient
        # num = Q @ context: (B, h, tgt, d)
        num = torch.matmul(q, context)
        # den = Q · z: (B, h, tgt)
        den = torch.einsum("bhld,bhd->bhl", q, z)
        den_expanded = den.unsqueeze(-1) + eps  # (B, h, tgt, 1)

        # grad through division: out = num / den
        # grad_num = grad_out / den
        grad_num = grad_out / den_expanded
        # grad_den = -sum(grad_out * num / den², dim=-1)
        grad_den = -(grad_out * num / (den_expanded * den_expanded)).sum(dim=-1)

        # grad through Q @ context → grad_q, grad_context
        grad_q_from_num = torch.matmul(grad_num, context.transpose(-1, -2))
        grad_context = torch.matmul(q.transpose(-2, -1), grad_num)

        # grad through Q · z → grad_q, grad_z
        grad_q_from_den = grad_den.unsqueeze(-1) * z.unsqueeze(-2)
        grad_z = torch.einsum("bht,bhtd->bhd", grad_den, q)

        grad_q = grad_q_from_num + grad_q_from_den

        # grad through context = K^T V → grad_k, grad_v
        grad_k_from_context = torch.matmul(v, grad_context.transpose(-1, -2))
        grad_v = torch.matmul(k, grad_context)

        # grad through z = sum(K) → grad_k
        grad_k_from_z = grad_z.unsqueeze(-2).expand_as(k)

        grad_k = grad_k_from_context + grad_k_from_z

        # Apply mask to gradients
        if mask.numel() > 0:
            mask_expanded = mask.unsqueeze(1).unsqueeze(-1).to(grad_k.dtype)
            grad_k = grad_k * mask_expanded
            grad_v = grad_v * mask_expanded

        return grad_q, grad_k, grad_v, None, None


def cuda_linear_cross_attn(q, k, v, mask, eps=1e-5):
    """CUDA fused linear cross-attention (autograd 지원)"""
    return _LinearCrossAttnCudaFn.apply(q, k, v, mask, eps)


# =========================================================================
# Module
# =========================================================================

class LinearCrossAttention(nn.Module):
    """
    Linear Cross-Attention (Grouped-Query 지원)
    - 시간/메모리 복잡도: O(N)
    - Q는 현재 디코더 상태 (B, tgt_len, d_model)
    - K, V는 인코더 출력 상태 (B, src_len, d_model)
    """
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int | None = None,
        dropout: float = 0.1,
        eps: float = 1e-5,
    ):
        super().__init__()
        if n_kv_heads is None:
            n_kv_heads = n_heads

        assert d_model % n_heads == 0, f"d_model({d_model}) must be divisible by n_heads({n_heads})"
        assert n_heads % n_kv_heads == 0, f"n_heads({n_heads}) must be divisible by n_kv_heads({n_kv_heads})"

        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_rep = n_heads // n_kv_heads
        self.d_head = d_model // n_heads
        self.eps = eps

        # 여기에 BitLinear가 아닌 일반 FP16/BF16 Linear를 사용
        # (어텐션 매핑은 높은 정밀도 요구)
        self.q_proj = nn.Linear(d_model, n_heads * self.d_head, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads * self.d_head, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads * self.d_head, bias=False)
        self.o_proj = nn.Linear(n_heads * self.d_head, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)

    def _repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        """KV 헤드를 n_rep번 반복하여 n_heads개로 확장
        (B, n_kv_heads, seq_len, d_head) -> (B, n_heads, seq_len, d_head)
        """
        if self.n_rep == 1:
            return x
        B, n_kv, seq_len, d_head = x.shape
        x = x.unsqueeze(2).expand(B, n_kv, self.n_rep, seq_len, d_head)
        return x.reshape(B, self.n_heads, seq_len, d_head)

    def forward(
        self,
        x: torch.Tensor,
        encoder_out: torch.Tensor,
        encoder_mask: torch.Tensor | None = None,
        src_doc_ids: torch.Tensor | None = None,
        tgt_doc_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x: 디코더 hidden state (B, tgt_len, d_model)
            encoder_out: 인코더 출력 (B, src_len, d_model)
            encoder_mask: 인코더 마스크 (B, src_len) - True가 유효 데이터
            src_doc_ids: (B, src_len) int — 소스 문서 ID (BOS cumsum 기반)
            tgt_doc_ids: (B, tgt_len) int — 타겟 문서 ID (BOS cumsum 기반)
        """
        B, tgt_len, _ = x.shape
        _, src_len, _ = encoder_out.shape

        # 1. Projections
        q = self.q_proj(x)
        k = self.k_proj(encoder_out)
        v = self.v_proj(encoder_out)

        # Reshape to multi-head: (B, head, seq_len, d_head)
        q = q.view(B, tgt_len, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, src_len, self.n_kv_heads, self.d_head).transpose(1, 2)
        v = v.view(B, src_len, self.n_kv_heads, self.d_head).transpose(1, 2)

        # GQA repeat
        k = self._repeat_kv(k)
        v = self._repeat_kv(v)

        # 2. Apply feature map to make keys and queries positive
        #    phi(x) = elu(x) + 1
        q = elu_feature_map(q)
        k = elu_feature_map(k)

        # 3. Apply Mask to K and V (PAD masking)
        if encoder_mask is not None:
            mask_4d = encoder_mask.view(B, 1, src_len, 1).to(k.dtype)
            k = k * mask_4d
            v = v * mask_4d

        # 4. Compute attention
        if src_doc_ids is not None and tgt_doc_ids is not None:
            # Per-document isolation: 각 문서의 context matrix를 따로 계산
            out = self._doc_isolated_forward(q, k, v, src_doc_ids, tgt_doc_ids, self.eps)
        else:
            out = self._pytorch_forward(q, k, v, self.eps)

        out = self.dropout(out)

        # Concat heads
        out = out.transpose(1, 2).contiguous().view(B, tgt_len, self.d_model)

        return self.o_proj(out)

    @staticmethod
    def _pytorch_forward(q, k, v, eps):
        """전체 소스를 하나의 context로 사용 (문서 격리 없음)"""
        kv = torch.matmul(k.transpose(-1, -2), v)
        z = k.sum(dim=-2)
        num = torch.matmul(q, kv)
        den = torch.einsum("bhld,bhd->bhl", q, z)
        return num / (den.unsqueeze(-1) + eps)

    @staticmethod
    def _doc_isolated_forward(q, k, v, src_doc_ids, tgt_doc_ids, eps):
        """Per-document linear cross-attention

        각 문서별로 context = K_d^T @ V_d를 따로 계산하여
        decoder의 문서 d 위치는 encoder의 문서 d 정보만 참조.

        Context matrix가 d_head × d_head (64×64 = 16KB)로 매우 작아
        문서 수만큼 루프해도 오버헤드 최소.
        """
        out = torch.zeros_like(q)
        max_doc = max(src_doc_ids.max().item(), tgt_doc_ids.max().item()) + 1

        for d in range(int(max_doc)):
            # Source positions for doc d
            src_mask_d = (src_doc_ids == d).unsqueeze(1).unsqueeze(-1).to(k.dtype)
            k_d = k * src_mask_d
            v_d = v * src_mask_d

            # Context for doc d
            context_d = torch.matmul(k_d.transpose(-1, -2), v_d)  # (B, H, D, D)
            z_d = k_d.sum(dim=-2)  # (B, H, D)

            # Output using doc d's context
            num_d = torch.matmul(q, context_d)
            den_d = torch.einsum("bhld,bhd->bhl", q, z_d)
            out_d = num_d / (den_d.unsqueeze(-1) + eps)

            # Apply only to tgt positions belonging to doc d
            tgt_mask_d = (tgt_doc_ids == d).unsqueeze(1).unsqueeze(-1).to(out_d.dtype)
            out = out + out_d * tgt_mask_d

        return out

