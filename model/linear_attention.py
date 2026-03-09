"""Linear Cross-Attention module

Grouped-Query Linear Attention (GQA) for efficient O(N) cross-attention.
Designed for integrating encoder context into a decoder without the O(N^2)
memory/compute bottleneck of standard Softmax attention.

Features:
- Feature map phi(x) = relu(x) + 1 implementation for positive keys/queries (≥1 guaranteed)
- Custom CUDA fused kernel for K^T V + Q(K^T V) + normalization
- Document-isolated CUDA kernel for per-doc context (scatter/gather 구조)
- PyTorch exact fallback when CUDA kernel unavailable

Doc-Isolated CUDA 커널 (cuda_doc_linear_attn_kernel.cu):
  Phase 1 (scatter): context[d,i,j] += k[s,i]*v[s,j]  (블록 독점 → atomic 불필요)
  Phase 2 (gather):  out[t,j] = q[t]·ctx[d_t,:,j] / (q[t]·z[d_t]+eps)
  → Python D-loop 없음, full-seq 중간 텐서 없음, O(seq_len) 메모리
"""
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# =========================================================================
# CUDA fused kernel (non-doc-isolated: JIT compile on first use)
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


# =========================================================================
# CUDA doc-isolated kernel (JIT compile on first use)
# =========================================================================
_CUDA_DOC_LINEAR_ATTN = None
_FALLBACK_WARNED = False

# 커널 버전 — 바뀌면 JIT 캐시 강제 무효화
_KERNEL_VERSION = "v3_fused"


def _load_cuda_doc_linear_attn():
    """doc-isolated scatter/gather CUDA 커널 로드

    - DDP 멀티프로세스: rank-0이 먼저 빌드 후 barrier, 나머지 rank는 로드만
    - 버전 불일치 시 자동 재빌드 (stale cache 방지)
    """
    global _CUDA_DOC_LINEAR_ATTN
    if _CUDA_DOC_LINEAR_ATTN is not None:
        return _CUDA_DOC_LINEAR_ATTN

    import hashlib
    from torch.utils.cpp_extension import load

    this_dir = os.path.dirname(__file__)
    cpp_src = os.path.join(this_dir, "cuda_doc_linear_attn_ext.cpp")
    cu_src  = os.path.join(this_dir, "cuda_doc_linear_attn_kernel.cu")

    # DDP rank 결정
    rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
    world = int(os.environ.get("LOCAL_WORLD_SIZE", os.environ.get("WORLD_SIZE", "1")))

    # 소스 해시 기반 이름 — 소스가 바뀌면 자동 재빌드
    src_hash = ""
    for p in [cpp_src, cu_src]:
        if os.path.exists(p):
            with open(p, 'rb') as fh:
                src_hash += hashlib.md5(fh.read()).hexdigest()
    ext_name = f"doc_linear_attn_cuda_{hashlib.md5(src_hash.encode()).hexdigest()[:8]}"

    if world > 1 and torch.distributed.is_initialized():
        # rank 0이 먼저 빌드, 나머지는 대기
        if rank == 0:
            _CUDA_DOC_LINEAR_ATTN = load(
                name=ext_name,
                sources=[cpp_src, cu_src],
                extra_cflags=["-O3"],
                extra_cuda_cflags=["-O3", "--use_fast_math"],
                verbose=False,
            )
        torch.distributed.barrier()
        if rank != 0:
            _CUDA_DOC_LINEAR_ATTN = load(
                name=ext_name,
                sources=[cpp_src, cu_src],
                extra_cflags=["-O3"],
                extra_cuda_cflags=["-O3", "--use_fast_math"],
                verbose=False,
            )
    else:
        _CUDA_DOC_LINEAR_ATTN = load(
            name=ext_name,
            sources=[cpp_src, cu_src],
            extra_cflags=["-O3"],
            extra_cuda_cflags=["-O3", "--use_fast_math"],
            verbose=False,
        )

    # fused 커널 존재 여부 검증
    if not hasattr(_CUDA_DOC_LINEAR_ATTN, 'doc_fused_forward'):
        import warnings
        warnings.warn(
            f"CUDA doc-linear-attn extension '{ext_name}' "
            f"lacks doc_fused_forward — stale cache? "
            f"Run: rm -rf ~/.cache/torch_extensions/ and retry.",
            RuntimeWarning,
        )
        _CUDA_DOC_LINEAR_ATTN = None
        raise RuntimeError("Stale CUDA extension cache")

    # 1회성 로드 성공 로그 (학습 로그에서 커널 사용 여부 확인용)
    if rank == 0:
        print(f"[LinearCrossAttn] CUDA fused kernel loaded: {ext_name} (version={_KERNEL_VERSION})")

    return _CUDA_DOC_LINEAR_ATTN


def _env_flag(name: str, default: str = "1") -> bool:
    value = os.environ.get(name, default).strip().lower()
    return value not in {"0", "false", "off", "no"}


HAS_CUDA_LINEAR_ATTN = torch.cuda.is_available()


# =========================================================================
# Feature Map
# =========================================================================

def elu_feature_map(x: torch.Tensor) -> torch.Tensor:
    """(deprecated) phi(x) = elu(x) + 1 — relu1p 로 대체됨"""
    return F.elu(x) + 1.0


def relu1p_feature_map(x: torch.Tensor) -> torch.Tensor:
    """(deprecated) phi(x) = relu(x) + 1 — gelu1p 로 대체됨"""
    return F.relu(x) + 1.0


def gelu1p_feature_map(x: torch.Tensor) -> torch.Tensor:
    """리니어 어텐션을 위한 양수 보장 특징 매핑: phi(x) = gelu(x) + 1

    relu+1과 달리 x < 0 영역에서도 smooth gradient가 전달되어
    attention weight 학습이 더 원활합니다.
    출력 하한 ≈ 0.83 (GELU 최솟값 ≈ -0.17 at x ≈ -0.75) → 양수 보장.
    """
    return F.gelu(x) + 1.0


# =========================================================================
# Autograd Function with CUDA forward + PyTorch backward
# =========================================================================

@torch.compiler.allow_in_graph
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
        den_expanded = (den.unsqueeze(-1) + eps).clamp(min=0.1)  # gradient explosion 방지

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
# Doc-Isolated Linear Cross-Attention — CUDA scatter/gather autograd Function
# =========================================================================

@torch.compiler.allow_in_graph
class _DocLinearAttnFn(torch.autograd.Function):
    """Document-isolated linear cross-attention — Fused CUDA kernel.

    Grid = (B*H*max_docs,) 로 문서별 블록을 할당하여:
      1. 블록 수 30x 증가 → GPU SM 점유율 대폭 상승
      2. context가 smem에만 존재 → 글로벌 메모리 트래픽 제거
      3. 단일 backward 커널: context 재계산 + grad 전부 smem 내 처리
      4. autograd 저장 텐서: context/z 제거 → 11.5MB/layer 절약
    """

    @staticmethod
    def forward(ctx, q, k, v, src_doc_ids, tgt_doc_ids, max_docs, eps):
        ext = _load_cuda_doc_linear_attn()
        # V3: returns (out, den, context, z) — context/z cached for backward
        if hasattr(ext, 'doc_v3_forward'):
            results = ext.doc_v3_forward(q, k, v, src_doc_ids, tgt_doc_ids, max_docs, eps)
            out, den, fwd_ctx, fwd_z = results[0], results[1], results[2], results[3]
            ctx.save_for_backward(q, k, v, out, den, fwd_ctx, fwd_z)
            ctx._v3_cached = True
        else:
            out, den = ext.doc_fused_forward(q, k, v, src_doc_ids, tgt_doc_ids, max_docs, eps)
            ctx.save_for_backward(q, k, v, out, den)
            ctx._v3_cached = False

        ctx.src_doc_ids = src_doc_ids
        ctx.tgt_doc_ids = tgt_doc_ids
        ctx.max_docs    = max_docs
        return out

    @staticmethod
    def backward(ctx, grad_out):
        ext = _load_cuda_doc_linear_attn()

        if ctx._v3_cached:
            # V3 cached: skip scatter_ctx recomputation
            q, k, v, out, den, fwd_ctx, fwd_z = ctx.saved_tensors
            grads = ext.doc_v3_backward_cached(
                q, k, v, out, den, grad_out.contiguous(),
                fwd_ctx, fwd_z,
                ctx.tgt_doc_ids, ctx.src_doc_ids, ctx.max_docs
            )
        elif hasattr(ext, 'doc_v3_backward'):
            q, k, v, out, den = ctx.saved_tensors
            grads = ext.doc_v3_backward(
                q, k, v, out, den, grad_out.contiguous(),
                ctx.src_doc_ids, ctx.tgt_doc_ids, ctx.max_docs
            )
        else:
            q, k, v, out, den = ctx.saved_tensors
            grads = ext.doc_fused_backward(
                q, k, v, out, den, grad_out.contiguous(),
                ctx.src_doc_ids, ctx.tgt_doc_ids, ctx.max_docs
            )
        return grads[0], grads[1], grads[2], None, None, None, None


def cuda_doc_linear_attn(q, k, v, src_doc_ids, tgt_doc_ids, max_docs, eps=1e-5):
    """Document-isolated linear cross-attention (CUDA fused, autograd 지원)"""
    return _DocLinearAttnFn.apply(q, k, v, src_doc_ids, tgt_doc_ids, max_docs, eps)


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
        max_docs: int | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x: 디코더 hidden state (B, tgt_len, d_model)
            encoder_out: 인코더 출력 (B, src_len, d_model)
            encoder_mask: 인코더 마스크 (B, src_len) - True가 유효 데이터
            src_doc_ids: (B, src_len) int — 소스 문서 ID (BOS cumsum 기반)
            tgt_doc_ids: (B, tgt_len) int — 타겟 문서 ID (BOS cumsum 기반)
            max_docs: int — 최대 문서 수 (pre-computed, .item() sync 제거용)
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
        #    phi(x) = relu(x) + 1  (≥1 보장 → z=sum(phi(K)) 0 방지)
        q = gelu1p_feature_map(q)
        k = gelu1p_feature_map(k)

        # 3. Apply Mask to K and V (PAD masking)
        if encoder_mask is not None:
            mask_4d = encoder_mask.view(B, 1, src_len, 1).to(k.dtype)
            k = k * mask_4d
            v = v * mask_4d

        # 4. Compute attention
        if src_doc_ids is not None and tgt_doc_ids is not None:
            # CUDA doc-isolated scatter/gather 경로
            # float32 변환은 CUDA C++ launcher 내부에서 처리 — Python 측 임시 텐서 제거
            # → autograd에 원본 dtype(bf16) 저장 → 레이어당 ~18MB 절약 × 12레이어 = ~216MB
            src_ids = src_doc_ids.to(torch.int32)
            tgt_ids = tgt_doc_ids.to(torch.int32)

            # max_docs: 외부에서 pre-computed → .item() 0회 (기존: 레이어당 2회×12=24회 sync)
            if max_docs is None:
                max_docs = int(torch.max(src_ids.max(), tgt_ids.max()).item()) + 1

            if q.is_cuda:
                try:
                    out = cuda_doc_linear_attn(
                        q, k, v, src_ids, tgt_ids, max_docs, self.eps
                    ).to(q.dtype)
                except Exception as e:
                    # CUDA 커널 미빌드 시 PyTorch fallback — 경고 1회
                    global _FALLBACK_WARNED
                    if not _FALLBACK_WARNED:
                        import warnings
                        warnings.warn(
                            f"CUDA doc-linear-attn 커널 실패, PyTorch loop fallback "
                            f"사용 (느림). 오류: {e}",
                            RuntimeWarning,
                        )
                        _FALLBACK_WARNED = True
                    out = self._doc_isolated_forward_pytorch(
                        q.float(), k.float(), v.float(), src_doc_ids, tgt_doc_ids, self.eps
                    ).to(q.dtype)
            else:
                out = self._doc_isolated_forward_pytorch(
                    q.float(), k.float(), v.float(), src_doc_ids, tgt_doc_ids, self.eps
                ).to(q.dtype)
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
    def _doc_isolated_forward_pytorch(q, k, v, src_doc_ids, tgt_doc_ids, eps):
        """CPU / CUDA 커널 미빌드 시 fallback — Python loop 기반.

        CUDA 커널과 동일한 결과를 내지만 Python loop(D회)를 사용하므로
        배치 규모가 크면 느림. 정확도 검증 및 CPU 환경용.
        """
        out = torch.zeros_like(q)
        max_doc = int(max(src_doc_ids.max().item(), tgt_doc_ids.max().item())) + 1

        for d in range(max_doc):
            src_mask_d = (src_doc_ids == d).unsqueeze(1).unsqueeze(-1).to(k.dtype)
            k_d = k * src_mask_d
            v_d = v * src_mask_d

            context_d = torch.matmul(k_d.transpose(-1, -2), v_d)
            z_d = k_d.sum(dim=-2)

            num_d = torch.matmul(q, context_d)
            den_d = torch.einsum("bhld,bhd->bhl", q, z_d)
            out_d = num_d / (den_d.unsqueeze(-1) + eps)

            tgt_mask_d = (tgt_doc_ids == d).unsqueeze(1).unsqueeze(-1).to(out_d.dtype)
            out = out + out_d * tgt_mask_d

        return out

