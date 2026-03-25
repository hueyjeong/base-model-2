"""AttentionLayer — 양방향 Linear Attention + SwiGLU FFN

양방향 Linear Attention: φ(Q) · (φ(K)^T · V) — O(Td²), softmax 없음.
모든 연산이 matmul + element-wise → ONNX 표준 op, CPU/GPU 모두 효율적.
문서 격리: reset_mask에서 문서 경계 파악 → 문서별 독립 KV 집계.
학습과 추론이 동일한 연산 — 배포 시 불일치 없음.

구조 (pre-norm):
    RMSNorm → Q/K/V proj (Int8Linear) → RoPE → Linear Attention → O proj → (+residual)
    RMSNorm → SwiGLU FFN → (+residual)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from model.bitlinear import Int8Linear
from model.encoder import RMSNorm, SwiGLUFFN
from model.mixing.full_attention import RotaryEmbedding, _apply_rotary


def _feature_map(x: Tensor) -> Tensor:
    """Linear attention feature map: elu(x) + 1 (양수 보장)"""
    return F.elu(x) + 1


class AttentionLayer(nn.Module):
    """양방향 Linear Attention + SwiGLU FFN

    Linear Attention: φ(Q) · (φ(K)^T · V) — O(Td²)
    학습과 추론 동일 연산. ONNX 표준 op만 사용.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        n_heads: int,
        n_kv_heads: int,
        headdim: int,
        max_seq_len: int = 4096,
        dropout: float = 0.1,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.headdim = headdim
        self.kv_repeat = n_heads // n_kv_heads

        d_kv = n_kv_heads * headdim

        # Pre-norm → Attention
        self.norm1 = RMSNorm(d_model, eps=eps)
        self.q_proj = Int8Linear(d_model, d_model, bias=False)
        self.k_proj = Int8Linear(d_model, d_kv, bias=False)
        self.v_proj = Int8Linear(d_model, d_kv, bias=False)
        self.o_proj = Int8Linear(d_model, d_model, bias=False)
        self.rope = RotaryEmbedding(headdim, max_seq_len=max_seq_len)

        # Pre-norm → FFN
        self.norm2 = RMSNorm(d_model, eps=eps)
        self.ffn = SwiGLUFFN(d_model, d_ff, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def _init_weights(self):
        """Q/K/V Xavier, O zero-init"""
        for proj in [self.q_proj, self.k_proj, self.v_proj]:
            nn.init.xavier_uniform_(proj.weight)
        nn.init.zeros_(self.o_proj.weight)

    def forward(
        self,
        x: Tensor,
        pad_mask: Tensor | None = None,
        reset_mask: Tensor | None = None,
    ) -> Tensor:
        # ── Attention ──
        h = self.norm1(x)
        attn_out = self._attention(h, pad_mask, reset_mask)
        x = x + self.dropout(attn_out)

        # ── FFN ──
        x = x + self.dropout(self.ffn(self.norm2(x)))

        return x

    def _attention(
        self, x: Tensor, pad_mask: Tensor | None, reset_mask: Tensor | None,
    ) -> Tensor:
        B, T, D = x.shape
        H = self.n_heads
        H_kv = self.n_kv_heads
        d = self.headdim

        q = self.q_proj(x).view(B, T, H, d)
        k = self.k_proj(x).view(B, T, H_kv, d)
        v = self.v_proj(x).view(B, T, H_kv, d)

        # RoPE
        q_t = q.transpose(1, 2)
        k_t = k.transpose(1, 2)
        q_t, k_t = self.rope(q_t, k_t)
        q = q_t.transpose(1, 2)
        k = k_t.transpose(1, 2)

        out = self._linear_attention(q, k, v, pad_mask, reset_mask)

        return self.o_proj(out)

    # ── Linear Attention (ONNX/CPU 호환) ──

    def _linear_attention(
        self, q: Tensor, k: Tensor, v: Tensor,
        pad_mask: Tensor | None, reset_mask: Tensor | None,
    ) -> Tensor:
        """양방향 Linear Attention: φ(Q) · (φ(K)^T · V)

        O(Td²) — T×T 행렬 생성 없음, ONNX 표준 op만 사용.
        문서 격리: reset_mask에서 문서별 독립 KV 집계.
        """
        B, T, H, d = q.shape
        H_kv = self.n_kv_heads

        # GQA expand
        if self.kv_repeat > 1:
            k = k.repeat_interleave(self.kv_repeat, dim=2)  # (B, T, H, d)
            v = v.repeat_interleave(self.kv_repeat, dim=2)

        # Feature map
        q = _feature_map(q)  # (B, T, H, d)
        k = _feature_map(k)

        # PAD 마스킹: PAD 위치의 K, V를 0으로
        if pad_mask is not None:
            mask_expand = pad_mask.unsqueeze(-1).unsqueeze(-1).to(k.dtype)  # (B, T, 1, 1)
            k = k * mask_expand
            v = v * mask_expand

        # (B, T, H, d) → (B, H, T, d)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if reset_mask is not None:
            out = self._linear_attn_with_docs(q, k, v, reset_mask)
        else:
            # 전역 linear attention
            kv = torch.einsum("bhsd,bhse->bhde", k, v)  # (B, H, d, d)
            out = torch.einsum("bhsd,bhde->bhse", q, kv)  # (B, H, T, d)
            # 정규화
            z = torch.einsum("bhsd,bhd->bhs", q, k.sum(dim=2))  # (B, H, T)
            out = out / (z.unsqueeze(-1).clamp(min=1e-6))

        out = out.transpose(1, 2).contiguous().reshape(B, T, H * d)
        return out

    def _linear_attn_with_docs(
        self, q: Tensor, k: Tensor, v: Tensor, reset_mask: Tensor,
    ) -> Tensor:
        """문서별 독립 linear attention

        배치는 벡터화, 문서는 n_docs loop (n_docs ~5-20, 작음).
        각 iteration에서 mask select → einsum → mask scatter.
        B loop 제거, 문서 loop만 유지.
        """
        B, H, T, d = q.shape

        # 문서 ID: (B, T)
        doc_id = reset_mask.int().cumsum(dim=1) - 1

        out = torch.zeros_like(q)
        n_docs = doc_id.max().item() + 1

        for doc in range(n_docs):
            # 이 문서에 속하는 토큰 마스크: (B, T)
            mask = (doc_id == doc)  # (B, T)
            if not mask.any():
                continue

            # 마스크 확장: (B, 1, T, 1)
            m = mask.unsqueeze(1).unsqueeze(-1).to(q.dtype)  # (B, 1, T, 1)

            # 이 문서의 K, V만 추출 (마스크 외 위치 0)
            k_doc = k * m  # (B, H, T, d)
            v_doc = v * m

            # KV 집계: (B, H, d, d)
            kv = torch.einsum("bhsd,bhse->bhde", k_doc, v_doc)
            # Q @ KV: (B, H, T, d)
            out_doc = torch.einsum("bhsd,bhde->bhse", q, kv)
            # 정규화
            k_sum = k_doc.sum(dim=2)  # (B, H, d)
            z = torch.einsum("bhsd,bhd->bhs", q, k_sum).clamp(min=1e-6)

            # 이 문서 위치에만 기록
            out += (out_doc / z.unsqueeze(-1)) * m

        return out



if __name__ == "__main__":
    print("=== AttentionLayer Smoke Test (Linear Attention) ===\n")

    d_model, d_ff = 256, 512
    n_heads, n_kv_heads, headdim = 4, 2, 64

    # ── Linear Attention 테스트 ──
    layer = AttentionLayer(d_model, d_ff, n_heads, n_kv_heads, headdim, )
    layer._init_weights()
    params = sum(p.numel() for p in layer.parameters())
    print(f"파라미터: {params:,}")

    # Forward
    x = torch.randn(2, 64, d_model)
    mask = torch.ones(2, 64, dtype=torch.bool)
    reset = torch.zeros(2, 64, dtype=torch.bool)
    reset[:, 0] = True
    reset[0, 32] = True  # 문서 2개

    out = layer(x, pad_mask=mask, reset_mask=reset)
    assert out.shape == (2, 64, d_model), f"shape: {out.shape}"
    print(f"Linear Attn forward OK: {out.shape}")

    out.sum().backward()
    print("Linear Attn backward OK")

    # 문서 격리 테스트 (eval 모드에서 — dropout 비활성)
    print("\n=== 문서 격리 테스트 ===")
    layer.eval()
    x = torch.randn(1, 128, d_model)
    mask = torch.ones(1, 128, dtype=torch.bool)
    reset = torch.zeros(1, 128, dtype=torch.bool)
    reset[0, 0] = True
    reset[0, 64] = True

    x2 = x.clone()
    x2[0, 64:] = 0.0

    with torch.no_grad():
        out1 = layer(x, mask, reset)
        out2 = layer(x2, mask, reset)
    diff = (out1[0, :64] - out2[0, :64]).abs().max().item()
    print(f"  문서 2 제거 후 문서 1 차이: {diff:.6f}")
    assert diff < 1e-4, f"문서 격리 실패: {diff}"
    print("  ✓ 문서 격리 OK")
    layer.train()

    # seq=4096 CPU 벤치마크
    import time
    print("\n=== CPU 벤치마크 (linear attention) ===")
    layer768 = AttentionLayer(768, 2048, 12, 4, 64, ).eval()
    x_big = torch.randn(1, 4096, 768)
    mask_big = torch.ones(1, 4096, dtype=torch.bool)
    reset_big = torch.zeros(1, 4096, dtype=torch.bool)
    reset_big[0, 0] = True

    # warmup
    with torch.no_grad():
        for _ in range(2):
            layer768(x_big, mask_big, reset_big)

    N = 5
    t0 = time.time()
    with torch.no_grad():
        for _ in range(N):
            layer768(x_big, mask_big, reset_big)
    dt = (time.time() - t0) / N * 1000
    print(f"  seq=4096, d=768: {dt:.0f}ms/layer (linear)")

    print("\n모든 테스트 통과!")
