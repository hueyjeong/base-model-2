"""AttentionLayer — Full 또는 Window Self-Attention + SwiGLU FFN

window_size=None이면 Full Attention, 정수이면 Window Attention.
flash_attn 패키지 있으면 사용 (mask 0bytes), 없으면 F.scaled_dot_product_attention fallback.

구조 (pre-norm):
    RMSNorm → Q/K/V proj (Int8Linear) → RoPE → GQA → Attention → O proj → (+residual)
    RMSNorm → SwiGLU FFN → (+residual)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.checkpoint import checkpoint

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from model.bitlinear import Int8Linear
from model.encoder import RMSNorm, SwiGLUFFN
from model.mixing.full_attention import RotaryEmbedding, _apply_rotary

# flash_attn 선택적 import
_FLASH_ATTN = False
try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    _FLASH_ATTN = True
except ImportError:
    pass


def _reset_mask_to_cu_seqlens(reset_mask: Tensor) -> tuple[Tensor, int]:
    """reset_mask(BOS 위치) → cu_seqlens 변환 (flash_attn_varlen_func용)

    Args:
        reset_mask: (B, T) bool — BOS 위치가 True

    Returns:
        cu_seqlens: (total_docs + 1,) int32 — 누적 문서 길이
        max_seqlen: int — 최대 문서 길이
    """
    B, T = reset_mask.shape
    cu_seqlens_list = [0]
    max_seqlen = 0

    for b in range(B):
        bos_pos = reset_mask[b].nonzero(as_tuple=True)[0]
        n_docs = bos_pos.size(0)
        for i in range(n_docs):
            start = bos_pos[i].item()
            end = bos_pos[i + 1].item() if i + 1 < n_docs else T
            doc_len = end - start
            cu_seqlens_list.append(cu_seqlens_list[-1] + doc_len)
            max_seqlen = max(max_seqlen, doc_len)

    return torch.tensor(cu_seqlens_list, dtype=torch.int32, device=reset_mask.device), max_seqlen


class AttentionLayer(nn.Module):
    """Full 또는 Window self-attention + SwiGLU FFN

    Args:
        d_model: 모델 차원
        d_ff: FFN 중간 차원
        n_heads: Q head 수
        n_kv_heads: KV head 수 (GQA)
        headdim: head 당 차원
        max_seq_len: 최대 시퀀스 길이
        window_size: None=Full Attention, int=Window Attention(w)
        dropout: 드롭아웃 비율
        eps: RMSNorm epsilon
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        n_heads: int,
        n_kv_heads: int,
        headdim: int,
        max_seq_len: int = 4096,
        window_size: int | None = None,
        dropout: float = 0.1,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.headdim = headdim
        self.kv_repeat = n_heads // n_kv_heads
        self.window_size = window_size

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
        """
        Args:
            x: (B, T, D)
            pad_mask: (B, T) bool — True=유효
            reset_mask: (B, T) bool — BOS 위치 True (문서 격리용)
        Returns:
            (B, T, D)
        """
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

        q = self.q_proj(x).view(B, T, H, d)       # (B, T, H, d)
        k = self.k_proj(x).view(B, T, H_kv, d)    # (B, T, H_kv, d)
        v = self.v_proj(x).view(B, T, H_kv, d)    # (B, T, H_kv, d)

        # RoPE — (B, H, T, d) 형태 필요
        q_t = q.transpose(1, 2)  # (B, H, T, d)
        k_t = k.transpose(1, 2)  # (B, H_kv, T, d)
        q_t, k_t = self.rope(q_t, k_t)
        q = q_t.transpose(1, 2)  # (B, T, H, d)
        k = k_t.transpose(1, 2)  # (B, T, H_kv, d)

        if _FLASH_ATTN and x.is_cuda and q.dtype in (torch.float16, torch.bfloat16):
            out = self._flash_attention(q, k, v, pad_mask, reset_mask)
        else:
            out = self._sdpa_attention(q, k, v, pad_mask, reset_mask, B, T, D)

        return self.o_proj(out)

    def _flash_attention(
        self, q: Tensor, k: Tensor, v: Tensor,
        pad_mask: Tensor | None, reset_mask: Tensor | None,
    ) -> Tensor:
        """flash_attn 사용 — mask 할당 0bytes"""
        B, T, H, d = q.shape

        if self.window_size is not None:
            # Window Attention
            w = self.window_size
            # flash_attn_func: (B, T, H, d) 형태 직접 사용, GQA 자동 처리
            out = flash_attn_func(
                q, k, v,
                window_size=(w // 2, w // 2),
                causal=False,
            )
        elif reset_mask is not None:
            # Full Attention + 문서 격리 → varlen
            cu_seqlens, max_seqlen = _reset_mask_to_cu_seqlens(reset_mask)
            # (B, T, H, d) → (total_tokens, H, d) — pad 제거
            q_flat = q.reshape(B * T, H, d)
            k_flat = k.reshape(B * T, self.n_kv_heads, d)
            v_flat = v.reshape(B * T, self.n_kv_heads, d)
            out_flat = flash_attn_varlen_func(
                q_flat, k_flat, v_flat,
                cu_seqlens_q=cu_seqlens, cu_seqlens_k=cu_seqlens,
                max_seqlen_q=max_seqlen, max_seqlen_k=max_seqlen,
                causal=False,
            )
            out = out_flat.reshape(B, T, H, d)
        else:
            # Full Attention, 패킹 없음
            out = flash_attn_func(q, k, v, causal=False)

        return out.reshape(B, T, H * d)

    def _sdpa_attention(
        self, q: Tensor, k: Tensor, v: Tensor,
        pad_mask: Tensor | None, reset_mask: Tensor | None,
        B: int, T: int, D: int,
    ) -> Tensor:
        """F.scaled_dot_product_attention fallback (CPU/비CUDA 환경)"""
        H = self.n_heads
        d = self.headdim

        # (B, T, H, d) → (B, H, T, d)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # GQA expand
        if self.kv_repeat > 1:
            k = k.repeat_interleave(self.kv_repeat, dim=1)
            v = v.repeat_interleave(self.kv_repeat, dim=1)

        # Attention mask
        attn_mask = self._make_mask(reset_mask, pad_mask, T, q.device)

        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=0.0, is_causal=False,
        )
        return out.transpose(1, 2).contiguous().reshape(B, T, D)

    def _make_mask(
        self, reset_mask: Tensor | None, pad_mask: Tensor | None,
        T: int, device: torch.device,
    ) -> Tensor | None:
        """Window + 문서 격리 + PAD 마스킹 → attention mask"""
        masks = []

        if self.window_size is not None:
            pos = torch.arange(T, device=device)
            band = (pos.unsqueeze(1) - pos.unsqueeze(0)).abs() <= (self.window_size // 2)
            masks.append(band)

        if reset_mask is not None:
            doc_id = (reset_mask.int().cumsum(dim=1) - 1)
            doc_mask = (doc_id.unsqueeze(2) == doc_id.unsqueeze(1))
            masks.append(doc_mask)

        if pad_mask is not None:
            key_mask = pad_mask.unsqueeze(1)
            masks.append(key_mask)

        if not masks:
            return None

        combined = masks[0]
        for m in masks[1:]:
            combined = combined & m

        # (T, T) → (1, 1, T, T), (B, T, T) → (B, 1, T, T)
        if combined.dim() == 2:
            return combined.unsqueeze(0).unsqueeze(0)
        return combined.unsqueeze(1)


if __name__ == "__main__":
    print("=== AttentionLayer Smoke Test ===\n")
    print(f"flash_attn: {'사용 가능' if _FLASH_ATTN else '미설치 (SDPA fallback)'}\n")

    d_model, d_ff = 256, 512
    n_heads, n_kv_heads, headdim = 4, 2, 64

    # Full Attention
    fa = AttentionLayer(d_model, d_ff, n_heads, n_kv_heads, headdim, window_size=None)
    fa._init_weights()
    params_fa = sum(p.numel() for p in fa.parameters())
    print(f"Full Attention 파라미터: {params_fa:,}")

    x = torch.randn(2, 64, d_model)
    mask = torch.ones(2, 64, dtype=torch.bool)
    reset = torch.zeros(2, 64, dtype=torch.bool)
    reset[:, 0] = True
    reset[0, 32] = True  # 문서 2개

    out = fa(x, pad_mask=mask, reset_mask=reset)
    assert out.shape == (2, 64, d_model), f"FA shape: {out.shape}"
    print(f"  FA forward OK: {out.shape}")

    out.sum().backward()
    print("  FA backward OK")

    # Window Attention (w=64)
    wa = AttentionLayer(d_model, d_ff, n_heads, n_kv_heads, headdim, window_size=64)
    wa._init_weights()
    x = torch.randn(2, 256, d_model)
    mask = torch.ones(2, 256, dtype=torch.bool)
    reset = torch.zeros(2, 256, dtype=torch.bool)
    reset[:, 0] = True

    out = wa(x, pad_mask=mask, reset_mask=reset)
    assert out.shape == (2, 256, d_model), f"WA shape: {out.shape}"
    print(f"\n  WA(w=64) forward OK: {out.shape}")

    out.sum().backward()
    print("  WA backward OK")

    # Window sizes
    for w in [32, 128, 256]:
        wa_test = AttentionLayer(d_model, d_ff, n_heads, n_kv_heads, headdim, window_size=w)
        out = wa_test(torch.randn(1, 128, d_model))
        assert out.shape == (1, 128, d_model)
        print(f"  WA(w={w}) OK")

    # 768 차원 (128M)
    fa768 = AttentionLayer(768, 2048, 12, 4, 64, window_size=None)
    p768 = sum(p.numel() for p in fa768.parameters())
    print(f"\n  d=768 FA params: {p768:,}")
    out768 = fa768(torch.randn(1, 32, 768))
    assert out768.shape == (1, 32, 768)
    print("  d=768 FA forward OK")

    print("\n모든 테스트 통과!")
