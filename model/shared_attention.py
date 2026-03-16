"""Shared Linear Self-Attention (Zamba 스타일)

1개의 공유 Q/K/V/O 프로젝션 + 삽입점별 LoRA adapter.
Zamba 연구: "One attention is all you need" — 1개의 공유 어텐션 블록을
여러 삽입점에서 재사용하되 LoRA로 차별화.

O(N) linear self-attention:
    context = φ(K)ᵀ @ V
    out = φ(Q) @ context / normalizer

Feature map: gelu(x) + 1 (linear_attention.py에서 재사용)
"""
import torch
import torch.nn as nn

from model.bitlinear import Int8Linear
from model.linear_attention import gelu1p_feature_map


class LoRA(nn.Module):
    """Low-Rank Adaptation

    down(d → rank) + up(rank → d), up은 zero-init.
    초기 상태에서 LoRA 출력은 0 → base projection과 동일.
    """

    def __init__(self, d_model: int, rank: int):
        super().__init__()
        self.down = nn.Linear(d_model, rank, bias=False)
        self.up = nn.Linear(rank, d_model, bias=False)
        nn.init.zeros_(self.up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(self.down(x))


class SharedLinearSelfAttention(nn.Module):
    """Shared Linear Self-Attention + per-insertion LoRA

    공유 프로젝션:
        q_proj, k_proj, v_proj, o_proj: FP Linear (어텐션 정밀도 필요)

    삽입점별 LoRA:
        각 삽입점마다 독립적인 LoRA adapter 4개(q, k, v, o)로
        공유 프로젝션을 미세 조정.

    Args:
        d_model: 입력/출력 차원
        n_heads: 어텐션 head 수
        n_insertion_points: 삽입점 수 (= LoRA 세트 수)
        lora_rank: LoRA rank
        dropout: 어텐션 dropout
        eps: normalizer epsilon
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_insertion_points: int = 3,
        lora_rank: int = 16,
        dropout: float = 0.1,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.eps = eps
        assert d_model % n_heads == 0

        # 공유 프로젝션 (INT8 QAT — CPU 인퍼런스 가속)
        self.q_proj = Int8Linear(d_model, d_model, bias=False)
        self.k_proj = Int8Linear(d_model, d_model, bias=False)
        self.v_proj = Int8Linear(d_model, d_model, bias=False)
        self.o_proj = Int8Linear(d_model, d_model, bias=False)

        # 삽입점별 LoRA adapters
        self.lora_q = nn.ModuleList([LoRA(d_model, lora_rank) for _ in range(n_insertion_points)])
        self.lora_k = nn.ModuleList([LoRA(d_model, lora_rank) for _ in range(n_insertion_points)])
        self.lora_v = nn.ModuleList([LoRA(d_model, lora_rank) for _ in range(n_insertion_points)])
        self.lora_o = nn.ModuleList([LoRA(d_model, lora_rank) for _ in range(n_insertion_points)])

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        insertion_idx: int,
        pad_mask: torch.Tensor | None = None,
        doc_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, T, d_model)
            insertion_idx: 몇 번째 삽입점인지 (0-based)
            pad_mask: (B, T) bool — True가 유효 데이터
            doc_ids: (B, T) int — 문서별 ID (0, 1, 2, ...). 문서 격리용.
        Returns:
            (B, T, d_model)
        """
        B, T, D = x.shape

        # 프로젝션 + LoRA
        q = self.q_proj(x) + self.lora_q[insertion_idx](x)
        k = self.k_proj(x) + self.lora_k[insertion_idx](x)
        v = self.v_proj(x) + self.lora_v[insertion_idx](x)

        # Multi-head reshape: (B, n_heads, T, d_head)
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        # Feature map
        q = gelu1p_feature_map(q)
        k = gelu1p_feature_map(k)

        # PAD 마스킹
        if pad_mask is not None:
            mask_4d = pad_mask.view(B, 1, T, 1).to(k.dtype)
            k = k * mask_4d
            v = v * mask_4d

        if doc_ids is not None:
            # 문서별 context 분리 — scatter_add 기반
            out = self._doc_isolated_attention(q, k, v, doc_ids)
        else:
            # 전체 시퀀스 단일 context (기존 동작)
            context = torch.matmul(k.transpose(-1, -2), v)
            z = k.sum(dim=-2)
            num = torch.matmul(q, context)
            den = torch.einsum("bhtd,bhd->bht", q, z)
            out = num / (den.unsqueeze(-1) + self.eps)

        out = self.dropout(out)

        # Concat heads + output projection + LoRA
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        out = self.o_proj(out) + self.lora_o[insertion_idx](out)

        return out

    def _doc_isolated_attention(
        self,
        q: torch.Tensor,  # (B, H, T, d_head)
        k: torch.Tensor,
        v: torch.Tensor,
        doc_ids: torch.Tensor,  # (B, T)
    ) -> torch.Tensor:
        """문서별 분리 Linear Self-Attention

        각 토큰은 같은 문서 내 토큰의 context만 참조.
        scatter_add로 문서별 K^T@V, sum(K) 계산 후 gather로 매핑.
        """
        B, H, T, d = q.shape
        max_docs = doc_ids.max().item() + 1

        # doc_ids를 (B, H, T, 1) 형태로 확장
        did = doc_ids.view(B, 1, T, 1).expand(B, H, T, 1)  # (B, H, T, 1)

        # 문서별 context: K^T @ V → (B, H, max_docs, d, d)
        # scatter_add는 1차원 인덱스에만 작동하므로, 문서별 loop 사용
        # (max_docs는 보통 10~30 — 루프 비용 미미)
        doc_ctx = q.new_zeros(B, H, max_docs, d, d)
        doc_z = q.new_zeros(B, H, max_docs, d)

        for doc in range(max_docs):
            mask = (doc_ids == doc).view(B, 1, T, 1)  # (B, 1, T, 1)
            k_d = k * mask.to(k.dtype)   # 다른 문서 토큰 0으로 마스킹
            v_d = v * mask.to(v.dtype)
            doc_ctx[:, :, doc] = torch.matmul(k_d.transpose(-1, -2), v_d)
            doc_z[:, :, doc] = k_d.sum(dim=-2)

        # 각 토큰이 자기 문서의 context를 참조
        # did_flat: (B, H, T) — 각 토큰의 문서 ID
        did_flat = doc_ids.view(B, 1, T).expand(B, H, T)

        # gather context: (B, H, T, d, d) — 각 토큰의 문서별 context
        ctx_idx = did_flat.view(B, H, T, 1, 1).expand(B, H, T, d, d)
        tok_ctx = doc_ctx.gather(2, ctx_idx)  # (B, H, T, d, d)

        # gather z: (B, H, T, d)
        z_idx = did_flat.view(B, H, T, 1).expand(B, H, T, d)
        tok_z = doc_z.gather(2, z_idx)  # (B, H, T, d)

        # Q @ context: (B, H, T, d)
        num = torch.einsum("bhtd,bhtde->bhte", q, tok_ctx)
        den = torch.einsum("bhtd,bhtd->bht", q, tok_z)

        return num / (den.unsqueeze(-1) + self.eps)
