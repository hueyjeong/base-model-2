"""Mixture of Experts (MoE) BitNet FFN

Switch Transformer 스타일의 MoE 라우팅:
- n_experts개의 BitNetFFN expert 중 top_k개만 활성화
- Load balancing auxiliary loss로 expert collapse 방지
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.encoder import BitNetFFN


class MoEBitNetFFN(nn.Module):
    """Mixture of Experts with BitNet FFN experts

    각 토큰을 softmax router로 top_k expert에 라우팅.
    활성화되지 않은 expert는 연산하지 않음.

    Args:
        d_model: 입력/출력 차원
        d_ff: expert FFN 중간 차원
        n_experts: expert 수
        top_k: 토큰당 활성 expert 수
        dropout: expert FFN 내부 dropout
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        n_experts: int = 16,
        top_k: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_experts = n_experts
        self.top_k = top_k

        # 라우터 (FP — softmax 정밀도 필요)
        self.router = nn.Linear(d_model, n_experts, bias=False)

        # Expert FFN 목록
        self.experts = nn.ModuleList([
            BitNetFFN(d_model, d_ff, dropout=dropout)
            for _ in range(n_experts)
        ])

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, T, d_model)
        Returns:
            (output, aux_loss):
                output: (B, T, d_model)
                aux_loss: scalar — load balancing loss
        """
        B, T, D = x.shape

        # 라우터 logits → softmax
        router_logits = self.router(x)  # (B, T, n_experts)
        router_probs = F.softmax(router_logits, dim=-1)

        # Top-k 선택
        top_probs, top_indices = torch.topk(router_probs, self.top_k, dim=-1)

        # Renormalize
        top_probs = top_probs / (top_probs.sum(dim=-1, keepdim=True) + 1e-8)

        # Expert dispatch (토큰 단위 라우팅)
        x_flat = x.view(-1, D)              # (B*T, D)
        out_flat = torch.zeros_like(x_flat)  # (B*T, D)
        top_indices_flat = top_indices.view(-1, self.top_k)  # (B*T, top_k)
        top_probs_flat = top_probs.view(-1, self.top_k)      # (B*T, top_k)

        for k_idx in range(self.top_k):
            expert_idx = top_indices_flat[:, k_idx]   # (B*T,)
            expert_w = top_probs_flat[:, k_idx]       # (B*T,)

            # argsort로 expert별 연속 배치 구성 (GPU sync 1회만)
            sorted_order = expert_idx.argsort()
            sorted_x = x_flat[sorted_order]
            counts = torch.bincount(expert_idx, minlength=self.n_experts)
            splits = counts.tolist()

            # expert별 연속 chunk 처리
            chunks = sorted_x.split(splits)
            out_chunks = []
            for e_idx, chunk in enumerate(chunks):
                if chunk.shape[0] > 0:
                    out_chunks.append(self.experts[e_idx](chunk))
                else:
                    out_chunks.append(chunk)

            sorted_out = torch.cat(out_chunks)
            sorted_out = sorted_out * expert_w[sorted_order].unsqueeze(-1)

            # 원래 순서로 복원
            inv_order = sorted_order.argsort()
            out_flat = out_flat + sorted_out[inv_order]

        output = out_flat.view(B, T, D)

        # Auxiliary loss (Switch Transformer 스타일)
        aux_loss = self._aux_loss(router_probs, top_indices)

        return output, aux_loss

    def _aux_loss(self, router_probs: torch.Tensor, top_indices: torch.Tensor) -> torch.Tensor:
        """Load balancing auxiliary loss

        expert별 토큰 할당 비율(f)과 라우터 확률 평균(p)의 곱을 최소화.
        균등 분배 시 f_i = 1/n_experts, p_i = 1/n_experts → loss = 1/n_experts
        """
        B, T, _ = router_probs.shape
        n_tokens = B * T

        one_hot = F.one_hot(top_indices, self.n_experts).float()
        f = one_hot.sum(dim=(0, 1, 2)) / (n_tokens * self.top_k)

        p = router_probs.mean(dim=(0, 1))

        aux_loss = self.n_experts * (f * p).sum()

        return aux_loss
