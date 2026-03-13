"""Mixture of Experts (MoE) BitNet FFN

Switch Transformer 스타일의 MoE 라우팅:
- n_experts개의 BitNetFFN expert 중 top_k개만 활성화
- Load balancing auxiliary loss로 expert collapse 방지
- Batched expert forward: 16회 sequential → 1회 bmm
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.encoder import BatchedBitNetFFN

# CUDA MoE dispatch 비활성화
# scatter/gather가 raw CUDA 커널이라 autograd 미지원 → expert gradient 차단
# TODO: torch.autograd.Function으로 scatter/gather backward 구현 후 재활성화
_CUDA_MOE_AVAILABLE = False
_moe_dispatch_forward = None


class MoEBitNetFFN(nn.Module):
    """Mixture of Experts with Batched BitNet FFN

    Batched expert: 모든 expert를 (E, capacity, D) 단일 텐서로 처리.
    16회 sequential forward → 1회 batched bmm.

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

        # Batched Expert FFN (E개를 단일 bmm으로 처리)
        self.experts = BatchedBitNetFFN(n_experts, d_model, d_ff, dropout)

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

        # Expert dispatch
        x_flat = x.view(-1, D)              # (B*T, D)
        out_flat = torch.zeros_like(x_flat)  # (B*T, D)
        top_indices_flat = top_indices.view(-1, self.top_k)  # (B*T, top_k)
        top_probs_flat = top_probs.view(-1, self.top_k)      # (B*T, top_k)

        for k_idx in range(self.top_k):
            expert_idx = top_indices_flat[:, k_idx]   # (B*T,)
            expert_w = top_probs_flat[:, k_idx]       # (B*T,)

            if _CUDA_MOE_AVAILABLE and x.is_cuda:
                # CUDA dispatch: GPU-only scatter/gather + batched expert
                out_flat = out_flat + _moe_dispatch_forward(
                    x_flat, expert_idx, expert_w,
                    self.experts, self.n_experts,
                )
            else:
                # CPU/CUDA fallback: batched expert forward
                out_flat = out_flat + self._batched_dispatch(
                    x_flat, expert_idx, expert_w,
                )

        output = out_flat.view(B, T, D)

        # Auxiliary loss (Switch Transformer 스타일)
        aux_loss = self._aux_loss(router_probs, top_indices)

        return output, aux_loss

    def _batched_dispatch(
        self, x_flat: torch.Tensor, expert_idx: torch.Tensor, expert_w: torch.Tensor,
    ) -> torch.Tensor:
        """Batched expert dispatch — 순수 텐서 연산 (no .item(), no Python loop)

        1. expert별 정렬 → 2. vectorized scatter into (E, capacity, D)
        → 3. batched expert forward → 4. vectorized gather + unsort
        """
        N, D = x_flat.shape
        E = self.n_experts

        # 1. expert별 정렬
        sorted_order = expert_idx.argsort(stable=True)
        sorted_x = x_flat[sorted_order]
        sorted_idx = expert_idx[sorted_order]

        # 2. expert 내 위치 계산 (vectorized — GPU 완전 활용)
        counts = torch.bincount(expert_idx, minlength=E)
        offsets = torch.zeros(E, device=x_flat.device, dtype=torch.long)
        offsets[1:] = counts[:-1].cumsum(0)
        local_pos = torch.arange(N, device=x_flat.device) - offsets[sorted_idx]

        # 고정 capacity (Python int 연산, GPU sync 없음)
        capacity = ((N + E - 1) // E) * 3  # 3× expected — expert collapse 초기에도 안전

        # Overflow 방지 (clamp → 초과 토큰은 마지막 슬롯에 머지, 학습 안정성용)
        local_pos = local_pos.clamp(max=capacity - 1)

        # 3. Scatter: sorted tokens → (E*capacity, D) flat buffer
        flat_idx = sorted_idx * capacity + local_pos
        padded = x_flat.new_zeros(E * capacity, D)
        padded = padded.index_copy(0, flat_idx, sorted_x)

        # 4. Batched expert forward
        expert_out = self.experts(padded.view(E, capacity, D))

        # 5. Gather + weight + unsort
        gathered = expert_out.view(-1, D)[flat_idx]
        weighted = gathered * expert_w[sorted_order].unsqueeze(-1)

        inv_order = sorted_order.argsort()
        return weighted[inv_order]

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
