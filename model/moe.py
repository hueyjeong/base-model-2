"""Mixture of Experts (MoE) BitNet FFN

Switch Transformer 스타일의 MoE 라우팅:
- n_experts개의 BitNetFFN expert 중 top_k개만 활성화
- Load balancing auxiliary loss로 expert collapse 방지
- Batched expert forward: 16회 sequential → 1회 bmm

Vectorized dispatch (autograd.Function):
- .item() 0회, Python loop 0회 — CPU-GPU sync 완전 제거
- torch.compile이 expert forward(bmm)를 자연스럽게 최적화
- scatter/gather backward로 라우터 + 입력 gradient 완전 전파
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.encoder import BatchedBitNetFFN


class _MoEScatter(torch.autograd.Function):
    """토큰 → expert 버퍼 scatter (autograd 지원, CPU sync 없음)

    Forward: x_flat[sorted_order] → buffer[expert, local_pos]
    Backward: grad_buffer[expert, local_pos] → grad_x_flat[sorted_order]

    .item() 없이 고정 capacity 버퍼에 텐서 연산만으로 scatter.
    """

    @staticmethod
    def forward(ctx, x_flat, sorted_order, sorted_expert, local_pos, E, capacity):
        N, D = x_flat.shape
        sorted_x = x_flat[sorted_order]

        buffer = x_flat.new_zeros(E, capacity, D)
        buffer[sorted_expert, local_pos] = sorted_x

        ctx.save_for_backward(sorted_order, sorted_expert, local_pos)
        ctx.N, ctx.D = N, D
        return buffer

    @staticmethod
    def backward(ctx, grad_buffer):
        sorted_order, sorted_expert, local_pos = ctx.saved_tensors
        sorted_grad = grad_buffer[sorted_expert, local_pos]
        grad_x = sorted_grad.new_zeros(ctx.N, ctx.D)
        grad_x[sorted_order] = sorted_grad
        return grad_x, None, None, None, None, None


class _MoEGather(torch.autograd.Function):
    """Expert 출력 → 원래 위치 gather (autograd 지원, CPU sync 없음)

    Forward: expert_out[expert, local_pos] × weight → out[sorted_order]
    Backward:
      - grad_expert_out: grad_out[sorted_order] × weight → scatter to buffer
      - grad_weight: sum(grad_out[sorted_order] × gathered, dim=-1)

    routing weight gradient 포함 — 라우터가 task loss로부터 학습 가능.
    """

    @staticmethod
    def forward(ctx, expert_out, sorted_w, sorted_order, sorted_expert, local_pos, N):
        D = expert_out.shape[-1]
        gathered = expert_out[sorted_expert, local_pos]
        weighted = gathered * sorted_w.unsqueeze(-1)

        out = weighted.new_zeros(N, D)
        out[sorted_order] = weighted

        ctx.save_for_backward(gathered, sorted_w, sorted_order, sorted_expert, local_pos)
        ctx.shapes = (expert_out.shape[0], expert_out.shape[1], D)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        gathered, sorted_w, sorted_order, sorted_expert, local_pos = ctx.saved_tensors
        E, capacity, D = ctx.shapes

        sorted_grad = grad_out[sorted_order]

        # grad expert_out: grad × weight → scatter back
        weighted_grad = sorted_grad * sorted_w.unsqueeze(-1)
        grad_expert = weighted_grad.new_zeros(E, capacity, D)
        grad_expert[sorted_expert, local_pos] = weighted_grad

        # grad routing weight: sum(grad × expert_output, dim=-1)
        grad_w = (sorted_grad * gathered).sum(dim=-1)

        return grad_expert, grad_w, None, None, None, None


class MoEBitNetFFN(nn.Module):
    """Mixture of Experts with Batched BitNet FFN

    Batched expert: 모든 expert를 (E, capacity, D) 단일 텐서로 처리.
    16회 sequential forward → 1회 batched bmm.

    Vectorized scatter/gather (autograd.Function):
    - .item() 0회 (기존 2회/dispatch) — CPU-GPU sync 제거
    - Python loop 0회 (기존 32회/dispatch) — 텐서 연산으로 대체
    - torch.compile 자연 호환 — graph break은 apply() 경계에서만 발생,
      expert forward(bmm)는 compile이 완전 최적화

    Args:
        d_model: 입력/출력 차원
        d_ff: expert FFN 중간 차원
        n_experts: expert 수
        top_k: 토큰당 활성 expert 수
        dropout: expert FFN 내부 dropout
        capacity_factor: expert당 버퍼 용량 = ceil(N/E) × factor
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        n_experts: int = 16,
        top_k: int = 1,
        dropout: float = 0.1,
        capacity_factor: float = 1.5,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_experts = n_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor

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

            out_flat = out_flat + self._dispatch(
                x_flat, expert_idx, expert_w,
            )

        output = out_flat.view(B, T, D)

        # Auxiliary loss (Switch Transformer 스타일)
        aux_loss = self._aux_loss(router_probs, top_indices)

        return output, aux_loss

    def _dispatch(
        self, x_flat: torch.Tensor, expert_idx: torch.Tensor, expert_w: torch.Tensor,
    ) -> torch.Tensor:
        """Vectorized expert dispatch — CPU sync 없음, Python loop 없음

        1. 토큰을 expert별로 정렬 (argsort — GPU)
        2. expert 내 로컬 위치 계산 (bincount + cumsum — GPU)
        3. scatter → (E, capacity, D) 버퍼 (autograd.Function)
        4. expert forward (batched bmm — compile 최적화 대상)
        5. gather → (N, D) 원래 위치 (autograd.Function)
        """
        N, D = x_flat.shape
        E = self.n_experts

        # 고정 capacity — .item() 불필요
        capacity = int(((N + E - 1) // E) * self.capacity_factor)

        # expert별 정렬
        sorted_order = expert_idx.argsort(stable=True)
        sorted_expert = expert_idx[sorted_order]

        # expert 내 로컬 위치 (GPU 텐서 연산만, CPU sync 없음)
        counts = torch.bincount(expert_idx, minlength=E)
        offsets = torch.zeros(E + 1, device=x_flat.device, dtype=torch.long)
        offsets[1:] = counts.cumsum(0)
        local_pos = torch.arange(N, device=x_flat.device) - offsets[sorted_expert]

        # capacity 초과 토큰 clamp (드물지만 안전장치 — 기존 CUDA dispatch와 동일)
        local_pos = local_pos.clamp(max=capacity - 1)

        # routing weight도 sorted order로
        sorted_w = expert_w[sorted_order]

        # Scatter: (N, D) → (E, capacity, D)
        padded = _MoEScatter.apply(x_flat, sorted_order, sorted_expert, local_pos, E, capacity)

        # Expert forward: batched bmm (torch.compile 최적화 대상)
        expert_out = self.experts(padded)  # (E, capacity, D)

        # Gather: (E, capacity, D) → (N, D) with weights
        return _MoEGather.apply(expert_out, sorted_w, sorted_order, sorted_expert, local_pos, N)

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
