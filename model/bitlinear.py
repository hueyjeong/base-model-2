"""BitLinear — 1.58-bit ternary quantization linear layer

BitNet b1.58 논문에 따른 구현:
- 가중치: absmean quantization → {-1, 0, +1}
- 활성화: 8-bit absmax per-token quantization
- 학습: Straight-Through Estimator (STE)
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# CUDA fused quantize_activations — 5+ PyTorch ops → 1 CUDA 커널
# autograd.Function backward에서 amax gradient 무시 → torch.compile NaN 방지
_CUDA_QUANT_ACT = False
try:
    from model.cuda_quant_act import cuda_quantize_activations_8bit, is_available as _quant_act_available
    if _quant_act_available():
        _CUDA_QUANT_ACT = True
except (ImportError, Exception):
    pass


def _ste_round(x: torch.Tensor) -> torch.Tensor:
    """Straight-Through Estimator round: 순전파는 반올림, 역전파는 identity"""
    return x + (x.round() - x).detach()


def quantize_weights_158(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """가중치를 1.58-bit ternary {-1, 0, +1}로 양자화

    absmean 스케일링 후 반올림:
        γ = mean(|W|)
        W_q = round(clip(W / γ, -1, 1))

    Returns:
        (quantized_weights, scale_factor)
    """
    gamma = w.abs().mean().clamp(min=1e-5)
    w_scaled = w / gamma
    w_clipped = w_scaled.clamp(-1.0, 1.0)
    w_quant = _ste_round(w_clipped)
    return w_quant, gamma


def quantize_activations_8bit(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """활성화를 8-bit per-token absmax 양자화

    Q_b = 127 (8-bit 범위)
    η = max(|x|) (per-token)
    x_q = clip(round(x × Q_b / η), -128, 127)

    CUDA 커널 사용 가능 시 fused 단일 커널로 실행 (5+ ops → 1).
    PyTorch 2.10+: amax backward inductor NaN 버그 수정됨 → compiler.disable 불필요.

    Returns:
        (quantized_activations, scale_factor)
    """
    if _CUDA_QUANT_ACT and x.is_cuda:
        return cuda_quantize_activations_8bit(x)

    Q_b = 127.0
    # per-token: 마지막 차원 기준
    eta = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-5)
    x_scaled = x * Q_b / eta
    x_quant = _ste_round(x_scaled.clamp(-128.0, 127.0))
    return x_quant, eta / Q_b


class BitLinear(nn.Module):
    """BitNet b1.58 Linear Layer

    학습 시:
        - FP32 가중치 유지, 순전파 시에만 ternary 양자화 적용
        - STE로 그래디언트 전파

    추론 시:
        - 양자화된 가중치로 정수 행렬곱 수행 가능
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # full-precision 가중치 (학습용)
        self.weight = nn.Parameter(torch.empty(out_features, in_features))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

        # RMSNorm before quantization (Sub-LayerNorm as in BitNet paper)
        self.norm = nn.LayerNorm(in_features, elementwise_affine=False)

        # 가중치 양자화 캐시 — optimizer.step() 전까지 불변이므로 재계산 불필요
        self._weight_version: int = -1
        self._w_quant_cache: torch.Tensor | None = None
        self._w_scale_cache: torch.Tensor | None = None

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, in_features)
        Returns:
            (batch, seq_len, out_features)
        """
        # 1. 입력 정규화
        x_norm = self.norm(x)

        # 2. 활성화 양자화 (8-bit)
        x_quant, x_scale = quantize_activations_8bit(x_norm)

        # 3. 가중치 양자화 (1.58-bit ternary) — 캐시 활용
        # detached 캐시: 양자화 결과만 저장 (그래프 미연결)
        # 매 forward마다 STE 연결을 새로 생성 → grad_accum/n_iterations 안전
        v = self.weight._version
        if v != self._weight_version:
            with torch.no_grad():
                gamma = self.weight.abs().mean().clamp(min=1e-5)
                w_scaled = self.weight / gamma
                w_quant_det = w_scaled.clamp(-1.0, 1.0).round()
            self._w_quant_cache = w_quant_det
            self._w_scale_cache = gamma
            self._weight_version = v

        # STE 재연결: forward는 캐시된 양자화값, backward는 weight로 직접 전파
        w_quant = self.weight + (self._w_quant_cache - self.weight).detach()
        w_scale = self._w_scale_cache

        # 4. 양자화된 행렬곱 + 스케일 복원
        out = F.linear(x_quant, w_quant, self.bias)
        out = out * (w_scale * x_scale)

        return out

    def quantization_loss(self) -> torch.Tensor:
        """가중치가 ternary {-1,0,+1}에 가까워지도록 하는 proximity loss

        L_prox = mean((W - γ * round(clip(W/γ, -1, 1)))²)

        학습 시 CE loss에 합산하여 latent weight의 ternary 근접성 유지.
        """
        with torch.no_grad():
            gamma = self.weight.abs().mean().clamp(min=1e-5)
            w_quant = (self.weight / gamma).clamp(-1.0, 1.0).round()
            target = gamma * w_quant
        return ((self.weight - target) ** 2).mean()

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"bias={self.bias is not None}, "
            f"quant=1.58bit"
        )


class TernaryLinear(nn.Module):
    """STE Ternary Weight Linear — activation quantization 없이 weight만 ternary QAT

    BitLinear 대비 장점:
      - LayerNorm, activation quantization, scale multiply 제거 → 커널 런치 ~8개 절약
      - 추론 시 FP32 activation × ternary weight이므로 act quant 불필요
      - STE로 ternary weight 학습은 유지
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.empty(out_features, in_features))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

        # STE 캐시
        self._weight_version: int = -1
        self._w_quant_cache: torch.Tensor | None = None
        self._w_scale_cache: torch.Tensor | None = None

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 가중치 양자화 (1.58-bit ternary) — 캐시 활용
        v = self.weight._version
        if v != self._weight_version:
            with torch.no_grad():
                gamma = self.weight.abs().mean().clamp(min=1e-5)
                w_scaled = self.weight / gamma
                w_quant_det = w_scaled.clamp(-1.0, 1.0).round()
            self._w_quant_cache = w_quant_det
            self._w_scale_cache = gamma
            self._weight_version = v

        # STE 재연결
        w_quant = self.weight + (self._w_quant_cache - self.weight).detach()
        w_scale = self._w_scale_cache

        # 양자화된 행렬곱 + 스케일 복원 (activation quantization 없음)
        # BF16 AMP: w_quant(fp32) → x.dtype으로 캐스팅
        if w_quant.dtype != x.dtype:
            w_quant = w_quant.to(x.dtype)
            w_scale = w_scale.to(x.dtype)
        out = F.linear(x, w_quant, self.bias)
        out = out * w_scale

        return out

    def quantization_loss(self) -> torch.Tensor:
        """가중치 proximity loss (BitLinear과 동일)"""
        with torch.no_grad():
            gamma = self.weight.abs().mean().clamp(min=1e-5)
            w_quant = (self.weight / gamma).clamp(-1.0, 1.0).round()
            target = gamma * w_quant
        return ((self.weight - target) ** 2).mean()

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"bias={self.bias is not None}, "
            f"quant=ternary_ste"
        )


class BatchedBitLinear(nn.Module):
    """Batched BitLinear: E개 expert 가중치를 단일 (E, out, in) 텐서에 저장

    MoE에서 16회 sequential expert forward 대신 1회 bmm으로 처리.
    수학적으로 E개의 독립 BitLinear와 동일.
    """

    def __init__(self, n_experts: int, in_features: int, out_features: int):
        super().__init__()
        self.n_experts = n_experts
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.empty(n_experts, out_features, in_features))

        # LayerNorm (elementwise_affine=False → 학습 파라미터 없음, E개 독립과 동일)
        self.norm = nn.LayerNorm(in_features, elementwise_affine=False)

        # STE 캐시: per-expert gamma
        self._weight_version: int = -1
        self._w_quant_cache: torch.Tensor | None = None
        self._w_scale_cache: torch.Tensor | None = None

        self._reset_parameters()

    def _reset_parameters(self):
        for e in range(self.n_experts):
            nn.init.kaiming_uniform_(self.weight.data[e], a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (E, capacity, in_features)
        Returns:
            (E, capacity, out_features)
        """
        # 1. Norm: (E, C, in) → per-(e,t) 정규화
        x_norm = self.norm(x)

        # 2. 활성화 양자화: per-token absmax, dim=-1 → (E, C) 각각 독립
        x_quant, x_scale = quantize_activations_8bit(x_norm)

        # 3. 가중치 양자화 — per-expert gamma, STE 캐시
        v = self.weight._version
        if v != self._weight_version:
            with torch.no_grad():
                # gamma: (E, 1, 1) — expert별 absmean
                gamma = self.weight.abs().mean(dim=(-2, -1), keepdim=True).clamp(min=1e-5)
                w_scaled = self.weight / gamma
                w_quant_det = w_scaled.clamp(-1.0, 1.0).round()
            self._w_quant_cache = w_quant_det
            self._w_scale_cache = gamma
            self._weight_version = v

        w_quant = self.weight + (self._w_quant_cache - self.weight).detach()
        w_scale = self._w_scale_cache  # (E, 1, 1)

        # 4. Batched matmul: (E, C, in) @ (E, in, out) → (E, C, out)
        out = torch.bmm(x_quant, w_quant.transpose(-2, -1))
        out = out * (w_scale * x_scale)

        return out

    def extra_repr(self) -> str:
        return (
            f"n_experts={self.n_experts}, "
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"quant=1.58bit_batched"
        )


class Int8Linear(nn.Linear):
    """INT8 Weight QAT Linear — CPU 인퍼런스 최적화

    BitLinear(ternary 1.58bit)보다 정밀하지만 FP보다 빠른 중간 양자화.

    학습 시:
        - FP weight 유지, 순전파에서 INT8 양자화 + STE
        - activation도 INT8 per-token absmax 양자화
    CPU 인퍼런스 시:
        - INT8 weight 고정 저장 → VNNI/AMX 가속
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__(in_features, out_features, bias=bias)
        self.norm = nn.LayerNorm(in_features, elementwise_affine=False)
        # weight 양자화 캐시 (optimizer.step() 전까지 불변)
        self._weight_version: int = -1
        self._w_quant_cache: torch.Tensor | None = None
        self._w_scale_cache: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # LayerNorm (BitLinear과 동일한 sub-layer norm)
        x = self.norm(x)

        # Activation: per-token absmax → INT8 + STE
        x_quant, x_scale = quantize_activations_8bit(x)

        # Weight: per-tensor absmax → INT8 + STE (캐시)
        Qb = 127.0
        v = self.weight._version
        # device 이동 감지 — .cuda()/.cpu() 후 캐시 무효화
        if self._w_quant_cache is not None and self._w_quant_cache.device != self.weight.device:
            self._weight_version = -1
        if v != self._weight_version:
            with torch.no_grad():
                w_eta = self.weight.abs().max().clamp(min=1e-5)
                w_scaled = self.weight * Qb / w_eta
                if self.training:
                    self._w_quant_cache = _ste_round(w_scaled.clamp(-128.0, 127.0))
                else:
                    self._w_quant_cache = w_scaled.clamp(-128.0, 127.0).round()
                self._w_scale_cache = w_eta / Qb
                self._weight_version = v

        # STE 재연결
        w_quant = self.weight + (self._w_quant_cache * self._w_scale_cache - self.weight).detach()

        # AMP autocast 존중: 양자화 값은 {-128~127} 정수 → bf16 정확 표현 가능
        # bf16 matmul = Tensor Core 활용, fp32 대비 ~2x throughput
        if torch.is_autocast_enabled():
            compute_dtype = torch.get_autocast_gpu_dtype()
            x_quant = x_quant.to(compute_dtype)
            w_quant = w_quant.to(compute_dtype)
            x_scale = x_scale.to(compute_dtype)

        out = F.linear(x_quant, w_quant, self.bias)
        return out * x_scale

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"bias={self.bias is not None}, "
            f"quant=int8_qat"
        )
