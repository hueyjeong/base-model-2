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

    Returns:
        (quantized_activations, scale_factor)
    """
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

        # 3. 가중치 양자화 (1.58-bit ternary)
        w_quant, w_scale = quantize_weights_158(self.weight)

        # 4. 양자화된 행렬곱 + 스케일 복원
        out = F.linear(x_quant, w_quant, self.bias)
        out = out * (w_scale * x_scale)

        return out

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"bias={self.bias is not None}, "
            f"quant=1.58bit"
        )
