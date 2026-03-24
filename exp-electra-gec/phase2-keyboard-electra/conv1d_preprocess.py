"""Conv1d 전처리 — 자모 토큰을 음절 단위 feature로 합성

depthwise conv1d(k=4) → RMSNorm → residual add.
자모 개별 토큰은 의미 없으므로 인접 토큰을 합쳐 음절 패턴을 만든다.
"""
import torch
import torch.nn as nn
from torch import Tensor

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from model.encoder import RMSNorm


class Conv1dPreprocess(nn.Module):
    """Depthwise Conv1d + RMSNorm → residual

    x → conv1d(depthwise, k=4) → RMSNorm → x + conv_out
    """

    def __init__(self, d_model: int, kernel_size: int = 4, eps: float = 1e-6):
        super().__init__()
        self.conv = nn.Conv1d(
            d_model, d_model, kernel_size,
            padding=kernel_size // 2, groups=d_model, bias=False,
        )
        self.norm = RMSNorm(d_model, eps=eps)

    def forward(self, x: Tensor, pad_mask: Tensor | None = None) -> Tensor:
        """
        Args:
            x: (B, T, D)
            pad_mask: (B, T) bool — True=유효
        Returns:
            (B, T, D)
        """
        B, T, D = x.shape
        h = x.transpose(1, 2)          # (B, D, T)
        conv_out = self.conv(h)
        if conv_out.size(2) != T:
            conv_out = conv_out[:, :, :T]
        conv_out = conv_out.transpose(1, 2)  # (B, T, D)
        conv_out = self.norm(conv_out)
        out = x + conv_out

        if pad_mask is not None:
            out = out * pad_mask.unsqueeze(-1).to(out.dtype)

        return out


if __name__ == "__main__":
    print("=== Conv1dPreprocess Smoke Test ===\n")

    d = 256
    mod = Conv1dPreprocess(d, kernel_size=4)
    params = sum(p.numel() for p in mod.parameters())
    print(f"파라미터: {params:,} (conv: {d}*4={d*4}, norm: {d})")

    for T in [64, 256, 1000, 4096]:
        x = torch.randn(2, T, d)
        mask = torch.ones(2, T, dtype=torch.bool)
        mask[0, T-5:] = False  # 마지막 5개 PAD

        out = mod(x, mask)
        assert out.shape == (2, T, d), f"shape mismatch: {out.shape}"
        # PAD 위치 제로 확인
        assert out[0, T-1].abs().sum().item() == 0.0, "PAD 위치가 0이 아님"
        print(f"  T={T:>4d}: OK")

    # backward
    x = torch.randn(2, 64, d, requires_grad=True)
    out = mod(x)
    out.sum().backward()
    assert x.grad is not None
    print("\nBackward OK")

    print("\n모든 테스트 통과!")
