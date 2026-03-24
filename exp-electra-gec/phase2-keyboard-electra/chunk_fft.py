"""ChunkFFT — 청크 단위 FFT로 전역 "분위기" 벡터 생성

시퀀스를 chunk_size(256) 단위로 분할 → rfft → 주파수 축 projection
→ mood vectors끼리 self-attention → 각 토큰에 해당 청크의 mood 주입.

16개 mood vector(n_chunks=4096/256)의 self-attention은 사실상 무료 (~125M FLOPs).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from model.encoder import RMSNorm


class ChunkFFT(nn.Module):
    """청크 분할 → rfft → 주파수 축 projection → mood attention → broadcast

    1. 입력을 chunk_size 단위로 분할 (부족분 zero-pad)
    2. 각 청크에 rfft (시퀀스 축)
    3. 주파수 축 learned projection → 청크당 1개 mood vector
    4. mood vectors끼리 small self-attention
    5. broadcast: 각 토큰에 해당 청크의 mood 더함
    """

    def __init__(self, d_model: int, chunk_size: int = 256, n_mood_heads: int = 2,
                 eps: float = 1e-6):
        super().__init__()
        self.chunk_size = chunk_size
        freq_bins = chunk_size // 2 + 1  # rfft 출력 크기

        # 주파수 축 projection: (freq_bins) → (1) per feature dim
        self.freq_proj = nn.Linear(freq_bins, 1, bias=False)
        self.mood_norm = RMSNorm(d_model, eps=eps)

        # mood vectors끼리 self-attention (n_chunks=16개, 사실상 무료)
        self.mood_attn = nn.MultiheadAttention(
            d_model, num_heads=n_mood_heads, batch_first=True, dropout=0.0,
        )
        self.out_norm = RMSNorm(d_model, eps=eps)

    def forward(self, x: Tensor, pad_mask: Tensor | None = None) -> Tensor:
        """
        Args:
            x: (B, T, D) — conv1d 전처리 후 입력
            pad_mask: (B, T) bool — True=유효
        Returns:
            (B, T, D) — x + mood broadcast
        """
        B, T, D = x.shape
        cs = self.chunk_size

        # chunk_size 배수로 zero-padding
        n_chunks = (T + cs - 1) // cs
        pad_len = n_chunks * cs - T
        if pad_len > 0:
            x_padded = F.pad(x, (0, 0, 0, pad_len))
        else:
            x_padded = x

        # 청크 분할
        chunks = x_padded.reshape(B, n_chunks, cs, D)  # (B, C, cs, D)

        # rfft (시퀀스 축) — float32로 수행
        freq = torch.fft.rfft(chunks.float(), dim=2)   # (B, C, freq_bins, D) complex
        freq_real = freq.real                            # 실수부만 사용

        # 주파수 축 projection → mood vector
        # (B, C, D, freq_bins) → Linear → (B, C, D, 1) → (B, C, D)
        mood = self.freq_proj(freq_real.permute(0, 1, 3, 2)).squeeze(-1)
        mood = mood.to(x.dtype)
        mood = self.mood_norm(mood)                      # (B, C, D)

        # mood self-attention
        mood = mood + self.mood_attn(mood, mood, mood, need_weights=False)[0]
        mood = self.out_norm(mood)                       # (B, C, D)

        # broadcast: 각 토큰에 해당 청크의 mood 주입
        chunk_idx = torch.arange(T, device=x.device) // cs  # (T,)
        out = x + mood[:, chunk_idx, :]

        if pad_mask is not None:
            out = out * pad_mask.unsqueeze(-1).to(out.dtype)

        return out


if __name__ == "__main__":
    print("=== ChunkFFT Smoke Test ===\n")

    d = 256
    mod = ChunkFFT(d, chunk_size=256, n_mood_heads=2)
    params = sum(p.numel() for p in mod.parameters())
    print(f"파라미터: {params:,}")
    for name, p in mod.named_parameters():
        print(f"  {name}: {p.shape} ({p.numel():,})")

    for T in [64, 256, 512, 1000, 4096]:
        x = torch.randn(2, T, d)
        mask = torch.ones(2, T, dtype=torch.bool)
        mask[0, -3:] = False

        out = mod(x, mask)
        assert out.shape == (2, T, d), f"T={T}: shape {out.shape}"
        assert out[0, -1].abs().sum().item() == 0.0, f"T={T}: PAD not zero"
        n_chunks = (T + 255) // 256
        print(f"  T={T:>4d} → {n_chunks} chunks: OK")

    # backward
    x = torch.randn(2, 512, d, requires_grad=True)
    out = mod(x)
    out.sum().backward()
    assert x.grad is not None
    print("\nBackward OK")

    # 768 차원 (128M 모델)
    mod768 = ChunkFFT(768, chunk_size=256, n_mood_heads=2)
    x768 = torch.randn(1, 4096, 768)
    out768 = mod768(x768)
    assert out768.shape == (1, 4096, 768)
    print(f"\nd=768, T=4096: OK (params={sum(p.numel() for p in mod768.parameters()):,})")

    print("\n모든 테스트 통과!")
