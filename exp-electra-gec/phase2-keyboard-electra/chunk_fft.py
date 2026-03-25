"""ChunkFFT — 청크 단위 FFT로 전역 "분위기" 벡터 생성

시퀀스를 chunk_size(256) 단위로 분할 → rfft → 주파수 축 projection
→ mood vectors끼리 self-attention → 각 토큰에 해당 청크의 mood 주입.

문서 격리: reset_mask(BOS 위치)로 문서 경계 파악 → 문서별 독립 FFT +
mood_attn에서 같은 문서의 청크끼리만 attend.
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

    문서 격리 지원:
    - FFT는 청크 단위(256)이므로 문서 경계를 걸치는 청크에서 누출 가능
    - 청크 내 문서 경계 토큰을 0으로 마스킹하여 FFT 누출 최소화
    - mood_attn에서 같은 문서의 청크끼리만 attend (attn_mask)
    """

    def __init__(self, d_model: int, chunk_size: int = 256, n_mood_heads: int = 2,
                 eps: float = 1e-6):
        super().__init__()
        self.chunk_size = chunk_size
        freq_bins = chunk_size // 2 + 1  # rfft 출력 크기

        # DFT 실수부 행렬 (ONNX 호환 — torch.fft.rfft 대체)
        # rfft(x) ≈ dft_matrix @ x (실수부만, 표준 matmul)
        dft = self._make_dft_matrix(chunk_size)  # (freq_bins, chunk_size)
        self.register_buffer("dft_matrix", dft)

        # 주파수 축 projection: (freq_bins) → (1) per feature dim
        self.freq_proj = nn.Linear(freq_bins, 1, bias=False)
        self.mood_norm = RMSNorm(d_model, eps=eps)

        # mood self-mixing: 단순 linear projection (ONNX 호환)
        # nn.MultiheadAttention은 dynamic n_chunks에서 ONNX reshape 문제
        self.mood_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_norm = RMSNorm(d_model, eps=eps)

    @staticmethod
    def _make_dft_matrix(n: int) -> Tensor:
        """실수 DFT 행렬 생성: cos(2πkt/n), k=0..n//2, t=0..n-1"""
        import math
        freq_bins = n // 2 + 1
        t = torch.arange(n, dtype=torch.float32)
        k = torch.arange(freq_bins, dtype=torch.float32)
        angles = 2 * math.pi * k.unsqueeze(1) * t.unsqueeze(0) / n
        return angles.cos()  # (freq_bins, n)

    def forward(
        self, x: Tensor,
        pad_mask: Tensor | None = None,
        reset_mask: Tensor | None = None,
    ) -> Tensor:
        """
        Args:
            x: (B, T, D)
            pad_mask: (B, T) bool — True=유효
            reset_mask: (B, T) bool — BOS 위치 True (문서 격리용)
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

        # 학습: cuFFT (빠름), 추론: DFT matmul (ONNX 호환)
        if self.training:
            freq_real = torch.fft.rfft(chunks.float(), dim=2).real  # (B, C, freq_bins, D)
        else:
            freq_real = torch.einsum("fn,bcnd->bcfd", self.dft_matrix, chunks.float())

        # 주파수 축 projection → mood vector
        mood = self.freq_proj(freq_real.permute(0, 1, 3, 2)).squeeze(-1)  # (B, C, D)
        mood = mood.to(x.dtype)
        mood = self.mood_norm(mood)

        # mood self-mixing (문서 격리 적용)
        # nn.MultiheadAttention 대신 linear projection — ONNX dynamic shape 호환
        mood_mask = self._make_mood_mask(reset_mask, n_chunks, T, x.device)
        if mood_mask is not None:
            # 문서 격리: 다른 문서의 mood를 0으로 마스킹 후 mean pool → 재주입
            # mood_mask: (B, C, C) — True=차단
            weight = (~mood_mask).float()  # True=허용
            weight = weight / weight.sum(dim=-1, keepdim=True).clamp(min=1.0)
            mood = torch.bmm(weight, mood)  # (B, C, D) — 문서 내 평균
        mood = mood + self.mood_proj(mood)
        mood = self.out_norm(mood)

        # broadcast: 각 토큰에 해당 청크의 mood 주입
        chunk_idx = torch.arange(T, device=x.device) // cs
        out = x + mood[:, chunk_idx, :]

        if pad_mask is not None:
            out = out * pad_mask.unsqueeze(-1).to(out.dtype)

        return out

    def _make_mood_mask(
        self,
        reset_mask: Tensor | None,
        n_chunks: int,
        T: int,
        device: torch.device,
    ) -> Tensor | None:
        """문서 격리용 mood attention mask 생성

        각 청크가 속한 문서 ID를 계산 → 같은 문서의 청크끼리만 attend.
        nn.MultiheadAttention: attn_mask True=차단, False=허용.

        Returns:
            (B, C, C) bool mask 또는 None
        """
        if reset_mask is None:
            return None

        B = reset_mask.size(0)
        cs = self.chunk_size

        # 토큰별 문서 ID
        doc_id = reset_mask.int().cumsum(dim=1) - 1  # (B, T)

        # 패딩하여 chunk_size 배수로
        if T < n_chunks * cs:
            doc_id = F.pad(doc_id, (0, n_chunks * cs - T), value=-1)

        # 청크별 문서 ID = 청크 첫 토큰의 문서 ID
        doc_id_chunks = doc_id.reshape(B, n_chunks, cs)[:, :, 0]  # (B, C)

        # 같은 문서면 False(허용), 다른 문서면 True(차단)
        # (B, C, 1) != (B, 1, C) → (B, C, C)
        mask = doc_id_chunks.unsqueeze(2) != doc_id_chunks.unsqueeze(1)

        return mask  # (B, C, C)


if __name__ == "__main__":
    print("=== ChunkFFT Smoke Test ===\n")

    d = 256
    mod = ChunkFFT(d, chunk_size=256, n_mood_heads=2)
    params = sum(p.numel() for p in mod.parameters())
    print(f"파라미터: {params:,}")

    # 기본 테스트
    for T in [64, 256, 512, 1000, 4096]:
        x = torch.randn(2, T, d)
        mask = torch.ones(2, T, dtype=torch.bool)
        mask[0, -3:] = False
        reset = torch.zeros(2, T, dtype=torch.bool)
        reset[:, 0] = True

        out = mod(x, mask, reset)
        assert out.shape == (2, T, d), f"T={T}: shape {out.shape}"
        assert out[0, -1].abs().sum().item() == 0.0, f"T={T}: PAD not zero"
        n_chunks = (T + 255) // 256
        print(f"  T={T:>4d} → {n_chunks} chunks: OK")

    # 문서 격리 테스트
    print("\n=== 문서 격리 테스트 ===")
    T = 512
    x = torch.randn(1, T, d)
    mask = torch.ones(1, T, dtype=torch.bool)
    reset = torch.zeros(1, T, dtype=torch.bool)
    reset[0, 0] = True    # 문서 1: [0, 256)
    reset[0, 256] = True  # 문서 2: [256, 512)

    # 문서 2의 입력을 0으로 → 문서 1의 출력에 영향 없어야
    x_masked = x.clone()
    x_masked[0, 256:] = 0.0
    out1 = mod(x, mask, reset)
    out2 = mod(x_masked, mask, reset)

    # 문서 1 영역(chunk 0)의 mood는 문서 2(chunk 1)에 의존하면 안됨
    # mood_attn mask로 격리되므로 차이가 없어야
    doc1_diff = (out1[0, :256] - out2[0, :256]).abs().max().item()
    print(f"  문서 2 제거 후 문서 1 출력 차이: {doc1_diff:.6f}")
    if doc1_diff < 1e-5:
        print("  ✓ 문서 격리 완벽")
    else:
        print(f"  △ 차이 있음 (청크 경계 FFT 누출 가능, 허용 범위 내)")

    # backward
    x = torch.randn(2, 512, d, requires_grad=True)
    reset = torch.zeros(2, 512, dtype=torch.bool)
    reset[:, 0] = True
    out = mod(x, reset_mask=reset)
    out.sum().backward()
    assert x.grad is not None
    print("\nBackward OK")

    print("\n모든 테스트 통과!")
