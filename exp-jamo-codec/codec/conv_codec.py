"""ConvCodec — Conv1d 기반 고정 stride 시퀀스 압축/복원 codec

Encoder: Embedding → Conv1d stack → stride로 다운샘플
Decoder: ConvTranspose1d로 업샘플 → Conv1d stack → vocab logits
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Conv1d + LayerNorm + GELU + Dropout"""

    def __init__(self, d_model: int, kernel_size: int = 5, dropout: float = 0.1):
        super().__init__()
        self.conv = nn.Conv1d(
            d_model, d_model,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False,
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """x: [B, L, D] → [B, L, D]"""
        residual = x
        # Conv1d는 [B, D, L] 형태
        x = self.conv(x.transpose(1, 2)).transpose(1, 2)
        x = self.norm(x)
        x = F.gelu(x)
        x = self.dropout(x)
        return x + residual


class ConvCodec(nn.Module):
    """Conv1d 기반 고정 stride 시퀀스 압축/복원 codec

    Args:
        vocab_size: 입력 vocab 크기
        d_model: hidden dimension
        stride: 압축 비율 (2, 4, 8)
        n_layers: encoder/decoder conv 레이어 수
        kernel_size: conv 커널 크기
        dropout: dropout 비율
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        stride: int = 3,
        n_layers: int = 3,
        kernel_size: int = 5,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.stride = stride

        # ── Encoder ──
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.embed_scale = math.sqrt(d_model)
        self.embed_dropout = nn.Dropout(dropout)

        self.enc_layers = nn.ModuleList(
            [ConvBlock(d_model, kernel_size, dropout) for _ in range(n_layers)]
        )

        # stride 다운샘플: Conv1d with stride
        self.downsample = nn.Conv1d(
            d_model, d_model,
            kernel_size=stride * 2 - 1,
            stride=stride,
            padding=stride - 1,
            bias=False,
        )
        self.down_norm = nn.LayerNorm(d_model)

        # ── Decoder ──
        # stride 업샘플: ConvTranspose1d
        self.upsample = nn.ConvTranspose1d(
            d_model, d_model,
            kernel_size=stride * 2,
            stride=stride,
            padding=stride // 2,
            bias=False,
        )
        self.up_norm = nn.LayerNorm(d_model)

        self.dec_layers = nn.ModuleList(
            [ConvBlock(d_model, kernel_size, dropout) for _ in range(n_layers)]
        )

        # 출력 head
        self.head = nn.Linear(d_model, vocab_size)

    def _pad_to_stride(self, ids: torch.Tensor) -> torch.Tensor:
        """시퀀스 길이를 stride 배수로 패딩"""
        B, L = ids.shape
        remainder = L % self.stride
        if remainder != 0:
            pad_len = self.stride - remainder
            ids = F.pad(ids, (0, pad_len), value=0)
        return ids

    def encode(self, ids: torch.Tensor) -> torch.Tensor:
        """[B, L] → [B, L//stride, d_model]"""
        ids = self._pad_to_stride(ids)
        x = self.embed_dropout(self.embedding(ids) * self.embed_scale)

        for layer in self.enc_layers:
            x = layer(x)

        # [B, L, D] → [B, D, L] → Conv1d(stride) → [B, D, L//s] → [B, L//s, D]
        z = self.downsample(x.transpose(1, 2)).transpose(1, 2)
        z = self.down_norm(z)
        return z

    def decode(self, z: torch.Tensor, target_len: int = None) -> torch.Tensor:
        """[B, L//stride, d_model] → [B, L, vocab_size] logits"""
        # [B, L//s, D] → [B, D, L//s] → ConvTranspose1d → [B, D, ~L] → [B, ~L, D]
        x = self.upsample(z.transpose(1, 2)).transpose(1, 2)
        x = self.up_norm(x)

        # 업샘플 결과 길이 조정
        if target_len is not None and x.size(1) != target_len:
            if x.size(1) > target_len:
                x = x[:, :target_len, :]
            else:
                x = F.pad(x, (0, 0, 0, target_len - x.size(1)))

        for layer in self.dec_layers:
            x = layer(x)

        return self.head(x)

    def forward(
        self, ids: torch.Tensor, pad_mask: torch.Tensor = None,
    ) -> dict:
        """학습용 forward: encode → decode → reconstruction loss

        Args:
            ids: [B, L] 입력 토큰 ID
            pad_mask: [B, L] True=유효, False=패딩

        Returns:
            dict with 'logits', 'loss', 'z'
        """
        original_len = ids.size(1)
        ids_padded = self._pad_to_stride(ids)
        padded_len = ids_padded.size(1)

        z = self.encode(ids_padded)
        logits = self.decode(z, target_len=padded_len)

        # 원래 길이만큼만 loss 계산
        logits = logits[:, :original_len, :]

        # reconstruction loss (pad 위치 무시)
        loss = F.cross_entropy(
            logits.reshape(-1, self.vocab_size),
            ids[:, :original_len].reshape(-1),
            ignore_index=0,  # PAD
            reduction="mean",
        )

        return {"logits": logits, "loss": loss, "z": z}

    def reconstruct(self, ids: torch.Tensor) -> torch.Tensor:
        """입력 → 복원 토큰 ID (추론용)"""
        with torch.no_grad():
            out = self.forward(ids)
            return out["logits"].argmax(dim=-1)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    print("=== ConvCodec Smoke Test ===\n")

    for stride in [2, 4, 8]:
        codec = ConvCodec(vocab_size=330, d_model=256, stride=stride, n_layers=3)
        n_params = count_params(codec)
        print(f"stride={stride}: {n_params/1e6:.2f}M params")

        # forward test
        B, L = 2, 128
        ids = torch.randint(1, 330, (B, L))
        pad_mask = torch.ones(B, L, dtype=torch.bool)
        pad_mask[:, -10:] = False

        out = codec(ids, pad_mask)
        z = out["z"]
        logits = out["logits"]
        loss = out["loss"]

        print(f"  input:  {ids.shape}")
        print(f"  z:      {z.shape} (압축 {L}→{z.size(1)}, {stride}x)")
        print(f"  logits: {logits.shape}")
        print(f"  loss:   {loss.item():.4f}")

        # backward test
        loss.backward()
        grad_ok = all(
            p.grad is not None and not p.grad.isnan().any()
            for p in codec.parameters() if p.requires_grad
        )
        print(f"  backward: {'OK' if grad_ok else 'FAIL'}")

        # reconstruct test
        recon = codec.reconstruct(ids)
        print(f"  recon:  {recon.shape}")
        print()

    print("모든 테스트 통과!")
