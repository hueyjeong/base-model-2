"""ConvCodec — Conv1d 기반 고정 stride 시퀀스 압축/복원 codec

Encoder: Embedding → Conv1d stack → stride로 다운샘플
Decoder: ConvTranspose1d로 업샘플 → Conv1d stack → vocab logits
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class HashNgramEmbedding(nn.Module):
    """Hash n-gram embedding — BLT 방식

    각 위치에서 n-gram(3~8)의 rolling polynomial hash → embedding table lookup → 합산.
    base embedding에 더해서 로컬 문맥 정보를 입력 단계에서 주입.
    """

    def __init__(
        self,
        d_model: int,
        ngram_sizes: tuple = (3, 4, 5, 6, 7, 8),
        table_size: int = 10000,
        embed_dim: int = 64,
    ):
        super().__init__()
        self.ngram_sizes = ngram_sizes
        self.table_size = table_size
        # n-gram 크기별 embedding table (작은 차원)
        self.tables = nn.ModuleList([
            nn.Embedding(table_size, embed_dim, padding_idx=0)
            for _ in ngram_sizes
        ])
        # 작은 차원 → d_model로 projection
        self.proj = nn.Linear(embed_dim, d_model, bias=False)
        # 초기화: 작은 값으로 (base embedding 대비 보조적 역할)
        for table in self.tables:
            nn.init.normal_(table.weight, std=0.02)
            table.weight.data[0].zero_()

    def _rolling_hash(self, ids: torch.Tensor, n: int) -> torch.Tensor:
        """Rolling polynomial hash로 n-gram 인덱스 계산

        Args:
            ids: [B, L] 토큰 ID (long)
            n: n-gram 크기

        Returns:
            [B, L] hash 인덱스 (0 = 길이 부족)
        """
        B, L = ids.shape
        if L < n:
            return torch.zeros(B, L, dtype=torch.long, device=ids.device)

        # polynomial hash: h = (id[0]*P^(n-1) + id[1]*P^(n-2) + ... + id[n-1]) % table_size
        P = 31
        MOD = self.table_size

        # powers[i] = P^i % MOD
        powers = torch.ones(n, dtype=torch.long, device=ids.device)
        for i in range(1, n):
            powers[i] = (powers[i - 1] * P) % MOD

        powers = powers.flip(0)  # [P^(n-1), P^(n-2), ..., P^0]

        # ids를 n-gram 윈도우로 unfold: [B, L-n+1, n]
        ids_long = ids.long()
        windows = ids_long.unfold(1, n, 1)  # [B, L-n+1, n]

        # hash 계산: 각 윈도우의 weighted sum
        hashes = (windows * powers.unsqueeze(0).unsqueeze(0)) % MOD
        hashes = hashes.sum(dim=-1) % MOD  # [B, L-n+1]
        hashes = hashes.clamp(min=1)  # 0은 padding_idx이므로 회피

        # 앞쪽 n-1 위치는 n-gram 불완전 → 0 (padding)
        pad = torch.zeros(B, n - 1, dtype=torch.long, device=ids.device)
        result = torch.cat([pad, hashes], dim=1)  # [B, L]
        return result

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """ids: [B, L] → n-gram embedding sum: [B, L, d_model]"""
        result = None
        for table, n in zip(self.tables, self.ngram_sizes):
            indices = self._rolling_hash(ids, n)  # [B, L]
            emb = table(indices)  # [B, L, embed_dim]
            if result is None:
                result = emb
            else:
                result = result + emb
        return self.proj(result)  # [B, L, d_model]


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
        use_hash_ngram: bool = False,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.stride = stride

        # ── Encoder ──
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.embed_scale = math.sqrt(d_model)
        self.embed_dropout = nn.Dropout(dropout)

        # Hash n-gram embeddings (BLT 방식)
        self.use_hash_ngram = use_hash_ngram
        if use_hash_ngram:
            self.hash_ngram = HashNgramEmbedding(d_model=d_model)

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

        if self.use_hash_ngram:
            x = x + self.hash_ngram(ids)

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

    # Hash n-gram 테스트
    print("--- Hash N-gram 테스트 ---\n")
    for n_layers in [1, 2, 3]:
        codec = ConvCodec(
            vocab_size=263, d_model=256, stride=16,
            n_layers=n_layers, use_hash_ngram=True,
        )
        n_params = count_params(codec)
        B, L = 2, 512
        ids = torch.randint(1, 263, (B, L))
        out = codec(ids)
        loss = out["loss"]
        loss.backward()
        grad_ok = all(
            p.grad is not None and not p.grad.isnan().any()
            for p in codec.parameters() if p.requires_grad
        )
        print(f"Conv {n_layers}L + hash_ngram: {n_params/1e6:.2f}M, "
              f"loss={loss.item():.4f}, backward={'OK' if grad_ok else 'FAIL'}")

    print("\n모든 테스트 통과!")
