"""CompositionCodec — BBPE 경계 + Conv 자모 composition

BBPE 토크나이저가 잘라준 토큰 경계 내에서,
자모 시퀀스를 Conv로 처리하여 토큰 벡터를 생성/복원.
임베딩 테이블 없이 Conv composition으로 표현 생성.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from codec.conv_codec import ConvBlock


class CompositionEncoder(nn.Module):
    """자모 시퀀스 → 토큰 벡터 (Conv + Global Avg Pool)

    각 BBPE 토큰의 자모 시퀀스를 Conv로 처리 → 하나의 벡터로 압축.

    Args:
        jamo_vocab: 자모 vocab 크기 (330)
        d_model: hidden dim (256)
        n_layers: Conv 레이어 수 (3~5)
        kernel_size: Conv 커널 크기 (7)
        dropout: dropout
    """

    def __init__(
        self,
        jamo_vocab: int = 330,
        d_model: int = 256,
        n_layers: int = 5,
        kernel_size: int = 7,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(jamo_vocab, d_model, padding_idx=0)
        self.embed_scale = math.sqrt(d_model)
        self.layers = nn.ModuleList([
            ConvBlock(d_model, kernel_size, dropout) for _ in range(n_layers)
        ])

    def forward(self, jamo_ids: torch.Tensor, jamo_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            jamo_ids: [B, n_tokens, max_jamo_len] 자모 ID
            jamo_mask: [B, n_tokens, max_jamo_len] 유효 위치 (True=유효)

        Returns:
            token_vecs: [B, n_tokens, d_model] 토큰별 벡터
        """
        B, T, J = jamo_ids.shape

        # [B*T, J]로 reshape하여 Conv 처리
        flat_ids = jamo_ids.reshape(B * T, J)
        flat_mask = jamo_mask.reshape(B * T, J)

        # 임베딩
        x = self.embedding(flat_ids) * self.embed_scale  # [B*T, J, D]

        # 패딩 위치 0으로 마스킹
        x = x * flat_mask.unsqueeze(-1).float()

        # Conv 레이어
        for layer in self.layers:
            x = layer(x)  # [B*T, J, D]
            x = x * flat_mask.unsqueeze(-1).float()  # 패딩 재마스킹

        # Global Average Pooling (유효 자모만)
        mask_sum = flat_mask.float().sum(dim=1, keepdim=True).clamp(min=1)  # [B*T, 1]
        x = (x * flat_mask.unsqueeze(-1).float()).sum(dim=1) / mask_sum  # [B*T, D]

        return x.reshape(B, T, self.d_model)


class CompositionDecoder(nn.Module):
    """토큰 벡터 → 자모 시퀀스 복원 (고정 길이 출력)

    Args:
        jamo_vocab: 자모 vocab 크기 (330)
        d_model: hidden dim (256)
        max_jamo_len: 최대 자모 길이 (32)
        n_layers: Conv 레이어 수
        kernel_size: Conv 커널 크기
        dropout: dropout
    """

    def __init__(
        self,
        jamo_vocab: int = 330,
        d_model: int = 256,
        max_jamo_len: int = 32,
        n_layers: int = 5,
        kernel_size: int = 7,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.max_jamo_len = max_jamo_len
        # 토큰 벡터 → 자모 위치별 벡터로 확장
        self.expand = nn.Linear(d_model, max_jamo_len * d_model)
        self.layers = nn.ModuleList([
            ConvBlock(d_model, kernel_size, dropout) for _ in range(n_layers)
        ])
        self.head = nn.Linear(d_model, jamo_vocab)

    def forward(self, token_vecs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            token_vecs: [B, n_tokens, d_model]

        Returns:
            logits: [B, n_tokens, max_jamo_len, jamo_vocab]
        """
        B, T, D = token_vecs.shape

        # [B*T, D] → [B*T, max_jamo_len, D]
        x = self.expand(token_vecs.reshape(B * T, D))  # [B*T, J*D]
        x = x.reshape(B * T, self.max_jamo_len, D)

        # Conv 레이어
        for layer in self.layers:
            x = layer(x)  # [B*T, J, D]

        # 자모 logits
        logits = self.head(x)  # [B*T, J, V]
        return logits.reshape(B, T, self.max_jamo_len, -1)


class CompositionCodec(nn.Module):
    """BBPE + Conv Composition Codec

    BBPE 토크나이저가 경계를 잡고, Conv가 자모에서 토큰 벡터를 composition/복원.

    Args:
        jamo_vocab: 자모 vocab (330)
        d_model: hidden dim (256)
        max_jamo_len: 최대 자모 길이 (32)
        n_layers: Conv 레이어 수 (3~5)
        kernel_size: Conv 커널 크기 (7)
        dropout: dropout
    """

    def __init__(
        self,
        jamo_vocab: int = 330,
        d_model: int = 256,
        max_jamo_len: int = 32,
        n_layers: int = 5,
        kernel_size: int = 7,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.jamo_vocab = jamo_vocab
        self.d_model = d_model
        self.max_jamo_len = max_jamo_len

        self.encoder = CompositionEncoder(
            jamo_vocab, d_model, n_layers, kernel_size, dropout,
        )
        self.decoder = CompositionDecoder(
            jamo_vocab, d_model, max_jamo_len, n_layers, kernel_size, dropout,
        )

    def forward(
        self,
        jamo_ids: torch.Tensor,
        jamo_mask: torch.Tensor,
        token_mask: torch.Tensor = None,
    ) -> dict:
        """학습용 forward: encode → decode → 복원 loss

        Args:
            jamo_ids: [B, T, J] 자모 ID
            jamo_mask: [B, T, J] 유효 자모 위치
            token_mask: [B, T] 유효 토큰 위치

        Returns:
            dict: logits, loss, z
        """
        # Encode
        z = self.encoder(jamo_ids, jamo_mask)  # [B, T, D]

        # Decode
        logits = self.decoder(z)  # [B, T, J, V]

        # Loss: 유효 토큰의 유효 자모 위치만
        B, T, J, V = logits.shape
        targets = jamo_ids  # [B, T, J]

        # [B*T*J, V] vs [B*T*J]
        flat_logits = logits.reshape(-1, V)
        flat_targets = targets.reshape(-1)

        # 유효 위치 마스크: 유효 토큰의 유효 자모
        if token_mask is not None:
            valid = (jamo_mask & token_mask.unsqueeze(-1)).reshape(-1)
        else:
            valid = jamo_mask.reshape(-1)

        loss = F.cross_entropy(
            flat_logits, flat_targets, ignore_index=0, reduction="none",
        )
        loss = loss[valid].mean()

        return {
            "logits": logits,
            "loss": loss,
            "z": z,
        }

    def encode(self, jamo_ids: torch.Tensor, jamo_mask: torch.Tensor) -> torch.Tensor:
        """인코딩만 (backbone 입력용)"""
        return self.encoder(jamo_ids, jamo_mask)

    def reconstruct(self, jamo_ids: torch.Tensor, jamo_mask: torch.Tensor) -> torch.Tensor:
        """복원 (평가용)"""
        with torch.no_grad():
            z = self.encoder(jamo_ids, jamo_mask)
            logits = self.decoder(z)
            return logits.argmax(dim=-1)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    print("=== CompositionCodec Smoke Test ===\n")

    for n_layers in [3, 4, 5]:
        codec = CompositionCodec(
            jamo_vocab=330, d_model=256, max_jamo_len=32,
            n_layers=n_layers, kernel_size=7,
        )
        n_params = count_params(codec)
        enc_params = count_params(codec.encoder)
        dec_params = count_params(codec.decoder)
        print(f"{n_layers}L: {n_params/1e6:.2f}M total "
              f"(enc {enc_params/1e6:.2f}M, dec {dec_params/1e6:.2f}M)")

        # forward test
        B, T, J = 2, 16, 32
        jamo_ids = torch.randint(1, 330, (B, T, J))
        jamo_mask = torch.ones(B, T, J, dtype=torch.bool)
        jamo_mask[:, :, 20:] = False  # 뒤쪽 패딩
        token_mask = torch.ones(B, T, dtype=torch.bool)
        token_mask[:, 12:] = False  # 뒤쪽 패딩

        out = codec(jamo_ids, jamo_mask, token_mask)
        print(f"  z: {out['z'].shape}, logits: {out['logits'].shape}, loss: {out['loss'].item():.4f}")

        # backward test
        out["loss"].backward()
        trainable = [p for p in codec.parameters() if p.grad is not None]
        grad_ok = all(not p.grad.isnan().any() for p in trainable)
        print(f"  backward: {'OK' if grad_ok else 'FAIL'}")
        print()

    print("모든 테스트 통과!")
