"""CrossAttentionCodec — BLT식 Local Encoder/Decoder

Conv 대신 cross-attention으로 bytes/자모 ↔ patches 매핑.
BLT의 Local Model 구조를 차용하되 고정 stride 사용.

Encoder: token embedding → local transformer → cross-attn (tokens → patches)
Decoder: cross-attn (patches → tokens) → local transformer → vocab logits
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x):
        rms = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * rms).to(x.dtype) * self.weight


class LocalTransformerLayer(nn.Module):
    """경량 로컬 트랜스포머 레이어 (self-attention + FFN)"""

    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True,
        )
        self.norm2 = RMSNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x, key_padding_mask=None):
        h = self.norm1(x)
        h, _ = self.self_attn(h, h, h, key_padding_mask=key_padding_mask)
        x = x + h
        x = x + self.ffn(self.norm2(x))
        return x


class CrossAttentionLayer(nn.Module):
    """Cross-attention: query가 key/value를 참조"""

    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.norm_q = RMSNorm(d_model)
        self.norm_kv = RMSNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True,
        )
        self.norm2 = RMSNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, query, kv, kv_padding_mask=None):
        q = self.norm_q(query)
        k = v = self.norm_kv(kv)
        h, _ = self.cross_attn(q, k, v, key_padding_mask=kv_padding_mask)
        query = query + h
        query = query + self.ffn(self.norm2(query))
        return query


class CrossAttentionCodec(nn.Module):
    """BLT식 Cross-Attention 기반 codec

    Args:
        vocab_size: 입력 vocab 크기
        d_model: hidden dimension
        stride: 압축 비율
        n_local_layers: local transformer 레이어 수
        n_heads: attention heads
        dropout: dropout 비율
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        stride: int = 4,
        n_local_layers: int = 2,
        n_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.stride = stride

        # ── 공유 ──
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.embed_scale = math.sqrt(d_model)

        # ── Encoder ──
        # Local transformer: 토큰 수준 문맥 인코딩
        self.enc_local = nn.ModuleList(
            [LocalTransformerLayer(d_model, n_heads, dropout)
             for _ in range(n_local_layers)]
        )
        # Patch queries: 학습 가능한 패치 위치 임베딩
        # forward에서 동적 생성 (길이 가변 대응)
        self.patch_query_proj = nn.Linear(d_model, d_model)
        # Cross-attention: tokens → patches
        self.enc_cross = CrossAttentionLayer(d_model, n_heads, dropout)
        self.enc_norm = RMSNorm(d_model)

        # ── Decoder ──
        # Cross-attention: patches → tokens
        self.dec_cross = CrossAttentionLayer(d_model, n_heads, dropout)
        # Local transformer: 토큰 수준 복원
        self.dec_local = nn.ModuleList(
            [LocalTransformerLayer(d_model, n_heads, dropout)
             for _ in range(n_local_layers)]
        )
        self.dec_norm = RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

        # 위치 인코딩 (sinusoidal)
        self._pos_cache = {}

    def _get_pos_encoding(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Sinusoidal positional encoding"""
        if seq_len in self._pos_cache:
            cached = self._pos_cache[seq_len]
            if cached.device == device:
                return cached

        pos = torch.arange(seq_len, device=device).unsqueeze(1).float()
        dim = torch.arange(0, self.d_model, 2, device=device).float()
        div = torch.exp(dim * (-math.log(10000.0) / self.d_model))

        pe = torch.zeros(seq_len, self.d_model, device=device)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)

        self._pos_cache[seq_len] = pe
        return pe

    def _pad_to_stride(self, ids: torch.Tensor) -> torch.Tensor:
        B, L = ids.shape
        remainder = L % self.stride
        if remainder != 0:
            pad_len = self.stride - remainder
            ids = F.pad(ids, (0, pad_len), value=0)
        return ids

    def _make_patch_queries(self, n_patches: int, batch_size: int,
                            token_hidden: torch.Tensor) -> torch.Tensor:
        """패치 쿼리 생성: 토큰 hidden의 stride 간격 평균으로 초기화"""
        B, L, D = token_hidden.shape
        # stride 간격으로 토큰 그룹의 평균을 패치 쿼리 초기값으로
        grouped = token_hidden.reshape(B, n_patches, self.stride, D)
        queries = grouped.mean(dim=2)  # [B, n_patches, D]
        queries = self.patch_query_proj(queries)
        # 위치 인코딩 추가
        queries = queries + self._get_pos_encoding(n_patches, queries.device)
        return queries

    def encode(self, ids: torch.Tensor) -> torch.Tensor:
        """[B, L] → [B, L//stride, d_model]"""
        ids = self._pad_to_stride(ids)
        B, L = ids.shape
        n_patches = L // self.stride

        # 토큰 임베딩 + 위치 인코딩
        x = self.embedding(ids) * self.embed_scale
        x = x + self._get_pos_encoding(L, x.device)

        # 패딩 마스크 (cross-attention용)
        pad_mask = (ids == 0)  # True = 무시

        # Local transformer
        for layer in self.enc_local:
            x = layer(x, key_padding_mask=pad_mask)

        # 패치 쿼리 생성
        patch_queries = self._make_patch_queries(n_patches, B, x)

        # Cross-attention: tokens → patches
        z = self.enc_cross(patch_queries, x, kv_padding_mask=pad_mask)
        z = self.enc_norm(z)
        return z

    def decode(self, z: torch.Tensor, target_len: int = None) -> torch.Tensor:
        """[B, L//stride, d_model] → [B, L, vocab_size] logits"""
        B, n_patches, D = z.shape
        L = n_patches * self.stride if target_len is None else target_len

        # 토큰 쿼리: 패치를 stride만큼 반복하여 초기값
        token_queries = z.unsqueeze(2).expand(-1, -1, self.stride, -1)
        token_queries = token_queries.reshape(B, n_patches * self.stride, D)
        if token_queries.size(1) > L:
            token_queries = token_queries[:, :L, :]
        elif token_queries.size(1) < L:
            token_queries = F.pad(token_queries, (0, 0, 0, L - token_queries.size(1)))

        # 위치 인코딩 추가
        token_queries = token_queries + self._get_pos_encoding(L, token_queries.device)

        # Cross-attention: patches → tokens
        x = self.dec_cross(token_queries, z)

        # Local transformer
        for layer in self.dec_local:
            x = layer(x)

        x = self.dec_norm(x)
        return self.head(x)

    def forward(self, ids: torch.Tensor, pad_mask: torch.Tensor = None) -> dict:
        """학습용 forward: encode → decode → reconstruction loss"""
        original_len = ids.size(1)
        ids_padded = self._pad_to_stride(ids)
        padded_len = ids_padded.size(1)

        z = self.encode(ids_padded)
        logits = self.decode(z, target_len=padded_len)

        logits = logits[:, :original_len, :]

        loss = F.cross_entropy(
            logits.reshape(-1, self.vocab_size),
            ids[:, :original_len].reshape(-1),
            ignore_index=0,
            reduction="mean",
        )

        return {"logits": logits, "loss": loss, "z": z}

    def reconstruct(self, ids: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            out = self.forward(ids)
            return out["logits"].argmax(dim=-1)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    print("=== CrossAttentionCodec Smoke Test ===\n")

    for stride in [4, 8, 16]:
        codec = CrossAttentionCodec(
            vocab_size=330, d_model=256, stride=stride,
            n_local_layers=2, n_heads=4,
        )
        n_params = count_params(codec)
        print(f"stride={stride}: {n_params/1e6:.2f}M params")

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

        loss.backward()
        grad_ok = all(
            p.grad is not None and not p.grad.isnan().any()
            for p in codec.parameters() if p.requires_grad
        )
        print(f"  backward: {'OK' if grad_ok else 'FAIL'}")
        print()

    # Conv 대비 파라미터 비교
    from codec.conv_codec import ConvCodec
    for stride in [4, 8, 16]:
        conv = ConvCodec(vocab_size=330, d_model=256, stride=stride, n_layers=3)
        xattn = CrossAttentionCodec(vocab_size=330, d_model=256, stride=stride, n_local_layers=2)
        print(f"stride={stride}: Conv {count_params(conv)/1e6:.2f}M vs XAttn {count_params(xattn)/1e6:.2f}M")

    print("\n모든 테스트 통과!")
