"""SimpleCodec — per-token encoder + parallel slot decoder.

설계 철학: 각 BBPE 토큰은 독립적으로 encode/decode. 시퀀스 개념 없음.
- 입력: [T, max_jamo] (T = 배치 내 총 토큰 수)
- 인코더: 각 토큰의 자모 → 1 벡터
- 디코더: 토큰 벡터 → max_jamo 슬롯 병렬 예측 (PAD 를 만나면 cut)

토큰 간 상호작용 0 (SegmentMaskedConv 같은 mask 불필요 — 배치 차원이 이미 독립).
메모리/코드 모두 단순. Decode-from-vec 자연스럽게 지원.

GPU 관점: T 가 그냥 batch 차원 → 모든 토큰 한 번에 병렬 처리.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Conv1d + LayerNorm + GELU + residual (segment mask 없음 — 필요 없음)."""

    def __init__(self, d_model: int, kernel_size: int = 5, dropout: float = 0.1):
        super().__init__()
        # even kernel 대응: pad=(k//2, (k-1)//2) 로 정확히 길이 보존
        self.kernel_size = kernel_size
        self.pad_left = kernel_size // 2
        self.pad_right = (kernel_size - 1) // 2
        self.conv = nn.Conv1d(
            d_model, d_model, kernel_size=kernel_size, padding=0, bias=False,
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [T, S, D] → [T, S, D]"""
        residual = x
        x_t = x.transpose(1, 2)  # [T, D, S]
        x_t = F.pad(x_t, (self.pad_left, self.pad_right))
        x_t = self.conv(x_t)  # [T, D, S]
        x = x_t.transpose(1, 2)  # [T, S, D]
        x = self.norm(x)
        x = F.gelu(x)
        x = self.dropout(x)
        return x + residual


class SimpleCodec(nn.Module):
    """Per-token codec.

    Args:
        jamo_vocab: 자모 vocab 크기 (기본 330)
        d_model: 히든 차원
        n_enc_layers / n_dec_layers: 인코더/디코더 Conv 레이어 수
        kernel_size: Conv 커널
        max_jamo: 토큰당 최대 자모 수 (패딩 + 슬롯 수)
        dropout: dropout 비율
    """

    def __init__(
        self,
        jamo_vocab: int = 330,
        d_model: int = 256,
        n_enc_layers: int = 5,
        n_dec_layers: int = 5,
        kernel_size: int = 5,
        max_jamo: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.jamo_vocab = jamo_vocab
        self.d_model = d_model
        self.max_jamo = max_jamo
        self.pad_id = 0

        # ── Encoder ──
        self.embedding = nn.Embedding(jamo_vocab, d_model, padding_idx=0)
        self.embed_scale = math.sqrt(d_model)
        self.enc_pos = nn.Embedding(max_jamo, d_model)
        self.enc_layers = nn.ModuleList([
            ConvBlock(d_model, kernel_size, dropout) for _ in range(n_enc_layers)
        ])
        self.enc_pool_proj = nn.Linear(d_model, d_model)

        # ── Decoder (parallel slot) ──
        self.dec_upsample = nn.Linear(d_model, d_model)
        self.dec_pos = nn.Embedding(max_jamo, d_model)
        self.dec_layers = nn.ModuleList([
            ConvBlock(d_model, kernel_size, dropout) for _ in range(n_dec_layers)
        ])
        self.head = nn.Linear(d_model, jamo_vocab)

        # 초기화
        nn.init.normal_(self.embedding.weight, std=1.0 / math.sqrt(d_model))
        self.embedding.weight.data[0].zero_()  # PAD
        nn.init.normal_(self.enc_pos.weight, std=0.02)
        nn.init.normal_(self.dec_pos.weight, std=0.02)

    # ─────────────────────────────────────────────
    #  Encoder: [T, max_jamo] → [T, D]
    # ─────────────────────────────────────────────
    def encode(self, jamo_ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """각 토큰을 1 벡터로 압축.

        Args:
            jamo_ids: [T, max_jamo] (PAD=0 로 padded)
            mask: [T, max_jamo] bool (유효 자모 위치)

        Returns:
            token_vecs: [T, D]
        """
        T, S = jamo_ids.shape
        x = self.embedding(jamo_ids) * self.embed_scale  # [T, S, D]

        pos_ids = torch.arange(S, device=x.device)
        x = x + self.enc_pos(pos_ids)  # [T, S, D]

        mask_f = mask.unsqueeze(-1).to(x.dtype)  # [T, S, 1]
        x = x * mask_f  # PAD 위치 0

        for layer in self.enc_layers:
            x = layer(x)
            x = x * mask_f  # 각 레이어 후 PAD 영역 재차단

        # Pool: 유효 자모 평균
        counts = mask.sum(dim=1, keepdim=True).clamp(min=1).to(x.dtype)  # [T, 1]
        pooled = (x * mask_f).sum(dim=1) / counts  # [T, D]
        return self.enc_pool_proj(pooled)

    # ─────────────────────────────────────────────
    #  Decoder: [T, D] → [T, max_jamo, V]
    # ─────────────────────────────────────────────
    def decode(self, token_vecs: torch.Tensor) -> torch.Tensor:
        """토큰 벡터 → max_jamo 슬롯 자모 logits.

        Args:
            token_vecs: [T, D]

        Returns:
            logits: [T, max_jamo, V]
        """
        T, D = token_vecs.shape
        S = self.max_jamo

        x = token_vecs.unsqueeze(1).expand(-1, S, -1).contiguous()  # [T, S, D]
        x = self.dec_upsample(x)

        pos_ids = torch.arange(S, device=x.device)
        x = x + self.dec_pos(pos_ids)  # [T, S, D]

        for layer in self.dec_layers:
            x = layer(x)

        return self.head(x)  # [T, S, V]

    # ─────────────────────────────────────────────
    #  Training forward
    # ─────────────────────────────────────────────
    def forward(self, jamo_ids: torch.Tensor, mask: torch.Tensor) -> dict:
        """학습용 forward.

        Args:
            jamo_ids: [T, max_jamo] (유효 자모 + PAD)
            mask: [T, max_jamo] bool

        Returns:
            dict: logits [T,S,V], loss scalar, z [T,D]
        """
        z = self.encode(jamo_ids, mask)  # [T, D]
        logits = self.decode(z)  # [T, S, V]

        # Target: 유효 위치 = jamo_ids, 나머지 = PAD(0)
        # jamo_ids 이미 PAD 로 채워져 있다면 그대로 사용
        target = jamo_ids.clone()
        target[~mask] = self.pad_id  # 안전하게 PAD 재지정

        # 전 슬롯 CE (PAD 위치 포함 → 모델이 PAD 예측 학습)
        loss = F.cross_entropy(
            logits.reshape(-1, self.jamo_vocab),
            target.reshape(-1),
        )

        # 정확도 (로깅용) — 전 슬롯 기준
        with torch.no_grad():
            pred = logits.argmax(dim=-1)  # [T, S]
            correct = (pred == target).float().mean()

        return {
            "logits": logits,
            "loss": loss,
            "z": z,
            "acc": correct,
        }

    # ─────────────────────────────────────────────
    #  Inference: token_vec → 자모 시퀀스 (PAD cut)
    # ─────────────────────────────────────────────
    @torch.no_grad()
    def decode_from_vec(self, token_vec: torch.Tensor) -> list:
        """token_vec → jamo_id 리스트 (PAD 만나면 cut).

        Args:
            token_vec: [D] (단일) 또는 [T, D] (배치)

        Returns:
            list[list[int]]: 각 토큰의 자모 ID 리스트
        """
        if token_vec.dim() == 1:
            token_vec = token_vec.unsqueeze(0)
        logits = self.decode(token_vec)  # [T, S, V]
        preds = logits.argmax(dim=-1)  # [T, S]

        results = []
        for t in range(preds.size(0)):
            seq = []
            for j in preds[t].tolist():
                if j == self.pad_id:
                    break
                seq.append(j)
            results.append(seq)
        return results


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    print("=== SimpleCodec Smoke Test ===\n")

    # 모델
    model = SimpleCodec(
        jamo_vocab=330, d_model=256,
        n_enc_layers=5, n_dec_layers=5,
        kernel_size=5, max_jamo=32,
    )
    print(f"params: {count_params(model)/1e6:.2f}M")

    # Dummy batch: T=100 토큰, max_jamo=32
    T = 100
    # 각 토큰의 실제 자모 수 3~20 랜덤
    lengths = torch.randint(3, 21, (T,))
    jamo_ids = torch.zeros(T, 32, dtype=torch.long)
    mask = torch.zeros(T, 32, dtype=torch.bool)
    for t in range(T):
        L = lengths[t].item()
        jamo_ids[t, :L] = torch.randint(1, 330, (L,))
        mask[t, :L] = True

    # Forward
    out = model(jamo_ids, mask)
    print(f"logits: {out['logits'].shape}  (기대 [{T},32,330])")
    print(f"loss: {out['loss'].item():.4f}")
    print(f"acc: {out['acc'].item()*100:.2f}%")
    print(f"z: {out['z'].shape}  (기대 [{T},256])")

    # Decode-from-vec
    z = out["z"][:3]
    results = model.decode_from_vec(z)
    print(f"\ndecode_from_vec (첫 3 토큰):")
    for i, seq in enumerate(results):
        print(f"  [{i}] len={len(seq)}: {seq[:10]}{'...' if len(seq) > 10 else ''}")

    print("\nOK — SimpleCodec forward + decode_from_vec 동작")
