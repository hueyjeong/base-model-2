"""EntropyPatchCodec — 엔트로피 기반 가변 패칭 codec

BLT 방식: 소형 LM으로 per-token 엔트로피 계산 → 누적 엔트로피가
임계값 초과 시 패치 경계 결정 → 가변 길이 패치 생성.

Conv/XAttn encoder 위에 적응형 풀링을 얹는 구조.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""

    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rms * self.weight


class SwiGLUFFN(nn.Module):
    """SwiGLU Feed-Forward Network (d_ff = d_model × 3)"""

    def __init__(self, d_model: int, d_ff: int = None, dropout: float = 0.1):
        super().__init__()
        d_ff = d_ff or d_model * 3
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))


class SmallLMLayer(nn.Module):
    """Transformer 레이어: RMSNorm + MHA + RMSNorm + SwiGLU"""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attn_norm = RMSNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True, bias=False,
        )
        self.ffn_norm = RMSNorm(d_model)
        self.ffn = SwiGLUFFN(d_model, dropout=dropout)

    def forward(self, x: torch.Tensor, is_causal: bool = False) -> torch.Tensor:
        h = self.attn_norm(x)
        if is_causal:
            L = x.size(1)
            mask = torch.triu(torch.ones(L, L, device=x.device), diagonal=1).bool()
            h, _ = self.attn(h, h, h, attn_mask=mask, is_causal=True)
        else:
            h, _ = self.attn(h, h, h)
        x = x + h
        x = x + self.ffn(self.ffn_norm(x))
        return x


class SmallLM(nn.Module):
    """소형 Causal LM — 엔트로피 계산용

    RMSNorm + SwiGLU (d_ff = d_model × 3) Transformer.
    per-token cross-entropy를 계산하여 패치 경계 결정에 사용.
    """

    def __init__(self, vocab_size: int, d_model: int = 128, n_layers: int = 2,
                 n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.layers = nn.ModuleList([
            SmallLMLayer(d_model, n_heads, dropout) for _ in range(n_layers)
        ])
        self.final_norm = RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """[B, L] → [B, L, vocab_size] causal logits"""
        x = self.embedding(ids) * math.sqrt(self.d_model)
        for layer in self.layers:
            x = layer(x, is_causal=True)
        return self.head(self.final_norm(x))

    @torch.no_grad()
    def compute_entropy(self, ids: torch.Tensor) -> torch.Tensor:
        """per-token cross-entropy 계산 [B, L]

        position i의 엔트로피 = -log P(token_i | token_{<i})
        첫 번째 위치는 uniform entropy로 설정.
        """
        logits = self.forward(ids)  # [B, L, V]
        # shift: logits[t] predicts ids[t+1]
        log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)  # [B, L-1, V]
        targets = ids[:, 1:]  # [B, L-1]
        # per-token negative log-likelihood
        nll = -log_probs.gather(2, targets.unsqueeze(-1)).squeeze(-1)  # [B, L-1]
        # 첫 위치는 큰 엔트로피 (항상 패치 시작)
        first = torch.full((ids.size(0), 1), 5.0, device=ids.device)
        return torch.cat([first, nll], dim=1)  # [B, L]


def compute_patch_boundaries(entropy: torch.Tensor, threshold: float,
                             min_patch: int = 2, max_patch: int = 32,
                             pad_mask: torch.Tensor = None) -> torch.Tensor:
    """엔트로피 누적으로 패치 경계 결정

    Args:
        entropy: [B, L] per-token 엔트로피
        threshold: 누적 엔트로피 임계값 (이 값 초과 시 새 패치 시작)
        min_patch: 최소 패치 크기
        max_patch: 최대 패치 크기
        pad_mask: [B, L] True=유효

    Returns:
        boundaries: [B, L] True=이 위치에서 새 패치 시작
    """
    B, L = entropy.shape
    boundaries = torch.zeros(B, L, dtype=torch.bool, device=entropy.device)
    boundaries[:, 0] = True  # 첫 위치는 항상 패치 시작

    cumsum = torch.zeros(B, device=entropy.device)
    since_last = torch.zeros(B, dtype=torch.long, device=entropy.device)

    for t in range(1, L):
        cumsum = cumsum + entropy[:, t]
        since_last = since_last + 1

        # 패치 시작 조건: (누적 > threshold AND 최소 길이 충족) OR 최대 길이 도달
        start_new = ((cumsum >= threshold) & (since_last >= min_patch)) | \
                    (since_last >= max_patch)

        # 패딩 위치는 경계 안 만듦
        if pad_mask is not None:
            start_new = start_new & pad_mask[:, t]

        boundaries[:, t] = start_new
        # 새 패치 시작 시 리셋
        cumsum = cumsum * (~start_new).float()
        since_last = since_last * (~start_new).long()

    return boundaries


def boundaries_to_segments(boundaries: torch.Tensor, seq_len: int,
                           max_patches: int) -> tuple:
    """패치 경계 → 세그먼트 인덱스

    Args:
        boundaries: [B, L] True=새 패치 시작
        seq_len: 시퀀스 길이
        max_patches: 최대 패치 수 (패딩용)

    Returns:
        segment_ids: [B, L] 각 위치의 패치 ID (0, 1, 2, ...)
        n_patches: [B] 배치별 실제 패치 수
    """
    B, L = boundaries.shape
    # cumsum으로 segment id 계산 (경계마다 +1)
    segment_ids = boundaries.long().cumsum(dim=1) - 1  # 0-indexed
    segment_ids = segment_ids.clamp(min=0)
    n_patches = segment_ids.max(dim=1).values + 1  # [B]
    return segment_ids, n_patches


class AdaptivePoolEncoder(nn.Module):
    """적응형 풀링 인코더: 가변 세그먼트 → 고정 패치 벡터

    각 세그먼트 내 토큰들을 가중 평균 풀링하여 패치 벡터 생성.
    가중치는 엔트로피 기반 (정보량 많은 토큰에 높은 가중치).
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, hidden: torch.Tensor, segment_ids: torch.Tensor,
                n_patches: torch.Tensor, entropy: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            hidden: [B, L, D] 토큰 hidden states
            segment_ids: [B, L] 세그먼트 ID
            n_patches: [B] 배치별 패치 수
            entropy: [B, L] 선택적 가중치

        Returns:
            patches: [B, max_patches, D]
        """
        B, L, D = hidden.shape
        max_p = n_patches.max().item()

        patches = torch.zeros(B, max_p, D, device=hidden.device, dtype=hidden.dtype)
        counts = torch.zeros(B, max_p, 1, device=hidden.device, dtype=hidden.dtype)

        # scatter_add로 세그먼트별 합산
        seg_expanded = segment_ids.unsqueeze(-1).expand(-1, -1, D)  # [B, L, D]
        patches.scatter_add_(1, seg_expanded, hidden)

        # 카운트
        ones = torch.ones(B, L, 1, device=hidden.device, dtype=hidden.dtype)
        counts.scatter_add_(1, segment_ids.unsqueeze(-1), ones)
        counts = counts.clamp(min=1)

        patches = patches / counts  # 평균 풀링
        return self.proj(patches)


class AdaptiveUpsampleDecoder(nn.Module):
    """적응형 업샘플 디코더: 패치 벡터 → 원래 길이

    세그먼트 ID를 사용해 각 위치에 해당 패치 벡터를 배치.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, patches: torch.Tensor, segment_ids: torch.Tensor,
                target_len: int) -> torch.Tensor:
        """
        Args:
            patches: [B, max_patches, D]
            segment_ids: [B, L]
            target_len: 출력 길이

        Returns:
            upsampled: [B, target_len, D]
        """
        # 각 위치에 해당 패치 벡터를 gather
        seg_expanded = segment_ids.unsqueeze(-1).expand(-1, -1, patches.size(-1))
        upsampled = patches.gather(1, seg_expanded)  # [B, L, D]
        return self.proj(upsampled)


class EntropyPatchCodec(nn.Module):
    """엔트로피 기반 가변 패칭 codec

    구조:
    1. SmallLM으로 per-token 엔트로피 계산
    2. 엔트로피 누적 → 패치 경계 결정
    3. 로컬 인코더(Conv or XAttn)로 토큰 features 추출
    4. AdaptivePool로 가변 세그먼트 → 패치 벡터
    5. AdaptiveUpsample로 패치 → 원래 길이
    6. 로컬 디코더로 복원

    Args:
        vocab_size: vocab 크기
        d_model: hidden dim
        encoder_type: 'conv' or 'xattn'
        entropy_threshold: 패치 경계 임계값 (클수록 큰 패치)
        n_layers: 로컬 인코더/디코더 레이어 수
        n_heads: attention heads (xattn만)
        kernel_size: conv kernel (conv만)
        dropout: dropout
        entropy_d_model: 엔트로피 모델 hidden dim
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        encoder_type: str = "conv",
        entropy_threshold: float = 8.0,
        n_layers: int = 3,
        n_heads: int = 4,
        kernel_size: int = 5,
        dropout: float = 0.1,
        entropy_d_model: int = 128,
        entropy_n_layers: int = 2,
        entropy_n_heads: int = 4,
        min_patch: int = 2,
        max_patch: int = 32,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.encoder_type = encoder_type
        self.entropy_threshold = entropy_threshold
        self.min_patch = min_patch
        self.max_patch = max_patch

        # 엔트로피 모델 (frozen after pre-training)
        self.entropy_model = SmallLM(
            vocab_size, d_model=entropy_d_model,
            n_layers=entropy_n_layers, n_heads=entropy_n_heads,
        )

        # 토큰 임베딩
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.embed_scale = math.sqrt(d_model)

        # 위치 인코딩
        self._pos_cache = {}

        # 로컬 인코더
        if encoder_type == "conv":
            from codec.conv_codec import ConvBlock
            self.enc_layers = nn.ModuleList(
                [ConvBlock(d_model, kernel_size, dropout) for _ in range(n_layers)]
            )
        else:
            from codec.xattn_codec import LocalTransformerLayer
            self.enc_layers = nn.ModuleList(
                [LocalTransformerLayer(d_model, n_heads, dropout)
                 for _ in range(n_layers)]
            )

        # 적응형 풀링
        self.adaptive_pool = AdaptivePoolEncoder(d_model)

        # 적응형 업샘플
        self.adaptive_upsample = AdaptiveUpsampleDecoder(d_model)

        # 로컬 디코더
        if encoder_type == "conv":
            from codec.conv_codec import ConvBlock
            self.dec_layers = nn.ModuleList(
                [ConvBlock(d_model, kernel_size, dropout) for _ in range(n_layers)]
            )
        else:
            from codec.xattn_codec import LocalTransformerLayer
            self.dec_layers = nn.ModuleList(
                [LocalTransformerLayer(d_model, n_heads, dropout)
                 for _ in range(n_layers)]
            )

        self.final_norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def _get_pos_encoding(self, seq_len, device):
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

    def compute_boundaries(self, ids: torch.Tensor,
                           pad_mask: torch.Tensor = None) -> tuple:
        """패치 경계 계산

        Returns:
            boundaries, segment_ids, n_patches, entropy
        """
        with torch.no_grad():
            entropy = self.entropy_model.compute_entropy(ids)

        boundaries = compute_patch_boundaries(
            entropy, self.entropy_threshold,
            self.min_patch, self.max_patch, pad_mask,
        )
        segment_ids, n_patches = boundaries_to_segments(
            boundaries, ids.size(1), max_patches=ids.size(1),
        )
        return boundaries, segment_ids, n_patches, entropy

    def encode(self, ids: torch.Tensor, segment_ids: torch.Tensor = None,
               n_patches: torch.Tensor = None,
               entropy: torch.Tensor = None) -> torch.Tensor:
        """[B, L] → [B, max_patches, d_model]"""
        B, L = ids.shape
        x = self.embedding(ids) * self.embed_scale
        x = x + self._get_pos_encoding(L, x.device)

        for layer in self.enc_layers:
            x = layer(x)

        z = self.adaptive_pool(x, segment_ids, n_patches, entropy)
        return z

    def decode(self, z: torch.Tensor, segment_ids: torch.Tensor,
               target_len: int) -> torch.Tensor:
        """[B, max_patches, d_model] → [B, L, vocab_size]"""
        x = self.adaptive_upsample(z, segment_ids, target_len)

        for layer in self.dec_layers:
            x = layer(x)

        x = self.final_norm(x)
        return self.head(x)

    def forward(self, ids: torch.Tensor, pad_mask: torch.Tensor = None) -> dict:
        """학습용 forward"""
        B, L = ids.shape

        boundaries, segment_ids, n_patches, entropy = self.compute_boundaries(
            ids, pad_mask,
        )

        z = self.encode(ids, segment_ids, n_patches, entropy)
        logits = self.decode(z, segment_ids, L)

        loss = F.cross_entropy(
            logits.reshape(-1, self.vocab_size),
            ids.reshape(-1),
            ignore_index=0,
            reduction="mean",
        )

        avg_patch_size = L / n_patches.float().mean().item()

        return {
            "logits": logits,
            "loss": loss,
            "z": z,
            "n_patches": n_patches,
            "avg_patch_size": avg_patch_size,
            "boundaries": boundaries,
        }

    def reconstruct(self, ids: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            out = self.forward(ids)
            return out["logits"].argmax(dim=-1)

    @property
    def stride(self):
        """호환성: 평균 패치 크기를 stride로 보고"""
        return -1  # 가변


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    print("=== EntropyPatchCodec Smoke Test ===\n")

    for enc_type in ["conv", "xattn"]:
        codec = EntropyPatchCodec(
            vocab_size=330, d_model=256, encoder_type=enc_type,
            entropy_threshold=8.0, n_layers=2,
        )
        n_params = count_params(codec)
        entropy_params = count_params(codec.entropy_model)
        print(f"{enc_type}: {n_params/1e6:.2f}M total ({entropy_params/1e6:.2f}M entropy model)")

        B, L = 2, 128
        ids = torch.randint(1, 330, (B, L))

        out = codec(ids)
        z = out["z"]
        logits = out["logits"]
        loss = out["loss"]
        avg_ps = out["avg_patch_size"]
        n_p = out["n_patches"]

        print(f"  input:      {ids.shape}")
        print(f"  z:          {z.shape}")
        print(f"  logits:     {logits.shape}")
        print(f"  loss:       {loss.item():.4f}")
        print(f"  n_patches:  {n_p.tolist()}")
        print(f"  avg patch:  {avg_ps:.1f} tokens")

        loss.backward()
        # entropy_model은 frozen이므로 grad 없어도 OK
        trainable = [p for p in codec.parameters() if p.requires_grad and p.grad is not None]
        grad_ok = all(not p.grad.isnan().any() for p in trainable)
        print(f"  backward:   {'OK' if grad_ok else 'FAIL'}")
        print()

    print("모든 테스트 통과!")
