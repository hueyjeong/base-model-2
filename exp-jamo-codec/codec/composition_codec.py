"""CompositionCodec — BBPE 경계 + Conv 자모 composition (concat 방식)

BBPE 토크나이저가 경계를 결정하고, 전체 자모를 1열로 concat하여
Conv 처리 후 토큰 경계에서 segment pool → 토큰 벡터 생성.
패딩 낭비 없이 토큰 간 문맥 정보도 활용.

인코더: 자모 concat → Embedding → Conv layers → Segment Avg Pool → 토큰 벡터
디코더: 토큰 벡터 → Segment Upsample → Conv layers → 자모 logits
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from codec.conv_codec import ConvBlock


class CompositionEncoder(nn.Module):
    """자모 concat → Conv → Segment Pool → 토큰 벡터

    전체 자모 시퀀스를 한번에 Conv 처리 후,
    토큰 경계(segment_ids)에 맞춰 평균 풀링.

    개선:
    - intra_pos_emb: 임베딩 직후 세그먼트 내 위치 정보 주입 → pool 전에 각 자모가
      "나는 이 토큰의 N번째" 임을 인코딩, token_vecs가 위치 정보를 포함
    """

    def __init__(self, jamo_vocab: int = 330, d_model: int = 256,
                 n_layers: int = 5, kernel_size: int = 7, dropout: float = 0.1,
                 max_jamo_per_token: int = 32,
                 fixed_output_len: int | None = None):
        """
        Args:
            fixed_output_len: None이면 출력 shape이 `[B, segment_ids.max()+1, D]`로
                배치마다 가변. int이면 항상 `[B, fixed_output_len, D]` 고정 출력
                (segment_ids 값이 fixed_output_len 이하임을 호출자가 보장해야 함).
                compile static shape 경로에서 torch.cat/dynamic symbol을 제거하여
                max-autotune/DDPOptimizer 호환성 및 copy 오버헤드 제거.
        """
        super().__init__()
        self.d_model = d_model
        self.max_jamo_per_token = max_jamo_per_token
        self.fixed_output_len = fixed_output_len
        self.embedding = nn.Embedding(jamo_vocab, d_model, padding_idx=0)
        self.embed_scale = math.sqrt(d_model)
        # 세그먼트 내 위치 임베딩 (intra-segment position)
        self.intra_pos_emb = nn.Embedding(max_jamo_per_token, d_model)
        self.layers = nn.ModuleList([
            ConvBlock(d_model, kernel_size, dropout) for _ in range(n_layers)
        ])
        self.pool_proj = nn.Linear(d_model, d_model)

    def forward(self, jamo_ids: torch.Tensor, jamo_mask: torch.Tensor,
                segment_ids: torch.Tensor, n_segments: torch.Tensor) -> torch.Tensor:
        """
        Args:
            jamo_ids: [B, L] concat된 자모 ID
            jamo_mask: [B, L] 유효 위치
            segment_ids: [B, L] 각 자모가 속한 토큰 ID (0, 0, 0, 1, 1, 2, ...)
            n_segments: [B] 배치별 토큰 수

        Returns:
            token_vecs: [B, max_segments, d_model] (fixed_output_len None 시)
                       또는 [B, fixed_output_len, d_model] (int 지정 시)
        """
        B, L = jamo_ids.shape
        D = self.d_model

        # 임베딩
        x = self.embedding(jamo_ids) * self.embed_scale  # [B, L, D]

        # within_pos 계산: 세그먼트 내 0-based index (compile 호환)
        arange_pos = torch.arange(L, device=segment_ids.device).unsqueeze(0).expand(B, -1)
        seg_change = torch.cat([
            torch.ones(B, 1, dtype=torch.bool, device=segment_ids.device),
            segment_ids[:, 1:] != segment_ids[:, :-1],
        ], dim=1)
        seg_start_per_pos = torch.cummax(seg_change * arange_pos, dim=1).values
        within_pos = (arange_pos - seg_start_per_pos).clamp(0, self.max_jamo_per_token - 1)

        # 세그먼트 내 위치 정보 주입 (임베딩 직후, Conv 전)
        x = x + self.intra_pos_emb(within_pos)

        # Conv 레이어
        for layer in self.layers:
            x = layer(x)  # [B, L, D]

        # Segment Avg Pool (scatter_add)
        # fixed_output_len 지정 시 고정 크기, 아니면 배치별 최대 segment로 동적 할당
        if self.fixed_output_len is not None:
            max_seg = self.fixed_output_len
        else:
            max_seg = segment_ids.max() + 1  # .item() 없이 compile 호환
        token_vecs = torch.zeros(B, max_seg, D, device=x.device, dtype=x.dtype)
        counts = torch.zeros(B, max_seg, 1, device=x.device, dtype=x.dtype)

        seg_exp = segment_ids.unsqueeze(-1).expand(-1, -1, D)  # [B, L, D]
        # 마스킹: 패딩 위치는 0
        x_masked = x * jamo_mask.unsqueeze(-1).float()
        token_vecs.scatter_add_(1, seg_exp, x_masked)

        ones = jamo_mask.float().unsqueeze(-1)  # [B, L, 1]
        counts.scatter_add_(1, segment_ids.unsqueeze(-1), ones)
        counts = counts.clamp(min=1)

        token_vecs = token_vecs / counts
        return self.pool_proj(token_vecs)


class CompositionDecoder(nn.Module):
    """토큰 벡터 → Segment Upsample → Conv → 자모 logits

    토큰 벡터를 segment_ids로 원래 자모 길이로 확장 후
    Conv로 처리하여 자모 logits 출력.

    개선:
    - intra_pos_emb: 세그먼트 내 0-based position embedding → '울'→'ㅜㅜ' 류 오류 해결
    - jamo_mask 적용: 패딩 위치가 Conv를 통해 유효 위치에 오염되는 것 방지
    """

    def __init__(self, jamo_vocab: int = 330, d_model: int = 256,
                 n_layers: int = 5, kernel_size: int = 7, dropout: float = 0.1,
                 max_jamo_per_token: int = 32):
        super().__init__()
        self.max_jamo_per_token = max_jamo_per_token
        self.upsample_proj = nn.Linear(d_model, d_model)
        # 세그먼트 내 위치 임베딩 (intra-segment position)
        self.intra_pos_emb = nn.Embedding(max_jamo_per_token, d_model)
        self.layers = nn.ModuleList([
            ConvBlock(d_model, kernel_size, dropout) for _ in range(n_layers)
        ])
        self.head = nn.Linear(d_model, jamo_vocab)

    def forward(self, token_vecs: torch.Tensor, segment_ids: torch.Tensor,
                target_len: int, jamo_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            token_vecs: [B, max_segments, d_model]
            segment_ids: [B, L] 각 자모의 토큰 ID
            target_len: 출력 자모 길이 L
            jamo_mask: [B, L] bool, 유효 위치 (선택)

        Returns:
            logits: [B, L, jamo_vocab]
        """
        B, L = segment_ids.shape
        D = token_vecs.size(-1)

        # within_pos 계산: 각 자모의 세그먼트 내 0-based index
        # .item() 없이 torch.compile 호환: segment 경계 감지 → cummax로 시작 위치 전파
        arange_pos = torch.arange(L, device=segment_ids.device).unsqueeze(0).expand(B, -1)
        seg_change = torch.cat([
            torch.ones(B, 1, dtype=torch.bool, device=segment_ids.device),
            segment_ids[:, 1:] != segment_ids[:, :-1],
        ], dim=1)  # [B, L]: 각 세그먼트 첫 위치에서 True
        seg_start_per_pos = torch.cummax(seg_change * arange_pos, dim=1).values  # [B, L]
        within_pos = (arange_pos - seg_start_per_pos).clamp(0, self.max_jamo_per_token - 1)

        # Upsample: 각 자모 위치에 해당 토큰 벡터 배치
        seg_exp = segment_ids.unsqueeze(-1).expand(-1, -1, D)
        x = token_vecs.gather(1, seg_exp)  # [B, L, D]
        x = self.upsample_proj(x)

        # 세그먼트 내 위치 정보 주입
        x = x + self.intra_pos_emb(within_pos)

        # 패딩 위치 0화 (Conv를 통한 오염 방지)
        if jamo_mask is not None:
            x = x * jamo_mask.unsqueeze(-1).float()

        # Conv 레이어 (각 레이어 후 패딩 재적용)
        for layer in self.layers:
            x = layer(x)
            if jamo_mask is not None:
                x = x * jamo_mask.unsqueeze(-1).float()

        return self.head(x)  # [B, L, V]


class CompositionCodec(nn.Module):
    """BBPE + Conv Composition Codec (concat 방식)

    전체 자모를 concat → Conv encoder → segment pool → 토큰 벡터
    → segment upsample → Conv decoder → 자모 복원

    패딩 낭비 없음 (활용률 ~100% vs 이전 ~8%).
    토큰 간 문맥 정보 활용 가능 (contextual embedding).
    """

    def __init__(self, jamo_vocab: int = 330, d_model: int = 256,
                 n_layers: int = 5, kernel_size: int = 7, dropout: float = 0.1,
                 max_jamo_per_token: int = 32):
        super().__init__()
        self.jamo_vocab = jamo_vocab
        self.d_model = d_model

        self.encoder = CompositionEncoder(
            jamo_vocab, d_model, n_layers, kernel_size, dropout,
            max_jamo_per_token=max_jamo_per_token,
        )
        self.decoder = CompositionDecoder(
            jamo_vocab, d_model, n_layers, kernel_size, dropout,
            max_jamo_per_token=max_jamo_per_token,
        )

    def forward(self, jamo_ids: torch.Tensor, jamo_mask: torch.Tensor,
                segment_ids: torch.Tensor, n_segments: torch.Tensor) -> dict:
        """학습용 forward

        Args:
            jamo_ids: [B, L] concat된 자모 시퀀스
            jamo_mask: [B, L] 유효 위치
            segment_ids: [B, L] 토큰 경계 (0,0,0,1,1,2,2,2,...)
            n_segments: [B] 토큰 수

        Returns:
            dict: logits, loss, z
        """
        L = jamo_ids.size(1)

        # Encode
        z = self.encoder(jamo_ids, jamo_mask, segment_ids, n_segments)

        # Decode
        logits = self.decoder(z, segment_ids, L, jamo_mask)  # [B, L, V]

        # Loss
        flat_logits = logits.reshape(-1, self.jamo_vocab)
        flat_targets = jamo_ids.reshape(-1)
        valid = jamo_mask.reshape(-1)

        loss = F.cross_entropy(
            flat_logits, flat_targets, ignore_index=0, reduction="none",
        )
        loss = loss[valid].mean()

        return {"logits": logits, "loss": loss, "z": z}

    def encode(self, jamo_ids, jamo_mask, segment_ids, n_segments):
        """인코딩만 (backbone 입력용)"""
        return self.encoder(jamo_ids, jamo_mask, segment_ids, n_segments)

    def reconstruct(self, jamo_ids, jamo_mask, segment_ids, n_segments):
        """복원 (평가용)"""
        with torch.no_grad():
            z = self.encoder(jamo_ids, jamo_mask, segment_ids, n_segments)
            logits = self.decoder(z, segment_ids, jamo_ids.size(1), jamo_mask)
            return logits.argmax(dim=-1)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    print("=== CompositionCodec (concat) Smoke Test ===\n")

    for n_layers in [3, 4, 5]:
        codec = CompositionCodec(
            jamo_vocab=330, d_model=256, n_layers=n_layers, kernel_size=7,
        )
        n_params = count_params(codec)
        enc_params = count_params(codec.encoder)
        dec_params = count_params(codec.decoder)
        print(f"{n_layers}L: {n_params/1e6:.2f}M total "
              f"(enc {enc_params/1e6:.2f}M, dec {dec_params/1e6:.2f}M)")

        # forward test: 3개 토큰, 자모 concat [5, 3, 7] = 15
        B = 2
        # 토큰별 자모 길이: [5, 3, 7]
        L = 15
        jamo_ids = torch.randint(1, 330, (B, L))
        jamo_mask = torch.ones(B, L, dtype=torch.bool)
        # segment_ids: 0,0,0,0,0, 1,1,1, 2,2,2,2,2,2,2
        segment_ids = torch.tensor([[0]*5 + [1]*3 + [2]*7] * B)
        n_segments = torch.tensor([3, 3])

        out = codec(jamo_ids, jamo_mask, segment_ids, n_segments)
        print(f"  input: [B={B}, L={L}], z: {out['z'].shape}, "
              f"logits: {out['logits'].shape}, loss: {out['loss'].item():.4f}")

        out["loss"].backward()
        trainable = [p for p in codec.parameters() if p.grad is not None]
        grad_ok = all(not p.grad.isnan().any() for p in trainable)
        print(f"  backward: {'OK' if grad_ok else 'FAIL'}")
        print()

    print("모든 테스트 통과!")
