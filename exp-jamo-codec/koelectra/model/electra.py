"""KoELECTRA Small v3 + Jamo-Codec 사전학습 모델.

구조:
  text → BBPE(150K) + 자모 분해(330) → jamo_ids[B,L] / segment_ids[B,L]
  → 마스킹 패치의 자모 → JAMO_MASK(=4) 치환
  → CompositionEncoder (공유, pretrained) → z [B, P, 256]
  → emb_proj(256→128) → pos_emb add → LN → dropout → hidden_proj(128→256)
  → Transformer (12L·256h·4H, norm_first) → h [B, P, 256]
  → (Generator) CompositionDecoder (공유, pretrained) → 자모 logits [B, L, 330]
  → argmax로 corrupted 자모 재구성 (stop-grad)
  → CompositionEncoder 재적용 → z_c [B, P, 256]
  → (Discriminator) 같은 공유 emb_proj + pos_emb + 별도 hidden_proj + Transformer
  → head Linear(256→1) → replaced_logits [B, P]

공유: codec_encoder, codec_decoder, emb_proj, pos_emb, emb_layer_norm
Gen 전용: gen_hidden_proj, generator
Disc 전용: disc_hidden_proj, discriminator, disc_head
"""
from __future__ import annotations

import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

# codec import를 위한 경로 설정
_EXP_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _EXP_ROOT not in sys.path:
    sys.path.insert(0, _EXP_ROOT)

from codec.composition_codec import CompositionEncoder, CompositionDecoder  # noqa: E402

from ..data.masking import scatter_any_per_patch  # noqa: E402


class TransformerStack(nn.Module):
    """nn.TransformerEncoderLayer 기반 스택.

    PyTorch 2.x는 batch_first=True, need_weights=False, bool key_padding_mask
    조건에서 SDPA → Flash Attention backend를 자동 선택한다.
    (`torch.nn.attention.sdpa_kernel([SDPBackend.FLASH_ATTENTION])` 컨텍스트로 강제 가능.)
    """

    def __init__(self, n_layers: int = 12, d_model: int = 256, n_heads: int = 4,
                 d_ff: int = 1024, dropout: float = 0.1):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)

    def forward(self, x: torch.Tensor, patch_pad_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, P, D]
            patch_pad_mask: [B, P] bool — True=유효 패치
        Returns:
            [B, P, D]
        """
        # TransformerEncoder는 key_padding_mask에서 True=무시
        key_padding_mask = ~patch_pad_mask
        return self.encoder(x, src_key_padding_mask=key_padding_mask)


class JamoKoElectra(nn.Module):
    """ELECTRA Small v3 + Jamo-Codec.

    Generator와 Discriminator가 CompositionEncoder/Decoder, emb_proj, pos_emb를 공유.
    """

    def __init__(
        self,
        jamo_vocab: int = 330,
        codec_d_model: int = 256,
        codec_n_layers: int = 6,
        codec_kernel_size: int = 7,
        max_jamo_per_token: int = 32,
        # ELECTRA
        embedding_size: int = 128,
        hidden_size: int = 256,
        n_heads: int = 4,
        d_ff: int = 1024,
        gen_layers: int = 12,
        disc_layers: int = 12,
        dropout: float = 0.1,
        max_patches: int = 512,
        gen_loss_weight: float = 50.0,
    ):
        super().__init__()
        self.jamo_vocab = jamo_vocab
        self.max_patches = max_patches
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.gen_loss_weight = gen_loss_weight

        # ── 공유 codec (pretrained로 초기화 예정) ──
        # fixed_output_len=max_patches로 codec 출력이 항상 [B, P, D]로 나옴.
        # → _pad_patches_to (torch.cat) 제거, compile dynamic 불필요.
        self.codec_encoder = CompositionEncoder(
            jamo_vocab=jamo_vocab, d_model=codec_d_model,
            n_layers=codec_n_layers, kernel_size=codec_kernel_size,
            dropout=dropout, max_jamo_per_token=max_jamo_per_token,
            fixed_output_len=max_patches,
        )
        self.codec_decoder = CompositionDecoder(
            jamo_vocab=jamo_vocab, d_model=codec_d_model,
            n_layers=codec_n_layers, kernel_size=codec_kernel_size,
            dropout=dropout, max_jamo_per_token=max_jamo_per_token,
        )

        # ── 공유 임베딩 계층 (ELECTRA 관행) ──
        self.emb_proj = nn.Linear(codec_d_model, embedding_size)
        self.pos_emb = nn.Embedding(max_patches, embedding_size)
        self.emb_layer_norm = nn.LayerNorm(embedding_size)
        self.emb_dropout = nn.Dropout(dropout)

        # ── Generator ──
        self.gen_hidden_proj = nn.Linear(embedding_size, hidden_size)
        self.generator = TransformerStack(
            n_layers=gen_layers, d_model=hidden_size,
            n_heads=n_heads, d_ff=d_ff, dropout=dropout,
        )
        # Generator 출력(256) → codec_decoder 입력(256) : dim 동일하므로 추가 proj 불필요

        # ── Discriminator ──
        self.disc_hidden_proj = nn.Linear(embedding_size, hidden_size)
        self.discriminator = TransformerStack(
            n_layers=disc_layers, d_model=hidden_size,
            n_heads=n_heads, d_ff=d_ff, dropout=dropout,
        )
        self.disc_head = nn.Linear(hidden_size, 1)

        self._init_weights()

    def _init_weights(self):
        """codec 외 신규 파라미터 초기화 (BERT 스타일 std=0.02)."""
        for name, p in self.named_parameters():
            if name.startswith("codec_"):
                continue  # codec은 pretrained로 별도 로드
            if p.dim() >= 2:
                nn.init.normal_(p, mean=0.0, std=0.02)
            elif "bias" in name:
                nn.init.zeros_(p)

    def load_codec_pretrained(self, ckpt_path: str, map_location="cpu"):
        """composition_6L_step*.pt 체크포인트에서 encoder/decoder 로드."""
        ckpt = torch.load(ckpt_path, map_location=map_location, weights_only=False)
        sd = ckpt["model"]
        enc_sd = {k[len("encoder."):]: v for k, v in sd.items() if k.startswith("encoder.")}
        dec_sd = {k[len("decoder."):]: v for k, v in sd.items() if k.startswith("decoder.")}
        missing_e, unexpected_e = self.codec_encoder.load_state_dict(enc_sd, strict=True)
        missing_d, unexpected_d = self.codec_decoder.load_state_dict(dec_sd, strict=True)
        return {
            "encoder_missing": missing_e, "encoder_unexpected": unexpected_e,
            "decoder_missing": missing_d, "decoder_unexpected": unexpected_d,
        }

    def codec_parameters(self):
        """Codec 파라미터 이터레이터 (LR 1/10 param group용)."""
        yield from self.codec_encoder.parameters()
        yield from self.codec_decoder.parameters()

    def non_codec_parameters(self):
        """Transformer + proj + head 파라미터."""
        codec_ids = {id(p) for p in self.codec_parameters()}
        for p in self.parameters():
            if id(p) not in codec_ids:
                yield p

    def _embed(self, z: torch.Tensor, patch_pad_mask: torch.Tensor) -> torch.Tensor:
        """codec 출력 → embedding_size(128) + pos_emb + LN + dropout."""
        B, P, _ = z.shape
        e = self.emb_proj(z)  # [B,P,128]
        positions = torch.arange(P, device=z.device).unsqueeze(0).expand(B, -1)
        e = e + self.pos_emb(positions)
        e = self.emb_layer_norm(e)
        e = self.emb_dropout(e)
        # 패딩 위치는 0화 (TransformerEncoder의 key_padding_mask와 중복이지만 안전)
        e = e * patch_pad_mask.unsqueeze(-1).to(e.dtype)
        return e

    def forward(
        self,
        jamo_ids: torch.Tensor,         # [B, L]
        jamo_mask: torch.Tensor,        # [B, L] bool, 유효 자모
        segment_ids: torch.Tensor,      # [B, L]
        n_segments: torch.Tensor,       # [B]
        masked_jamo_ids: torch.Tensor,  # [B, L] 마스킹 적용본
        per_jamo_mask: torch.Tensor,    # [B, L] 마스킹 패치에 속한 유효 자모
        masked_patch_mask: torch.Tensor,  # [B, P] 마스킹된 패치 자체
    ) -> dict:
        B, L = jamo_ids.shape
        P = self.max_patches
        device = jamo_ids.device

        # 패치 단위 패딩 마스크: segment_ids가 실제 토큰 < n_segments인 위치
        pos = torch.arange(P, device=device).unsqueeze(0).expand(B, -1)
        patch_pad_mask = pos < n_segments.unsqueeze(-1)  # [B, P] True=유효

        # ── (1) Generator: 마스킹본 인코딩 ──
        # codec_encoder는 fixed_output_len=max_patches로 항상 [B, P, D] 반환
        z_masked = self.codec_encoder(masked_jamo_ids, jamo_mask, segment_ids, n_segments)
        e_masked = self._embed(z_masked, patch_pad_mask)
        h_gen = self.gen_hidden_proj(e_masked)  # [B, P, 256]
        h_gen = self.generator(h_gen, patch_pad_mask)

        # Generator head: codec_decoder로 자모 logits 복원
        jamo_logits = self.codec_decoder(h_gen, segment_ids, L, jamo_mask)  # [B, L, V]

        # Gen loss: 마스킹 패치에 속한 유효 자모 위치만
        gen_target_mask = per_jamo_mask  # [B, L] bool
        V = self.jamo_vocab
        ce = F.cross_entropy(
            jamo_logits.reshape(-1, V),
            jamo_ids.reshape(-1),
            reduction="none",
        ).reshape(B, L)
        denom = gen_target_mask.sum().clamp(min=1)
        gen_loss = (ce * gen_target_mask.float()).sum() / denom

        # ── (2) Corrupted 재구성 (stop-gradient) ──
        with torch.no_grad():
            sampled = jamo_logits.argmax(-1)  # [B, L]
            jamo_corrupted = torch.where(gen_target_mask, sampled, jamo_ids)
            diff = (sampled != jamo_ids) & gen_target_mask
            # 패치 단위 replaced 라벨: 해당 패치에 속한 자모 중 하나라도 다르면 replaced
            replaced = scatter_any_per_patch(diff, segment_ids, P)  # [B, P]
            # 학습 신호: 마스킹된 패치만 평가 대상 (비마스킹 패치는 replaced=False)
            # ELECTRA 원논문은 모든 패치를 평가하나, 마스킹 패치만 replaced 가능하므로
            # 모든 패치를 포함해도 결과는 동일 (비마스킹=항상 원본=replaced=False).

        # ── (3) Discriminator: corrupted 자모 재인코딩 ──
        z_corrupted = self.codec_encoder(jamo_corrupted, jamo_mask, segment_ids, n_segments)
        e_corrupted = self._embed(z_corrupted, patch_pad_mask)
        h_disc = self.disc_hidden_proj(e_corrupted)
        h_disc = self.discriminator(h_disc, patch_pad_mask)
        disc_logits = self.disc_head(h_disc).squeeze(-1)  # [B, P]

        # Disc loss: 유효 패치만
        valid = patch_pad_mask
        disc_loss = F.binary_cross_entropy_with_logits(
            disc_logits[valid], replaced[valid].float()
        )

        # Disc 정확도 + 패치 활용률 (로깅용)
        with torch.no_grad():
            disc_pred = disc_logits > 0
            disc_correct = (disc_pred == replaced) & valid
            disc_acc = disc_correct.sum().float() / valid.sum().clamp(min=1).float()
            replaced_rate = (replaced & valid).sum().float() / valid.sum().clamp(min=1).float()
            masked_tokens = gen_target_mask.sum().float()
            # 패치 활용률: 배치 평균 n_segments / max_patches
            # 1.0에 가까울수록 transformer가 의미있는 위치에 계산 집중.
            # 낮으면 (<0.8) padding 위치 계산 낭비가 커서 varlen attention 이득 큼.
            patch_util = n_segments.float().mean() / float(self.max_patches)

        total_loss = disc_loss + self.gen_loss_weight * gen_loss
        return {
            "total_loss": total_loss,
            "gen_loss": gen_loss.detach(),
            "disc_loss": disc_loss.detach(),
            "disc_acc": disc_acc,
            "replaced_rate": replaced_rate,
            "masked_tokens": masked_tokens,
            "patch_util": patch_util,
        }


def _pad_patches_to(z: torch.Tensor, P: int) -> torch.Tensor:
    """DEPRECATED — CompositionEncoder(fixed_output_len=P) 사용으로 불필요해짐.
    profile_run.py 등 외부 임포트 호환성을 위해 유지.
    """
    B, cur_P, D = z.shape
    if cur_P == P:
        return z
    if cur_P > P:
        return z[:, :P]
    pad = torch.zeros(B, P - cur_P, D, device=z.device, dtype=z.dtype)
    return torch.cat([z, pad], dim=1)


# ── smoke test ──
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--codec_ckpt", type=str,
                    default="exp-jamo-codec/checkpoints/composition_6L_step600000.pt")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    print("=== JamoKoElectra smoke test ===")
    model = JamoKoElectra(
        codec_d_model=256, codec_n_layers=6, codec_kernel_size=7,
        embedding_size=128, hidden_size=256, n_heads=4, d_ff=1024,
        gen_layers=12, disc_layers=12, max_patches=64,
        gen_loss_weight=50.0,
    ).to(args.device)

    if os.path.exists(args.codec_ckpt):
        info = model.load_codec_pretrained(args.codec_ckpt)
        print("codec load:", {k: (len(v) if isinstance(v, list) else v) for k, v in info.items()})
    else:
        print(f"[경고] codec_ckpt 없음: {args.codec_ckpt} — random init")

    total_params = sum(p.numel() for p in model.parameters())
    codec_params = sum(p.numel() for p in model.codec_parameters())
    tf_params = total_params - codec_params
    print(f"params: total={total_params/1e6:.2f}M "
          f"(codec={codec_params/1e6:.2f}M, transformer+proj={tf_params/1e6:.2f}M)")

    # 더미 배치
    B, L, P = 2, 256, 64
    dev = args.device
    jamo_ids = torch.randint(10, 330, (B, L), device=dev)
    jamo_mask = torch.ones(B, L, dtype=torch.bool, device=dev)
    # segment_ids: 0~31까지 각 8자모
    n_seg = 32
    segment_ids = torch.arange(n_seg, device=dev).repeat_interleave(L // n_seg).unsqueeze(0).expand(B, -1).contiguous()
    n_segments = torch.tensor([n_seg, n_seg], device=dev)

    from ..data.masking import make_patch_mask, apply_mask
    masked_patch_mask = make_patch_mask(n_segments, max_patches=P, mask_ratio=0.20)
    masked_jamo_ids, per_jamo_mask = apply_mask(jamo_ids, segment_ids, jamo_mask, masked_patch_mask)

    out = model(jamo_ids, jamo_mask, segment_ids, n_segments,
                masked_jamo_ids, per_jamo_mask, masked_patch_mask)
    for k, v in out.items():
        if torch.is_tensor(v):
            print(f"  {k}: {v.item():.4f}")

    out["total_loss"].backward()
    has_nan = any(p.grad is not None and p.grad.isnan().any() for p in model.parameters())
    print(f"backward: {'FAIL (NaN)' if has_nan else 'OK'}")
