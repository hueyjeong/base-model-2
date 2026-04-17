"""KoELECTRA Small v3 + SimpleCodec 사전학습 모델.

구조:
  text → BBPE(150K) + 자모 분해(330)
  → jamo_ids[B,P,S], jamo_mask[B,P,S], token_pad_mask[B,P], special_token_mask[B,P]
  → 마스킹 토큰의 자모 → JAMO_MASK(=4) 치환
  → SimpleCodec.encode (frozen, reshape 후 per-token) → z [B, P, 256]
  → emb_proj(256→128) + pos_emb + LN + dropout
  → gen_hidden_proj(128→256) → Generator Transformer → h_gen [B, P, 256]
  → SimpleCodec.decode (frozen) → 자모 logits [B, P, S, 330]
  → argmax 로 corrupted 자모 재구성 (stop-grad)
  → SimpleCodec.encode 재적용 → z_c [B, P, 256]
  → disc_hidden_proj(128→256) → Discriminator Transformer → h_disc [B, P, 256]
  → Linear(256→1) → replaced_logits [B, P]

Codec 은 학습 전체 동안 freeze. Transformer + proj + head 만 학습.
"""
from __future__ import annotations

import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

_EXP_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _EXP_ROOT not in sys.path:
    sys.path.insert(0, _EXP_ROOT)

from codec.simple_codec import SimpleCodec  # noqa: E402


class TransformerStack(nn.Module):
    """nn.TransformerEncoderLayer 기반 스택.

    batch_first + need_weights=False + bool key_padding_mask → PyTorch 2.x SDPA
    가 Flash backend 자동 선택.
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
        x: [B, P, D]
        patch_pad_mask: [B, P] bool — True=유효 토큰
        """
        key_padding_mask = ~patch_pad_mask
        return self.encoder(x, src_key_padding_mask=key_padding_mask)


class JamoKoElectra(nn.Module):
    """ELECTRA Small v3 + Jamo-Codec (SimpleCodec frozen).

    Generator/Discriminator 가 codec, emb_proj, pos_emb, emb_layer_norm 을 공유.
    """

    def __init__(
        self,
        jamo_vocab: int = 330,
        # Codec (SimpleCodec hparams — checkpoint 와 일치해야 함)
        codec_d_model: int = 256,
        codec_n_enc_layers: int = 5,
        codec_n_dec_layers: int = 5,
        codec_kernel_size: int = 5,
        max_jamo_per_token: int = 32,
        codec_dropout: float = 0.1,
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
        self.max_jamo_per_token = max_jamo_per_token
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.gen_loss_weight = gen_loss_weight

        # ── SimpleCodec (frozen) ──
        self.codec = SimpleCodec(
            jamo_vocab=jamo_vocab,
            d_model=codec_d_model,
            n_enc_layers=codec_n_enc_layers,
            n_dec_layers=codec_n_dec_layers,
            kernel_size=codec_kernel_size,
            max_jamo=max_jamo_per_token,
            dropout=codec_dropout,
        )
        for p in self.codec.parameters():
            p.requires_grad = False
        self.codec.eval()

        # ── 공유 embedding 계층 ──
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

        # ── Discriminator ──
        self.disc_hidden_proj = nn.Linear(embedding_size, hidden_size)
        self.discriminator = TransformerStack(
            n_layers=disc_layers, d_model=hidden_size,
            n_heads=n_heads, d_ff=d_ff, dropout=dropout,
        )
        self.disc_head = nn.Linear(hidden_size, 1)

        self._init_weights()

    def _init_weights(self):
        """codec 외 신규 파라미터 BERT 스타일 초기화."""
        for name, p in self.named_parameters():
            if name.startswith("codec."):
                continue
            if p.dim() >= 2:
                nn.init.normal_(p, mean=0.0, std=0.02)
            elif "bias" in name:
                nn.init.zeros_(p)

    def train(self, mode: bool = True):
        """codec 은 항상 eval (frozen + Dropout 고정)."""
        super().train(mode)
        self.codec.eval()
        return self

    def load_codec_pretrained(self, ckpt_path: str, map_location="cpu"):
        """SimpleCodec 체크포인트(`checkpoints/simple_codec_final.pt`) 로드.

        ckpt["model"] 은 SimpleCodec top-level state_dict (prefix 없음).
        """
        ckpt = torch.load(ckpt_path, map_location=map_location, weights_only=False)
        sd = ckpt["model"]
        missing, unexpected = self.codec.load_state_dict(sd, strict=True)
        return {"missing": missing, "unexpected": unexpected}

    def codec_parameters(self):
        """Codec 파라미터 이터레이터 (optimizer 에서 제외 용)."""
        yield from self.codec.parameters()

    def non_codec_parameters(self):
        """학습 대상 (Transformer + proj + head)."""
        codec_ids = {id(p) for p in self.codec_parameters()}
        for p in self.parameters():
            if id(p) not in codec_ids:
                yield p

    # ─────────────────────────────────────────────
    def _embed(self, z: torch.Tensor, patch_pad_mask: torch.Tensor) -> torch.Tensor:
        """codec 출력 → embedding_size + pos_emb + LN + dropout."""
        B, P, _ = z.shape
        e = self.emb_proj(z)
        positions = torch.arange(P, device=z.device).unsqueeze(0).expand(B, -1)
        e = e + self.pos_emb(positions)
        e = self.emb_layer_norm(e)
        e = self.emb_dropout(e)
        e = e * patch_pad_mask.unsqueeze(-1).to(e.dtype)
        return e

    def forward(
        self,
        jamo_ids: torch.Tensor,           # [B, P, S]
        jamo_mask: torch.Tensor,          # [B, P, S] bool
        token_pad_mask: torch.Tensor,     # [B, P] bool (유효 토큰)
        masked_jamo_ids: torch.Tensor,    # [B, P, S] (masked 토큰 전 슬롯 JAMO_MASK)
        masked_jamo_mask: torch.Tensor,   # [B, P, S] (masked 토큰 전 슬롯 True)
        masked_patch_mask: torch.Tensor,  # [B, P] bool (마스킹된 토큰)
        per_jamo_mask: torch.Tensor,      # [B, P, S] bool (masked & 실자모 — loss target)
    ) -> dict:
        B, P, S = jamo_ids.shape

        # ── (1) Generator encode ──
        # masked 토큰은 전 슬롯 JAMO_MASK + jamo_mask 전 True 로 codec 에 saturate 신호
        m_flat = masked_jamo_ids.reshape(B * P, S)
        m_mask_flat = masked_jamo_mask.reshape(B * P, S)
        z_flat = self.codec.encode(m_flat, m_mask_flat)
        z_masked = z_flat.view(B, P, -1)

        e_masked = self._embed(z_masked, token_pad_mask)
        h_gen = self.gen_hidden_proj(e_masked)  # [B, P, hidden]
        h_gen = self.generator(h_gen, token_pad_mask)  # [B, P, hidden]

        # ── Generator decode ──
        h_gen_flat = h_gen.reshape(B * P, -1)  # [B*P, hidden=256=d_model]
        logits_flat = self.codec.decode(h_gen_flat)  # [B*P, S, V]
        jamo_logits = logits_flat.view(B, P, S, -1)

        # Gen loss: per_jamo_mask 위치만
        V = self.jamo_vocab
        ce = F.cross_entropy(
            jamo_logits.reshape(-1, V),
            jamo_ids.reshape(-1),
            reduction="none",
        ).reshape(B, P, S)
        denom = per_jamo_mask.sum().clamp(min=1)
        gen_loss = (ce * per_jamo_mask.float()).sum() / denom

        # ── (2) Corrupted 재구성 (stop-gradient) ──
        with torch.no_grad():
            sampled = jamo_logits.argmax(-1)  # [B, P, S]
            jamo_corrupted = torch.where(per_jamo_mask, sampled, jamo_ids)
            diff = ((sampled != jamo_ids) & per_jamo_mask).any(dim=-1)  # [B, P]
            replaced = diff & masked_patch_mask  # 비마스킹 토큰은 False

        # ── (3) Discriminator ──
        c_flat = jamo_corrupted.reshape(B * P, S)
        z_c_flat = self.codec.encode(c_flat, jamo_mask.reshape(B * P, S))
        z_corrupted = z_c_flat.view(B, P, -1)

        e_corrupted = self._embed(z_corrupted, token_pad_mask)
        h_disc = self.disc_hidden_proj(e_corrupted)
        h_disc = self.discriminator(h_disc, token_pad_mask)
        disc_logits = self.disc_head(h_disc).squeeze(-1)  # [B, P]

        # Disc loss: 유효 토큰만
        valid = token_pad_mask
        disc_loss = F.binary_cross_entropy_with_logits(
            disc_logits[valid], replaced[valid].float()
        )

        with torch.no_grad():
            disc_pred = disc_logits > 0
            disc_correct = (disc_pred == replaced) & valid
            disc_acc = disc_correct.sum().float() / valid.sum().clamp(min=1).float()
            replaced_rate = (replaced & valid).sum().float() / valid.sum().clamp(min=1).float()
            masked_tokens = masked_patch_mask.sum().float()
            patch_util = valid.sum().float() / (B * P)

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


# ── smoke test ──
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--codec_ckpt", type=str,
                    default="checkpoints/simple_codec_final.pt")
    ap.add_argument("--device", type=str,
                    default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    print("=== JamoKoElectra smoke test ===")
    model = JamoKoElectra(
        codec_d_model=256, codec_n_enc_layers=5, codec_n_dec_layers=5,
        codec_kernel_size=5, max_jamo_per_token=32,
        embedding_size=128, hidden_size=256, n_heads=4, d_ff=1024,
        gen_layers=12, disc_layers=12, max_patches=64,
        gen_loss_weight=50.0,
    ).to(args.device)

    if os.path.exists(args.codec_ckpt):
        info = model.load_codec_pretrained(args.codec_ckpt, map_location=args.device)
        print(f"codec load: missing={len(info['missing'])}, "
              f"unexpected={len(info['unexpected'])}")
    else:
        print(f"[경고] codec_ckpt 없음: {args.codec_ckpt} — random init")

    total_params = sum(p.numel() for p in model.parameters())
    codec_params = sum(p.numel() for p in model.codec_parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"params: total={total_params/1e6:.2f}M "
          f"(codec frozen={codec_params/1e6:.2f}M, "
          f"trainable={trainable/1e6:.2f}M)")
    assert trainable == total_params - codec_params, "codec freeze 실패"

    # Dummy batch
    B, P, S = 2, 64, 32
    dev = args.device
    jamo_ids = torch.randint(10, 330, (B, P, S), device=dev)
    jamo_mask = torch.zeros(B, P, S, dtype=torch.bool, device=dev)
    jamo_mask[:, :, :8] = True  # 각 토큰 8자모
    token_pad_mask = torch.zeros(B, P, dtype=torch.bool, device=dev)
    token_pad_mask[:, :50] = True  # 앞 50 토큰만 유효
    n_tokens = torch.tensor([50, 50], device=dev)

    from koelectra.data.masking import make_patch_mask, apply_mask
    masked_patch_mask = make_patch_mask(n_tokens, max_patches=P, mask_ratio=0.20)
    masked_jamo_ids, masked_jamo_mask, per_jamo_mask = apply_mask(
        jamo_ids, jamo_mask, masked_patch_mask,
    )

    out = model(jamo_ids, jamo_mask, token_pad_mask,
                masked_jamo_ids, masked_jamo_mask,
                masked_patch_mask, per_jamo_mask)
    for k, v in out.items():
        if torch.is_tensor(v):
            print(f"  {k}: {v.item():.4f}")

    out["total_loss"].backward()
    has_nan = any(p.grad is not None and p.grad.isnan().any()
                  for p in model.parameters())
    codec_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                     for p in model.codec_parameters())
    print(f"backward: {'FAIL (NaN)' if has_nan else 'OK'}")
    print(f"codec grad 존재: {codec_grad} (False 기대 — freeze)")
