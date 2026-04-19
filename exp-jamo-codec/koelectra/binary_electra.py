"""Binary ELECTRA — codec 없이 BBPE token id 를 18-bit binary 로 직접 입력.

설계 (변종 A: pure binary, k=0):
  bbpe_ids[B,P]
  → int_to_bits → bits[B,P,18] (±1 정규화)
  → bit_proj(18→128) + pos_emb + LN + dropout
  → gen_hidden_proj(128→256) → Generator Transformer → h_gen[B,P,256]
  → gen_head(256→18) → per-bit logit
  → BCE-with-logits (masked 위치만)
  → sigmoid > 0 → sampled bits → integer (clamp to vocab)
  → corrupted = where(masked, sampled, original)
  → 다시 binary encode → Discriminator → disc_head → replaced 판정

Codec 완전 제거. softmax(153600) 회피 — Linear(hidden, 18) 만 사용.
Mask 토큰: vocab_size 자체를 reserve (= 153600, binary 18 bit 안에 존재).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def int_to_bits(ids: torch.Tensor, n_bits: int) -> torch.Tensor:
    """[...] long → [..., n_bits] float in {0.0, 1.0}.

    LSB 가 bit[0]. Vectorized bit shift.
    """
    shifts = torch.arange(n_bits, device=ids.device, dtype=ids.dtype)
    return ((ids.unsqueeze(-1) >> shifts) & 1).to(torch.float32)


def bits_to_int(bits: torch.Tensor) -> torch.Tensor:
    """[..., n_bits] {0,1} → [...] long. LSB 가 bit[0]."""
    n_bits = bits.size(-1)
    powers = (1 << torch.arange(n_bits, device=bits.device)).long()
    return (bits.long() * powers).sum(-1)


class TransformerStack(nn.Module):
    """nn.TransformerEncoderLayer 기반 스택. SDPA flash 자동 선택."""

    def __init__(self, n_layers: int = 14, d_model: int = 256, n_heads: int = 4,
                 d_ff: int = 1024, dropout: float = 0.1):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_ff, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)

    def forward(self, x: torch.Tensor, patch_pad_mask: torch.Tensor) -> torch.Tensor:
        return self.encoder(x, src_key_padding_mask=~patch_pad_mask)


class BinaryElectra(nn.Module):
    """Binary encoding ELECTRA — codec 우회 + per-bit BCE generator.

    Generator/Discriminator 가 bit_proj, pos_emb, emb_layer_norm 공유 (원 ELECTRA 구조).
    """

    def __init__(
        self,
        vocab_size: int = 153600,
        bbpe_bits: int = 18,
        embedding_size: int = 128,
        hidden_size: int = 256,
        n_heads: int = 4,
        d_ff: int = 1024,
        gen_layers: int = 14,
        disc_layers: int = 14,
        dropout: float = 0.1,
        max_patches: int = 512,
        gen_loss_weight: float = 50.0,
        mask_id: int | None = None,  # 기본 = vocab_size (vocab 밖, bbpe_bits 안)
        embedding_dim_k: int = 0,    # 0 → pure binary (variant A), k>0 → hybrid (C)
    ):
        super().__init__()
        assert (1 << bbpe_bits) >= vocab_size + 1, \
            f"bbpe_bits={bbpe_bits} → 2^{bbpe_bits}={1<<bbpe_bits} < vocab+1={vocab_size+1}"
        self.vocab_size = vocab_size
        self.bbpe_bits = bbpe_bits
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.gen_loss_weight = gen_loss_weight
        self.mask_id = mask_id if mask_id is not None else vocab_size
        self.embedding_dim_k = embedding_dim_k

        # ── 공유 embedding 계층 ──
        # binary bits (±1) + 선택적 compressed embedding lookup → embedding_size
        # variant A (k=0): input = bbpe_bits
        # variant C (k>0): input = bbpe_bits + k, Embedding(vocab+1, k) 추가
        # mask_id = vocab_size 용 슬롯 포함하려 size = vocab_size + 1
        if embedding_dim_k > 0:
            self.token_embedding = nn.Embedding(
                vocab_size + 1, embedding_dim_k, padding_idx=0,
            )
            input_dim = bbpe_bits + embedding_dim_k
        else:
            self.token_embedding = None
            input_dim = bbpe_bits
        self.bit_proj = nn.Linear(input_dim, embedding_size)
        self.pos_emb = nn.Embedding(max_patches, embedding_size)
        self.emb_layer_norm = nn.LayerNorm(embedding_size)
        self.emb_dropout = nn.Dropout(dropout)

        # ── Generator ──
        self.gen_hidden_proj = nn.Linear(embedding_size, hidden_size)
        self.generator = TransformerStack(
            n_layers=gen_layers, d_model=hidden_size,
            n_heads=n_heads, d_ff=d_ff, dropout=dropout,
        )
        # per-bit logit head — softmax(153600) 대체
        self.gen_head = nn.Linear(hidden_size, bbpe_bits)

        # ── Discriminator ──
        self.disc_hidden_proj = nn.Linear(embedding_size, hidden_size)
        self.discriminator = TransformerStack(
            n_layers=disc_layers, d_model=hidden_size,
            n_heads=n_heads, d_ff=d_ff, dropout=dropout,
        )
        self.disc_head = nn.Linear(hidden_size, 1)

        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if p.dim() >= 2:
                nn.init.normal_(p, mean=0.0, std=0.02)
            elif "bias" in name:
                nn.init.zeros_(p)
        # token_embedding 은 std=1/sqrt(k) 로 덮어씀 (compressed dim 에 맞춰 분산 조절)
        if self.token_embedding is not None:
            k = self.embedding_dim_k
            nn.init.normal_(self.token_embedding.weight, mean=0.0, std=1.0 / (k ** 0.5))
            self.token_embedding.weight.data[0].zero_()  # padding_idx

    # ─────────────────────────────────────────────
    def _embed_bits(self, ids: torch.Tensor, token_pad_mask: torch.Tensor) -> torch.Tensor:
        """ids[B, P] → emb[B, P, embedding_size]. binary ±1 (+선택적 embedding) → linear → +pos → LN → dropout."""
        B, P = ids.shape
        bits = int_to_bits(ids, self.bbpe_bits)    # [B, P, bits] in {0,1}
        bits = bits * 2.0 - 1.0                     # ±1 정규화
        if self.token_embedding is not None:
            # ids 는 vocab_size (mask_id) 까지 가능 — Embedding size = vocab_size + 1
            safe_ids = ids.clamp(max=self.vocab_size)
            emb = self.token_embedding(safe_ids)    # [B, P, k]
            x = torch.cat([bits, emb], dim=-1)      # [B, P, bits + k]
        else:
            x = bits
        e = self.bit_proj(x)
        positions = torch.arange(P, device=ids.device).unsqueeze(0).expand(B, -1)
        e = e + self.pos_emb(positions)
        e = self.emb_layer_norm(e)
        e = self.emb_dropout(e)
        e = e * token_pad_mask.unsqueeze(-1).to(e.dtype)
        return e

    def forward(
        self,
        bbpe_ids: torch.Tensor,           # [B, P] long — 원본
        token_pad_mask: torch.Tensor,     # [B, P] bool — 유효 토큰
        masked_bbpe_ids: torch.Tensor,    # [B, P] long — masked 위치는 mask_id
        masked_patch_mask: torch.Tensor,  # [B, P] bool — 마스킹된 토큰
    ) -> dict:
        B, P = bbpe_ids.shape
        device = bbpe_ids.device

        # ── (1) Generator path ──
        e_masked = self._embed_bits(masked_bbpe_ids, token_pad_mask)
        h_gen = self.gen_hidden_proj(e_masked)
        h_gen = self.generator(h_gen, token_pad_mask)            # [B, P, hidden]
        bit_logits = self.gen_head(h_gen)                         # [B, P, 18]

        # ── (2) Per-bit BCE loss (masked positions only) ──
        target_bits = int_to_bits(bbpe_ids, self.bbpe_bits)       # [B, P, 18]
        bce = F.binary_cross_entropy_with_logits(
            bit_logits, target_bits, reduction="none",
        )
        # bit dim 평균 → token dim 마스킹 평균
        bce_per_token = bce.mean(-1)                              # [B, P]
        denom = masked_patch_mask.sum().clamp(min=1)
        gen_loss = (bce_per_token * masked_patch_mask.float()).sum() / denom

        # ── (3) Sample corrupted ids ──
        with torch.no_grad():
            sampled_bits = (bit_logits > 0).long()                # [B, P, 18]
            sampled_ids = bits_to_int(sampled_bits)               # [B, P]
            # vocab 범위로 clamp (vocab+ 만큼은 무효 토큰)
            sampled_ids = sampled_ids.clamp(max=self.vocab_size - 1)
            corrupted_ids = torch.where(masked_patch_mask, sampled_ids, bbpe_ids)
            replaced = (corrupted_ids != bbpe_ids) & masked_patch_mask  # [B, P]

        # ── (4) Discriminator path ──
        e_corrupted = self._embed_bits(corrupted_ids, token_pad_mask)
        h_disc = self.disc_hidden_proj(e_corrupted)
        h_disc = self.discriminator(h_disc, token_pad_mask)
        disc_logits = self.disc_head(h_disc).squeeze(-1)          # [B, P]

        valid = token_pad_mask
        disc_loss = F.binary_cross_entropy_with_logits(
            disc_logits[valid], replaced[valid].float()
        )

        # ── (5) Metrics ──
        with torch.no_grad():
            disc_pred = disc_logits > 0
            disc_acc = ((disc_pred == replaced) & valid).sum().float() / valid.sum().clamp(min=1).float()
            replaced_rate = (replaced & valid).sum().float() / valid.sum().clamp(min=1).float()
            masked_tokens = masked_patch_mask.sum().float()
            patch_util = valid.sum().float() / (B * P)
            # generator token-level accuracy (masked 위치만)
            gen_correct = (sampled_ids == bbpe_ids) & masked_patch_mask
            gen_acc = gen_correct.sum().float() / masked_patch_mask.sum().clamp(min=1).float()
            # per-bit accuracy (masked 위치 평균)
            bit_pred = (bit_logits > 0).long()
            bit_correct = (bit_pred == target_bits.long()).float()  # [B, P, 18]
            bit_acc = (bit_correct * masked_patch_mask.unsqueeze(-1).float()).sum() / \
                      (masked_patch_mask.sum().clamp(min=1).float() * self.bbpe_bits)

        total_loss = disc_loss + self.gen_loss_weight * gen_loss
        return {
            "total_loss": total_loss,
            "gen_loss": gen_loss.detach(),
            "disc_loss": disc_loss.detach(),
            "disc_acc": disc_acc,
            "gen_acc": gen_acc,
            "bit_acc": bit_acc,
            "replaced_rate": replaced_rate,
            "masked_tokens": masked_tokens,
            "patch_util": patch_util,
        }


# ── smoke test ──
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=str,
                    default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    print("=== BinaryElectra smoke test (변종 A: pure binary, k=0) ===")
    model = BinaryElectra(
        vocab_size=153600, bbpe_bits=18,
        embedding_size=128, hidden_size=256, n_heads=4, d_ff=1024,
        gen_layers=14, disc_layers=14, max_patches=64,
        gen_loss_weight=50.0,
    ).to(args.device)

    total = sum(p.numel() for p in model.parameters())
    gen_params = sum(p.numel() for p in model.generator.parameters())
    disc_params = sum(p.numel() for p in model.discriminator.parameters())
    head_params = (model.gen_head.weight.numel() + model.gen_head.bias.numel() +
                   model.disc_head.weight.numel() + model.disc_head.bias.numel())
    emb_params = (model.bit_proj.weight.numel() + model.bit_proj.bias.numel() +
                  model.pos_emb.weight.numel() +
                  model.emb_layer_norm.weight.numel() + model.emb_layer_norm.bias.numel())
    print(f"params: total={total/1e6:.2f}M  "
          f"(emb={emb_params/1e3:.1f}K, gen={gen_params/1e6:.2f}M, "
          f"disc={disc_params/1e6:.2f}M, head={head_params}B)")

    # Dummy batch
    B, P = 2, 64
    dev = args.device
    bbpe_ids = torch.randint(0, 153600, (B, P), device=dev)
    token_pad_mask = torch.zeros(B, P, dtype=torch.bool, device=dev)
    token_pad_mask[:, :50] = True
    n_tokens = torch.tensor([50, 50], device=dev)

    # Mask 20%
    from koelectra.data.masking import make_patch_mask
    masked_patch_mask = make_patch_mask(n_tokens, max_patches=P, mask_ratio=0.20)
    masked_bbpe_ids = torch.where(masked_patch_mask, model.mask_id, bbpe_ids)

    out = model(bbpe_ids, token_pad_mask, masked_bbpe_ids, masked_patch_mask)
    for k, v in out.items():
        if torch.is_tensor(v):
            print(f"  {k}: {v.item():.4f}")

    out["total_loss"].backward()
    has_nan = any(p.grad is not None and p.grad.isnan().any()
                  for p in model.parameters())
    print(f"backward: {'FAIL (NaN)' if has_nan else 'OK'}")

    # bit encode/decode round-trip 검증
    test_ids = torch.tensor([0, 1, 4, 153599, 153600], device=dev)
    bits = int_to_bits(test_ids, 18)
    back = bits_to_int(bits.long())
    print(f"round-trip: {test_ids.tolist()} → {back.tolist()} "
          f"({'OK' if (test_ids == back).all() else 'FAIL'})")
