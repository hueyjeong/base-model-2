"""SACodec — single self-attention block 기반 per-token codec.

SimpleCodec (5 conv enc + 5 conv dec) 대비:
- Encoder: 1 transformer block (SDPA → Flash backend 자동)
- Decoder: 1 transformer block
- 동일 인터페이스: encode(jamo_ids[T, S], mask[T, S]) → z[T, D],
                  decode(z[T, D]) → logits[T, S, V]
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SACodec(nn.Module):
    def __init__(
        self,
        jamo_vocab: int = 330,
        d_model: int = 256,
        n_heads: int = 8,
        d_ff: int = 1024,
        max_jamo: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.jamo_vocab = jamo_vocab
        self.d_model = d_model
        self.max_jamo = max_jamo
        self.pad_id = 0

        self.embedding = nn.Embedding(jamo_vocab, d_model, padding_idx=0)
        self.embed_scale = math.sqrt(d_model)
        self.enc_pos = nn.Embedding(max_jamo, d_model)

        self.enc_attn = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_ff, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.enc_pool_proj = nn.Linear(d_model, d_model)

        self.dec_upsample = nn.Linear(d_model, d_model)
        self.dec_pos = nn.Embedding(max_jamo, d_model)
        self.dec_attn = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_ff, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.head = nn.Linear(d_model, jamo_vocab)

        nn.init.normal_(self.embedding.weight, std=1.0 / math.sqrt(d_model))
        self.embedding.weight.data[0].zero_()
        nn.init.normal_(self.enc_pos.weight, std=0.02)
        nn.init.normal_(self.dec_pos.weight, std=0.02)

    def encode(self, jamo_ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """[T, S] → [T, D]."""
        T, S = jamo_ids.shape
        x = self.embedding(jamo_ids) * self.embed_scale
        pos_ids = torch.arange(S, device=x.device)
        x = x + self.enc_pos(pos_ids)

        # key_padding_mask: True 가 무시
        kpm = ~mask
        x = self.enc_attn(x, src_key_padding_mask=kpm)

        mf = mask.unsqueeze(-1).to(x.dtype)
        counts = mask.sum(dim=1, keepdim=True).clamp(min=1).to(x.dtype)
        pooled = (x * mf).sum(dim=1) / counts
        return self.enc_pool_proj(pooled)

    def decode(self, token_vecs: torch.Tensor) -> torch.Tensor:
        """[T, D] → [T, max_jamo, V]."""
        T, D = token_vecs.shape
        S = self.max_jamo
        x = token_vecs.unsqueeze(1).expand(-1, S, -1).contiguous()
        x = self.dec_upsample(x)
        pos_ids = torch.arange(S, device=x.device)
        x = x + self.dec_pos(pos_ids)
        x = self.dec_attn(x)  # 전 슬롯 활성
        return self.head(x)

    def forward(self, jamo_ids, mask):
        z = self.encode(jamo_ids, mask)
        logits = self.decode(z)
        target = jamo_ids.clone()
        target[~mask] = self.pad_id
        loss = F.cross_entropy(
            logits.reshape(-1, self.jamo_vocab),
            target.reshape(-1),
        )
        with torch.no_grad():
            pred = logits.argmax(-1)
            acc = (pred == target).float().mean()
        return {"logits": logits, "loss": loss, "z": z, "acc": acc}

    @torch.no_grad()
    def decode_from_vec(self, token_vec: torch.Tensor):
        if token_vec.dim() == 1:
            token_vec = token_vec.unsqueeze(0)
        logits = self.decode(token_vec)
        preds = logits.argmax(dim=-1)
        results = []
        for t in range(preds.size(0)):
            seq = []
            for j in preds[t].tolist():
                if j == self.pad_id:
                    break
                seq.append(j)
            results.append(seq)
        return results


if __name__ == "__main__":
    print("=== SACodec smoke ===")
    m = SACodec()
    print(f"params: {sum(p.numel() for p in m.parameters())/1e6:.2f}M")
    T = 100
    j = torch.randint(1, 330, (T, 32))
    mask = torch.zeros(T, 32, dtype=torch.bool)
    for t in range(T):
        L = torch.randint(3, 21, (1,)).item()
        mask[t, :L] = True
    out = m(j, mask)
    print(f"loss={out['loss'].item():.4f}, acc={out['acc'].item():.3f}")
    print(f"z shape={out['z'].shape}")
