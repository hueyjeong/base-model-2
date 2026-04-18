"""HeadCodec — inter-slot interaction 없는 minimal codec.

각 슬롯은 token_vec + slot_pos 만 보고 자모 logits 직접 산출.
Conv/attention 둘 다 없음. BERT MLM head 스타일.

장점: 매우 빠름, 메모리 작음, embarrassingly parallel
단점: 토큰 내 자모 간 일관성 강제 신호 없음 → 학습 난이도 ↑
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class HeadCodec(nn.Module):
    def __init__(
        self,
        jamo_vocab: int = 330,
        d_model: int = 256,
        max_jamo: int = 32,
        dec_hidden: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.jamo_vocab = jamo_vocab
        self.d_model = d_model
        self.max_jamo = max_jamo
        self.pad_id = 0

        self.embedding = nn.Embedding(jamo_vocab, d_model, padding_idx=0)
        self.embed_scale = math.sqrt(d_model)
        # encoder/decoder 가 다른 slot pos (encoder 는 input position, decoder 는 output position)
        self.enc_pos = nn.Embedding(max_jamo, d_model)
        self.dec_pos = nn.Embedding(max_jamo, d_model)

        # Encoder: mean pool + linear (no spatial mixing)
        self.enc_proj = nn.Linear(d_model, d_model)

        # Decoder: per-slot MLP (no inter-slot)
        # token_vec + slot_pos → MLP(d → dec_hidden → d) → head(d → V)
        self.dec_mlp = nn.Sequential(
            nn.Linear(d_model, dec_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dec_hidden, d_model),
        )
        self.dec_norm = nn.LayerNorm(d_model)
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
        mf = mask.unsqueeze(-1).to(x.dtype)
        counts = mask.sum(dim=1, keepdim=True).clamp(min=1).to(x.dtype)
        pooled = (x * mf).sum(dim=1) / counts
        return self.enc_proj(pooled)

    def decode(self, token_vecs: torch.Tensor) -> torch.Tensor:
        """[T, D] → [T, max_jamo, V]. Per-slot 독립 (no inter-slot mixing)."""
        T, D = token_vecs.shape
        S = self.max_jamo
        x = token_vecs.unsqueeze(1).expand(-1, S, -1).contiguous()  # [T, S, D]
        pos_ids = torch.arange(S, device=x.device)
        x = x + self.dec_pos(pos_ids)
        # Residual MLP, per-slot
        x = self.dec_norm(x + self.dec_mlp(x))
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
    print("=== HeadCodec smoke ===")
    m = HeadCodec()
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
