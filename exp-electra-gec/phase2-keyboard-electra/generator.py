"""TransformerGenerator — ELECTRA Generator (소형 Transformer + MLM Head)

FP16/BF16 학습, INT8 QAT 미적용.
nn.TransformerEncoder 사용 (표준 구현).
~2M 파라미터 (d=128, 4L, d_ff=512).
"""
import torch
import torch.nn as nn
from torch import Tensor

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from config import GeneratorConfig


class TransformerGenerator(nn.Module):
    """ELECTRA Generator — 소형 Transformer Encoder + MLM Head"""

    def __init__(self, cfg: GeneratorConfig):
        super().__init__()
        self.cfg = cfg

        self.embedding = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=cfg.pad_id)
        self.pos_embedding = nn.Embedding(cfg.max_seq_len, cfg.d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.d_ff,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=cfg.n_layers)
        self.mlm_head = nn.Linear(cfg.d_model, cfg.vocab_size)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.embedding.weight, std=0.02)
        nn.init.normal_(self.pos_embedding.weight, std=0.02)
        if self.cfg.pad_id is not None:
            nn.init.zeros_(self.embedding.weight[self.cfg.pad_id])
        nn.init.xavier_uniform_(self.mlm_head.weight)
        nn.init.zeros_(self.mlm_head.bias)

    def forward(self, input_ids: Tensor, pad_mask: Tensor | None = None) -> Tensor:
        """
        Args:
            input_ids: (B, T)
            pad_mask: (B, T) bool — True=유효

        Returns:
            mlm_logits: (B, T, vocab_size)
        """
        B, T = input_ids.shape
        pos = torch.arange(T, device=input_ids.device)
        x = self.embedding(input_ids) + self.pos_embedding(pos)

        # nn.TransformerEncoder: src_key_padding_mask에서 True=무시
        src_key_pad = ~pad_mask if pad_mask is not None else None
        x = self.encoder(x, src_key_padding_mask=src_key_pad)

        return self.mlm_head(x)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


if __name__ == "__main__":
    print("=== TransformerGenerator Smoke Test ===\n")

    cfg = GeneratorConfig(d_model=128, n_layers=4, d_ff=512, n_heads=2)
    gen = TransformerGenerator(cfg)
    n_params = gen.count_parameters()
    print(f"파라미터: {n_params:,} ({n_params/1e6:.2f}M)")

    # Forward
    B, T = 2, 64
    ids = torch.randint(1, 303, (B, T))
    mask = torch.ones(B, T, dtype=torch.bool)
    mask[0, T-5:] = False

    logits = gen(ids, mask)
    assert logits.shape == (B, T, 303), f"shape: {logits.shape}"
    print(f"Forward OK: {logits.shape}")

    # Backward
    logits.sum().backward()
    assert gen.embedding.weight.grad is not None
    print("Backward OK")

    # 파라미터 상세
    for name, p in gen.named_parameters():
        if "." not in name or name.count(".") <= 1:
            print(f"  {name}: {p.shape} ({p.numel():,})")

    print("\n모든 테스트 통과!")
