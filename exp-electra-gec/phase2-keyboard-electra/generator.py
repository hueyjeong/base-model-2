"""TransformerGenerator — ELECTRA Generator (Transformer + INT8 QAT + MLM Head)

nn.TransformerEncoder 기반, Int8Linear MLM head.
Mamba2 fused CUDA kernel과 workspace 충돌 방지를 위해 Transformer 사용.
"""
import torch
import torch.nn as nn
from torch import Tensor

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.dirname(__file__))

from config import GeneratorConfig
class TransformerGenerator(nn.Module):
    """ELECTRA Generator — Transformer Encoder + Int8Linear MLM Head"""

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

        self.gradient_checkpointing = False
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.embedding.weight, std=0.02)
        nn.init.normal_(self.pos_embedding.weight, std=0.02)
        if self.cfg.pad_id is not None:
            nn.init.zeros_(self.embedding.weight[self.cfg.pad_id])

    def forward(
        self, input_ids: Tensor, pad_mask: Tensor | None = None,
        return_hidden: bool = False,
    ) -> Tensor:
        """
        Args:
            input_ids: (B, T)
            pad_mask: (B, T) bool — True=유효
            return_hidden: True면 hidden states 반환 (MLM head 미적용)

        Returns:
            return_hidden=False: mlm_logits (B, T, vocab_size)
            return_hidden=True: hidden (B, T, d_model)
        """
        B, T = input_ids.shape
        pos = torch.arange(T, device=input_ids.device)
        x = self.embedding(input_ids) + self.pos_embedding(pos)

        # nn.TransformerEncoder: src_key_padding_mask에서 True=무시
        src_key_pad = ~pad_mask if pad_mask is not None else None
        x = self.encoder(x, src_key_padding_mask=src_key_pad)

        if return_hidden:
            return x
        return self.mlm_head(x)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


if __name__ == "__main__":
    print("=== TransformerGenerator Smoke Test ===\n")

    cfg = GeneratorConfig(d_model=256, n_layers=6, d_ff=1024, n_heads=4)
    gen = TransformerGenerator(cfg)
    n_params = gen.count_parameters()
    print(f"파라미터: {n_params:,} ({n_params/1e6:.2f}M)")

    # Forward
    B, T = 2, 64
    ids = torch.randint(7, 303, (B, T))
    ids[:, 0] = cfg.bos_id
    mask = torch.ones(B, T, dtype=torch.bool)
    mask[0, T-5:] = False

    logits = gen(ids, mask)
    assert logits.shape == (B, T, 303), f"shape: {logits.shape}"
    print(f"Forward OK: {logits.shape}")

    # return_hidden
    hidden = gen(ids, mask, return_hidden=True)
    assert hidden.shape == (B, T, 256), f"hidden shape: {hidden.shape}"
    print(f"Hidden OK: {hidden.shape}")

    # Backward
    logits.sum().backward()
    assert gen.embedding.weight.grad is not None
    print("Backward OK")

    print("\n모든 테스트 통과!")
