"""DiamondEncoder — Conv1d + ChunkFFT 전처리 → 다이아몬드 FA/WA 스택

Embedding → Conv1d → ChunkFFT → AttentionLayer × N → RMSNorm.
layer_spec으로 FA/WA 구성을 유연하게 지정.

forward()는 hidden states를 반환 (head 없음).
RTD head, Tag head는 ElectraRTD 또는 fine-tune 코드에서 추가.
"""
import math

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.checkpoint import checkpoint

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from model.encoder import RMSNorm

from config import DiscriminatorConfig
from conv1d_preprocess import Conv1dPreprocess
from chunk_fft import ChunkFFT
from attention_layer import AttentionLayer


def _parse_layer_spec(spec: str) -> tuple[str, int | None]:
    """레이어 사양 문자열 파싱

    "fa" → ("fa", None)
    "wa:64" → ("wa", 64)
    """
    if spec == "fa":
        return "fa", None
    if spec.startswith("wa:"):
        return "wa", int(spec.split(":")[1])
    raise ValueError(f"알 수 없는 layer spec: {spec}")


class DiamondEncoder(nn.Module):
    """커스텀 Diamond 인코더

    Args:
        cfg: DiscriminatorConfig
    """

    def __init__(self, cfg: DiscriminatorConfig):
        super().__init__()
        self.cfg = cfg
        self.gradient_checkpointing = False

        # 임베딩
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=cfg.pad_id)
        self.embed_scale = math.sqrt(cfg.d_model)
        self.embed_dropout = nn.Dropout(cfg.dropout)

        # Conv1d + ChunkFFT 전처리
        self.conv_preprocess = Conv1dPreprocess(
            cfg.d_model, kernel_size=cfg.conv_kernel, eps=cfg.rms_norm_eps,
        )
        self.chunk_fft = ChunkFFT(
            cfg.d_model, chunk_size=cfg.chunk_fft_size, eps=cfg.rms_norm_eps,
        )

        # 다이아몬드 레이어 스택
        self.layers = nn.ModuleList()
        for spec in cfg.layer_spec:
            _, window_size = _parse_layer_spec(spec)
            self.layers.append(AttentionLayer(
                d_model=cfg.d_model,
                d_ff=cfg.d_ff,
                n_heads=cfg.n_heads,
                n_kv_heads=cfg.n_kv_heads,
                headdim=cfg.headdim,
                max_seq_len=cfg.max_seq_len,
                window_size=window_size,
                dropout=cfg.dropout,
                eps=cfg.rms_norm_eps,
            ))

        # Final norm
        self.final_norm = RMSNorm(cfg.d_model, eps=cfg.rms_norm_eps)

        self._init_weights()

    def _init_weights(self):
        """가중치 초기화"""
        embed_std = 1.0 / math.sqrt(self.cfg.d_model)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=embed_std)
        if self.cfg.pad_id is not None:
            nn.init.zeros_(self.embedding.weight[self.cfg.pad_id])

        for layer in self.layers:
            layer._init_weights()

    def forward(
        self,
        input_ids: Tensor,
        pad_mask: Tensor | None = None,
    ) -> Tensor:
        """
        Args:
            input_ids: (B, T) 토큰 ID
            pad_mask: (B, T) bool — True=유효

        Returns:
            hidden: (B, T, D) — 최종 hidden states (head 없음)
        """
        x = self.embedding(input_ids) * self.embed_scale
        x = self.embed_dropout(x)

        # 문서 경계 감지 (패킹 시 BOS 위치에서 격리)
        reset_mask = (input_ids == self.cfg.bos_id)

        # Conv1d + ChunkFFT 전처리
        x = self.conv_preprocess(x, pad_mask)
        x = self.chunk_fft(x, pad_mask)

        # 다이아몬드 레이어 스택
        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                x = checkpoint(layer, x, pad_mask, reset_mask, use_reentrant=False)
            else:
                x = layer(x, pad_mask=pad_mask, reset_mask=reset_mask)

        return self.final_norm(x)

    def count_parameters(self) -> dict[str, int]:
        """파라미터 수 집계"""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)

        categories = {}
        for name, p in self.named_parameters():
            cat = name.split(".")[0]
            categories[cat] = categories.get(cat, 0) + p.numel()

        return {"total": total, "trainable": trainable, **categories}


if __name__ == "__main__":
    print("=== DiamondEncoder Smoke Test ===\n")

    # ── Small 모델 ──
    from config import make_small_config
    small_cfg = make_small_config()
    enc_small = DiamondEncoder(small_cfg.disc)
    counts = enc_small.count_parameters()
    print(f"Small (d={small_cfg.disc.d_model}):")
    print(f"  총 파라미터: {counts['total']:,}")
    for k, v in counts.items():
        if k not in ("total", "trainable"):
            print(f"  {k}: {v:,}")

    # Forward
    B, T = 2, 128
    ids = torch.randint(1, 303, (B, T))
    ids[:, 0] = small_cfg.disc.bos_id
    mask = torch.ones(B, T, dtype=torch.bool)
    mask[0, T-10:] = False

    out = enc_small(ids, mask)
    assert out.shape == (B, T, small_cfg.disc.d_model), f"shape: {out.shape}"
    assert out[0, T-1].abs().sum().item() < 1e-6, "PAD not zero"
    print(f"  Forward OK: {out.shape}")

    out.sum().backward()
    assert enc_small.embedding.weight.grad is not None
    print("  Backward OK")

    # ── 128M 모델 ──
    from config import make_128m_config
    big_cfg = make_128m_config()
    enc_big = DiamondEncoder(big_cfg.disc)
    counts_big = enc_big.count_parameters()
    print(f"\n128M (d={big_cfg.disc.d_model}):")
    print(f"  총 파라미터: {counts_big['total']:,} ({counts_big['total']/1e6:.1f}M)")
    for k, v in counts_big.items():
        if k not in ("total", "trainable"):
            print(f"  {k}: {v:,}")

    # Forward (작은 seq로 메모리 절약)
    ids = torch.randint(1, 303, (1, 32))
    ids[:, 0] = big_cfg.disc.bos_id
    out = enc_big(ids)
    assert out.shape == (1, 32, 768)
    print(f"  Forward OK: {out.shape}")

    out.sum().backward()
    print("  Backward OK")

    # Gradient checkpointing
    enc_small.gradient_checkpointing = True
    ids = torch.randint(1, 303, (2, 64))
    ids[:, 0] = small_cfg.disc.bos_id
    out = enc_small(ids)
    out.sum().backward()
    print("\n  Gradient checkpointing OK")

    print("\n모든 테스트 통과!")
