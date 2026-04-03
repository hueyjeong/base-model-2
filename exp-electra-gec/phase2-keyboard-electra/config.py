"""Phase 2 Keyboard ELECTRA — BiMamba2 기반 설정

Discriminator: DenseEditorConfig (128M, BiMamba2, INT8 QAT)
Generator: GeneratorConfig (32M, BiMamba2, BF16)
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from model.dense_editor_config import DenseEditorConfig


@dataclass
class GeneratorConfig:
    """Transformer Generator 설정 (INT8 QAT)

    nn.TransformerEncoder + Int8Linear MLM head.
    Mamba2 fused CUDA kernel과 충돌 방지를 위해 Transformer 사용.
    """
    d_model: int = 256
    n_layers: int = 6
    d_ff: int = 1024
    vocab_size: int = 303
    n_heads: int = 4
    max_seq_len: int = 4096
    dropout: float = 0.1

    # 토큰 ID
    pad_id: int = 0
    bos_id: int = 2
    mask_id: int = 6


@dataclass
class ElectraConfig:
    """ELECTRA RTD 학습 설정"""
    disc: DenseEditorConfig = field(default_factory=DenseEditorConfig)
    gen: GeneratorConfig = field(default_factory=GeneratorConfig)

    mask_prob: float = 0.15         # 마스킹 비율
    disc_loss_weight: float = 50.0  # Discriminator loss 가중치
    gen_loss_weight: float = 1.0    # Generator loss 가중치
    temperature: float = 1.0        # Generator 샘플링 온도

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> ElectraConfig:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        disc = DenseEditorConfig(**data.pop("disc"))
        gen = GeneratorConfig(**data.pop("gen"))
        return cls(disc=disc, gen=gen, **data)


def make_electra_config(**overrides) -> ElectraConfig:
    """128M Disc + 32M Gen BiMamba2 프리셋

    Discriminator: d=640, 15L, d_ff=1707, BiMamba2, INT8 QAT (~127M)
    Generator: d=384, 10L, d_ff=1024, BiMamba2, BF16 (~31M)
    """
    disc = DenseEditorConfig(
        d_model=768,
        n_layers=20,
        d_ff=2048,
        vocab_size=303,
        n_tags=608,                 # 2 + 2×303 (GEC fine-tune용, RTD 시 미사용)
        max_seq_len=4096,
        mixing_type="attention",
        int8_qat=True,
        n_heads=24,
        headdim=32,
        attn_n_kv_heads=4,          # GQA
        dropout=0.1,
    )

    gen = GeneratorConfig(
        d_model=768,
        n_layers=4,
        d_ff=3072,
        n_heads=12,
        vocab_size=303,
        max_seq_len=4096,
    )

    cfg = ElectraConfig(disc=disc, gen=gen)
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


# 하위 호환 (pretrain_rtd.py --preset 인자 대응)
make_small_config = make_electra_config
make_base_config = make_electra_config
make_large_config = make_electra_config


if __name__ == "__main__":
    import tempfile

    print("=== Config Smoke Test ===\n")

    cfg = make_electra_config()
    print(f"Disc: d={cfg.disc.d_model}, layers={cfg.disc.n_layers}, "
          f"d_ff={cfg.disc.d_ff}, mixing={cfg.disc.mixing_type}, "
          f"int8_qat={cfg.disc.int8_qat}")
    print(f"Gen: d={cfg.gen.d_model}, layers={cfg.gen.n_layers}, "
          f"d_ff={cfg.gen.d_ff}")
    print(f"mask_prob={cfg.mask_prob}, disc_weight={cfg.disc_loss_weight}, "
          f"temp={cfg.temperature}")

    # JSON round-trip
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
        path = f.name
    cfg.save(path)
    loaded = ElectraConfig.load(path)
    assert loaded.disc.d_model == cfg.disc.d_model
    assert loaded.disc.mixing_type == cfg.disc.mixing_type
    assert loaded.disc.int8_qat == cfg.disc.int8_qat
    assert loaded.gen.d_model == cfg.gen.d_model
    assert loaded.gen.n_heads == cfg.gen.n_heads
    assert loaded.mask_prob == cfg.mask_prob
    os.unlink(path)
    print("\nJSON round-trip OK")

    print("\n모든 테스트 통과!")
