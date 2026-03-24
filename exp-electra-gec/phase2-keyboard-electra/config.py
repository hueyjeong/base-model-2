"""Phase 2 Keyboard ELECTRA — 설정 dataclass

Discriminator (커스텀 인코더), Generator (소형 Transformer), ELECTRA RTD 설정.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field


@dataclass
class DiscriminatorConfig:
    """커스텀 Diamond 인코더 설정"""
    d_model: int = 256
    d_ff: int = 512
    vocab_size: int = 303
    n_tags: int = 608           # 2 + 2*vocab_size (GEC fine-tune용)
    max_seq_len: int = 4096

    # Attention
    n_heads: int = 4
    n_kv_heads: int = 2         # GQA (n_heads // n_kv_heads = repeat factor)
    headdim: int = 64           # d_model // n_heads

    # Conv1d 전처리
    conv_kernel: int = 4

    # ChunkFFT
    chunk_fft_size: int = 256

    # 정규화/드롭아웃
    dropout: float = 0.1
    rms_norm_eps: float = 1e-6

    # 토큰 ID (keyboard tokenizer)
    pad_id: int = 0
    bos_id: int = 2
    eos_id: int = 3
    mask_id: int = 6

    # 다이아몬드 레이어 사양
    # "fa" = Full Attention, "wa:N" = Window Attention(w=N)
    layer_spec: list[str] = field(default_factory=lambda: [
        "fa", "wa:64", "wa:32", "wa:64", "fa",
    ])

    def __post_init__(self):
        assert self.d_model > 0
        assert self.d_model == self.n_heads * self.headdim, \
            f"d_model({self.d_model}) != n_heads({self.n_heads}) * headdim({self.headdim})"
        assert self.n_heads % self.n_kv_heads == 0, \
            f"n_heads({self.n_heads}) must be divisible by n_kv_heads({self.n_kv_heads})"
        assert self.n_tags == 2 + 2 * self.vocab_size


@dataclass
class GeneratorConfig:
    """소형 Transformer Generator 설정 (FP16, INT8 QAT 미적용)"""
    d_model: int = 128
    n_layers: int = 4
    d_ff: int = 512
    vocab_size: int = 303
    n_heads: int = 2
    max_seq_len: int = 4096
    dropout: float = 0.1
    pad_id: int = 0


@dataclass
class ElectraConfig:
    """ELECTRA RTD 학습 설정"""
    disc: DiscriminatorConfig = field(default_factory=DiscriminatorConfig)
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
        disc = DiscriminatorConfig(**data.pop("disc"))
        gen = GeneratorConfig(**data.pop("gen"))
        return cls(disc=disc, gen=gen, **data)


# ── 프리셋 ──

def make_small_config(**overrides) -> ElectraConfig:
    """Small Discriminator (~8M) + Generator (~2M)"""
    disc = DiscriminatorConfig(
        d_model=256, d_ff=512,
        n_heads=4, n_kv_heads=2, headdim=64,
        layer_spec=["fa", "wa:64", "wa:32", "wa:64", "fa"],
    )
    gen = GeneratorConfig(d_model=128, n_layers=4, d_ff=512, n_heads=2)
    cfg = ElectraConfig(disc=disc, gen=gen)
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def make_128m_config(**overrides) -> ElectraConfig:
    """128M Discriminator + Generator (~2M)"""
    disc = DiscriminatorConfig(
        d_model=768, d_ff=2048,
        n_heads=12, n_kv_heads=4, headdim=64,
        layer_spec=[
            "fa",
            "wa:64", "wa:32", "wa:64", "wa:128",
            "wa:256",
            "wa:128", "wa:64", "wa:32", "wa:64",
            "fa",
        ],
    )
    gen = GeneratorConfig(d_model=128, n_layers=4, d_ff=512, n_heads=2)
    cfg = ElectraConfig(disc=disc, gen=gen)
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


if __name__ == "__main__":
    import tempfile, os

    print("=== Config Smoke Test ===\n")

    # Small 프리셋
    small = make_small_config()
    print(f"Small disc: d={small.disc.d_model}, layers={len(small.disc.layer_spec)}, "
          f"d_ff={small.disc.d_ff}, heads={small.disc.n_heads}")
    print(f"Small gen: d={small.gen.d_model}, layers={small.gen.n_layers}")
    print(f"  layer_spec: {small.disc.layer_spec}")

    # 128M 프리셋
    big = make_128m_config()
    print(f"\n128M disc: d={big.disc.d_model}, layers={len(big.disc.layer_spec)}, "
          f"d_ff={big.disc.d_ff}, heads={big.disc.n_heads}")
    print(f"  layer_spec: {big.disc.layer_spec}")

    # JSON round-trip
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
        path = f.name
    small.save(path)
    loaded = ElectraConfig.load(path)
    assert loaded.disc.d_model == small.disc.d_model
    assert loaded.disc.layer_spec == small.disc.layer_spec
    assert loaded.gen.d_model == small.gen.d_model
    assert loaded.mask_prob == small.mask_prob
    os.unlink(path)
    print("\nJSON round-trip OK")

    print("\n모든 테스트 통과!")
