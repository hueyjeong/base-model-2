"""DenseEditor 설정 (Config)

Dense (MoE 없음) 인코더-only 편집 태깅 모델의 하이퍼파라미터 관리.
d_model=256 고정 (L1 캐시 적중), mixing_type에 따라 depth 가변.
CPU 인퍼런스 최적화 목적.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field


@dataclass
class DenseEditorConfig:
    """DenseEditor 모델 설정

    Attributes:
        d_model: 모델 히든 차원 (256 고정 — 64KB weight → L1 적중)
        n_layers: 레이어 수 (mixing_type별 가변)
        d_ff: BitNetFFN 중간 차원
        vocab_size: 토크나이저 어휘 크기
        n_tags: 편집 태그 수 (2 + 2*vocab_size)
        max_seq_len: 최대 시퀀스 길이
        mixing_type: mixing layer 종류
    """
    # 모델 차원
    d_model: int = 256
    n_layers: int = 46
    d_ff: int = 682          # d_model * 8/3 (SwiGLU 최적)

    # 토크나이저/시퀀스
    vocab_size: int = 303
    n_tags: int = 608         # 2 + 2*303
    max_seq_len: int = 2048

    # 정규화/드롭아웃
    dropout: float = 0.1
    rms_norm_eps: float = 1e-6

    # 토큰 ID
    pad_id: int = 0
    bos_id: int = 2

    # Mixing layer 선택
    mixing_type: str = "rwkv"  # mamba|fnet|tcn|rwkv|retnet|xlstm

    # Mixing 공통 (RWKV, Mamba, RetNet, xLSTM)
    n_heads: int = 8
    headdim: int = 32          # d_model // n_heads

    # Mamba 전용
    mamba_d_state: int = 16
    mamba_d_conv: int = 4
    mamba_expand: int = 2

    # TCN 전용
    tcn_kernel_size: int = 7
    tcn_n_dilations: int = 6   # dilation: [1, 2, 4, 8, 16, 32]

    # RetNet 전용
    retnet_gamma_min: float = 0.8
    retnet_gamma_max: float = 0.999

    def save(self, path: str) -> None:
        """설정을 JSON 파일로 저장"""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "DenseEditorConfig":
        """JSON 파일에서 설정 로드"""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)

    def __post_init__(self):
        """파라미터 유효성 검증"""
        assert self.d_model > 0, "d_model은 양수여야 함"
        assert self.d_model % self.n_heads == 0, \
            f"d_model({self.d_model})은 n_heads({self.n_heads})로 나누어떨어져야 함"
        assert self.d_model // self.n_heads == self.headdim, \
            f"headdim({self.headdim})은 d_model//n_heads({self.d_model // self.n_heads})이어야 함"
        assert self.n_tags == 2 + 2 * self.vocab_size, \
            f"n_tags({self.n_tags})는 2 + 2*vocab_size({2 + 2 * self.vocab_size})이어야 함"
        valid_types = {"mamba", "fnet", "tcn", "rwkv", "retnet", "xlstm"}
        assert self.mixing_type in valid_types, \
            f"mixing_type '{self.mixing_type}'은 {valid_types} 중 하나여야 함"


# ── 128M 파라미터 프리셋 (d_model=256 고정, depth 가변) ──

DENSE_128M_PRESETS: dict[str, dict] = {
    # per-layer: ~1.57M (FFN만, mixing 파라미터 없음)
    "fnet": dict(
        d_model=256, n_layers=80, d_ff=2048,
        mixing_type="fnet",
    ),
    # per-layer: ~601K (depthwise conv + pointwise BitLinear + FFN)
    "tcn": dict(
        d_model=256, n_layers=213, d_ff=682,
        mixing_type="tcn",
        tcn_kernel_size=7, tcn_n_dilations=6,
    ),
    # per-layer: ~1.40M → 128M/1.40M ≈ 91 layers
    "mamba": dict(
        d_model=256, n_layers=91, d_ff=682,
        mixing_type="mamba",
        mamba_d_state=16, mamba_d_conv=4, mamba_expand=2,
    ),
    # per-layer: ~1.20M → 128M/1.20M ≈ 107 layers
    "rwkv": dict(
        d_model=256, n_layers=107, d_ff=682,
        mixing_type="rwkv",
    ),
    # per-layer: ~1.18M → 128M/1.18M ≈ 108 layers
    "retnet": dict(
        d_model=256, n_layers=108, d_ff=682,
        mixing_type="retnet",
    ),
    # per-layer: ~1.05M → 128M/1.05M ≈ 122 layers
    "xlstm": dict(
        d_model=256, n_layers=122, d_ff=682,
        mixing_type="xlstm",
    ),
}


def make_preset(mixing_type: str) -> DenseEditorConfig:
    """미리 정의된 128M 프리셋에서 설정 생성"""
    if mixing_type not in DENSE_128M_PRESETS:
        raise ValueError(f"알 수 없는 프리셋: {mixing_type}. 가능: {list(DENSE_128M_PRESETS)}")
    return DenseEditorConfig(**DENSE_128M_PRESETS[mixing_type])
