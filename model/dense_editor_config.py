"""DenseEditor 설정 (Config)

Dense (MoE 없음) 인코더-only 편집 태깅 모델의 하이퍼파라미터 관리.
mixing_type에 따라 depth 자동 계산 (128M 파라미터 타겟).
CPU 인퍼런스 벤치마크 결과 d_model=640이 스위트스팟.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field


@dataclass
class DenseEditorConfig:
    """DenseEditor 모델 설정"""
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
        valid_types = {"mamba", "fnet", "tcn", "rwkv", "retnet", "xlstm", "mlstm"}
        assert self.mixing_type in valid_types, \
            f"mixing_type '{self.mixing_type}'은 {valid_types} 중 하나여야 함"


# ── mixing layer별 projection 수 (양방향 기준) ──

# (방향당 proj 수, output proj 포함 여부)
MIXING_PROJ_COUNT: dict[str, int] = {
    "fnet": 0,     # mixing 파라미터 없음
    "tcn": 1,      # 1 pointwise proj (depthwise는 작음)
    "rwkv": 5,     # r,k,v,o,g (양방향 각각)
    "retnet": 5,   # q,k,v,o,g
    "mamba": 0,    # 특수 (in_proj 2x width)
    "xlstm": 4,    # i,f,z,o
    "mlstm": 5,    # q,k,v,i,f
}


def calc_layer_params(d_model: int, mixing_type: str) -> tuple[int, int, int]:
    """아키텍처별 레이어당 파라미터 수 계산

    Returns:
        (mix_params, ffn_params, total_per_layer)
    """
    d = d_model
    # 전 아키텍처 동일 FFN 비율 (8/3 ≈ 2.66x)
    # FNet은 mixing=0이므로 FFN을 키우는 대신 레이어를 늘려 FFT 반복이 더 효과적
    dff = int(d * 8 / 3)

    nmix = MIXING_PROJ_COUNT[mixing_type]
    if mixing_type == "mamba":
        di = d * 2  # expand=2
        ds = 16
        dtr = max(d // 16, 1)
        # 양방향: 2 × (in_proj + out_proj + x_proj + dt_proj)
        mix_params = 2 * (d * 2 * di + di * d + (dtr + 2 * ds) * di + dtr * di)
    elif mixing_type == "tcn":
        mix_params = d * 7 * 6 + d * d  # 6 depthwise + 1 pointwise
    else:
        # 양방향: 2 × nmix × d² + output d²
        mix_params = 2 * nmix * d * d + d * d

    ffn_params = 3 * d * dff  # gate + up + down
    total = mix_params + ffn_params + 2 * d  # + norms
    return mix_params, ffn_params, total


def calc_n_layers(d_model: int, mixing_type: str, target_params: int = 128_000_000) -> int:
    """128M 파라미터 타겟에 맞는 레이어 수 계산"""
    overhead = 303 * d_model + 608 * d_model + d_model  # embedding + tag_head + norm
    _, _, per_layer = calc_layer_params(d_model, mixing_type)
    return max(1, (target_params - overhead) // per_layer)


def make_config(
    mixing_type: str,
    d_model: int = 640,
    target_params: int = 128_000_000,
    **overrides,
) -> DenseEditorConfig:
    """d_model과 mixing_type으로 128M 설정 자동 생성"""
    n_layers = calc_n_layers(d_model, mixing_type, target_params)
    dff = int(d_model * 8 / 3)
    n_heads = d_model // 32
    headdim = 32

    kwargs = dict(
        d_model=d_model,
        n_layers=n_layers,
        d_ff=dff,
        mixing_type=mixing_type,
        n_heads=n_heads,
        headdim=headdim,
    )
    kwargs.update(overrides)
    return DenseEditorConfig(**kwargs)


# 하위 호환: d=256 프리셋
def make_preset(mixing_type: str) -> DenseEditorConfig:
    """d=256 128M 프리셋 (하위 호환)"""
    return make_config(mixing_type, d_model=256)
