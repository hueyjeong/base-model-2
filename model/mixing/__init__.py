"""Mixing Layer 레지스트리

mixing_type 문자열로 적절한 mixing layer 클래스를 생성한다.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from model.dense_editor_config import DenseEditorConfig
    from model.mixing.base import MixingLayer


def create_mixing_layer(cfg: "DenseEditorConfig") -> "MixingLayer":
    """설정에 따라 적절한 mixing layer 인스턴스 생성"""
    t = cfg.mixing_type

    if t == "rwkv":
        from model.mixing.rwkv_wrap import BiRWKVMixing
        return BiRWKVMixing(cfg)
    elif t == "fnet":
        from model.mixing.fnet import FNetMixing
        return FNetMixing(cfg)
    elif t == "tcn":
        from model.mixing.tcn import TCNMixing
        return TCNMixing(cfg)
    elif t == "retnet":
        from model.mixing.retnet import BiRetentionMixing
        return BiRetentionMixing(cfg)
    elif t == "mamba":
        from model.mixing.bi_mamba import BiMambaMixing
        return BiMambaMixing(cfg)
    elif t == "mamba2":
        from model.mixing.bi_mamba2 import BiMamba2Mixing
        return BiMamba2Mixing(cfg)
    elif t == "xlstm":
        from model.mixing.xlstm import BiSLSTMMixing
        return BiSLSTMMixing(cfg)
    elif t == "xlstm_mamba":
        from model.mixing.xlstm import BiSLSTMMambaMixing
        return BiSLSTMMambaMixing(cfg)
    elif t == "mlstm":
        from model.mixing.mlstm import BiMLSTMMixing
        return BiMLSTMMixing(cfg)
    elif t == "attention":
        from model.mixing.full_attention import FullAttentionMixing
        return FullAttentionMixing(cfg)
    else:
        raise ValueError(f"알 수 없는 mixing_type: {t}")
