"""모델 설정 (Config)

BitNet-Mamba Encoder-Decoder 모델의 모든 하이퍼파라미터를 관리한다.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class BitMambaSeq2SeqConfig:
    """BitNet-Mamba Seq2Seq 모델 설정

    Attributes:
        d_model: 모델 히든 차원 (너비)
        n_encoder_layers: 인코더 레이어 수
        n_decoder_layers: 디코더 레이어 수
        d_inner: Mamba 내부 확장 차원
        d_state: SSM 상태 차원
        d_conv: Mamba 1D conv 커널 크기
        dt_rank: Δ projection rank
        d_ff: BitNet FFN 중간 차원
        n_heads: Cross-attention 헤드 수
        vocab_size: 토크나이저 어휘 크기
        max_seq_len: 최대 시퀀스 길이
        dropout: 드롭아웃 비율
        tie_embeddings: 인코더/디코더 임베딩 공유 여부
        tie_lm_head: LM Head와 임베딩 weight tying 여부
        pad_id: 패딩 토큰 ID
        rms_norm_eps: RMSNorm epsilon
    """
    # 모델 차원
    d_model: int = 768
    n_encoder_layers: int = 6
    n_decoder_layers: int = 10

    # Mamba 파라미터
    d_inner: int = 1536       # 2 × d_model
    d_state: int = 16
    d_conv: int = 4
    dt_rank: int = 48         # d_model // 16

    # BitNet FFN 파라미터
    d_ff: int = 1280

    # Cross-Attention 파라미터
    n_heads: int = 12

    # 토크나이저/시퀀스
    vocab_size: int = 64000
    max_seq_len: int = 512

    # 정규화
    dropout: float = 0.1
    rms_norm_eps: float = 1e-6

    # 임베딩
    tie_embeddings: bool = True
    tie_lm_head: bool = True
    pad_id: int = 0

    def save(self, path: str) -> None:
        """설정을 JSON 파일로 저장"""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "BitMambaSeq2SeqConfig":
        """JSON 파일에서 설정 로드"""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)

    def __post_init__(self):
        """파라미터 유효성 검증"""
        assert self.d_model > 0, "d_model must be positive"
        assert self.d_model % self.n_heads == 0, \
            f"d_model({self.d_model}) must be divisible by n_heads({self.n_heads})"
        assert self.d_inner > 0, "d_inner must be positive"
        assert self.d_state > 0, "d_state must be positive"
