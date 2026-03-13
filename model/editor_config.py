"""BitEditor 설정 (Config)

비자기회귀 편집 태깅 모델의 모든 하이퍼파라미터를 관리한다.
RWKV-6 양방향 SSM + MoE BitNet FFN + Shared Linear Self-Attention + Edit Tagging.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict


@dataclass
class BitEditorConfig:
    """BitEditor 모델 설정

    Attributes:
        d_model: 모델 히든 차원
        n_rwkv_layers: BiRWKV 레이어 수
        d_inner: RWKV 내부 차원 (= d_model, head 분할)
        d_ff: MoE expert FFN 중간 차원
        n_experts: MoE expert 수
        top_k: MoE top-k 라우팅
        n_heads: RWKV head 수
        headdim: RWKV head 차원
        n_attn_heads: Shared Linear Self-Attention head 수
        attn_insertion_points: Shared Attn 삽입 레이어 인덱스 (0-based)
        lora_rank: Shared Attn LoRA rank
        vocab_size: 토크나이저 어휘 크기
        n_tags: 편집 태그 수 (KEEP + DELETE + REPLACE_vocab + INSERT_vocab)
        max_seq_len: 최대 시퀀스 길이
        dropout: 드롭아웃 비율
        rms_norm_eps: RMSNorm epsilon
        pad_id: 패딩 토큰 ID
        bos_id: BOS 토큰 ID
        aux_loss_weight: MoE auxiliary loss 가중치
        n_iterations: iterative refinement 반복 횟수
    """
    # 모델 차원
    d_model: int = 384
    n_rwkv_layers: int = 10

    # RWKV-6 파라미터
    d_inner: int = 384          # = d_model (head 분할용)
    n_heads: int = 12           # RWKV head 수
    headdim: int = 32           # head 차원 (d_inner // n_heads)

    # MoE 파라미터
    d_ff: int = 512             # expert FFN 중간 차원
    n_experts: int = 16
    top_k: int = 1

    # Shared Linear Self-Attention 파라미터
    n_attn_heads: int = 24
    attn_insertion_points: tuple[int, ...] = (3, 7, 9)  # 0-based: layer 4, 8, 10 뒤
    lora_rank: int = 16

    # 토크나이저/시퀀스
    vocab_size: int = 303
    n_tags: int = 608           # KEEP(1) + DELETE(1) + REPLACE(303) + INSERT(303)
    max_seq_len: int = 2048

    # 정규화
    dropout: float = 0.1
    rms_norm_eps: float = 1e-6

    # 토큰 ID
    pad_id: int = 0
    bos_id: int = 1

    # 학습
    aux_loss_weight: float = 0.01
    n_iterations: int = 3

    def save(self, path: str) -> None:
        """설정을 JSON 파일로 저장"""
        d = asdict(self)
        # tuple → list for JSON
        d["attn_insertion_points"] = list(d["attn_insertion_points"])
        with open(path, "w", encoding="utf-8") as f:
            json.dump(d, f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "BitEditorConfig":
        """JSON 파일에서 설정 로드"""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # list → tuple
        if "attn_insertion_points" in data:
            data["attn_insertion_points"] = tuple(data["attn_insertion_points"])
        return cls(**data)

    def __post_init__(self):
        """파라미터 유효성 검증"""
        assert self.d_model > 0, "d_model must be positive"
        assert self.d_inner > 0, "d_inner must be positive"
        assert self.d_inner % self.n_heads == 0, \
            f"d_inner({self.d_inner}) must be divisible by n_heads({self.n_heads})"
        assert self.d_inner // self.n_heads == self.headdim, \
            f"headdim({self.headdim}) must equal d_inner//n_heads({self.d_inner // self.n_heads})"
        assert self.n_tags == 2 + 2 * self.vocab_size, \
            f"n_tags({self.n_tags}) must equal 2 + 2*vocab_size({2 + 2 * self.vocab_size})"
        for pt in self.attn_insertion_points:
            assert 0 <= pt < self.n_rwkv_layers, \
                f"attn_insertion_point {pt} out of range [0, {self.n_rwkv_layers})"
