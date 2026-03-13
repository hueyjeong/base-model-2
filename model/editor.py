"""BitEditor — 비자기회귀 편집 태깅 모델

인코더-only 구조로, 입력 토큰마다 편집 태그를 예측한다.
RWKV-6 양방향 SSM + MoE BitNet FFN + Shared Linear Self-Attention.

구조:
    Embedding (vocab × d_model, BF16)
    ├── BitEditorLayer × n_rwkv_layers
    │   ├── BiRWKV (양방향 RWKV-6)
    │   ├── Dropout → RMSNorm + residual
    │   ├── MoE BitNetFFN (n_experts, top_k)
    │   └── Dropout → RMSNorm + residual
    │   (* 특정 레이어 뒤 Shared Linear Self-Attention 삽입)
    ├── Final RMSNorm
    └── Edit Tag Head (Linear: d_model → n_tags)
"""
import math

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from model.editor_config import BitEditorConfig
from model.encoder import RMSNorm
from model.bi_rwkv import BiRWKV
from model.moe import MoEBitNetFFN
from model.shared_attention import SharedLinearSelfAttention


class BitEditorLayer(nn.Module):
    """BitEditor 단일 레이어

    RMSNorm → BiRWKV → Dropout → (+residual)
    RMSNorm → MoE FFN → Dropout → (+residual)

    pre-norm 패턴 — gradient 안정성 확보 (deep network 필수).
    """

    def __init__(self, cfg: BitEditorConfig):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model, eps=cfg.rms_norm_eps)
        self.bi_rwkv = BiRWKV(cfg.d_model, cfg.n_heads, cfg.headdim)
        self.norm2 = RMSNorm(cfg.d_model, eps=cfg.rms_norm_eps)
        self.moe_ffn = MoEBitNetFFN(
            d_model=cfg.d_model,
            d_ff=cfg.d_ff,
            n_experts=cfg.n_experts,
            top_k=cfg.top_k,
            dropout=cfg.dropout,
        )
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(
        self, x: torch.Tensor, pad_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            (output, aux_loss)
        """
        # pre-norm → BiRWKV → residual
        x = x + self.dropout(self.bi_rwkv(self.norm1(x), pad_mask=pad_mask))

        # pre-norm → MoE FFN → residual
        normed = self.norm2(x)
        ffn_out, aux_loss = self.moe_ffn(normed)
        x = x + self.dropout(ffn_out)

        return x, aux_loss


class BitEditor(nn.Module):
    """BitEditor 메인 모델

    비자기회귀 편집 태깅: 각 입력 토큰에 대해 편집 태그(KEEP/DELETE/REPLACE/INSERT) 예측.
    Iterative refinement으로 복잡한 편집도 처리 가능.
    """

    def __init__(self, cfg: BitEditorConfig):
        super().__init__()
        self.cfg = cfg
        self.gradient_checkpointing = False

        # 임베딩 (BF16, sqrt(d_model) 스케일링)
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=cfg.pad_id)
        self.embed_scale = math.sqrt(cfg.d_model)
        self.embed_dropout = nn.Dropout(cfg.dropout)

        # BiRWKV + MoE 레이어 스택
        self.layers = nn.ModuleList([
            BitEditorLayer(cfg) for _ in range(cfg.n_rwkv_layers)
        ])

        # Shared Linear Self-Attention (1개 블록, 여러 삽입점에서 재사용)
        n_insertions = len(cfg.attn_insertion_points)
        self.shared_attn = SharedLinearSelfAttention(
            d_model=cfg.d_model,
            n_heads=cfg.n_attn_heads,
            n_insertion_points=n_insertions,
            lora_rank=cfg.lora_rank,
            dropout=cfg.dropout,
        )
        self.attn_norms = nn.ModuleList([
            RMSNorm(cfg.d_model, eps=cfg.rms_norm_eps)
            for _ in range(n_insertions)
        ])
        self.attn_dropouts = nn.ModuleList([
            nn.Dropout(cfg.dropout)
            for _ in range(n_insertions)
        ])
        self.attn_insertion_set = set(cfg.attn_insertion_points)

        # 삽입점 인덱스 매핑 (forward에서 반복 생성 회피)
        self._insertion_map = {pt: idx for idx, pt in enumerate(cfg.attn_insertion_points)}

        # Final norm + tag head
        self.final_norm = RMSNorm(cfg.d_model, eps=cfg.rms_norm_eps)
        self.tag_head = nn.Linear(cfg.d_model, cfg.n_tags)

        # 가중치 초기화
        self._init_weights()

    def _init_weights(self):
        """가중치 초기화"""
        # 임베딩
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        if self.cfg.pad_id is not None:
            nn.init.zeros_(self.embedding.weight[self.cfg.pad_id])

        # Tag head
        nn.init.xavier_uniform_(self.tag_head.weight)
        nn.init.zeros_(self.tag_head.bias)

        # BiRWKV LoRA zero-init
        for layer in self.layers:
            layer.bi_rwkv._init_weights()

    def forward(
        self,
        input_ids: torch.Tensor,
        pad_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            input_ids: (B, T) — 입력 토큰 ID
            pad_mask: (B, T) bool — True가 유효 데이터

        Returns:
            (tag_logits, total_aux_loss):
                tag_logits: (B, T, n_tags) — 편집 태그 logit
                total_aux_loss: scalar — MoE aux loss 합산
        """
        # 임베딩
        x = self.embedding(input_ids) * self.embed_scale
        x = self.embed_dropout(x)

        total_aux_loss = x.new_zeros(())

        # 레이어 순회
        for i, layer in enumerate(self.layers):
            if self.gradient_checkpointing and self.training:
                x, aux_loss = checkpoint(
                    layer, x, pad_mask, use_reentrant=False
                )
            else:
                x, aux_loss = layer(x, pad_mask=pad_mask)
            total_aux_loss = total_aux_loss + aux_loss

            # Shared Attention 삽입점 확인
            if i in self._insertion_map:
                ins_idx = self._insertion_map[i]
                residual = x
                attn_out = self.shared_attn(x, insertion_idx=ins_idx, pad_mask=pad_mask)
                attn_out = self.attn_dropouts[ins_idx](attn_out)
                x = self.attn_norms[ins_idx](residual + attn_out)

        # Final norm + tag head
        x = self.final_norm(x)
        tag_logits = self.tag_head(x.float())  # FP32 logits

        return tag_logits, total_aux_loss * self.cfg.aux_loss_weight

    def count_parameters(self) -> dict[str, int]:
        """파라미터 수 집계"""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)

        # 카테고리별 집계
        categories = {}
        for name, p in self.named_parameters():
            cat = name.split(".")[0]
            categories[cat] = categories.get(cat, 0) + p.numel()

        return {
            "total": total,
            "trainable": trainable,
            **categories,
        }

    def estimate_active_params(self) -> int:
        """토큰당 활성 파라미터 수 추정 (MoE top_k 기준)"""
        total = sum(p.numel() for p in self.parameters())

        # MoE expert 파라미터 = 전체 expert 파라미터 × (1 - top_k/n_experts)
        expert_params = 0
        for layer in self.layers:
            expert_params += sum(p.numel() for p in layer.moe_ffn.experts.parameters())

        inactive_ratio = 1.0 - self.cfg.top_k / self.cfg.n_experts
        inactive_params = int(expert_params * inactive_ratio)

        return total - inactive_params
