"""DenseEditor — Dense (MoE 없음) 인코더-only 편집 태깅 모델

교체 가능한 mixing layer + Dense BitNetFFN 구조.
CPU 인퍼런스 최적화 목적, d_model=256 (L1 캐시 적중).

구조:
    Embedding (vocab × d_model)
    ├── DenseEditorLayer × n_layers
    │   ├── RMSNorm → MixingLayer → (+residual)
    │   └── RMSNorm → BitNetFFN → (+residual)
    ├── Final RMSNorm
    └── Tag Head (BitLinear: d_model → n_tags)
"""
import math

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from model.dense_editor_config import DenseEditorConfig
from model.encoder import RMSNorm, BitNetFFN, Int8FFN, SwiGLUFFN
from model.bitlinear import BitLinear, Int8Linear
from model.mixing import create_mixing_layer


class DenseEditorLayer(nn.Module):
    """DenseEditor 단일 레이어

    pre-norm 패턴:
        RMSNorm → MixingLayer → Dropout → (+residual)
        RMSNorm → BitNetFFN → Dropout → (+residual)
    """

    def __init__(self, cfg: DenseEditorConfig):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model, eps=cfg.rms_norm_eps)
        self.mixing = create_mixing_layer(cfg)
        self.norm2 = RMSNorm(cfg.d_model, eps=cfg.rms_norm_eps)
        if cfg.mixing_type in ("attention", "hybrid"):
            self.ffn = SwiGLUFFN(cfg.d_model, cfg.d_ff, dropout=cfg.dropout)
        elif getattr(cfg, 'int8_qat', False):
            self.ffn = Int8FFN(cfg.d_model, cfg.d_ff, dropout=cfg.dropout, fused_gate_up=True)
        else:
            self.ffn = BitNetFFN(cfg.d_model, cfg.d_ff, dropout=cfg.dropout, fused_gate_up=True)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(
        self, x: torch.Tensor, pad_mask: torch.Tensor | None = None,
        reset_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # pre-norm → Mixing → residual (reset_mask로 문서 경계 state 리셋)
        x = x + self.dropout(self.mixing(self.norm1(x), pad_mask=pad_mask, reset_mask=reset_mask))
        # pre-norm → FFN → residual
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


class DenseEditor(nn.Module):
    """DenseEditor 메인 모델

    6종 mixing layer 중 하나를 선택하여 사용하는 인코더-only 편집 태깅 모델.
    Dense 구조 (MoE 없음)로 CPU 인퍼런스에 최적화.
    """

    def __init__(self, cfg: DenseEditorConfig):
        super().__init__()
        self.cfg = cfg
        self.gradient_checkpointing = False

        # 임베딩
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=cfg.pad_id)
        self.embed_scale = math.sqrt(cfg.d_model)
        self.embed_dropout = nn.Dropout(cfg.dropout)

        # 레이어 스택
        self.layers = nn.ModuleList([
            DenseEditorLayer(cfg) for _ in range(cfg.n_layers)
        ])

        # Final norm + tag head
        self.final_norm = RMSNorm(cfg.d_model, eps=cfg.rms_norm_eps)
        if cfg.mixing_type in ("attention", "hybrid") or getattr(cfg, 'int8_qat', False):
            self.tag_head = Int8Linear(cfg.d_model, cfg.n_tags, bias=False)
        else:
            self.tag_head = BitLinear(cfg.d_model, cfg.n_tags)

        self._init_weights()

    def _init_weights(self):
        """가중치 초기화"""
        # Xavier 근사 — d_model에 비례하는 std (BERT 0.02는 d=768 기준)
        embed_std = 1.0 / math.sqrt(self.cfg.d_model)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=embed_std)
        if self.cfg.pad_id is not None:
            nn.init.zeros_(self.embedding.weight[self.cfg.pad_id])

        # Deep network scaling: 1/sqrt(2*n_layers) for residual
        scale = (2 * self.cfg.n_layers) ** -0.5
        for layer in self.layers:
            if hasattr(layer.mixing, '_init_weights'):
                layer.mixing._init_weights()

    def forward(
        self,
        input_ids: torch.Tensor,
        pad_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            input_ids: (B, T) — 입력 토큰 ID
            pad_mask: (B, T) bool — True가 유효 데이터

        Returns:
            tag_logits: (B, T, n_tags)
        """
        x = self.embedding(input_ids) * self.embed_scale
        x = self.embed_dropout(x)

        # 문서 경계 감지 (패킹 시 BOS 위치에서 state 리셋)
        reset_mask = (input_ids == self.cfg.bos_id)

        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                x = checkpoint(layer, x, pad_mask, reset_mask, use_reentrant=False)
            else:
                x = layer(x, pad_mask=pad_mask, reset_mask=reset_mask)

        x = self.final_norm(x)
        # tag_head 내부에서 LayerNorm + quantization 수행 → float 캐스팅은 CE loss 직전으로 이동
        tag_logits = self.tag_head(x)

        return tag_logits

    def count_parameters(self) -> dict[str, int]:
        """파라미터 수 집계"""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)

        categories = {}
        for name, p in self.named_parameters():
            parts = name.split(".")
            cat = parts[0]
            categories[cat] = categories.get(cat, 0) + p.numel()

        return {"total": total, "trainable": trainable, **categories}


if __name__ == "__main__":
    from model.dense_editor_config import DenseEditorConfig, make_preset

    print("=" * 60)
    print("DenseEditor 모델 검증")
    print("=" * 60)

    for mixing_type in ["fnet", "tcn", "rwkv", "retnet", "mamba", "xlstm", "attention", "hybrid"]:
        print(f"\n--- {mixing_type.upper()} ---")
        cfg = make_preset(mixing_type)

        # 작은 모델로 테스트 (메모리 절약)
        cfg.n_layers = min(cfg.n_layers, 4)
        model = DenseEditor(cfg)

        counts = model.count_parameters()
        per_layer = (counts["total"] - counts.get("embedding", 0)
                     - counts.get("final_norm", 0) - counts.get("tag_head", 0))
        per_layer //= cfg.n_layers

        print(f"  d_model={cfg.d_model}, d_ff={cfg.d_ff}, n_layers(test)={cfg.n_layers}")
        print(f"  총 파라미터(test): {counts['total']:,}")
        print(f"  레이어당 파라미터: ~{per_layer:,}")

        # 128M 프리셋의 실제 레이어 수에서 추정
        full_cfg = make_preset(mixing_type)
        est_total = counts.get("embedding", 0) + counts.get("final_norm", 0) \
                    + counts.get("tag_head", 0) + per_layer * full_cfg.n_layers
        print(f"  128M 프리셋 (n_layers={full_cfg.n_layers}): ~{est_total / 1e6:.1f}M 추정")

        # Forward pass 검증
        input_ids = torch.randint(1, cfg.vocab_size, (2, 64))
        logits = model(input_ids)
        print(f"  Forward OK: input={input_ids.shape} → logits={logits.shape}")

        # Backward pass 검증
        loss = logits.sum()
        loss.backward()
        has_grad = model.embedding.weight.grad is not None
        print(f"  Backward OK (embedding grad: {has_grad})")
        model.zero_grad()

        del model

    print("\n" + "=" * 60)
    print("모든 mixing type 검증 완료!")
