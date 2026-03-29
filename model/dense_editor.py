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


class InsertDecoder(nn.Module):
    """INSERT 위치에서 삽입 토큰을 autoregressive 생성

    Mamba2Block (causal, forward only)을 사용.
    encoder hidden state를 초기 입력으로 받고, 이후 토큰을 순차 생성.
    """

    def __init__(self, d_model: int, vocab_size: int, n_layers: int = 1,
                 max_insert_len: int = 16, eos_id: int = 3,
                 d_state: int = 64, headdim: int = 64):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_insert_len = max_insert_len
        self.eos_id = eos_id

        # encoder hidden → 디코더 초기 입력 변환
        self.context_proj = nn.Linear(d_model, d_model)

        # 토큰 임베딩 (인코더와 공유 가능하지만 별도 유지)
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.embed_scale = math.sqrt(d_model)

        # Mamba2Block 레이어 (causal, forward direction only)
        from model.mixing.bi_mamba2 import Mamba2Block
        self.layers = nn.ModuleList([
            Mamba2Block(d_model, d_state=d_state, headdim=headdim, expand=2)
            for _ in range(n_layers)
        ])
        self.norm = RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, enc_hidden: torch.Tensor,
                target_tokens: torch.Tensor | None = None,
                target_mask: torch.Tensor | None = None,
                ) -> torch.Tensor:
        """학습 시 teacher-forced forward

        Args:
            enc_hidden: (N_ins, d_model) — INSERT 위치의 인코더 hidden state
            target_tokens: (N_ins, max_len) — 정답 삽입 시퀀스 (EOS 포함)
            target_mask: (N_ins, max_len) bool — 유효 위치

        Returns:
            logits: (N_ins, max_len, vocab_size) — 각 위치의 토큰 예측
        """
        N_ins = enc_hidden.shape[0]
        if N_ins == 0:
            return torch.zeros(0, 1, self.vocab_size, device=enc_hidden.device)

        max_len = target_tokens.shape[1] if target_tokens is not None else self.max_insert_len

        # 디코더 입력 구성: [context_proj(enc_hidden), embed(tok_0), embed(tok_1), ...]
        # 첫 위치: encoder hidden → context_proj
        ctx = self.context_proj(enc_hidden).unsqueeze(1)  # (N_ins, 1, d)

        if target_tokens is not None:
            # Teacher forcing: 정답 토큰을 입력으로 (마지막 토큰은 제외 — 예측 대상)
            tok_embed = self.embed(target_tokens[:, :-1]) * self.embed_scale  # (N_ins, max_len-1, d)
            x = torch.cat([ctx, tok_embed], dim=1)  # (N_ins, max_len, d)
        else:
            x = ctx  # 추론 시 첫 토큰만

        # Mamba2 레이어 (causal — 자연스럽게 왼쪽만 참조)
        for layer in self.layers:
            x = x + layer(x)  # residual (단순 구조)

        x = self.norm(x)
        logits = self.head(x)  # (N_ins, max_len, vocab)
        return logits

    @torch.no_grad()
    def generate(self, enc_hidden: torch.Tensor) -> list[list[int]]:
        """추론 시 autoregressive 생성

        Args:
            enc_hidden: (N_ins, d_model) — INSERT 위치의 인코더 hidden state

        Returns:
            생성된 토큰 시퀀스 리스트 (각 INSERT 위치별)
        """
        N_ins = enc_hidden.shape[0]
        if N_ins == 0:
            return []

        device = enc_hidden.device
        # 첫 입력: context_proj(enc_hidden)
        current = self.context_proj(enc_hidden).unsqueeze(1)  # (N_ins, 1, d)
        generated = [[] for _ in range(N_ins)]
        active = torch.ones(N_ins, dtype=torch.bool, device=device)

        for step in range(self.max_insert_len):
            # Forward (전체 시퀀스 — Mamba는 stateful이므로 전체 입력 필요)
            x = current
            for layer in self.layers:
                x = x + layer(x)
            x = self.norm(x)
            logits = self.head(x[:, -1, :])  # 마지막 위치의 예측 (N_ins, vocab)
            next_token = logits.argmax(dim=-1)  # (N_ins,)

            for i in range(N_ins):
                if active[i]:
                    tok = next_token[i].item()
                    if tok == self.eos_id:
                        active[i] = False
                    else:
                        generated[i].append(tok)

            if not active.any():
                break

            # 다음 입력 추가
            next_embed = self.embed(next_token).unsqueeze(1) * self.embed_scale  # (N_ins, 1, d)
            current = torch.cat([current, next_embed], dim=1)

        return generated


class DenseEditor(nn.Module):
    """DenseEditor 메인 모델

    하이브리드 모드: 인코더 태깅 + INSERT 위치 autoregressive 디코더.
    """

    def __init__(self, cfg: DenseEditorConfig):
        super().__init__()
        self.cfg = cfg
        self.gradient_checkpointing = False
        self.hybrid_mode = getattr(cfg, 'hybrid_decoder', False)

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

        # INSERT 디코더 (하이브리드 모드)
        if self.hybrid_mode:
            self.insert_decoder = InsertDecoder(
                d_model=cfg.d_model,
                vocab_size=cfg.vocab_size,
                n_layers=getattr(cfg, 'decoder_n_layers', 1),
                max_insert_len=getattr(cfg, 'max_insert_len', 16),
                eos_id=getattr(cfg, 'eos_id', 3),
                d_state=cfg.mamba2_d_state,
                headdim=cfg.mamba2_headdim,
            )

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
        edit_tags: torch.Tensor | None = None,
        insert_targets: torch.Tensor | None = None,
        insert_target_mask: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            input_ids: (B, T) — 입력 토큰 ID
            pad_mask: (B, T) bool — True가 유효 데이터
            edit_tags: (B, T) — 정답 태그 (하이브리드: INSERT_START 위치 탐지용)
            insert_targets: (N_ins, max_insert_len) — 삽입 시퀀스 정답
            insert_target_mask: (N_ins, max_insert_len) bool — 유효 위치

        Returns:
            레거시: tag_logits (B, T, n_tags)
            하이브리드: (tag_logits, decoder_logits)
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
        tag_logits = self.tag_head(x)

        # 하이브리드: edit_tags에서 INSERT_START 위치를 GPU에서 직접 추출
        if self.hybrid_mode and edit_tags is not None and insert_targets is not None:
            INSERT_START = 2 + self.cfg.vocab_size
            insert_mask = (edit_tags == INSERT_START)  # (B, T) bool
            if insert_mask.any():
                enc_hidden = x[insert_mask]  # (N_ins, d) — GPU에서 직접
                decoder_logits = self.insert_decoder(
                    enc_hidden, insert_targets, insert_target_mask)
                return tag_logits, decoder_logits

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
