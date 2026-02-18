"""BitMamba Seq2Seq — 전체 Encoder-Decoder 모델

FP16 Embedding → Encoder(Mamba + BitNet) → Decoder(Mamba + CrossAttn + BitNet) → LM Head
"""
import torch
import torch.nn as nn

from model.config import BitMambaSeq2SeqConfig
from model.encoder import Encoder, RMSNorm
from model.decoder import Decoder


class BitMambaSeq2Seq(nn.Module):
    """BitNet + Mamba Encoder-Decoder Seq2Seq 모델

    특징:
        - FP16 임베딩 (경량)
        - Mamba SSM 기반 시퀀스 모델링
        - BitNet 1.58b ternary FFN
        - Cross-attention (디코더 → 인코더)
        - 임베딩 weight tying (옵션)
    """

    def __init__(self, config: BitMambaSeq2SeqConfig):
        super().__init__()
        self.config = config

        # --- 임베딩 (FP16) ---
        self.encoder_embedding = nn.Embedding(
            config.vocab_size, config.d_model, padding_idx=config.pad_id,
        )
        # 임베딩을 FP16으로 유지
        self.encoder_embedding = self.encoder_embedding.half()

        if config.tie_embeddings:
            self.decoder_embedding = self.encoder_embedding
        else:
            self.decoder_embedding = nn.Embedding(
                config.vocab_size, config.d_model, padding_idx=config.pad_id,
            ).half()

        # 임베딩 스케일링 (d_model ** 0.5)
        self.embed_scale = config.d_model ** 0.5

        # 드롭아웃
        self.embed_dropout = nn.Dropout(config.dropout)

        # --- 인코더 ---
        self.encoder = Encoder(
            n_layers=config.n_encoder_layers,
            d_model=config.d_model,
            d_inner=config.d_inner,
            d_state=config.d_state,
            d_conv=config.d_conv,
            dt_rank=config.dt_rank,
            d_ff=config.d_ff,
            dropout=config.dropout,
            rms_norm_eps=config.rms_norm_eps,
        )

        # --- 디코더 ---
        self.decoder = Decoder(
            n_layers=config.n_decoder_layers,
            d_model=config.d_model,
            d_inner=config.d_inner,
            d_state=config.d_state,
            d_conv=config.d_conv,
            dt_rank=config.dt_rank,
            d_ff=config.d_ff,
            n_heads=config.n_heads,
            dropout=config.dropout,
            rms_norm_eps=config.rms_norm_eps,
        )

        # --- 최종 정규화 ---
        self.final_norm = RMSNorm(config.d_model, eps=config.rms_norm_eps)

        # --- LM Head ---
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        if config.tie_lm_head:
            # LM Head와 디코더 임베딩 weight tying
            # 임베딩이 FP16이므로 lm_head도 해당 weight 사용
            self.lm_head.weight = self.decoder_embedding.weight

        self._init_weights()

    def _init_weights(self):
        """가중치 초기화"""
        for name, p in self.named_parameters():
            if "embedding" in name:
                # 임베딩은 별도 초기화
                if p.dim() >= 2:
                    nn.init.normal_(p, mean=0.0, std=0.02)
            elif p.dim() >= 2 and "weight" in name:
                nn.init.xavier_uniform_(p)

    def _make_src_mask(self, src_ids: torch.Tensor) -> torch.Tensor:
        """소스 패딩 마스크: (B, src_len) — True=유효, False=패딩"""
        return src_ids != self.config.pad_id

    def encode(
        self,
        src_ids: torch.Tensor,
        src_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """인코더 forward

        Args:
            src_ids: (B, src_len) — 소스 토큰 ID
            src_mask: (B, src_len) — 소스 패딩 마스크 (옵션)

        Returns:
            encoder_out: (B, src_len, d_model)
        """
        # 임베딩 (FP16 → FP32 변환)
        x = self.encoder_embedding(src_ids).float() * self.embed_scale
        x = self.embed_dropout(x)

        # 인코더 스택
        encoder_out = self.encoder(x)
        return encoder_out

    def decode(
        self,
        tgt_ids: torch.Tensor,
        encoder_out: torch.Tensor,
        encoder_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """디코더 forward → logits

        Args:
            tgt_ids: (B, tgt_len) — 타겟 토큰 ID
            encoder_out: (B, src_len, d_model)
            encoder_mask: (B, src_len)

        Returns:
            logits: (B, tgt_len, vocab_size)
        """
        # 임베딩 (FP16 → FP32 변환)
        x = self.decoder_embedding(tgt_ids).float() * self.embed_scale
        x = self.embed_dropout(x)

        # 디코더 스택
        x = self.decoder(x, encoder_out, encoder_mask)

        # 최종 정규화 + LM Head
        x = self.final_norm(x)

        # LM Head: weight tying 시 임베딩이 FP16이므로 FP32로 변환 후 matmul
        if self.config.tie_lm_head:
            logits = torch.nn.functional.linear(
                x, self.lm_head.weight.float(), self.lm_head.bias
            )
        else:
            logits = self.lm_head(x)

        return logits

    def forward(
        self,
        src_ids: torch.Tensor,
        tgt_ids: torch.Tensor,
        src_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """전체 Encoder-Decoder forward

        Args:
            src_ids: (B, src_len)
            tgt_ids: (B, tgt_len)
            src_mask: (B, src_len) — True=유효, False=패딩

        Returns:
            logits: (B, tgt_len, vocab_size)
        """
        if src_mask is None:
            src_mask = self._make_src_mask(src_ids)

        encoder_out = self.encode(src_ids, src_mask)
        logits = self.decode(tgt_ids, encoder_out, src_mask)

        return logits

    def count_parameters(self, exclude_embeddings: bool = False) -> dict:
        """파라미터 수 카운팅

        Returns:
            dict: 컴포넌트별 파라미터 수
        """
        counts = {}

        # 임베딩
        embed_params = sum(
            p.numel() for n, p in self.named_parameters()
            if "embedding" in n
        )
        counts["embedding"] = embed_params

        # 인코더
        encoder_params = sum(p.numel() for p in self.encoder.parameters())
        counts["encoder"] = encoder_params

        # 디코더
        decoder_params = sum(p.numel() for p in self.decoder.parameters())
        counts["decoder"] = decoder_params

        # LM Head (tying 시 0)
        if self.config.tie_lm_head:
            counts["lm_head"] = 0  # 임베딩과 공유
        else:
            counts["lm_head"] = sum(p.numel() for p in self.lm_head.parameters())

        # Final norm
        counts["final_norm"] = sum(p.numel() for p in self.final_norm.parameters())

        # 총합
        total = sum(counts.values())
        total_no_embed = total - counts["embedding"]

        counts["total"] = total
        counts["total_excl_embedding"] = total_no_embed

        return counts
