"""BitMamba Seq2Seq — 전체 Encoder-Decoder 모델

FP16 Embedding → Encoder(Mamba + BitNet) → Decoder(Mamba concat + BitNet) → LM Head
Cross-Attention 없이 encoder 출력을 decoder 입력에 concat하여 Mamba state로 전달.
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
        - Cross-Attention 없음 (Mamba concat 방식)
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
            mamba_version=config.mamba_version,
            headdim=config.headdim,
            ngroups=config.ngroups,
            chunk_size=config.chunk_size,
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
            dropout=config.dropout,
            rms_norm_eps=config.rms_norm_eps,
            mamba_version=config.mamba_version,
            headdim=config.headdim,
            ngroups=config.ngroups,
            chunk_size=config.chunk_size,
        )

        # --- 최종 정규화 ---
        self.final_norm = RMSNorm(config.d_model, eps=config.rms_norm_eps)

        # --- LM Head ---
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        if config.tie_lm_head:
            # LM Head와 디코더 임베딩 weight tying
            # 임베딩이 FP16이므로 lm_head도 해당 weight 사용
            self.lm_head.weight = self.decoder_embedding.weight

        # --- Copy Gate (Trial B) ---
        if getattr(config, "use_copy_gate", False):
            self.copy_gate = nn.Linear(config.d_model, 1)
            # 게이트 초기값을 보수적으로 설정 (생성 분포 p_gen 쪽으로 쏠리도록 유도)
            nn.init.constant_(self.copy_gate.bias, 2.0)

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
            src_mask: (B, src_len) — 소스 패딩 마스크 (True=유효, False=패딩)

        Returns:
            encoder_out: (B, src_len, d_model)
        """
        # Document isolation: BOS 위치에서 SSM state 리셋
        reset_mask = (src_ids == self.config.bos_id)  # (B, src_len)

        # 임베딩 (FP16 → FP32 변환)
        x = self.encoder_embedding(src_ids).float() * self.embed_scale
        x = self.embed_dropout(x)
        if self.encoder.gradient_checkpointing:
            x.requires_grad_(True)

        # 패딩 위치를 완전히 0으로 만들면 역전파 시 MambaBlock에서 NaN 유발 가능성 존재
        # 극소값 1e-5를 남겨두어 gradient flow를 안전하게 유지
        if src_mask is not None:
            x = x * (src_mask.unsqueeze(-1).float() + 1e-5)  # (B, src_len, 1)

        # 인코더 스택
        encoder_out = self.encoder(x, reset_mask=reset_mask)
        return encoder_out

    def decode(
        self,
        tgt_ids: torch.Tensor,
        encoder_out: torch.Tensor,
        encoder_mask: torch.Tensor | None = None,
        return_hidden: bool = False,
        src_ids: torch.Tensor | None = None,
        source_bias: float = 0.0,
        src_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """디코더 forward → logits (또는 hidden states)

        Encoder 출력을 target 임베딩 앞에 concat하여 Mamba로 처리.
        Cross-Attention 없이 Mamba의 recurrent state로 encoder 정보 전달.

        Args:
            tgt_ids: (B, tgt_len) — 타겟 토큰 ID
            encoder_out: (B, src_len, d_model)
            encoder_mask: (B, src_len) — 패딩 마스크 (True=유효, False=패딩)
            return_hidden: True면 lm_head 적용 전 hidden states 반환
                           (chunked cross-entropy용)

        Returns:
            return_hidden=False: logits (B, tgt_len, vocab_size)
            return_hidden=True:  hidden (B, tgt_len, d_model)
        """
        src_len = encoder_out.size(1)

        # Document isolation: BOS 위치에서 SSM state 리셋
        reset_mask = (tgt_ids == self.config.bos_id)  # (B, tgt_len)

        # Cross-attention 문서 격리: 각 문서별로 context matrix 분리
        src_doc_ids = (src_ids == self.config.bos_id).int().cumsum(dim=1) - 1  # (B, src_len)
        tgt_doc_ids = reset_mask.int().cumsum(dim=1) - 1  # (B, tgt_len)

        # 타겟 임베딩 (FP16 → FP32 변환)
        tgt_emb = self.decoder_embedding(tgt_ids).float() * self.embed_scale
        tgt_emb = self.embed_dropout(tgt_emb)
        if self.decoder.gradient_checkpointing:
            tgt_emb.requires_grad_(True)

        # 패딩 위치의 encoder_out 마스킹 시 극소값 잔존으로 완전한 0.0 회피
        if encoder_mask is not None:
            encoder_out = encoder_out * (encoder_mask.unsqueeze(-1).float() + 1e-5)

        # 디코더 스택 (Target Embedding만 Mamba로 들어가고, encoder_out은 Linear Cross-Attention으로 전달됨)
        x = self.decoder(tgt_emb, encoder_out=encoder_out, encoder_mask=encoder_mask,
                         reset_mask=reset_mask, src_doc_ids=src_doc_ids,
                         tgt_doc_ids=tgt_doc_ids)

        # 최종 정규화
        x = self.final_norm(x)

        # Chunked CE용: hidden states 반환
        if return_hidden:
            return x

        # LM Head: weight tying 시 임베딩이 FP16이므로 FP32로 변환 후 matmul
        if self.config.tie_lm_head:
            logits = torch.nn.functional.linear(
                x, self.lm_head.weight.float(), self.lm_head.bias
            )
        else:
            logits = self.lm_head(x)

        # --- Trial B: Copy Gate ---
        use_copy_gate = getattr(self.config, "use_copy_gate", False)
        if use_copy_gate and src_ids is not None and hasattr(self, "copy_gate"):
            # Gate 값 산출 (Logit Space 보정용)
            raw_gate = torch.sigmoid(self.copy_gate(x))  # (B, tgt_len, 1)
            # Gate Collapse 방지: 생성 모델이 최소 50%의 gradient를 받도록 강제
            gate = 0.5 + 0.5 * raw_gate
            
            # p_copy: src_ids 내의 unigram 분포 산출 (one-hot counting)
            B, V = logits.size(0), logits.size(-1)
            one_hot = torch.zeros(B, V, dtype=logits.dtype, device=logits.device)
            one_hot.scatter_add_(1, src_ids, torch.ones_like(src_ids, dtype=logits.dtype))
            one_hot[:, self.config.pad_id] = 0.0
            
            # 확률화
            p_copy_sum = one_hot.sum(dim=-1, keepdim=True).clamp(min=1e-5)
            p_copy = (one_hot / p_copy_sum).unsqueeze(1)  # (B, 1, V)
            
            # Logit Space 변환 (BF16 수치 안정성을 위해 1e-5로 하한선 상향)
            copy_logits = torch.log(p_copy.clamp(min=1e-5))
            
            # Logit 레벨에서 Gate 혼합 (그래디언트 보존)
            logits = gate * logits + (1.0 - gate) * copy_logits

        # --- Trial A: Source-Aware Logit Bias ---
        if source_bias > 0.0 and src_ids is not None:
            B, V = logits.size(0), logits.size(-1)
            src_mask_vocab = torch.zeros(B, V, dtype=logits.dtype, device=logits.device)
            
            if src_weights is not None:
                # src_weights에 기반하여 bias mask 생성 (동일 토큰이 여러 번 등장하면 최대 가중치 유지)
                src_mask_vocab.scatter_reduce_(1, src_ids, src_weights.to(logits.dtype), reduce="amax", include_self=False)
            else:
                src_mask_vocab.scatter_(1, src_ids, 1.0)
                
            src_mask_vocab[:, self.config.pad_id] = 0.0
            
            # 원문에 등장한 토큰들에게 source_bias 가산 (src_weights 값에 비례)
            logits = logits + src_mask_vocab.unsqueeze(1) * source_bias

        return logits

    def forward(
        self,
        src_ids: torch.Tensor,
        tgt_ids: torch.Tensor,
        src_mask: torch.Tensor | None = None,
        source_bias: float = 0.0,
        src_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """전체 Encoder-Decoder forward

        Args:
            src_ids: (B, src_len)
            tgt_ids: (B, tgt_len)
            src_mask: (B, src_len) — True=유효, False=패딩
            source_bias: Trial A Source-Aware Logit Bias 강도

        Returns:
            logits: (B, tgt_len, vocab_size)
        """
        if src_mask is None:
            src_mask = self._make_src_mask(src_ids)

        encoder_out = self.encode(src_ids, src_mask)
        logits = self.decode(
            tgt_ids=tgt_ids,
            encoder_out=encoder_out,
            encoder_mask=src_mask,
            src_ids=src_ids,
            source_bias=source_bias,
            src_weights=src_weights,
        )

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

        # Copy Gate (Trial B)
        if getattr(self.config, "use_copy_gate", False) and hasattr(self, "copy_gate"):
            counts["copy_gate"] = sum(p.numel() for p in self.copy_gate.parameters())

        # Final norm
        counts["final_norm"] = sum(p.numel() for p in self.final_norm.parameters())

        # 총합
        total = sum(counts.values())
        total_no_embed = total - counts["embedding"]

        counts["total"] = total
        counts["total_excl_embedding"] = total_no_embed

        return counts
