"""CompositionCodec — BBPE 경계 + Conv 자모 composition (concat 방식)

BBPE 토크나이저가 경계를 결정하고, 전체 자모를 1열로 concat하여
Conv 처리 후 토큰 경계에서 segment pool → 토큰 벡터 생성.
패딩 낭비 없이 토큰 간 문맥 정보도 활용.

인코더: 자모 concat → Embedding → Conv layers → Segment Avg Pool → 토큰 벡터
디코더: 토큰 벡터 → Segment Upsample → Conv layers → 자모 logits
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from codec.conv_codec import ConvBlock


class SegmentMaskedConvBlock(nn.Module):
    """Conv1d + LayerNorm + GELU + Dropout — segment 경계를 넘지 않는 conv.

    표준 ConvBlock과 동일한 파라미터 shape(weight [D, D, k], no bias, padding=k//2)이라
    기존 체크포인트 state_dict 로드 호환. 차이점은 forward에서 kernel 내 각 위치를
    center와 같은 segment인 자모만 기여하도록 mask 처리 → 토큰 경계에서 정보 리크 0.

    구현: k번의 shift + mask + matmul. unfold 없이 메모리 효율적.
    """

    def __init__(self, d_model: int, kernel_size: int = 7, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.kernel_size = kernel_size
        self.pad = kernel_size // 2
        self.conv = nn.Conv1d(
            d_model, d_model,
            kernel_size=kernel_size,
            padding=self.pad,
            bias=False,
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, segment_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, L, D]
            segment_ids: [B, L]  (pad 영역은 음수/특수값으로 들어와야 안전)
        Returns:
            [B, L, D]
        """
        residual = x
        B, L, D = x.shape
        k = self.kernel_size
        pad = self.pad

        w = self.conv.weight  # [D_out, D_in, k]

        out = torch.zeros_like(x)
        for j in range(k):
            offset = j - pad  # -pad..+pad
            # 각 위치 l에서 kernel의 j번째 tap은 x[l+offset]를 봄
            if offset > 0:
                x_shift = F.pad(x[:, offset:, :], (0, 0, 0, offset))
                seg_shift = F.pad(segment_ids[:, offset:], (0, offset), value=-1)
            elif offset < 0:
                neg = -offset
                x_shift = F.pad(x[:, :L - neg, :], (0, 0, neg, 0))
                seg_shift = F.pad(segment_ids[:, :L - neg], (neg, 0), value=-1)
            else:
                x_shift = x
                seg_shift = segment_ids

            mask = (seg_shift == segment_ids).unsqueeze(-1).to(x.dtype)  # [B, L, 1]
            x_masked = x_shift * mask
            out = out + x_masked @ w[:, :, j].T  # [B, L, D]

        out = self.norm(out)
        out = F.gelu(out)
        out = self.dropout(out)
        return out + residual


class CompositionEncoder(nn.Module):
    """자모 concat → Conv → Segment Pool → 토큰 벡터

    전체 자모 시퀀스를 한번에 Conv 처리 후,
    토큰 경계(segment_ids)에 맞춰 평균 풀링.

    개선:
    - intra_pos_emb: 임베딩 직후 세그먼트 내 위치 정보 주입 → pool 전에 각 자모가
      "나는 이 토큰의 N번째" 임을 인코딩, token_vecs가 위치 정보를 포함
    """

    def __init__(self, jamo_vocab: int = 330, d_model: int = 256,
                 n_layers: int = 5, kernel_size: int = 7, dropout: float = 0.1,
                 max_jamo_per_token: int = 32,
                 fixed_output_len: int | None = None,
                 segment_masked: bool = False):
        """
        Args:
            fixed_output_len: None이면 출력 shape이 `[B, segment_ids.max()+1, D]`로
                배치마다 가변. int이면 항상 `[B, fixed_output_len, D]` 고정 출력
                (segment_ids 값이 fixed_output_len 이하임을 호출자가 보장해야 함).
                compile static shape 경로에서 torch.cat/dynamic symbol을 제거하여
                max-autotune/DDPOptimizer 호환성 및 copy 오버헤드 제거.
            segment_masked: True면 conv가 토큰 경계를 못 넘도록 SegmentMaskedConvBlock
                사용 (이웃 토큰 정보 리크 0). False면 기존 ConvBlock.
        """
        super().__init__()
        self.d_model = d_model
        self.max_jamo_per_token = max_jamo_per_token
        self.fixed_output_len = fixed_output_len
        self.segment_masked = segment_masked
        self.embedding = nn.Embedding(jamo_vocab, d_model, padding_idx=0)
        self.embed_scale = math.sqrt(d_model)
        # 세그먼트 내 위치 임베딩 (intra-segment position)
        self.intra_pos_emb = nn.Embedding(max_jamo_per_token, d_model)
        block_cls = SegmentMaskedConvBlock if segment_masked else ConvBlock
        self.layers = nn.ModuleList([
            block_cls(d_model, kernel_size, dropout) for _ in range(n_layers)
        ])
        self.pool_proj = nn.Linear(d_model, d_model)

    def forward(self, jamo_ids: torch.Tensor, jamo_mask: torch.Tensor,
                segment_ids: torch.Tensor, n_segments: torch.Tensor) -> torch.Tensor:
        """
        Args:
            jamo_ids: [B, L] concat된 자모 ID
            jamo_mask: [B, L] 유효 위치
            segment_ids: [B, L] 각 자모가 속한 토큰 ID (0, 0, 0, 1, 1, 2, ...)
            n_segments: [B] 배치별 토큰 수

        Returns:
            token_vecs: [B, max_segments, d_model] (fixed_output_len None 시)
                       또는 [B, fixed_output_len, d_model] (int 지정 시)
        """
        B, L = jamo_ids.shape
        D = self.d_model

        # 임베딩
        x = self.embedding(jamo_ids) * self.embed_scale  # [B, L, D]

        # within_pos 계산: 세그먼트 내 0-based index (compile 호환)
        arange_pos = torch.arange(L, device=segment_ids.device).unsqueeze(0).expand(B, -1)
        seg_change = torch.cat([
            torch.ones(B, 1, dtype=torch.bool, device=segment_ids.device),
            segment_ids[:, 1:] != segment_ids[:, :-1],
        ], dim=1)
        seg_start_per_pos = torch.cummax(seg_change * arange_pos, dim=1).values
        within_pos = (arange_pos - seg_start_per_pos).clamp(0, self.max_jamo_per_token - 1)

        # 세그먼트 내 위치 정보 주입 (임베딩 직후, Conv 전)
        x = x + self.intra_pos_emb(within_pos)

        # Conv 레이어
        if self.segment_masked:
            # pad 영역(jamo_mask=False)은 segment_ids=0과 섞일 수 있으므로 -1로 치환
            seg_for_conv = torch.where(jamo_mask, segment_ids, torch.full_like(segment_ids, -1))
            for layer in self.layers:
                x = layer(x, seg_for_conv)  # [B, L, D]
        else:
            for layer in self.layers:
                x = layer(x)  # [B, L, D]

        # Segment Avg Pool (scatter_add)
        # fixed_output_len 지정 시 고정 크기, 아니면 배치별 최대 segment로 동적 할당
        if self.fixed_output_len is not None:
            max_seg = self.fixed_output_len
        else:
            max_seg = segment_ids.max() + 1  # .item() 없이 compile 호환
        token_vecs = torch.zeros(B, max_seg, D, device=x.device, dtype=x.dtype)
        counts = torch.zeros(B, max_seg, 1, device=x.device, dtype=x.dtype)

        seg_exp = segment_ids.unsqueeze(-1).expand(-1, -1, D)  # [B, L, D]
        # 마스킹: 패딩 위치는 0
        x_masked = x * jamo_mask.unsqueeze(-1).float()
        token_vecs.scatter_add_(1, seg_exp, x_masked)

        ones = jamo_mask.float().unsqueeze(-1)  # [B, L, 1]
        counts.scatter_add_(1, segment_ids.unsqueeze(-1), ones)
        counts = counts.clamp(min=1)

        token_vecs = token_vecs / counts
        return self.pool_proj(token_vecs)


class ParallelSlotDecoder(nn.Module):
    """Per-token self-attention 병렬 디코더.

    각 token_vec을 max_slot 슬롯으로 broadcast → + intra_pos_emb → per-token self-attention →
    자모 logits [B, P, max_slot, V]. Segment_masked conv 기반 CompositionDecoder 대체:
    - 토큰 간 interaction 완전 차단 (각 토큰을 독립 시퀀스로 transformer 처리)
    - fixed max_slot으로 within_pos 범위 완전 학습 → decode_from_vec 일반화 확실
    - 학습 target: 각 토큰 첫 N 자모 (원본) + (max_slot - N) PAD

    연산량: Encoder forward는 그대로(가변), decoder만 B*P batch × max_slot 길이의
    transformer. ELECTRA 등 encoder-only downstream 경로에는 영향 없음.
    """

    def __init__(self, jamo_vocab: int = 330, d_model: int = 256,
                 max_slot: int = 16, n_layers: int = 2, n_heads: int = 4,
                 dropout: float = 0.1, d_ff_mult: int = 4):
        super().__init__()
        self.jamo_vocab = jamo_vocab
        self.d_model = d_model
        self.max_slot = max_slot
        self.intra_pos_emb = nn.Embedding(max_slot, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * d_ff_mult,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, jamo_vocab)

    def forward(self, token_vecs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            token_vecs: [B, P, D]
        Returns:
            logits: [B, P, max_slot, V]
        """
        B, P, D = token_vecs.shape
        S = self.max_slot
        # Broadcast: [B, P, D] → [B, P, S, D]
        x = token_vecs.unsqueeze(2).expand(-1, -1, S, -1).contiguous()
        # + intra_pos_emb
        pos_ids = torch.arange(S, device=token_vecs.device)
        pos = self.intra_pos_emb(pos_ids).view(1, 1, S, D)
        x = x + pos
        # Per-token self-attention: [B*P, S, D]
        x = x.view(B * P, S, D)
        x = self.transformer(x)
        x = x.view(B, P, S, D)
        return self.head(x)


class CompositionDecoder(nn.Module):
    """토큰 벡터 → Segment Upsample → Conv → 자모 logits

    토큰 벡터를 segment_ids로 원래 자모 길이로 확장 후
    Conv로 처리하여 자모 logits 출력.

    개선:
    - intra_pos_emb: 세그먼트 내 0-based position embedding → '울'→'ㅜㅜ' 류 오류 해결
    - jamo_mask 적용: 패딩 위치가 Conv를 통해 유효 위치에 오염되는 것 방지
    """

    def __init__(self, jamo_vocab: int = 330, d_model: int = 256,
                 n_layers: int = 5, kernel_size: int = 7, dropout: float = 0.1,
                 max_jamo_per_token: int = 32,
                 segment_masked: bool = False):
        super().__init__()
        self.max_jamo_per_token = max_jamo_per_token
        self.segment_masked = segment_masked
        self.upsample_proj = nn.Linear(d_model, d_model)
        # 세그먼트 내 위치 임베딩 (intra-segment position)
        self.intra_pos_emb = nn.Embedding(max_jamo_per_token, d_model)
        block_cls = SegmentMaskedConvBlock if segment_masked else ConvBlock
        self.layers = nn.ModuleList([
            block_cls(d_model, kernel_size, dropout) for _ in range(n_layers)
        ])
        self.head = nn.Linear(d_model, jamo_vocab)

    def forward(self, token_vecs: torch.Tensor, segment_ids: torch.Tensor,
                target_len: int, jamo_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            token_vecs: [B, max_segments, d_model]
            segment_ids: [B, L] 각 자모의 토큰 ID
            target_len: 출력 자모 길이 L
            jamo_mask: [B, L] bool, 유효 위치 (선택)

        Returns:
            logits: [B, L, jamo_vocab]
        """
        B, L = segment_ids.shape
        D = token_vecs.size(-1)

        # within_pos 계산: 각 자모의 세그먼트 내 0-based index
        # .item() 없이 torch.compile 호환: segment 경계 감지 → cummax로 시작 위치 전파
        arange_pos = torch.arange(L, device=segment_ids.device).unsqueeze(0).expand(B, -1)
        seg_change = torch.cat([
            torch.ones(B, 1, dtype=torch.bool, device=segment_ids.device),
            segment_ids[:, 1:] != segment_ids[:, :-1],
        ], dim=1)  # [B, L]: 각 세그먼트 첫 위치에서 True
        seg_start_per_pos = torch.cummax(seg_change * arange_pos, dim=1).values  # [B, L]
        within_pos = (arange_pos - seg_start_per_pos).clamp(0, self.max_jamo_per_token - 1)

        # Upsample: 각 자모 위치에 해당 토큰 벡터 배치
        seg_exp = segment_ids.unsqueeze(-1).expand(-1, -1, D)
        x = token_vecs.gather(1, seg_exp)  # [B, L, D]
        x = self.upsample_proj(x)

        # 세그먼트 내 위치 정보 주입
        x = x + self.intra_pos_emb(within_pos)

        # 패딩 위치 0화 (Conv를 통한 오염 방지)
        if jamo_mask is not None:
            x = x * jamo_mask.unsqueeze(-1).float()

        # Conv 레이어 (각 레이어 후 패딩 재적용)
        if self.segment_masked:
            if jamo_mask is not None:
                seg_for_conv = torch.where(jamo_mask, segment_ids, torch.full_like(segment_ids, -1))
            else:
                seg_for_conv = segment_ids
            for layer in self.layers:
                x = layer(x, seg_for_conv)
                if jamo_mask is not None:
                    x = x * jamo_mask.unsqueeze(-1).float()
        else:
            for layer in self.layers:
                x = layer(x)
                if jamo_mask is not None:
                    x = x * jamo_mask.unsqueeze(-1).float()

        return self.head(x)  # [B, L, V]


class CompositionCodec(nn.Module):
    """BBPE + Conv Composition Codec (concat 방식)

    전체 자모를 concat → Conv encoder → segment pool → 토큰 벡터
    → segment upsample → Conv decoder → 자모 복원

    패딩 낭비 없음 (활용률 ~100% vs 이전 ~8%).
    토큰 간 문맥 정보 활용 가능 (contextual embedding).
    """

    def __init__(self, jamo_vocab: int = 330, d_model: int = 256,
                 n_layers: int = 5, kernel_size: int = 7, dropout: float = 0.1,
                 max_jamo_per_token: int = 32,
                 segment_masked: bool = False,
                 parallel_decoder: bool = False,
                 decoder_layers: int = 2,
                 decoder_heads: int = 4,
                 fixed_output_len: int | None = None):
        """
        Args:
            parallel_decoder: True면 CompositionDecoder(conv) 대신 ParallelSlotDecoder 사용.
                Encoder는 가변 입력 유지, decoder만 각 토큰을 max_jamo_per_token 슬롯으로
                확장해 self-attention. decode_from_vec 완벽 대응. Dataset은 가변 원본 구조
                (fixed_slot/append_pad_slot 불필요).
            fixed_output_len: encoder 출력 token 수 고정. None이면 `segment_ids.max()+1`로
                배치마다 가변 → torch.compile recompile 누적으로 CPU RAM 성장. 고정값
                지정 시 static shape → recompile 없음. fixed_slot=True 코퍼스는
                `max_seq_len // max_jamo_per_token`을 기본값으로 쓰면 됨.
        """
        super().__init__()
        self.jamo_vocab = jamo_vocab
        self.d_model = d_model
        self.segment_masked = segment_masked
        self.parallel_decoder = parallel_decoder
        self.max_slot = max_jamo_per_token

        self.encoder = CompositionEncoder(
            jamo_vocab, d_model, n_layers, kernel_size, dropout,
            max_jamo_per_token=max_jamo_per_token,
            segment_masked=segment_masked,
            fixed_output_len=fixed_output_len,
        )
        if parallel_decoder:
            self.decoder = ParallelSlotDecoder(
                jamo_vocab=jamo_vocab, d_model=d_model,
                max_slot=max_jamo_per_token,
                n_layers=decoder_layers, n_heads=decoder_heads,
                dropout=dropout,
            )
        else:
            self.decoder = CompositionDecoder(
                jamo_vocab, d_model, n_layers, kernel_size, dropout,
                max_jamo_per_token=max_jamo_per_token,
                segment_masked=segment_masked,
            )

    def forward(self, jamo_ids: torch.Tensor, jamo_mask: torch.Tensor,
                segment_ids: torch.Tensor, n_segments: torch.Tensor) -> dict:
        """학습용 forward

        Args:
            jamo_ids: [B, L] concat된 자모 시퀀스
            jamo_mask: [B, L] 유효 위치
            segment_ids: [B, L] 토큰 경계 (0,0,0,1,1,2,2,2,...)
            n_segments: [B] 토큰 수

        Returns:
            dict: logits, loss, z
        """
        B, L = jamo_ids.shape
        device = jamo_ids.device

        # Encode (가변 입력, 변화 없음)
        z = self.encoder(jamo_ids, jamo_mask, segment_ids, n_segments)  # [B, P, D]

        if self.parallel_decoder:
            # === Parallel decoder 경로 ===
            # Decoder output: [B, P, max_slot, V]
            logits_slot = self.decoder(z)
            S = self.max_slot
            P = logits_slot.size(1)

            # Target 재구성: 각 토큰의 실제 자모를 슬롯 앞부분에 배치, 나머지는 PAD(0)
            # within_pos 계산 (encoder와 동일)
            arange_pos = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)
            seg_change = torch.cat([
                torch.ones(B, 1, dtype=torch.bool, device=device),
                segment_ids[:, 1:] != segment_ids[:, :-1],
            ], dim=1)
            seg_start_per_pos = torch.cummax(seg_change * arange_pos, dim=1).values
            within_pos = (arange_pos - seg_start_per_pos).clamp(0, S - 1)

            # target_slot[B, P, S] = PAD, jamo_mask=True인 자모만 scatter
            target_slot = torch.zeros(B, P, S, dtype=torch.long, device=device)
            b_idx = torch.arange(B, device=device).unsqueeze(1).expand(-1, L)
            valid = jamo_mask  # [B, L]
            target_slot[b_idx[valid], segment_ids[valid], within_pos[valid]] = jamo_ids[valid]

            # slot 유효성: 유효 segment(n_segments 이내)만 loss에 포함
            # n_segments 이상의 P는 padding 토큰 → loss 무시
            seg_idx = torch.arange(P, device=device).unsqueeze(0).expand(B, -1)  # [B, P]
            n_seg_b = n_segments.unsqueeze(1)  # [B, 1]
            seg_valid = seg_idx < n_seg_b  # [B, P]
            # slot_loss_mask: [B, P, S] — 유효 segment의 모든 슬롯
            slot_loss_mask = seg_valid.unsqueeze(-1).expand(-1, -1, S)  # [B, P, S]

            flat_logits = logits_slot.reshape(-1, self.jamo_vocab)
            flat_targets = target_slot.reshape(-1)
            flat_mask = slot_loss_mask.reshape(-1)
            loss = F.cross_entropy(flat_logits, flat_targets, reduction="none")
            loss = (loss * flat_mask.float()).sum() / flat_mask.sum().clamp(min=1)

            return {"logits": logits_slot, "loss": loss, "z": z,
                    "target_slot": target_slot, "slot_loss_mask": slot_loss_mask}

        # === 기존 conv decoder 경로 ===
        logits = self.decoder(z, segment_ids, L, jamo_mask)  # [B, L, V]

        # Loss — jamo_mask로만 필터 (ignore_index 제거: PAD target도 학습 대상)
        flat_logits = logits.reshape(-1, self.jamo_vocab)
        flat_targets = jamo_ids.reshape(-1)
        valid = jamo_mask.reshape(-1)

        loss = F.cross_entropy(flat_logits, flat_targets, reduction="none")
        loss = loss[valid].mean()

        return {"logits": logits, "loss": loss, "z": z}

    def encode(self, jamo_ids, jamo_mask, segment_ids, n_segments):
        """인코딩만 (backbone 입력용)"""
        return self.encoder(jamo_ids, jamo_mask, segment_ids, n_segments)

    def reconstruct(self, jamo_ids, jamo_mask, segment_ids, n_segments):
        """복원 (평가용)"""
        with torch.no_grad():
            z = self.encoder(jamo_ids, jamo_mask, segment_ids, n_segments)
            logits = self.decoder(z, segment_ids, jamo_ids.size(1), jamo_mask)
            return logits.argmax(dim=-1)

    @torch.no_grad()
    def decode_from_vec(self, token_vecs: torch.Tensor, n_segments,
                        max_slot: int = None, pad_id: int = 0):
        """token_vecs로부터 각 토큰별 가변 길이 자모 시퀀스 복원 (PAD 조기 종료).

        Args:
            token_vecs: [B, P, D]
            n_segments: [B] 유효 토큰 수
            max_slot: 토큰당 할당 슬롯 수. None이면 decoder의 max_slot 사용
            pad_id: PAD 자모 ID (기본 0)

        Returns:
            List[List[List[int]]]: results[b][p] = 토큰 p의 자모 ID 리스트 (PAD 전까지)
        """
        B, P, D = token_vecs.shape
        device = token_vecs.device

        if self.parallel_decoder:
            # ParallelSlotDecoder: [B, P, max_slot, V]. max_slot은 decoder 고정값 사용
            S = self.decoder.max_slot if max_slot is None else max_slot
            if max_slot is not None and max_slot != self.decoder.max_slot:
                raise ValueError(
                    f"parallel_decoder 구조에서는 max_slot이 학습값({self.decoder.max_slot})으로 고정됩니다. "
                    f"다른 값 지정 불가."
                )
            logits = self.decoder(token_vecs)  # [B, P, S, V]
            preds = logits.argmax(-1)  # [B, P, S]

            n_segs_list = n_segments.tolist() if torch.is_tensor(n_segments) else list(n_segments)
            results = []
            for b, n in enumerate(n_segs_list):
                tokens = []
                for p in range(n):
                    slot_preds = preds[b, p].tolist()
                    jamo_seq = []
                    for j in slot_preds:
                        if j == pad_id:
                            break
                        jamo_seq.append(j)
                    tokens.append(jamo_seq)
                results.append(tokens)
            return results

        # === Conv decoder 경로 (이전 버전) ===
        if max_slot is None:
            max_slot = 16

        seg_per_token = torch.arange(P, device=device).unsqueeze(1).expand(-1, max_slot).reshape(-1)
        segment_ids = seg_per_token.unsqueeze(0).expand(B, -1).contiguous()
        L = P * max_slot
        jamo_mask = torch.ones(B, L, dtype=torch.bool, device=device)

        n_segs_list = n_segments.tolist() if torch.is_tensor(n_segments) else list(n_segments)
        for b, n in enumerate(n_segs_list):
            if n < P:
                jamo_mask[b, n * max_slot:] = False

        logits = self.decoder(token_vecs, segment_ids, L, jamo_mask)  # [B, L, V]
        preds = logits.argmax(-1)  # [B, L]

        results = []
        for b, n in enumerate(n_segs_list):
            tokens = []
            for p in range(n):
                slot_preds = preds[b, p * max_slot:(p + 1) * max_slot].tolist()
                jamo_seq = []
                for j in slot_preds:
                    if j == pad_id:
                        break
                    jamo_seq.append(j)
                tokens.append(jamo_seq)
            results.append(tokens)
        return results


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    print("=== CompositionCodec (concat) Smoke Test ===\n")

    for n_layers in [3, 4, 5]:
        codec = CompositionCodec(
            jamo_vocab=330, d_model=256, n_layers=n_layers, kernel_size=7,
        )
        n_params = count_params(codec)
        enc_params = count_params(codec.encoder)
        dec_params = count_params(codec.decoder)
        print(f"{n_layers}L: {n_params/1e6:.2f}M total "
              f"(enc {enc_params/1e6:.2f}M, dec {dec_params/1e6:.2f}M)")

        # forward test: 3개 토큰, 자모 concat [5, 3, 7] = 15
        B = 2
        # 토큰별 자모 길이: [5, 3, 7]
        L = 15
        jamo_ids = torch.randint(1, 330, (B, L))
        jamo_mask = torch.ones(B, L, dtype=torch.bool)
        # segment_ids: 0,0,0,0,0, 1,1,1, 2,2,2,2,2,2,2
        segment_ids = torch.tensor([[0]*5 + [1]*3 + [2]*7] * B)
        n_segments = torch.tensor([3, 3])

        out = codec(jamo_ids, jamo_mask, segment_ids, n_segments)
        print(f"  input: [B={B}, L={L}], z: {out['z'].shape}, "
              f"logits: {out['logits'].shape}, loss: {out['loss'].item():.4f}")

        out["loss"].backward()
        trainable = [p for p in codec.parameters() if p.grad is not None]
        grad_ok = all(not p.grad.isnan().any() for p in trainable)
        print(f"  backward: {'OK' if grad_ok else 'FAIL'}")
        print()

    print("모든 테스트 통과!")
