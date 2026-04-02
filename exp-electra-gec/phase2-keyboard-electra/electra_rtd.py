"""ElectraRTD — ELECTRA Replaced Token Detection 학습 래퍼

Generator + Discriminator joint training.

흐름:
1. 원본 토큰 mask_prob(15%) 위치 마스킹 (special token 제외)
2. Generator MLM → multinomial 샘플링으로 대체
3. Discriminator가 모든 위치에서 real/replaced 이진 분류
4. gen_loss(MLM) + disc_loss(RTD binary CE) 반환
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.dirname(__file__))

from config import ElectraConfig
from model.dense_editor import DenseEditor
from generator import TransformerGenerator


class ElectraRTD(nn.Module):
    """ELECTRA RTD 학습 래퍼"""

    def __init__(self, cfg: ElectraConfig):
        super().__init__()
        self.cfg = cfg

        self.generator = TransformerGenerator(cfg.gen)
        self.discriminator = DenseEditor(cfg.disc)
        self.rtd_head = nn.Linear(cfg.disc.d_model, 2)

        nn.init.xavier_uniform_(self.rtd_head.weight)
        nn.init.zeros_(self.rtd_head.bias)

    def forward(
        self,
        input_ids: Tensor,
        pad_mask: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """
        Args:
            input_ids: (B, T) 원본 토큰
            pad_mask: (B, T) bool — True=유효

        Returns:
            dict with keys:
                gen_loss: Generator MLM loss (scalar)
                disc_loss: Discriminator RTD loss (scalar)
                rtd_acc: RTD 정확도 (scalar, 모니터링용)
                gen_acc: Generator MLM 정확도 (scalar, 모니터링용)
                replaced_ratio: 실제 replaced 비율 (scalar)
        """
        B, T = input_ids.shape
        device = input_ids.device
        disc_cfg = self.cfg.disc

        # ── 1. 마스킹 (special token 제외) ──
        is_special = (
            (input_ids == disc_cfg.pad_id) |
            (input_ids == disc_cfg.bos_id) |
            (input_ids == disc_cfg.eos_id) |
            (input_ids == disc_cfg.mask_id)
        )
        can_mask = ~is_special
        if pad_mask is not None:
            can_mask = can_mask & pad_mask

        mask_probs = torch.rand(B, T, device=device)
        mask_positions = can_mask & (mask_probs < self.cfg.mask_prob)

        # 마스크 위치가 0이면 학습 신호 없음 → 빈 배치 방어
        if not mask_positions.any():
            zero = torch.tensor(0.0, device=device, requires_grad=True)
            return {
                "gen_loss": zero, "disc_loss": zero,
                "rtd_acc": torch.tensor(0.0, device=device),
                "gen_acc": torch.tensor(0.0, device=device),
                "replaced_ratio": torch.tensor(0.0, device=device),
            }

        # ── 2. Generator 입력: 마스크 위치를 [MASK]로 ──
        gen_input = input_ids.clone()
        gen_input[mask_positions] = disc_cfg.mask_id

        # ── 3. Generator forward → 마스크 위치에서만 MLM head 실행 ──
        gen_hidden = self.generator(gen_input, pad_mask, return_hidden=True)  # (B, T, d_gen)
        gen_logits_masked = self.generator.mlm_head(gen_hidden[mask_positions])  # (N_mask, V)

        gen_loss = F.cross_entropy(
            gen_logits_masked,               # (N_mask, V)
            input_ids[mask_positions],       # (N_mask,) 원본이 정답
        )

        # ── 4. 마스크 위치에서만 샘플링 ──
        with torch.no_grad():
            gen_probs_masked = F.softmax(gen_logits_masked.float() / self.cfg.temperature, dim=-1)
            sampled_tokens = torch.multinomial(gen_probs_masked, 1).squeeze(-1)  # (N_mask,)

        disc_input = input_ids.clone()
        disc_input[mask_positions] = sampled_tokens

        # ── 5. RTD 라벨 ──
        # Generator가 정답을 맞히면 real(0), 틀리면 replaced(1)
        rtd_labels = (disc_input != input_ids).long()

        # ── 6. Discriminator forward → RTD loss ──
        disc_hidden = self.discriminator.forward_hidden(disc_input, pad_mask)  # (B, T, D)
        rtd_logits = self.rtd_head(disc_hidden)                  # (B, T, 2)

        valid_mask = pad_mask if pad_mask is not None else \
            torch.ones(B, T, dtype=torch.bool, device=device)

        disc_loss = F.cross_entropy(
            rtd_logits[valid_mask],    # (N_valid, 2)
            rtd_labels[valid_mask],    # (N_valid,)
        )

        # ── 모니터링 (no_grad, graph 밖) ──
        with torch.no_grad():
            gen_preds = gen_logits_masked.argmax(dim=-1)
            gen_acc = (gen_preds == input_ids[mask_positions]).float().mean()
            rtd_preds = rtd_logits.argmax(dim=-1)
            rtd_acc = (rtd_preds[valid_mask] == rtd_labels[valid_mask]).float().mean()
            replaced_ratio = rtd_labels[valid_mask].float().mean()

        return {
            "gen_loss": gen_loss,
            "disc_loss": disc_loss,
            "rtd_acc": rtd_acc,
            "gen_acc": gen_acc,
            "replaced_ratio": replaced_ratio,
        }

    def get_total_loss(self, outputs: dict[str, Tensor]) -> Tensor:
        """학습용 총 loss"""
        return (
            self.cfg.gen_loss_weight * outputs["gen_loss"] +
            self.cfg.disc_loss_weight * outputs["disc_loss"]
        )

    # ── Split backward 지원: Gen forward→backward → Disc forward→backward ──
    # Gen activation을 Disc forward 전에 해제하여 메모리 확보

    def forward_gen(
        self,
        input_ids: Tensor,
        pad_mask: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Phase 1: Generator forward + MLM loss + 샘플링"""
        B, T = input_ids.shape
        device = input_ids.device
        disc_cfg = self.cfg.disc

        is_special = (
            (input_ids == disc_cfg.pad_id) |
            (input_ids == disc_cfg.bos_id) |
            (input_ids == disc_cfg.eos_id) |
            (input_ids == disc_cfg.mask_id)
        )
        can_mask = ~is_special
        if pad_mask is not None:
            can_mask = can_mask & pad_mask

        mask_probs = torch.rand(B, T, device=device)
        mask_positions = can_mask & (mask_probs < self.cfg.mask_prob)

        gen_input = input_ids.clone()
        gen_input[mask_positions] = disc_cfg.mask_id

        gen_hidden = self.generator(gen_input, pad_mask, return_hidden=True)
        gen_logits_masked = self.generator.mlm_head(gen_hidden[mask_positions])

        gen_loss = F.cross_entropy(gen_logits_masked, input_ids[mask_positions])

        with torch.no_grad():
            gen_probs_masked = F.softmax(gen_logits_masked.float() / self.cfg.temperature, dim=-1)
            sampled_tokens = torch.multinomial(gen_probs_masked, 1).squeeze(-1)
            gen_preds = gen_logits_masked.argmax(dim=-1)
            gen_acc = (gen_preds == input_ids[mask_positions]).float().mean()

        disc_input = input_ids.clone()
        disc_input[mask_positions] = sampled_tokens
        rtd_labels = (disc_input != input_ids).long()

        valid_mask = pad_mask if pad_mask is not None else \
            torch.ones(B, T, dtype=torch.bool, device=device)

        return {
            "gen_loss": gen_loss,
            "gen_acc": gen_acc,
            # detach: disc phase에서 사용, gen graph 의존성 없음
            "disc_input": disc_input.detach(),
            "rtd_labels": rtd_labels.detach(),
            "valid_mask": valid_mask.detach(),
            "replaced_ratio": rtd_labels[valid_mask].float().mean(),
        }

    def forward_disc(
        self,
        disc_input: Tensor,
        pad_mask: Tensor | None,
        rtd_labels: Tensor,
        valid_mask: Tensor,
    ) -> dict[str, Tensor]:
        """Phase 2: Discriminator forward + RTD loss"""
        disc_hidden = self.discriminator.forward_hidden(disc_input, pad_mask)
        rtd_logits = self.rtd_head(disc_hidden)

        disc_loss = F.cross_entropy(
            rtd_logits[valid_mask],
            rtd_labels[valid_mask],
        )

        with torch.no_grad():
            rtd_preds = rtd_logits.argmax(dim=-1)
            rtd_acc = (rtd_preds[valid_mask] == rtd_labels[valid_mask]).float().mean()

        return {"disc_loss": disc_loss, "rtd_acc": rtd_acc}


if __name__ == "__main__":
    print("=== ElectraRTD Smoke Test ===\n")

    from config import make_electra_config
    cfg = make_electra_config()
    model = ElectraRTD(cfg)

    gen_params = model.generator.count_parameters()
    disc_params = sum(p.numel() for p in model.discriminator.parameters())
    rtd_params = sum(p.numel() for p in model.rtd_head.parameters())
    total = sum(p.numel() for p in model.parameters())
    print(f"Generator: {gen_params:,} ({gen_params/1e6:.2f}M)")
    print(f"Discriminator: {disc_params:,} ({disc_params/1e6:.2f}M)")
    print(f"RTD head: {rtd_params:,}")
    print(f"총: {total:,} ({total/1e6:.2f}M)")

    # Forward
    B, T = 2, 64
    ids = torch.randint(7, 303, (B, T))  # special token 회피
    ids[:, 0] = cfg.disc.bos_id
    ids[:, -1] = cfg.disc.eos_id
    mask = torch.ones(B, T, dtype=torch.bool)
    mask[0, T-3:] = False

    outputs = model(ids, mask)
    print(f"\ngen_loss: {outputs['gen_loss'].item():.4f}")
    print(f"disc_loss: {outputs['disc_loss'].item():.4f}")
    print(f"rtd_acc: {outputs['rtd_acc'].item():.4f}")
    print(f"gen_acc: {outputs['gen_acc'].item():.4f}")
    print(f"replaced_ratio: {outputs['replaced_ratio'].item():.4f}")

    # Backward
    total_loss = model.get_total_loss(outputs)
    print(f"total_loss: {total_loss.item():.4f}")
    total_loss.backward()
    assert model.generator.embedding.weight.grad is not None
    assert model.discriminator.embedding.weight.grad is not None
    print("\nBackward OK")

    print("\n모든 테스트 통과!")
