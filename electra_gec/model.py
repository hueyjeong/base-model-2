"""KoELECTRA-Base-v3 + Two-head GECToR 모델

pretrained ELECTRA discriminator encoder 위에
Action Head (4-class) + Content Head (vocab-tied)를 얹은 GEC 편집 태깅 모델.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from transformers import AutoModel

# HuggingFace LOAD REPORT 경고 숨기기 (RTD head UNEXPECTED는 정상)
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)


# 액션 상수
ACTION_KEEP = 0
ACTION_DELETE = 1
ACTION_REPLACE = 2
ACTION_INSERT = 3


class KoELECTRAGECToR(nn.Module):
    """KoELECTRA encoder + Two-head GEC 편집 태깅 모델

    Args:
        model_name: HuggingFace 모델 이름
        dropout: head 드롭아웃 비율
    """

    def __init__(
        self,
        model_name: str = "monologg/koelectra-base-v3-discriminator",
        dropout: float = 0.1,
    ):
        super().__init__()
        self.electra = AutoModel.from_pretrained(model_name)
        d = self.electra.config.hidden_size   # 768
        V = self.electra.config.vocab_size     # 35000

        self.d_model = d
        self.vocab_size = V
        self.dropout = nn.Dropout(dropout)

        # Action head: KEEP/DELETE/REPLACE/INSERT
        self.action_head = nn.Linear(d, 4)
        nn.init.xavier_uniform_(self.action_head.weight)
        nn.init.zeros_(self.action_head.bias)

        # Content head (tied): h @ embedding.T + bias
        self.content_bias = nn.Parameter(torch.zeros(V))

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            action_logits: (B, T, 4)
            content_logits: (B, T, V)
        """
        h = self.electra(input_ids, attention_mask=attention_mask).last_hidden_state
        h = self.dropout(h)  # (B, T, 768)

        action_logits = self.action_head(h)  # (B, T, 4)

        # Tied content head: h @ word_embeddings.T + bias
        embed_w = self.electra.embeddings.word_embeddings.weight  # (V, 768)
        content_logits = F.linear(h, embed_w, self.content_bias)  # (B, T, V)

        return action_logits, content_logits

    # ── Freeze/Unfreeze ──

    def freeze_encoder(self):
        """encoder 전체 동결, heads만 학습"""
        for p in self.electra.parameters():
            p.requires_grad = False

    def unfreeze_top_layers(self, n: int = 6, unfreeze_embeddings: bool = False):
        """상위 n개 encoder layer + heads 학습, 선택적 embedding unfreeze"""
        for p in self.electra.parameters():
            p.requires_grad = False
        total_layers = self.electra.config.num_hidden_layers
        for i in range(total_layers - n, total_layers):
            for p in self.electra.encoder.layer[i].parameters():
                p.requires_grad = True
        if unfreeze_embeddings:
            for p in self.electra.embeddings.parameters():
                p.requires_grad = True

    def unfreeze_all(self):
        """전체 모델 학습"""
        for p in self.parameters():
            p.requires_grad = True

    def count_trainable(self) -> int:
        """학습 가능 파라미터 수"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    # ── 추론 ──

    @torch.no_grad()
    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        keep_bias: float = 0.0,
        conf_threshold: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """추론: greedy decode + keep_bias + 확신도 필터링

        Returns:
            actions: (B, T) 액션 ID
            contents: (B, T) 콘텐츠 토큰 ID
            confidence: (B, T) 확신도
        """
        self.eval()
        action_logits, content_logits = self.forward(input_ids, attention_mask)

        # Keep bias: KEEP 선호도 조정
        action_logits[:, :, ACTION_KEEP] += keep_bias

        action_probs = F.softmax(action_logits, dim=-1)
        content_probs = F.softmax(content_logits, dim=-1)

        actions = action_logits.argmax(dim=-1)
        contents = content_logits.argmax(dim=-1)

        # 확신도: KEEP/DELETE → action_prob만, REPLACE/INSERT → action × content
        action_conf = action_probs.max(dim=-1).values
        content_conf = content_probs.max(dim=-1).values

        is_edit = (actions == ACTION_DELETE) | (actions == ACTION_REPLACE) | (actions == ACTION_INSERT)
        confidence = torch.where(is_edit, action_conf * content_conf, action_conf)

        # 확신도 미달 → KEEP으로 강제
        if conf_threshold > 0:
            below = (confidence < conf_threshold) & is_edit
            actions = torch.where(below, torch.zeros_like(actions), actions)

        return actions, contents, confidence


def apply_two_head_tags(
    input_ids: list[int],
    actions: list[int],
    contents: list[int],
) -> list[int]:
    """Two-head 태그 적용하여 교정된 토큰 시퀀스 생성"""
    result = []
    for tok, act, cont in zip(input_ids, actions, contents):
        if act == ACTION_KEEP:
            result.append(tok)
        elif act == ACTION_DELETE:
            pass
        elif act == ACTION_REPLACE:
            result.append(cont)
        elif act == ACTION_INSERT:
            result.append(tok)
            result.append(cont)
    return result


if __name__ == "__main__":
    print("=== KoELECTRAGECToR Smoke Test ===\n")

    model = KoELECTRAGECToR()
    total = sum(p.numel() for p in model.parameters())
    print(f"총 파라미터: {total:,}")

    # Forward pass
    B, T = 2, 64
    ids = torch.randint(1, 35000, (B, T))
    mask = torch.ones(B, T, dtype=torch.long)
    act_logits, cont_logits = model(ids, mask)
    print(f"action_logits: {act_logits.shape}")
    print(f"content_logits: {cont_logits.shape}")

    # Backward
    loss = act_logits.sum() + cont_logits.sum()
    loss.backward()
    print(f"backward OK")

    # Freeze/unfreeze 테스트
    model.freeze_encoder()
    print(f"\nfreeze_encoder → trainable: {model.count_trainable():,}")

    model.unfreeze_top_layers(6)
    print(f"unfreeze_top_6 → trainable: {model.count_trainable():,}")

    model.unfreeze_all()
    print(f"unfreeze_all → trainable: {model.count_trainable():,}")

    # Predict
    actions, contents, conf = model.predict(ids, mask, keep_bias=1.0)
    print(f"\npredict → actions: {actions.shape}, contents: {contents.shape}")

    # apply_two_head_tags
    src = [10, 20, 30, 40]
    acts = [ACTION_KEEP, ACTION_REPLACE, ACTION_DELETE, ACTION_INSERT]
    conts = [0, 99, 0, 55]
    result = apply_two_head_tags(src, acts, conts)
    assert result == [10, 99, 40, 55], f"apply 실패: {result}"
    print(f"apply_two_head_tags OK: {result}")

    print("\n모든 테스트 통과!")
