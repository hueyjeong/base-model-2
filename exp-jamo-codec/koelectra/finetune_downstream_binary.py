"""Binary ELECTRA downstream fine-tune — NSMC / KLUE-NLI/YNAT/STS/RE / PAWS-X-KO.

finetune_downstream.py 의 binary 버전. codec 없음, bbpe_ids 직접 입력.

사용 예:
    python -m koelectra.finetune_downstream_binary --task klue_nli \
        --ckpt checkpoints/b_small/binary_step_20000.pt \
        --max_patches 128 --batch_size 32 --epochs 3 --lr 3e-5 --bf16

Pair 포맷: [BOS] s1 [EOS] [BOS] s2 [EOS] (pretrain multi-doc packing 과 일치)
"""
from __future__ import annotations

import argparse
import os
import random
import sys
import time
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

_THIS = os.path.abspath(os.path.dirname(__file__))
_EXP_ROOT = os.path.abspath(os.path.join(_THIS, ".."))
_PROJECT_ROOT = os.path.abspath(os.path.join(_EXP_ROOT, ".."))
for p in (_EXP_ROOT, _PROJECT_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from koelectra.binary_electra import BinaryElectra, int_to_bits  # noqa: E402
from koelectra.data.bbpe_dataset import load_bbpe_tokenizer  # noqa: E402


def _pearson_np(x: np.ndarray, y: np.ndarray) -> float:
    x = x.astype(np.float64); y = y.astype(np.float64)
    xm, ym = x - x.mean(), y - y.mean()
    denom = np.sqrt((xm ** 2).sum() * (ym ** 2).sum())
    if denom < 1e-12:
        return float("nan")
    return float((xm * ym).sum() / denom)


def _rank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(x), dtype=np.float64)
    return ranks


# ─────────────────────────────────────────────
TASK_CONFIGS = {
    "nsmc": {
        "type": "classification", "n_classes": 2, "pair": False,
        "loader": "tsv",
        "train_path": "corpus/nsmc/ratings_train.txt",
        "val_path": "corpus/nsmc/ratings_test.txt",
        "text_fields": ["document"],
        "label_field": "label",
    },
    "klue_nli": {
        "type": "classification", "n_classes": 3, "pair": True,
        "loader": "hf", "hf_name": "klue", "hf_subset": "nli",
        "train_split": "train", "val_split": "validation",
        "text_fields": ["premise", "hypothesis"],
        "label_field": "label",
    },
    "klue_ynat": {
        "type": "classification", "n_classes": 7, "pair": False,
        "loader": "hf", "hf_name": "klue", "hf_subset": "ynat",
        "train_split": "train", "val_split": "validation",
        "text_fields": ["title"],
        "label_field": "label",
    },
    "klue_sts": {
        "type": "regression", "n_classes": 1, "pair": True,
        "loader": "hf", "hf_name": "klue", "hf_subset": "sts",
        "train_split": "train", "val_split": "validation",
        "text_fields": ["sentence1", "sentence2"],
        "label_field": "labels",
        "label_nested": "label",
    },
    "paws_x_ko": {
        "type": "classification", "n_classes": 2, "pair": True,
        "loader": "hf", "hf_name": "paws-x", "hf_subset": "ko",
        "train_split": "train", "val_split": "validation",
        "text_fields": ["sentence1", "sentence2"],
        "label_field": "label",
    },
    "klue_re": {
        "type": "classification", "n_classes": 30, "pair": True,
        "loader": "hf", "hf_name": "klue", "hf_subset": "re",
        "train_split": "train", "val_split": "validation",
        "text_fields": ["sentence"],
        "entity_fields": [("subject_entity", "word"), ("object_entity", "word")],
        "label_field": "label",
    },
}


# ─────────────────────────────────────────────
def load_task_data(task: str) -> Tuple[List, List]:
    cfg = TASK_CONFIGS[task]
    if cfg["loader"] == "tsv":
        train = _load_tsv(cfg["train_path"], cfg["text_fields"], cfg["label_field"])
        val = _load_tsv(cfg["val_path"], cfg["text_fields"], cfg["label_field"])
    else:
        from datasets import load_dataset
        ds_train = load_dataset(cfg["hf_name"], cfg["hf_subset"], split=cfg["train_split"])
        ds_val = load_dataset(cfg["hf_name"], cfg["hf_subset"], split=cfg["val_split"])
        train = _extract_hf(ds_train, cfg)
        val = _extract_hf(ds_val, cfg)
    return train, val


def _load_tsv(path: str, text_fields: List[str], label_field: str):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split("\t")
        col_text = [header.index(t) for t in text_fields]
        col_label = header.index(label_field)
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) <= max(col_text + [col_label]):
                continue
            texts = [parts[i] for i in col_text]
            if not all(t.strip() for t in texts):
                continue
            try:
                label = int(parts[col_label])
            except ValueError:
                continue
            items.append((texts, label))
    return items


def _extract_hf(ds, cfg):
    items = []
    entity_fields = cfg.get("entity_fields", [])
    for row in ds:
        texts = [row[k] for k in cfg["text_fields"]]
        for outer, inner in entity_fields:
            v = row.get(outer)
            if isinstance(v, dict) and inner in v:
                texts.append(str(v[inner]))
        if not all(isinstance(t, str) and t.strip() for t in texts):
            continue
        lbl = row[cfg["label_field"]]
        if cfg.get("label_nested"):
            lbl = lbl[cfg["label_nested"]]
        if cfg["type"] == "regression":
            label = float(lbl)
        else:
            label = int(lbl)
        items.append((texts, label))
    return items


# ─────────────────────────────────────────────
class DownstreamDatasetBinary(Dataset):
    """bbpe_ids 직접 사용. pair: [BOS] s1 [EOS] [BOS] s2 [EOS]."""

    def __init__(self, items, task: str, bbpe, max_patches: int,
                 bos_id: int, eos_id: int):
        cfg = TASK_CONFIGS[task]
        self.items = items
        self.pair = cfg["pair"]
        self.is_regression = (cfg["type"] == "regression")
        self.bbpe = bbpe
        self.P = max_patches
        self.bos_id = bos_id
        self.eos_id = eos_id

        # BBPE prefetch
        rust_tok = self.bbpe.backend_tokenizer
        all_texts = []
        self._idx_ranges = []
        pos = 0
        for texts, _ in items:
            start = pos
            all_texts.extend(texts)
            pos += len(texts)
            self._idx_ranges.append((start, pos))
        encs = rust_tok.encode_batch(all_texts, add_special_tokens=False)
        self._bbpe_ids = [e.ids for e in encs]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int):
        _, label = self.items[idx]
        start, end = self._idx_ranges[idx]

        # 각 문장의 BBPE id 리스트
        ids_per_sent = [list(self._bbpe_ids[i]) for i in range(start, end)]

        # [BOS] s1_ids [EOS] [BOS] s2_ids [EOS] (pair) or [BOS] s [EOS]
        # 자리 확보: 문장 수 n 에 대해 특수 토큰 2n 개 예약
        n_sent = len(ids_per_sent)
        budget = self.P - 2 * n_sent
        if budget < 0:
            # 극단: 문장 너무 많으면 토큰 0 → 방어
            budget = 0
        # 각 문장 길이 균등 분배
        per_sent_max = max(budget // max(n_sent, 1), 0)

        seq_ids: List[int] = []
        for ids in ids_per_sent:
            seq_ids.append(self.bos_id)
            trimmed = ids[:per_sent_max]
            seq_ids.extend(trimmed)
            seq_ids.append(self.eos_id)
        # 최종 truncate (안전)
        if len(seq_ids) > self.P:
            seq_ids = seq_ids[: self.P - 1] + [self.eos_id]

        P = self.P
        bbpe_ids = torch.zeros(P, dtype=torch.long)
        token_pad_mask = torch.zeros(P, dtype=torch.bool)
        L = min(len(seq_ids), P)
        if L > 0:
            bbpe_ids[:L] = torch.tensor(seq_ids[:L], dtype=torch.long)
            token_pad_mask[:L] = True

        label_t = (torch.tensor(label, dtype=torch.float32) if self.is_regression
                   else torch.tensor(label, dtype=torch.long))

        return {
            "bbpe_ids": bbpe_ids,
            "token_pad_mask": token_pad_mask,
            "label": label_t,
        }


def collate(samples):
    return {
        "bbpe_ids": torch.stack([s["bbpe_ids"] for s in samples]),
        "token_pad_mask": torch.stack([s["token_pad_mask"] for s in samples]),
        "label": torch.stack([s["label"] for s in samples]),
    }


# ─────────────────────────────────────────────
class BinaryDownstreamHead(nn.Module):
    def __init__(self, electra: BinaryElectra, n_outputs: int,
                 dropout: float = 0.1, is_regression: bool = False):
        super().__init__()
        self.bbpe_bits = electra.bbpe_bits
        self.vocab_size = electra.vocab_size
        self.token_embedding = electra.token_embedding  # variant A 면 None
        self.bit_proj = electra.bit_proj
        self.pos_emb = electra.pos_emb
        self.emb_layer_norm = electra.emb_layer_norm
        self.emb_dropout = electra.emb_dropout
        self.disc_hidden_proj = electra.disc_hidden_proj
        self.discriminator = electra.discriminator
        self.is_regression = is_regression
        hidden = electra.hidden_size

        if is_regression:
            self.head = nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, n_outputs),
            )
            for m in self.head:
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, std=0.02)
                    nn.init.zeros_(m.bias)
            self.head[-1].bias.data.fill_(2.5)
        else:
            self.head = nn.Linear(hidden, n_outputs)
            nn.init.normal_(self.head.weight, std=0.02)
            nn.init.zeros_(self.head.bias)
        self.head_dropout = nn.Dropout(dropout)

    def forward(self, bbpe_ids: torch.Tensor, token_pad_mask: torch.Tensor):
        B, P = bbpe_ids.shape
        bits = int_to_bits(bbpe_ids, self.bbpe_bits) * 2.0 - 1.0
        if self.token_embedding is not None:
            safe_ids = bbpe_ids.clamp(max=self.vocab_size)
            x = torch.cat([bits, self.token_embedding(safe_ids)], dim=-1)
        else:
            x = bits
        e = self.bit_proj(x)
        positions = torch.arange(P, device=bbpe_ids.device).unsqueeze(0).expand(B, -1)
        e = e + self.pos_emb(positions)
        e = self.emb_layer_norm(e)
        e = self.emb_dropout(e)
        e = e * token_pad_mask.unsqueeze(-1).to(e.dtype)

        h = self.disc_hidden_proj(e)
        h = self.discriminator(h, token_pad_mask)  # [B, P, hidden]

        if self.is_regression:
            mask_f = token_pad_mask.unsqueeze(-1).to(h.dtype)
            pooled = (h * mask_f).sum(1) / mask_f.sum(1).clamp(min=1)
            pooled = self.head_dropout(pooled)
            return self.head(pooled)
        else:
            cls_vec = h[:, 0, :]  # BOS pool
            cls_vec = self.head_dropout(cls_vec)
            return self.head(cls_vec)


# ─────────────────────────────────────────────
@torch.no_grad()
def evaluate_classification(model, loader, device, amp_dtype, use_amp):
    model.eval()
    correct = 0; total = 0; loss_sum = 0.0; n_batches = 0
    for batch in loader:
        bbpe_ids = batch["bbpe_ids"].to(device, non_blocking=True)
        token_pad_mask = batch["token_pad_mask"].to(device, non_blocking=True)
        label = batch["label"].to(device, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
            logits = model(bbpe_ids, token_pad_mask)
            loss = F.cross_entropy(logits, label)
        pred = logits.argmax(-1)
        correct += (pred == label).sum().item()
        total += label.size(0)
        loss_sum += loss.item()
        n_batches += 1
    model.train()
    return {"acc": correct / max(total, 1),
            "loss": loss_sum / max(n_batches, 1), "n": total}


@torch.no_grad()
def evaluate_regression(model, loader, device, amp_dtype, use_amp):
    model.eval()
    preds_all = []; labels_all = []; loss_sum = 0.0; n_batches = 0
    for batch in loader:
        bbpe_ids = batch["bbpe_ids"].to(device, non_blocking=True)
        token_pad_mask = batch["token_pad_mask"].to(device, non_blocking=True)
        label = batch["label"].to(device, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
            out = model(bbpe_ids, token_pad_mask).squeeze(-1)
            loss = F.mse_loss(out, label)
        preds_all.append(out.float().cpu().numpy())
        labels_all.append(label.cpu().numpy())
        loss_sum += loss.item()
        n_batches += 1
    preds = np.concatenate(preds_all); labels = np.concatenate(labels_all)
    valid = np.isfinite(preds)
    if (~valid).any():
        preds = preds[valid]; labels = labels[valid]
    if len(preds) < 2 or preds.std() < 1e-9:
        pearson = spearman = float("nan")
    else:
        pearson = float(_pearson_np(preds, labels))
        spearman = float(_pearson_np(_rank(preds), _rank(labels)))
    model.train()
    return {"pearson": pearson, "spearman": spearman,
            "mse": loss_sum / max(n_batches, 1), "n": len(preds)}


# ─────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", type=str, required=True, choices=list(TASK_CONFIGS.keys()))
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--max_patches", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--eval_batch_size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--warmup_ratio", type=float, default=0.1)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--log_every", type=int, default=100)
    ap.add_argument("--no_tf32", action="store_true")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--tokenizer_path", type=str, default=None,
                    help="HF id 또는 로컬 디렉토리. 미지정 시 ckpt args 에서 자동 감지.")
    args = ap.parse_args()

    if not args.no_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.bfloat16 if args.bf16 else torch.float32
    use_amp = args.bf16 and device.type == "cuda"

    cfg = TASK_CONFIGS[args.task]
    print(f"[Task] {args.task} — type={cfg['type']}, pair={cfg['pair']}")

    print(f"[Load] {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    saved_args = ckpt.get("args", {})
    print(f"[Ckpt] step={ckpt.get('step')}, "
          f"hidden={saved_args.get('hidden_size')}, "
          f"disc_L={saved_args.get('disc_layers')}, "
          f"bits={saved_args.get('bbpe_bits')}")

    electra = BinaryElectra(
        vocab_size=saved_args.get("vocab_size", 153600),
        bbpe_bits=saved_args.get("bbpe_bits", 18),
        embedding_size=saved_args.get("embedding_size", 128),
        hidden_size=saved_args.get("hidden_size", 256),
        n_heads=saved_args.get("n_heads", 4),
        d_ff=saved_args.get("d_ff", 1024),
        gen_layers=saved_args.get("gen_layers", 14),
        disc_layers=saved_args.get("disc_layers", 14),
        dropout=saved_args.get("dropout", 0.1),
        max_patches=saved_args.get("max_patches", 512),
        gen_loss_weight=saved_args.get("gen_loss_weight", 50.0),
        embedding_dim_k=saved_args.get("embedding_dim_k", 0),
    )
    missing, unexpected = electra.load_state_dict(ckpt["model"], strict=False)
    print(f"[State] missing={len(missing)}, unexpected={len(unexpected)}")

    n_out = 1 if cfg["type"] == "regression" else cfg["n_classes"]
    model = BinaryDownstreamHead(
        electra, n_outputs=n_out,
        dropout=0.1, is_regression=(cfg["type"] == "regression"),
    ).to(device)
    if args.compile:
        import torch._dynamo as _dynamo
        _dynamo.config.suppress_errors = True
        _dynamo.config.cache_size_limit = 64
        model = torch.compile(model)
        print("[Compile] torch.compile 활성")
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] trainable={trainable/1e6:.2f}M")

    print(f"[Data] loading {args.task}...")
    train_items, val_items = load_task_data(args.task)
    print(f"[Data] train={len(train_items)}, val={len(val_items)}")

    tok_path = args.tokenizer_path or saved_args.get(
        "tokenizer_path", "LGAI-EXAONE/K-EXAONE-236B-A23B"
    )
    print(f"[Tok] 로드: {tok_path}")
    bbpe = load_bbpe_tokenizer(tok_path)
    bos_id = int(bbpe.bos_token_id)
    eos_id = int(bbpe.eos_token_id)

    train_ds = DownstreamDatasetBinary(train_items, args.task, bbpe,
                                        max_patches=args.max_patches,
                                        bos_id=bos_id, eos_id=eos_id)
    val_ds = DownstreamDatasetBinary(val_items, args.task, bbpe,
                                      max_patches=args.max_patches,
                                      bos_id=bos_id, eos_id=eos_id)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.eval_batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )

    total_steps = len(train_loader) * args.epochs
    warmup_steps = max(int(total_steps * args.warmup_ratio), 1)
    print(f"[Sched] total_steps={total_steps}, warmup={warmup_steps}, lr={args.lr}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
        betas=(0.9, 0.999), eps=1e-8,
    )

    def lr_at(step):
        if step < warmup_steps:
            return args.lr * step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return args.lr * max(1 - progress, 0.0)

    eval_fn = (evaluate_regression if cfg["type"] == "regression"
               else evaluate_classification)
    loss_fn = (F.mse_loss if cfg["type"] == "regression"
               else F.cross_entropy)
    primary_metric = "spearman" if cfg["type"] == "regression" else "acc"

    model.train()
    global_step = 0
    best_metric = -1e9
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        loss_sum = 0.0; n_log = 0
        for batch in train_loader:
            bbpe_ids = batch["bbpe_ids"].to(device, non_blocking=True)
            token_pad_mask = batch["token_pad_mask"].to(device, non_blocking=True)
            label = batch["label"].to(device, non_blocking=True)

            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                out = model(bbpe_ids, token_pad_mask)
                if cfg["type"] == "regression":
                    loss = loss_fn(out.squeeze(-1), label)
                else:
                    loss = loss_fn(out, label)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            for pg in optimizer.param_groups:
                pg["lr"] = lr_at(global_step)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            loss_sum += loss.item(); n_log += 1; global_step += 1

            if global_step % args.log_every == 0:
                elapsed = time.time() - t0
                sps = global_step / max(elapsed, 1e-6)
                print(f"ep{epoch} step {global_step:>5d}/{total_steps} | "
                      f"loss {loss_sum/n_log:.4f} | "
                      f"lr {lr_at(global_step):.2e} | "
                      f"{sps:.2f} step/s", flush=True)
                loss_sum = 0.0; n_log = 0

        metrics = eval_fn(model, val_loader, device, amp_dtype, use_amp)
        print(f"[Eval epoch {epoch}] " +
              " | ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                         for k, v in metrics.items()), flush=True)
        current = metrics[primary_metric]
        if current > best_metric:
            best_metric = current

    print(f"\n[Done] {args.task} best {primary_metric}={best_metric:.4f}")


if __name__ == "__main__":
    main()
