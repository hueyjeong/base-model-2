"""NSMC 감성분류 fine-tune (discriminator 재활용).

사용 예:
    python -m koelectra.finetune_nsmc \
        --ckpt checkpoints/electra_step_10000.pt \
        --train_tsv corpus/nsmc/ratings_train.txt \
        --test_tsv  corpus/nsmc/ratings_test.txt \
        --max_patches 64 --batch_size 64 --epochs 3 --lr 2e-5 --bf16

Discriminator 전용 경로 재활용 (codec + emb_proj + pos_emb + disc_hidden_proj + discriminator)
→ BOS 토큰 representation 에 Linear(hidden, 2) 분류 head.
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

from tok.jamo_tokenizer import JamoTokenizer  # noqa: E402
from koelectra.data.bbpe_token_dataset import (  # noqa: E402
    load_bbpe_tokenizer, decompose_token,
    JAMO_BOS, JAMO_EOS, JAMO_PAD,
)
from koelectra.model.electra import JamoKoElectra  # noqa: E402


# ─────────────────────────────────────────────
# Dataset (NSMC TSV → per-sample 단일 문서)
# ─────────────────────────────────────────────
class NSMCDataset(Dataset):
    def __init__(self, tsv_path: str, bbpe, jamo, max_patches: int, max_jamo: int):
        self.bbpe = bbpe
        self.jamo = jamo
        self.P = max_patches
        self.S = max_jamo
        self.items: List[Tuple[str, int]] = []
        with open(tsv_path, "r", encoding="utf-8") as f:
            f.readline()  # header
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 3:
                    continue
                text, label = parts[1], parts[2]
                if not text.strip():
                    continue
                try:
                    label = int(label)
                except ValueError:
                    continue
                self.items.append((text, label))

        # BBPE encode 전체 prefetch (작은 데이터라 메모리 여유)
        texts = [t for t, _ in self.items]
        rust_tok = self.bbpe.backend_tokenizer
        encs = rust_tok.encode_batch(texts, add_special_tokens=False)
        self.bbpe_ids_cache = [e.ids for e in encs]

        # Jamo 분해 cache
        self._tok_cache: dict = {}

    def _decompose_id(self, tid: int) -> List[List[int]]:
        entry = self._tok_cache.get(tid)
        if entry is not None:
            return list(entry)
        tok_str = self.bbpe.decode([tid])
        base = decompose_token(tok_str, self.jamo)
        if len(base) <= self.S:
            entry = (base,)
        else:
            import re
            parts_seqs: List[List[int]] = []
            parts = re.split(r"( )", tok_str)
            for part in parts:
                if not part:
                    continue
                pj = decompose_token(part, self.jamo)
                if len(pj) <= self.S:
                    parts_seqs.append(pj)
                else:
                    for ch in part:
                        cj = decompose_token(ch, self.jamo)
                        if cj:
                            parts_seqs.append(cj[:self.S])
            entry = tuple(parts_seqs)
        self._tok_cache[tid] = entry
        return list(entry)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int):
        text, label = self.items[idx]
        bbpe_ids = self.bbpe_ids_cache[idx]

        jamo_seqs: List[List[int]] = []
        for tid in bbpe_ids:
            jamo_seqs.extend(self._decompose_id(tid))

        # [BOS] + 토큰들 + [EOS], max_patches 초과 시 truncate (EOS 는 유지)
        tokens = [([JAMO_BOS], True)]
        for seq in jamo_seqs:
            if len(tokens) + 1 >= self.P:  # EOS 자리 확보
                break
            tokens.append((seq, False))
        tokens.append(([JAMO_EOS], True))

        P, S = self.P, self.S
        jamo_ids = torch.zeros(P, S, dtype=torch.long)
        jamo_mask = torch.zeros(P, S, dtype=torch.bool)
        token_pad_mask = torch.zeros(P, dtype=torch.bool)

        for p, (seq, is_sp) in enumerate(tokens):
            if is_sp:
                jamo_ids[p, :] = seq[0]
                jamo_mask[p, :] = True
            else:
                L = min(len(seq), S)
                if L > 0:
                    jamo_ids[p, :L] = torch.tensor(seq[:L], dtype=torch.long)
                    jamo_mask[p, :L] = True
            token_pad_mask[p] = True

        return {
            "jamo_ids": jamo_ids,
            "jamo_mask": jamo_mask,
            "token_pad_mask": token_pad_mask,
            "label": label,
        }


def collate(samples):
    return {
        "jamo_ids": torch.stack([s["jamo_ids"] for s in samples]),
        "jamo_mask": torch.stack([s["jamo_mask"] for s in samples]),
        "token_pad_mask": torch.stack([s["token_pad_mask"] for s in samples]),
        "label": torch.tensor([s["label"] for s in samples], dtype=torch.long),
    }


# ─────────────────────────────────────────────
# Classifier wrapper (discriminator 경로만)
# ─────────────────────────────────────────────
class NSMCClassifier(nn.Module):
    def __init__(self, electra: JamoKoElectra, n_classes: int = 2, dropout: float = 0.1):
        super().__init__()
        self.codec = electra.codec
        self.emb_proj = electra.emb_proj
        self.pos_emb = electra.pos_emb
        self.emb_layer_norm = electra.emb_layer_norm
        self.emb_dropout = electra.emb_dropout
        self.disc_hidden_proj = electra.disc_hidden_proj
        self.discriminator = electra.discriminator
        self.cls_head = nn.Linear(electra.hidden_size, n_classes)
        self.cls_dropout = nn.Dropout(dropout)
        nn.init.normal_(self.cls_head.weight, std=0.02)
        nn.init.zeros_(self.cls_head.bias)

    def forward(self, jamo_ids, jamo_mask, token_pad_mask):
        B, P, S = jamo_ids.shape
        # codec encode (per-token)
        z_flat = self.codec.encode(
            jamo_ids.reshape(B * P, S),
            jamo_mask.reshape(B * P, S),
        )
        z = z_flat.view(B, P, -1)

        # embedding + pos + LN
        e = self.emb_proj(z)
        positions = torch.arange(P, device=z.device).unsqueeze(0).expand(B, -1)
        e = e + self.pos_emb(positions)
        e = self.emb_layer_norm(e)
        e = self.emb_dropout(e)
        e = e * token_pad_mask.unsqueeze(-1).to(e.dtype)

        # discriminator transformer
        h = self.disc_hidden_proj(e)
        h = self.discriminator(h, token_pad_mask)  # [B, P, hidden]

        # pool: BOS 위치 (index 0) representation
        cls_vec = h[:, 0, :]  # [B, hidden]
        cls_vec = self.cls_dropout(cls_vec)
        logits = self.cls_head(cls_vec)  # [B, n_classes]
        return logits


# ─────────────────────────────────────────────
# Train / Eval
# ─────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, loader, device, amp_dtype, use_amp):
    model.eval()
    total = 0
    correct = 0
    loss_sum = 0.0
    n_batches = 0
    for batch in loader:
        jamo_ids = batch["jamo_ids"].to(device, non_blocking=True)
        jamo_mask = batch["jamo_mask"].to(device, non_blocking=True)
        token_pad_mask = batch["token_pad_mask"].to(device, non_blocking=True)
        label = batch["label"].to(device, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
            logits = model(jamo_ids, jamo_mask, token_pad_mask)
            loss = F.cross_entropy(logits, label)
        pred = logits.argmax(-1)
        correct += (pred == label).sum().item()
        total += label.size(0)
        loss_sum += loss.item()
        n_batches += 1
    model.train()
    return {
        "acc": correct / max(total, 1),
        "loss": loss_sum / max(n_batches, 1),
        "n": total,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--train_tsv", type=str, default="corpus/nsmc/ratings_train.txt")
    ap.add_argument("--test_tsv", type=str, default="corpus/nsmc/ratings_test.txt")
    ap.add_argument("--max_patches", type=int, default=64)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--eval_batch_size", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--warmup_ratio", type=float, default=0.1)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--no_tf32", action="store_true")
    args = ap.parse_args()

    if not args.no_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.bfloat16 if args.bf16 else torch.float32
    use_amp = args.bf16 and device.type == "cuda"

    # ── 체크포인트 ──
    print(f"[Load] {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    saved_args = ckpt.get("args", {})
    print(f"[Ckpt] step={ckpt.get('step')}, "
          f"hidden_size={saved_args.get('hidden_size')}, "
          f"disc_layers={saved_args.get('disc_layers')}")

    # Electra 재구성 (체크포인트 hparams)
    electra = JamoKoElectra(
        jamo_vocab=330,
        codec_d_model=saved_args.get("codec_d_model", 256),
        codec_n_enc_layers=saved_args.get("codec_n_enc_layers", 5),
        codec_n_dec_layers=saved_args.get("codec_n_dec_layers", 5),
        codec_kernel_size=saved_args.get("codec_kernel_size", 5),
        max_jamo_per_token=saved_args.get("max_jamo_per_token", 32),
        codec_dropout=saved_args.get("dropout", 0.1),
        embedding_size=saved_args.get("embedding_size", 128),
        hidden_size=saved_args.get("hidden_size", 256),
        n_heads=saved_args.get("n_heads", 4),
        d_ff=saved_args.get("d_ff", 1024),
        gen_layers=saved_args.get("gen_layers", 12),
        disc_layers=saved_args.get("disc_layers", 12),
        dropout=saved_args.get("dropout", 0.1),
        max_patches=saved_args.get("max_patches", 512),
        gen_loss_weight=saved_args.get("gen_loss_weight", 50.0),
    )
    missing, unexpected = electra.load_state_dict(ckpt["model"], strict=False)
    print(f"[State] missing={len(missing)}, unexpected={len(unexpected)}")
    if missing:
        print(f"  missing sample: {missing[:3]}")
    if unexpected:
        print(f"  unexpected sample: {unexpected[:3]}")

    model = NSMCClassifier(electra, n_classes=2, dropout=0.1).to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] trainable={trainable/1e6:.2f}M")

    # ── 토크나이저 + 데이터 ──
    print("[Tok] BBPE + Jamo 로드")
    bbpe = load_bbpe_tokenizer()
    jamo = JamoTokenizer()

    print(f"[Data] train: {args.train_tsv}")
    train_ds = NSMCDataset(
        args.train_tsv, bbpe, jamo,
        max_patches=args.max_patches,
        max_jamo=saved_args.get("max_jamo_per_token", 32),
    )
    print(f"[Data] test:  {args.test_tsv}")
    test_ds = NSMCDataset(
        args.test_tsv, bbpe, jamo,
        max_patches=args.max_patches,
        max_jamo=saved_args.get("max_jamo_per_token", 32),
    )
    print(f"[Data] train={len(train_ds)}, test={len(test_ds)}")

    # cache warmup (worker 에서 복제되지만 한 번 돌려 fork 시 공유)
    for i in (0, 1):
        train_ds[i]

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.eval_batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )

    # ── Optimizer + Schedule ──
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

    # ── Train ──
    model.train()
    global_step = 0
    best_test_acc = 0.0
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        loss_sum = 0.0
        corr = 0
        tot = 0
        for i, batch in enumerate(train_loader):
            jamo_ids = batch["jamo_ids"].to(device, non_blocking=True)
            jamo_mask = batch["jamo_mask"].to(device, non_blocking=True)
            token_pad_mask = batch["token_pad_mask"].to(device, non_blocking=True)
            label = batch["label"].to(device, non_blocking=True)

            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                logits = model(jamo_ids, jamo_mask, token_pad_mask)
                loss = F.cross_entropy(logits, label)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            for pg in optimizer.param_groups:
                pg["lr"] = lr_at(global_step)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            pred = logits.argmax(-1)
            corr += (pred == label).sum().item()
            tot += label.size(0)
            loss_sum += loss.item()
            global_step += 1

            if global_step % args.log_every == 0:
                elapsed = time.time() - t0
                sps = global_step / max(elapsed, 1e-6)
                print(f"ep{epoch} step {global_step:>5d}/{total_steps} | "
                      f"loss {loss_sum/args.log_every:.4f} | "
                      f"acc {corr/max(tot,1):.4f} | "
                      f"lr {lr_at(global_step):.2e} | "
                      f"{sps:.2f} step/s", flush=True)
                loss_sum = 0.0
                corr = 0
                tot = 0

        # epoch 끝 test eval
        metrics = evaluate(model, test_loader, device, amp_dtype, use_amp)
        print(f"[Eval epoch {epoch}] test_acc={metrics['acc']:.4f} "
              f"test_loss={metrics['loss']:.4f} (n={metrics['n']})", flush=True)
        if metrics["acc"] > best_test_acc:
            best_test_acc = metrics["acc"]

    print(f"\n[Done] best test_acc={best_test_acc:.4f}")


if __name__ == "__main__":
    main()
