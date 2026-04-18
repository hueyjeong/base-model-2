"""KoELECTRA NER fine-tune (KLUE-NER, BIO 13-tag, entity-level F1)."""
from __future__ import annotations

import argparse
import os
import random
import re
import sys
import time
from typing import List

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

NER_LABELS = ["B-DT", "I-DT", "B-LC", "I-LC", "B-OG", "I-OG",
              "B-PS", "I-PS", "B-QT", "I-QT", "B-TI", "I-TI", "O"]
NER_O = 12
N_NER = len(NER_LABELS)


def load_klue_ner():
    from datasets import load_dataset
    train = load_dataset("klue", "ner", split="train")
    val = load_dataset("klue", "ner", split="validation")
    return train, val


# ─────────────────────────────────────────────
# Char-level NER tag → BBPE token tag (decode length 누적)
# ─────────────────────────────────────────────
def align_char_tags_to_bbpe(sentence: str, char_tags: List[int],
                             bbpe, max_chars: int = None):
    """sentence 와 char-level ner_tags → (bbpe_ids, bbpe_tags).

    각 BBPE 토큰의 첫 char 의 tag 채택 (FIRST 전략).
    I-X 가 토큰 시작에 오면 B-X 로 변환 (entity 시작 보장).
    """
    bbpe_ids = bbpe.encode(sentence, add_special_tokens=False)
    bbpe_tags = []
    char_pos = 0
    for tid in bbpe_ids:
        tok_str = bbpe.decode([tid])
        if char_pos < len(char_tags):
            tag = char_tags[char_pos]
            if tag != NER_O and tag % 2 == 1:  # I-X
                tag = tag - 1                  # → B-X
            bbpe_tags.append(tag)
        else:
            bbpe_tags.append(NER_O)
        char_pos += len(tok_str)
    return bbpe_ids, bbpe_tags


# ─────────────────────────────────────────────
class NERDataset(Dataset):
    def __init__(self, hf_ds, bbpe, jamo, max_patches: int, max_jamo: int):
        self.bbpe = bbpe
        self.jamo = jamo
        self.P = max_patches
        self.S = max_jamo
        self.items = []
        for ex in hf_ds:
            sent = ex["sentence"]
            tags = ex["ner_tags"]
            if not sent.strip():
                continue
            ids, btags = align_char_tags_to_bbpe(sent, tags, bbpe)
            self.items.append((ids, btags))
        self._tok_cache: dict = {}

    def _decompose_id(self, tid):
        entry = self._tok_cache.get(tid)
        if entry is not None:
            return list(entry)
        tok_str = self.bbpe.decode([tid])
        base = decompose_token(tok_str, self.jamo)
        if len(base) <= self.S:
            entry = (base,)
        else:
            parts_seqs = []
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

    def __getitem__(self, idx):
        bbpe_ids, bbpe_tags = self.items[idx]
        token_units = []
        for tid, tag in zip(bbpe_ids, bbpe_tags):
            seqs = self._decompose_id(tid)
            for seq in seqs:
                token_units.append((seq, tag))

        all_units = [([JAMO_BOS], NER_O, True)]
        for seq, tag in token_units:
            if len(all_units) + 1 >= self.P:
                break
            all_units.append((seq, tag, False))
        all_units.append(([JAMO_EOS], NER_O, True))

        P, S = self.P, self.S
        jamo_ids = torch.zeros(P, S, dtype=torch.long)
        jamo_mask = torch.zeros(P, S, dtype=torch.bool)
        token_pad_mask = torch.zeros(P, dtype=torch.bool)
        token_tags = torch.full((P,), -100, dtype=torch.long)

        for p, (seq, tag, is_sp) in enumerate(all_units):
            if is_sp:
                jamo_ids[p, :] = seq[0]
                jamo_mask[p, :] = True
            else:
                L = min(len(seq), S)
                if L > 0:
                    jamo_ids[p, :L] = torch.tensor(seq[:L], dtype=torch.long)
                    jamo_mask[p, :L] = True
            token_pad_mask[p] = True
            if not is_sp:
                token_tags[p] = tag

        return {"jamo_ids": jamo_ids, "jamo_mask": jamo_mask,
                "token_pad_mask": token_pad_mask, "token_tags": token_tags}


def collate(samples):
    return {
        "jamo_ids": torch.stack([s["jamo_ids"] for s in samples]),
        "jamo_mask": torch.stack([s["jamo_mask"] for s in samples]),
        "token_pad_mask": torch.stack([s["token_pad_mask"] for s in samples]),
        "token_tags": torch.stack([s["token_tags"] for s in samples]),
    }


# ─────────────────────────────────────────────
class NERHead(nn.Module):
    def __init__(self, electra: JamoKoElectra, n_tags: int, dropout: float = 0.1):
        super().__init__()
        self.codec = electra.codec
        self.emb_proj = electra.emb_proj
        self.pos_emb = electra.pos_emb
        self.emb_layer_norm = electra.emb_layer_norm
        self.emb_dropout = electra.emb_dropout
        self.disc_hidden_proj = electra.disc_hidden_proj
        self.discriminator = electra.discriminator
        self.head = nn.Linear(electra.hidden_size, n_tags)
        self.dropout = nn.Dropout(dropout)
        nn.init.normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    def forward(self, jamo_ids, jamo_mask, token_pad_mask):
        B, P, S = jamo_ids.shape
        z_flat = self.codec.encode(jamo_ids.reshape(B * P, S),
                                    jamo_mask.reshape(B * P, S))
        z = z_flat.view(B, P, -1)
        e = self.emb_proj(z)
        positions = torch.arange(P, device=z.device).unsqueeze(0).expand(B, -1)
        e = e + self.pos_emb(positions)
        e = self.emb_layer_norm(e)
        e = self.emb_dropout(e)
        e = e * token_pad_mask.unsqueeze(-1).to(e.dtype)
        h = self.disc_hidden_proj(e)
        h = self.discriminator(h, token_pad_mask)
        h = self.dropout(h)
        return self.head(h)


# ─────────────────────────────────────────────
def bio_to_entities(tags):
    """tag list (BIO str) → set of (etype, start, end)."""
    entities = []
    cur = None
    for i, t in enumerate(tags):
        if t == "O":
            if cur:
                entities.append((cur[0], cur[1], i))
                cur = None
        elif t.startswith("B-"):
            if cur:
                entities.append((cur[0], cur[1], i))
            cur = (t[2:], i)
        elif t.startswith("I-"):
            if cur and cur[0] == t[2:]:
                pass
            else:
                if cur:
                    entities.append((cur[0], cur[1], i))
                cur = (t[2:], i)
    if cur:
        entities.append((cur[0], cur[1], len(tags)))
    return set(entities)


def entity_f1(pred_seqs, true_seqs):
    tp = fp = fn = 0
    for ps, ts in zip(pred_seqs, true_seqs):
        pe = bio_to_entities(ps)
        te = bio_to_entities(ts)
        tp += len(pe & te)
        fp += len(pe - te)
        fn += len(te - pe)
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    f1 = 2 * p * r / max(p + r, 1e-9)
    return {"precision": p, "recall": r, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


@torch.no_grad()
def evaluate(model, loader, device, amp_dtype, use_amp):
    model.eval()
    pred_seqs, true_seqs = [], []
    loss_sum = 0.0
    n_batches = 0
    for batch in loader:
        jamo_ids = batch["jamo_ids"].to(device, non_blocking=True)
        jamo_mask = batch["jamo_mask"].to(device, non_blocking=True)
        token_pad_mask = batch["token_pad_mask"].to(device, non_blocking=True)
        token_tags = batch["token_tags"].to(device, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
            logits = model(jamo_ids, jamo_mask, token_pad_mask)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                    token_tags.reshape(-1),
                                    ignore_index=-100)
        loss_sum += loss.item()
        n_batches += 1
        preds = logits.argmax(-1)
        for b in range(preds.size(0)):
            mask = token_tags[b] != -100
            p_tags = [NER_LABELS[t.item()] for t in preds[b][mask]]
            t_tags = [NER_LABELS[t.item()] for t in token_tags[b][mask]]
            pred_seqs.append(p_tags)
            true_seqs.append(t_tags)
    metrics = entity_f1(pred_seqs, true_seqs)
    metrics["loss"] = loss_sum / max(n_batches, 1)
    metrics["n"] = sum(len(s) for s in pred_seqs)
    model.train()
    return metrics


def main():
    ap = argparse.ArgumentParser()
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
    ap.add_argument("--log_every", type=int, default=200)
    args = ap.parse_args()

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

    print(f"[Load] {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    saved_args = ckpt.get("args", {})

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
    electra.load_state_dict(ckpt["model"], strict=False)
    model = NERHead(electra, n_tags=N_NER, dropout=0.1).to(device)
    print(f"[Model] trainable={sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.2f}M")

    bbpe = load_bbpe_tokenizer()
    jamo = JamoTokenizer()

    print("[Data] loading klue/ner...")
    train_hf, val_hf = load_klue_ner()
    max_jamo = saved_args.get("max_jamo_per_token", 32)
    train_ds = NERDataset(train_hf, bbpe, jamo, args.max_patches, max_jamo)
    val_ds = NERDataset(val_hf, bbpe, jamo, args.max_patches, max_jamo)
    print(f"[Data] train={len(train_ds)}, val={len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, collate_fn=collate,
                              pin_memory=(device.type == "cuda"),
                              persistent_workers=(args.num_workers > 0))
    val_loader = DataLoader(val_ds, batch_size=args.eval_batch_size,
                            num_workers=args.num_workers, collate_fn=collate,
                            pin_memory=(device.type == "cuda"),
                            persistent_workers=(args.num_workers > 0))

    total_steps = len(train_loader) * args.epochs
    warmup_steps = max(int(total_steps * args.warmup_ratio), 1)
    print(f"[Sched] total_steps={total_steps}, warmup={warmup_steps}, lr={args.lr}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                   weight_decay=args.weight_decay,
                                   betas=(0.9, 0.999), eps=1e-8)

    def lr_at(step):
        if step < warmup_steps:
            return args.lr * step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return args.lr * max(1 - progress, 0.0)

    model.train()
    global_step = 0
    best = -1.0
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        loss_sum = 0.0
        n = 0
        for batch in train_loader:
            jamo_ids = batch["jamo_ids"].to(device, non_blocking=True)
            jamo_mask = batch["jamo_mask"].to(device, non_blocking=True)
            token_pad_mask = batch["token_pad_mask"].to(device, non_blocking=True)
            token_tags = batch["token_tags"].to(device, non_blocking=True)
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                logits = model(jamo_ids, jamo_mask, token_pad_mask)
                loss = F.cross_entropy(logits.reshape(-1, N_NER),
                                        token_tags.reshape(-1),
                                        ignore_index=-100)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            for pg in optimizer.param_groups:
                pg["lr"] = lr_at(global_step)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            loss_sum += loss.item()
            n += 1
            global_step += 1
            if global_step % args.log_every == 0:
                sps = global_step / max(time.time() - t0, 1e-6)
                print(f"ep{epoch} step {global_step:>5d}/{total_steps} | "
                      f"loss {loss_sum/n:.4f} | lr {lr_at(global_step):.2e} | "
                      f"{sps:.2f} step/s", flush=True)
                loss_sum = 0.0
                n = 0
        m = evaluate(model, val_loader, device, amp_dtype, use_amp)
        print(f"[Eval epoch {epoch}] precision={m['precision']:.4f} "
              f"recall={m['recall']:.4f} f1={m['f1']:.4f} "
              f"loss={m['loss']:.4f} (n={m['n']})", flush=True)
        if m["f1"] > best:
            best = m["f1"]

    print(f"\n[Done] klue_ner best f1={best:.4f}")


if __name__ == "__main__":
    main()
