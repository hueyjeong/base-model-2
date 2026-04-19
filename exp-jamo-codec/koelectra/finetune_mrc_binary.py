"""Binary ELECTRA MRC fine-tune (KLUE-MRC, span extraction, EM + F1).

finetune_mrc.py 의 binary 버전. codec 없음, bbpe_ids 직접 입력.

입력 포맷: [BOS] question [EOS] [BOS] context [EOS] (pretrain 과 일치)
  context 토큰 i → 입력 시퀀스 idx = 1 + len(q) + 1 + 1 + i
                                   = len(q) + 3 + i
  (BOS + q + EOS + BOS 이후)
"""
from __future__ import annotations

import argparse
import collections
import os
import random
import re
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


def load_klue_mrc():
    from datasets import load_dataset
    train = load_dataset("klue", "mrc", split="train")
    val = load_dataset("klue", "mrc", split="validation")
    return train, val


# ─────────────────────────────────────────────
def char_span_to_bbpe(context_bbpe_ids: List[int], bbpe,
                       answer_start: int, answer_text: str):
    """context BBPE token 시퀀스에서 정답 시작/끝 token idx 반환."""
    char_pos = 0
    start_tok = end_tok = None
    answer_end_char = answer_start + len(answer_text)
    for ti, tid in enumerate(context_bbpe_ids):
        tok_str = bbpe.decode([tid])
        next_pos = char_pos + len(tok_str)
        if start_tok is None and char_pos <= answer_start < next_pos:
            start_tok = ti
        if start_tok is not None and char_pos < answer_end_char <= next_pos:
            end_tok = ti
            break
        char_pos = next_pos
    if start_tok is None:
        return None, None
    if end_tok is None:
        end_tok = len(context_bbpe_ids) - 1
    return start_tok, end_tok


# ─────────────────────────────────────────────
class MRCDatasetBinary(Dataset):
    """입력: [BOS] q [EOS] [BOS] c [EOS]
    → input_offset (context 의 첫 token 이 들어갈 위치) = 1 + len(q) + 1 + 1 = len(q) + 3
    """

    def __init__(self, hf_ds, bbpe, max_patches: int,
                 bos_id: int, eos_id: int,
                 skip_impossible: bool = True):
        self.bbpe = bbpe
        self.P = max_patches
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.items = []
        rust_tok = bbpe.backend_tokenizer
        skipped = 0
        for ex in hf_ds:
            ctx = ex["context"]
            q = ex["question"]
            ans = ex["answers"]
            if skip_impossible and ex.get("is_impossible", False):
                skipped += 1
                continue
            if not ans["text"] or not ans["answer_start"]:
                skipped += 1
                continue
            answer_text = ans["text"][0]
            answer_start = int(ans["answer_start"][0])
            q_ids = rust_tok.encode(q, add_special_tokens=False).ids
            c_ids = rust_tok.encode(ctx, add_special_tokens=False).ids
            s_tok, e_tok = char_span_to_bbpe(c_ids, bbpe, answer_start, answer_text)
            if s_tok is None:
                skipped += 1
                continue
            # [BOS] q [EOS] [BOS] c [EOS]
            input_offset = 1 + len(q_ids) + 1 + 1
            max_c_len = self.P - input_offset - 1  # 끝 EOS 자리
            if max_c_len <= 0:
                skipped += 1
                continue
            if e_tok >= max_c_len:
                skipped += 1
                continue
            c_ids_trunc = c_ids[:max_c_len]
            start_in_input = input_offset + s_tok
            end_in_input = input_offset + e_tok
            self.items.append((list(q_ids), list(c_ids_trunc),
                               start_in_input, end_in_input,
                               answer_text, ctx))
        self._skipped = skipped

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        q_ids, c_ids, start_pos, end_pos, ans_text, ctx_text = self.items[idx]
        P = self.P

        seq_ids: List[int] = []
        seq_ids.append(self.bos_id)
        seq_ids.extend(q_ids)
        seq_ids.append(self.eos_id)
        seq_ids.append(self.bos_id)
        seq_ids.extend(c_ids)
        seq_ids.append(self.eos_id)

        if len(seq_ids) > P:
            seq_ids = seq_ids[:P]
            seq_ids[-1] = self.eos_id  # truncated 시 EOS 강제

        bbpe_ids = torch.zeros(P, dtype=torch.long)
        token_pad_mask = torch.zeros(P, dtype=torch.bool)
        L = len(seq_ids)
        bbpe_ids[:L] = torch.tensor(seq_ids, dtype=torch.long)
        token_pad_mask[:L] = True

        if start_pos >= L:
            start_pos = 0
            end_pos = 0
        if end_pos >= L:
            end_pos = L - 1

        return {
            "bbpe_ids": bbpe_ids,
            "token_pad_mask": token_pad_mask,
            "start_pos": torch.tensor(start_pos, dtype=torch.long),
            "end_pos": torch.tensor(end_pos, dtype=torch.long),
            "answer_text": ans_text,
            "q_len": len(q_ids),
            "c_ids": c_ids,
        }


def collate(samples):
    return {
        "bbpe_ids": torch.stack([s["bbpe_ids"] for s in samples]),
        "token_pad_mask": torch.stack([s["token_pad_mask"] for s in samples]),
        "start_pos": torch.stack([s["start_pos"] for s in samples]),
        "end_pos": torch.stack([s["end_pos"] for s in samples]),
        "answer_text": [s["answer_text"] for s in samples],
        "q_len": [s["q_len"] for s in samples],
        "c_ids": [s["c_ids"] for s in samples],
    }


# ─────────────────────────────────────────────
class BinaryMRCHead(nn.Module):
    def __init__(self, electra: BinaryElectra):
        super().__init__()
        self.bbpe_bits = electra.bbpe_bits
        self.bit_proj = electra.bit_proj
        self.pos_emb = electra.pos_emb
        self.emb_layer_norm = electra.emb_layer_norm
        self.emb_dropout = electra.emb_dropout
        self.disc_hidden_proj = electra.disc_hidden_proj
        self.discriminator = electra.discriminator
        self.qa_head = nn.Linear(electra.hidden_size, 2)
        nn.init.normal_(self.qa_head.weight, std=0.02)
        nn.init.zeros_(self.qa_head.bias)

    def forward(self, bbpe_ids: torch.Tensor, token_pad_mask: torch.Tensor):
        B, P = bbpe_ids.shape
        bits = int_to_bits(bbpe_ids, self.bbpe_bits) * 2.0 - 1.0
        e = self.bit_proj(bits)
        positions = torch.arange(P, device=bbpe_ids.device).unsqueeze(0).expand(B, -1)
        e = e + self.pos_emb(positions)
        e = self.emb_layer_norm(e)
        e = self.emb_dropout(e)
        e = e * token_pad_mask.unsqueeze(-1).to(e.dtype)
        h = self.disc_hidden_proj(e)
        h = self.discriminator(h, token_pad_mask)
        logits = self.qa_head(h)  # [B, P, 2]
        return logits[..., 0], logits[..., 1]


# ─────────────────────────────────────────────
def _normalize(s: str) -> str:
    s = re.sub(r"[\s\.\,\?\!]", "", s)
    return s.lower()


def korean_f1(pred: str, gold: str) -> float:
    p = list(_normalize(pred))
    g = list(_normalize(gold))
    if not p or not g:
        return 0.0
    common = collections.Counter(p) & collections.Counter(g)
    same = sum(common.values())
    if same == 0:
        return 0.0
    precision = same / len(p)
    recall = same / len(g)
    return 2 * precision * recall / (precision + recall)


def korean_em(pred: str, gold: str) -> float:
    return float(_normalize(pred) == _normalize(gold))


@torch.no_grad()
def evaluate(model, loader, device, amp_dtype, use_amp, bbpe):
    model.eval()
    em_sum = 0.0; f1_sum = 0.0; n = 0; loss_sum = 0.0; n_b = 0
    for batch in loader:
        bbpe_ids = batch["bbpe_ids"].to(device, non_blocking=True)
        token_pad_mask = batch["token_pad_mask"].to(device, non_blocking=True)
        start_pos = batch["start_pos"].to(device, non_blocking=True)
        end_pos = batch["end_pos"].to(device, non_blocking=True)

        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
            sl, el = model(bbpe_ids, token_pad_mask)
            loss = (F.cross_entropy(sl, start_pos) + F.cross_entropy(el, end_pos)) / 2
        loss_sum += loss.item(); n_b += 1

        sl = sl.masked_fill(~token_pad_mask, -1e4)
        el = el.masked_fill(~token_pad_mask, -1e4)
        s_pred = sl.argmax(-1)
        e_pred = el.argmax(-1)

        for b in range(s_pred.size(0)):
            sp = s_pred[b].item()
            ep = e_pred[b].item()
            if ep < sp:
                ep = sp
            q_len = batch["q_len"][b]
            c_ids = batch["c_ids"][b]
            # [BOS] q [EOS] [BOS] c [EOS] → context 첫 토큰 위치 = 1 + q_len + 1 + 1
            input_offset = 1 + q_len + 1 + 1
            ctx_s = sp - input_offset
            ctx_e = ep - input_offset
            if ctx_s < 0 or ctx_e < 0 or ctx_s >= len(c_ids):
                pred_text = ""
            else:
                ctx_e = min(ctx_e, len(c_ids) - 1)
                pred_text = bbpe.decode(c_ids[ctx_s:ctx_e + 1])
            gold = batch["answer_text"][b]
            em_sum += korean_em(pred_text, gold)
            f1_sum += korean_f1(pred_text, gold)
            n += 1
    model.train()
    return {
        "em": em_sum / max(n, 1),
        "f1": f1_sum / max(n, 1),
        "loss": loss_sum / max(n_b, 1),
        "n": n,
    }


# ─────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--max_patches", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--eval_batch_size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--warmup_ratio", type=float, default=0.1)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--log_every", type=int, default=200)
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--tokenizer_path", type=str, default=None)
    args = ap.parse_args()

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

    print(f"[Load] {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    saved_args = ckpt.get("args", {})

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
    electra.load_state_dict(ckpt["model"], strict=False)
    model = BinaryMRCHead(electra).to(device)
    print(f"[Model] trainable={sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.2f}M")
    if args.compile:
        import torch._dynamo as _dynamo
        _dynamo.config.suppress_errors = True
        _dynamo.config.cache_size_limit = 64
        model = torch.compile(model)
        print("[Compile] torch.compile 활성")

    tok_path = args.tokenizer_path or saved_args.get(
        "tokenizer_path", "LGAI-EXAONE/K-EXAONE-236B-A23B"
    )
    print(f"[Tok] 로드: {tok_path}")
    bbpe = load_bbpe_tokenizer(tok_path)
    bos_id = int(bbpe.bos_token_id)
    eos_id = int(bbpe.eos_token_id)

    print("[Data] loading klue/mrc...")
    train_hf, val_hf = load_klue_mrc()
    train_ds = MRCDatasetBinary(train_hf, bbpe, args.max_patches, bos_id, eos_id)
    val_ds = MRCDatasetBinary(val_hf, bbpe, args.max_patches, bos_id, eos_id)
    print(f"[Data] train={len(train_ds)} (skipped {train_ds._skipped}), "
          f"val={len(val_ds)} (skipped {val_ds._skipped})")

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
    best_f1 = -1.0
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        loss_sum = 0.0; n = 0
        for batch in train_loader:
            bbpe_ids = batch["bbpe_ids"].to(device, non_blocking=True)
            token_pad_mask = batch["token_pad_mask"].to(device, non_blocking=True)
            start_pos = batch["start_pos"].to(device, non_blocking=True)
            end_pos = batch["end_pos"].to(device, non_blocking=True)
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                sl, el = model(bbpe_ids, token_pad_mask)
                loss = (F.cross_entropy(sl, start_pos) + F.cross_entropy(el, end_pos)) / 2
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            for pg in optimizer.param_groups:
                pg["lr"] = lr_at(global_step)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            loss_sum += loss.item(); n += 1; global_step += 1
            if global_step % args.log_every == 0:
                sps = global_step / max(time.time() - t0, 1e-6)
                print(f"ep{epoch} step {global_step:>5d}/{total_steps} | "
                      f"loss {loss_sum/n:.4f} | lr {lr_at(global_step):.2e} | "
                      f"{sps:.2f} step/s", flush=True)
                loss_sum = 0.0; n = 0
        m = evaluate(model, val_loader, device, amp_dtype, use_amp, bbpe)
        print(f"[Eval epoch {epoch}] em={m['em']:.4f} f1={m['f1']:.4f} "
              f"loss={m['loss']:.4f} (n={m['n']})", flush=True)
        if m["f1"] > best_f1:
            best_f1 = m["f1"]

    print(f"\n[Done] klue_mrc best f1={best_f1:.4f}")


if __name__ == "__main__":
    main()
