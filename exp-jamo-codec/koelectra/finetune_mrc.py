"""KoELECTRA MRC fine-tune (KLUE-MRC, span extraction, EM + F1)."""
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

from tok.jamo_tokenizer import JamoTokenizer  # noqa: E402
from koelectra.data.bbpe_token_dataset import (  # noqa: E402
    load_bbpe_tokenizer, decompose_token,
    JAMO_BOS, JAMO_EOS, JAMO_PAD,
)
from koelectra.model.electra import JamoKoElectra  # noqa: E402

JAMO_SEP = 5


def load_klue_mrc():
    from datasets import load_dataset
    train = load_dataset("klue", "mrc", split="train")
    val = load_dataset("klue", "mrc", split="validation")
    return train, val


# ─────────────────────────────────────────────
# 정답 char span → context BBPE token idx 매핑 (decode 길이 누적)
# ─────────────────────────────────────────────
def char_span_to_bbpe(context_bbpe_ids: List[int], bbpe,
                       answer_start: int, answer_text: str):
    """context BBPE token 시퀀스에서 정답 시작/끝 token idx 반환.

    answer_start: char 단위 시작 위치 (context 내)
    answer_text: 정답 문자열 (길이로 char 종료 위치 추정)
    """
    char_pos = 0
    start_tok = end_tok = None
    answer_end_char = answer_start + len(answer_text)
    for ti, tid in enumerate(context_bbpe_ids):
        tok_str = bbpe.decode([tid])
        next_pos = char_pos + len(tok_str)
        # 시작 토큰 = 정답 시작 char 가 처음 이 토큰 범위에 들어가는 시점
        if start_tok is None and char_pos <= answer_start < next_pos:
            start_tok = ti
        # 끝 토큰 = 정답 끝 char 가 마지막으로 이 토큰 범위에 들어가는 시점
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
class MRCDataset(Dataset):
    """
    입력 구성: [BOS] question_tokens [SEP] context_tokens [EOS]
    Answer span 은 입력 시퀀스 내 token idx 로 변환되어 저장.
    Truncate: context 가 길면 뒤에서 자름. answer 가 잘리면 sample skip.
    """

    def __init__(self, hf_ds, bbpe, jamo, max_patches: int, max_jamo: int,
                 skip_impossible: bool = True):
        self.bbpe = bbpe
        self.jamo = jamo
        self.P = max_patches
        self.S = max_jamo
        self.items = []  # (q_ids, c_ids, start_tok_in_input, end_tok, answer_text, context_for_eval)
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
            # 입력 시퀀스 layout: [BOS] q [SEP] c [EOS]
            # c 토큰 i → 입력 시퀀스 idx = 1 + len(q) + 1 + i
            input_offset = 1 + len(q_ids) + 1
            # truncation 적용 후 c_ids 길이 제한
            max_c_len = self.P - input_offset - 1  # EOS 자리
            if max_c_len <= 0:
                skipped += 1
                continue
            if e_tok >= max_c_len:
                skipped += 1
                continue
            c_ids_trunc = c_ids[:max_c_len]
            start_in_input = input_offset + s_tok
            end_in_input = input_offset + e_tok
            self.items.append((q_ids, c_ids_trunc, start_in_input, end_in_input,
                                answer_text, ctx))
        self._tok_cache: dict = {}
        self._skipped = skipped

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
        q_ids, c_ids, start_pos, end_pos, ans_text, ctx_text = self.items[idx]
        # decompose 가 토큰 분할을 만들 수 있으나 단순화 위해 token=BBPE 단위 1:1 가정 (그대로 enumerate)
        # 단, jamo_seq 가 multi-segment 면 한 BBPE token 이 multi position 차지 → MRC start/end 어긋남.
        # 이 문제 회피: decompose 결과 첫 segment 만 사용 (split 안 된 것만 학습 대상; 분할 토큰은 그냥 첫 part)
        all_units: List[Tuple[list, bool]] = [([JAMO_BOS], True)]
        for tid in q_ids:
            seqs = self._decompose_id(tid)
            all_units.append((seqs[0], False))
        all_units.append(([JAMO_SEP], True))
        for tid in c_ids:
            seqs = self._decompose_id(tid)
            all_units.append((seqs[0], False))
        all_units.append(([JAMO_EOS], True))

        P, S = self.P, self.S
        if len(all_units) > P:
            all_units = all_units[:P]
            # truncation 으로 EOS 가 사라지면 마지막 토큰을 EOS 로 강제
            all_units[-1] = ([JAMO_EOS], True)

        jamo_ids = torch.zeros(P, S, dtype=torch.long)
        jamo_mask = torch.zeros(P, S, dtype=torch.bool)
        token_pad_mask = torch.zeros(P, dtype=torch.bool)
        for p, (seq, is_sp) in enumerate(all_units):
            if is_sp:
                jamo_ids[p, :] = seq[0]
                jamo_mask[p, :] = True
            else:
                L = min(len(seq), S)
                if L > 0:
                    jamo_ids[p, :L] = torch.tensor(seq[:L], dtype=torch.long)
                    jamo_mask[p, :L] = True
            token_pad_mask[p] = True

        # start/end 가 truncated 영역에 들어갔으면 0 으로 (BOS — 무효 표시)
        if start_pos >= len(all_units):
            start_pos = 0
            end_pos = 0
        if end_pos >= len(all_units):
            end_pos = len(all_units) - 1

        return {
            "jamo_ids": jamo_ids,
            "jamo_mask": jamo_mask,
            "token_pad_mask": token_pad_mask,
            "start_pos": torch.tensor(start_pos, dtype=torch.long),
            "end_pos": torch.tensor(end_pos, dtype=torch.long),
            "answer_text": ans_text,
            # eval: 예측 토큰 idx → context substring 으로 변환할 때 필요한 정보
            "q_len": len(q_ids),
            "c_ids": c_ids,
        }


def collate(samples):
    return {
        "jamo_ids": torch.stack([s["jamo_ids"] for s in samples]),
        "jamo_mask": torch.stack([s["jamo_mask"] for s in samples]),
        "token_pad_mask": torch.stack([s["token_pad_mask"] for s in samples]),
        "start_pos": torch.stack([s["start_pos"] for s in samples]),
        "end_pos": torch.stack([s["end_pos"] for s in samples]),
        "answer_text": [s["answer_text"] for s in samples],
        "q_len": [s["q_len"] for s in samples],
        "c_ids": [s["c_ids"] for s in samples],
    }


# ─────────────────────────────────────────────
class MRCHead(nn.Module):
    def __init__(self, electra: JamoKoElectra):
        super().__init__()
        self.codec = electra.codec
        self.emb_proj = electra.emb_proj
        self.pos_emb = electra.pos_emb
        self.emb_layer_norm = electra.emb_layer_norm
        self.emb_dropout = electra.emb_dropout
        self.disc_hidden_proj = electra.disc_hidden_proj
        self.discriminator = electra.discriminator
        self.qa_head = nn.Linear(electra.hidden_size, 2)
        nn.init.normal_(self.qa_head.weight, std=0.02)
        nn.init.zeros_(self.qa_head.bias)

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
        logits = self.qa_head(h)  # [B, P, 2]
        return logits[..., 0], logits[..., 1]


# ─────────────────────────────────────────────
# KorQuAD-style char F1
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
    em_sum = 0.0
    f1_sum = 0.0
    n = 0
    loss_sum = 0.0
    n_b = 0
    for batch in loader:
        jamo_ids = batch["jamo_ids"].to(device, non_blocking=True)
        jamo_mask = batch["jamo_mask"].to(device, non_blocking=True)
        token_pad_mask = batch["token_pad_mask"].to(device, non_blocking=True)
        start_pos = batch["start_pos"].to(device, non_blocking=True)
        end_pos = batch["end_pos"].to(device, non_blocking=True)

        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
            sl, el = model(jamo_ids, jamo_mask, token_pad_mask)
            loss = (F.cross_entropy(sl, start_pos) + F.cross_entropy(el, end_pos)) / 2
        loss_sum += loss.item()
        n_b += 1

        # invalid 위치 (pad) 마스킹 — 매우 작은 값
        sl = sl.masked_fill(~token_pad_mask, -1e4)
        el = el.masked_fill(~token_pad_mask, -1e4)
        s_pred = sl.argmax(-1)
        e_pred = el.argmax(-1)

        for b in range(s_pred.size(0)):
            sp = s_pred[b].item()
            ep = e_pred[b].item()
            if ep < sp:
                ep = sp  # 안전장치
            q_len = batch["q_len"][b]
            c_ids = batch["c_ids"][b]
            input_offset = 1 + q_len + 1
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
    model = MRCHead(electra).to(device)
    print(f"[Model] trainable={sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.2f}M")

    bbpe = load_bbpe_tokenizer()
    jamo = JamoTokenizer()

    print("[Data] loading klue/mrc...")
    train_hf, val_hf = load_klue_mrc()
    max_jamo = saved_args.get("max_jamo_per_token", 32)
    train_ds = MRCDataset(train_hf, bbpe, jamo, args.max_patches, max_jamo)
    val_ds = MRCDataset(val_hf, bbpe, jamo, args.max_patches, max_jamo)
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
        loss_sum = 0.0
        n = 0
        for batch in train_loader:
            jamo_ids = batch["jamo_ids"].to(device, non_blocking=True)
            jamo_mask = batch["jamo_mask"].to(device, non_blocking=True)
            token_pad_mask = batch["token_pad_mask"].to(device, non_blocking=True)
            start_pos = batch["start_pos"].to(device, non_blocking=True)
            end_pos = batch["end_pos"].to(device, non_blocking=True)
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                sl, el = model(jamo_ids, jamo_mask, token_pad_mask)
                loss = (F.cross_entropy(sl, start_pos)
                        + F.cross_entropy(el, end_pos)) / 2
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
        m = evaluate(model, val_loader, device, amp_dtype, use_amp, bbpe)
        print(f"[Eval epoch {epoch}] em={m['em']:.4f} f1={m['f1']:.4f} "
              f"loss={m['loss']:.4f} (n={m['n']})", flush=True)
        if m["f1"] > best_f1:
            best_f1 = m["f1"]

    print(f"\n[Done] klue_mrc best f1={best_f1:.4f}")


if __name__ == "__main__":
    main()
