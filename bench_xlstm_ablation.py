"""xLSTM 개선 ablation 테스트

각 변형을 2000 step 오버핏 테스트하여 edit recall 비교.
"""
import argparse
import gc
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))

from model.dense_editor_config import make_config
from model.dense_editor import DenseEditor
from model.edit_tags import TAG_KEEP
from training.noising import DenoisingNoiser, NoiseConfig
from training.editor_dataset import EditorDataset

PROJECT_ROOT = os.path.dirname(__file__)

def load_tokenizer(name="keyboard"):
    from keyboard_tokenizer.keyboard_wrapper import KeyboardTokenizer
    return KeyboardTokenizer(os.path.join(PROJECT_ROOT, "keyboard_tokenizer", "keyboard_tokenizer.json"))


# 테스트 변형 목록
VARIANTS = {
    "baseline":    {},  # o_proj만
    "conv":        {"xlstm_use_conv": True},
    "silu_gate":   {"xlstm_use_silu_gate": True},
    "decay_bias":  {"xlstm_use_decay_bias": True},
    "conv+decay":  {"xlstm_use_conv": True, "xlstm_use_decay_bias": True},
    "ds2":         {"xlstm_d_state": 2},
    "ds4":         {"xlstm_d_state": 4},
    "ds4+conv":    {"xlstm_d_state": 4, "xlstm_use_conv": True},
    "hybrid":      {},  # Phase 3: sLSTM-Mamba 하이브리드
    "mamba_ref":   {},  # Mamba 기준선
}


def run_one(variant_name, overrides, corpus, text_key, d_model, max_steps,
            log_interval, seq_len, batch_size, use_bf16, target_params):
    """단일 변형 오버핏 테스트"""
    if variant_name == "mamba_ref":
        mixing_type = "mamba"
    elif variant_name == "hybrid":
        mixing_type = "xlstm_mamba"
    else:
        mixing_type = "xlstm"
    cfg = make_config(mixing_type, d_model=d_model, target_params=target_params,
                      **overrides)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DenseEditor(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    tokenizer = load_tokenizer()
    noise_cfg = NoiseConfig(token_mask_ratio=0.0)
    noiser = DenoisingNoiser(tokenizer, noise_cfg, seed=42, use_korean_errors=True)

    dataset = EditorDataset(
        corpus, tokenizer, noiser,
        vocab_size=tokenizer.vocab_size,
        max_seq_len=seq_len,
        text_key=text_key,
        seed=42, rank=0, world_size=1, pack=True,
    )
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=2,
                        pin_memory=True, drop_last=True, prefetch_factor=4,
                        persistent_workers=True)

    amp_ctx = (torch.amp.autocast("cuda", dtype=torch.bfloat16)
               if use_bf16 and device.type == "cuda"
               else torch.enable_grad())

    model.train()
    data_iter = iter(loader)
    history = []

    t0 = time.time()
    for step in range(max_steps):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        input_ids = batch["input_ids"].to(device, non_blocking=True)
        edit_tags = batch["edit_tags"].to(device, non_blocking=True)
        pad_mask = batch["pad_mask"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with amp_ctx:
            logits = model(input_ids, pad_mask)
            targets = torch.where(pad_mask, edit_tags,
                                  torch.tensor(-100, dtype=torch.long, device=device))
            loss = criterion(logits.view(-1, cfg.n_tags), targets.view(-1))

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if (step + 1) % log_interval == 0:
            with torch.no_grad():
                preds = logits.argmax(dim=-1)
                valid = pad_mask
                n_tok = valid.sum().item()
                correct = (preds[valid] == edit_tags[valid]).sum().item()
                tag_acc = correct / max(n_tok, 1)

                pred_edit = preds[valid] != TAG_KEEP
                true_edit = edit_tags[valid] != TAG_KEEP
                tp = (pred_edit & true_edit).sum().item()
                fp = (pred_edit & ~true_edit).sum().item()
                fn = (~pred_edit & true_edit).sum().item()
                edit_p = tp / max(tp + fp, 1)
                edit_r = tp / max(tp + fn, 1)

            history.append({
                "step": step + 1,
                "loss": loss.item(),
                "tag_acc": tag_acc,
                "edit_p": edit_p,
                "edit_r": edit_r,
            })
            elapsed = time.time() - t0
            print(f"    step {step+1:>5}: loss={loss.item():.4f} "
                  f"acc={tag_acc:.2%} P={edit_p:.2%} R={edit_r:.2%} "
                  f"({elapsed:.0f}s)")

    del model, optimizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "n_layers": cfg.n_layers,
        "n_params": n_params,
        "final": history[-1] if history else {},
        "history": history,
    }


def main():
    parser = argparse.ArgumentParser(description="xLSTM ablation 테스트")
    parser.add_argument("--d_model", type=int, default=640)
    parser.add_argument("--corpus", type=str, nargs="+",
                        default=["corpus/val_50k.jsonl"])
    parser.add_argument("--text_key", type=str, default="text")
    parser.add_argument("--max_steps", type=int, default=2000)
    parser.add_argument("--log_interval", type=int, default=200)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--target_params", type=int, default=128_000_000)
    parser.add_argument("--variants", nargs="+",
                        default=list(VARIANTS.keys()),
                        choices=list(VARIANTS.keys()))
    args = parser.parse_args()

    print(f"=== xLSTM Ablation 테스트 ===")
    print(f"d={args.d_model}, seq={args.seq_len}, batch={args.batch_size}, "
          f"steps={args.max_steps}\n")

    all_results = {}
    for name in args.variants:
        overrides = VARIANTS[name]
        opts = ", ".join(f"{k}={v}" for k, v in overrides.items()) or "(기본)"
        print(f"--- {name} [{opts}] ---")
        try:
            r = run_one(name, overrides, args.corpus, args.text_key,
                        args.d_model, args.max_steps, args.log_interval,
                        args.seq_len, args.batch_size, args.bf16,
                        args.target_params)
            all_results[name] = r
            f = r["final"]
            print(f"  => {r['n_layers']}L {r['n_params']/1e6:.1f}M | "
                  f"loss={f.get('loss',0):.4f} R={f.get('edit_r',0):.2%}\n")
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback; traceback.print_exc()
            print()

    # 최종 비교
    print(f"\n{'='*70}")
    print(f"{'Variant':<12} {'L':>3} {'Params':>7} {'Loss':>8} "
          f"{'Acc':>7} {'EditP':>7} {'EditR':>7}")
    print("-" * 70)
    for name in args.variants:
        if name in all_results:
            r = all_results[name]
            f = r["final"]
            print(f"{name:<12} {r['n_layers']:>3} {r['n_params']/1e6:>6.1f}M "
                  f"{f.get('loss',0):>8.4f} {f.get('tag_acc',0):>6.2%} "
                  f"{f.get('edit_p',0):>6.2%} {f.get('edit_r',0):>6.2%}")


if __name__ == "__main__":
    main()
