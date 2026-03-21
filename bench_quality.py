"""DenseEditor 모델 품질 벤치마크 — 오버핏 테스트

소량 데이터(1000문장)에서 각 아키텍처의 수렴 속도와 태그 정확도를 비교.

Usage:
    python bench_quality.py --d_model 640 --max_steps 2000 --corpus corpus/val_50k.jsonl
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

# 토크나이저
PROJECT_ROOT = os.path.dirname(__file__)
def load_tokenizer(name="keyboard"):
    from keyboard_tokenizer.keyboard_wrapper import KeyboardTokenizer
    return KeyboardTokenizer(os.path.join(PROJECT_ROOT, "keyboard_tokenizer", "keyboard_tokenizer.json"))


MIXING_TYPES = ["xlstm", "mlstm", "rwkv", "retnet", "mamba", "mamba2", "fnet", "tcn", "attention", "hybrid"]


def bench_quality_one(
    mixing_type: str, d_model: int, corpus: str, text_key: str,
    max_steps: int, log_interval: int, seq_len: int, batch_size: int,
    use_bf16: bool, target_params: int, grad_accum: int = 1,
    noise_preset: str = "default",
    **config_overrides,
):
    """단일 아키텍처 오버핏 테스트

    grad_accum > 1이면 batch_size를 키워도 effective batch를 유지할 수 있음.
    effective_batch = batch_size * grad_accum
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = load_tokenizer()
    cfg = make_config(mixing_type, d_model=d_model, target_params=target_params,
                      vocab_size=tokenizer.vocab_size,
                      n_tags=2 + 2 * tokenizer.vocab_size,
                      max_seq_len=seq_len,
                      pad_id=tokenizer.pad_id, bos_id=tokenizer.bos_id,
                      **config_overrides)

    model = DenseEditor(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, fused=torch.cuda.is_available())
    # edit_loss_weight: non-KEEP 태그에 2배 가중치 (KEEP 편향 완화)
    edit_loss_weight = 2.0
    tag_weights = torch.ones(cfg.n_tags, device=device)
    tag_weights[1:] = edit_loss_weight  # index 0 = TAG_KEEP
    criterion = nn.CrossEntropyLoss(ignore_index=-100, weight=tag_weights)

    # 노이즈 (토큰 레벨 비활성)
    noise_cfg = NoiseConfig(
        token_mask_ratio=0.0, token_delete_ratio=0.0, text_infill_ratio=0.0,
        weight_preset=noise_preset,
    )
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

    amp_ctx = torch.amp.autocast("cuda", dtype=torch.bfloat16) if use_bf16 and device.type == "cuda" else torch.enable_grad()

    model.train()
    data_iter = iter(loader)
    results = []
    micro_step = 0

    for step in range(max_steps):
        optimizer.zero_grad(set_to_none=True)

        # gradient accumulation
        for ga in range(grad_accum):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                batch = next(data_iter)

            input_ids = batch["input_ids"].to(device, non_blocking=True)
            edit_tags = batch["edit_tags"].to(device, non_blocking=True)
            pad_mask = batch["pad_mask"].to(device, non_blocking=True)

            with amp_ctx:
                logits = model(input_ids, pad_mask)
                targets = torch.where(pad_mask, edit_tags, torch.tensor(-100, dtype=torch.long, device=device))
                loss = criterion(logits.view(-1, cfg.n_tags), targets.view(-1))
                if grad_accum > 1:
                    loss = loss / grad_accum

            loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if (step + 1) % log_interval == 0:
            with torch.no_grad():
                # 마지막 micro-batch의 logits로 메트릭 계산
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

            # loss를 원래 스케일로 복원 (grad_accum으로 나눈 경우)
            loss_val = loss.item() * grad_accum if grad_accum > 1 else loss.item()
            results.append({
                "step": step + 1,
                "loss": loss_val,
                "tag_acc": tag_acc,
                "edit_p": edit_p,
                "edit_r": edit_r,
            })

    # 정리
    del model, optimizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "mixing_type": mixing_type,
        "n_layers": cfg.n_layers,
        "n_params": n_params,
        "final": results[-1] if results else {},
        "history": results,
    }


def main():
    parser = argparse.ArgumentParser(description="DenseEditor 품질 벤치마크")
    parser.add_argument("--d_model", type=int, default=640)
    parser.add_argument("--corpus", type=str, nargs="+", required=True)
    parser.add_argument("--text_key", type=str, default="text")
    parser.add_argument("--max_steps", type=int, default=2000)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=1,
                        help="Gradient accumulation steps (effective_batch = batch_size * grad_accum)")
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--target_params", type=int, default=128_000_000)
    parser.add_argument("--n_layers", type=int, default=None,
                        help="레이어 수 직접 지정 (미지정 시 target_params로 자동 계산)")
    parser.add_argument("--mixing_types", nargs="+", default=MIXING_TYPES)
    parser.add_argument("--noise_preset", type=str, default="default",
                        choices=["default", "realistic"],
                        help="한국어 오류 가중치 프리셋 (default | realistic)")
    args = parser.parse_args()

    eff_batch = args.batch_size * args.grad_accum
    print(f"=== DenseEditor 품질 벤치마크 (오버핏 테스트) ===")
    print(f"d_model={args.d_model}, seq_len={args.seq_len}, batch={args.batch_size}, "
          f"grad_accum={args.grad_accum}, effective_batch={eff_batch}")
    print(f"max_steps={args.max_steps}, corpus={args.corpus}\n")

    all_results = {}

    for mt in args.mixing_types:
        print(f"--- {mt.upper()} ---")
        try:
            overrides = {}
            if args.n_layers is not None:
                overrides["n_layers"] = args.n_layers
            r = bench_quality_one(
                mt, args.d_model, args.corpus, args.text_key,
                args.max_steps, args.log_interval, args.seq_len, args.batch_size,
                args.bf16, args.target_params, grad_accum=args.grad_accum,
                noise_preset=args.noise_preset,
                **overrides,
            )
            all_results[mt] = r
            f = r["final"]
            print(f"  {r['n_layers']}L {r['n_params']/1e6:.1f}M | "
                  f"loss={f.get('loss',0):.4f} acc={f.get('tag_acc',0):.2%} "
                  f"P={f.get('edit_p',0):.2%} R={f.get('edit_r',0):.2%}")
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback; traceback.print_exc()

    # 최종 비교
    print(f"\n{'='*60}")
    print(f"{'Arch':<10} {'Layers':>6} {'Params':>7} {'Loss':>8} {'TagAcc':>8} {'EditP':>8} {'EditR':>8}")
    print("-" * 60)
    for mt in args.mixing_types:
        if mt in all_results:
            r = all_results[mt]
            f = r["final"]
            print(f"{mt:<10} {r['n_layers']:>6} {r['n_params']/1e6:>6.1f}M "
                  f"{f.get('loss',0):>8.4f} {f.get('tag_acc',0):>7.2%} "
                  f"{f.get('edit_p',0):>7.2%} {f.get('edit_r',0):>7.2%}")


if __name__ == "__main__":
    main()
