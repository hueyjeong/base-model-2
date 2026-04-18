"""KoELECTRA 학습 step 의 시간/메모리 프로파일.

freeze vs unfreeze 모드의 bottleneck 식별용.
1 GPU eager 모드로 짧게 (warmup 3 + active 5) 측정.

사용:
  PYTHONPATH=exp-jamo-codec python -m koelectra.profile_train --mode freeze
  PYTHONPATH=exp-jamo-codec python -m koelectra.profile_train --mode unfreeze
  PYTHONPATH=exp-jamo-codec python -m koelectra.profile_train --mode encoder_only
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import torch
import torch.nn.functional as F
from torch.profiler import profile, record_function, ProfilerActivity

_THIS = os.path.abspath(os.path.dirname(__file__))
_EXP_ROOT = os.path.abspath(os.path.join(_THIS, ".."))
_PROJECT_ROOT = os.path.abspath(os.path.join(_EXP_ROOT, ".."))
for p in (_EXP_ROOT, _PROJECT_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from tok.jamo_tokenizer import JamoTokenizer  # noqa: E402
from koelectra.data.bbpe_token_dataset import (  # noqa: E402
    load_bbpe_tokenizer, BBPETokenDataset, _worker_init_fn,
)
from koelectra.data.masking import make_patch_mask, apply_mask  # noqa: E402
from koelectra.model.electra import JamoKoElectra  # noqa: E402
from koelectra.train import collate_batch  # noqa: E402
from torch.utils.data import DataLoader


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["freeze", "unfreeze", "encoder_only"],
                    default="freeze")
    ap.add_argument("--codec_ckpt", default="checkpoints/simple_codec_final.pt")
    ap.add_argument("--corpus", default="corpus/k-exaone_coverage_5_len1000.parquet")
    ap.add_argument("--max_patches", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--bf16", action="store_true", default=True)
    ap.add_argument("--recon_weight", type=float, default=0.5)
    ap.add_argument("--n_active", type=int, default=5)
    ap.add_argument("--out_dir", default="/tmp/profile_koelectra")
    args = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(0)

    device = torch.device("cuda")
    amp_dtype = torch.bfloat16

    print(f"[Mode] {args.mode}, batch={args.batch_size}, max_patches={args.max_patches}")
    bbpe = load_bbpe_tokenizer()
    jamo = JamoTokenizer()

    model = JamoKoElectra(
        codec_d_model=256, codec_n_enc_layers=5, codec_n_dec_layers=5,
        codec_kernel_size=5, max_jamo_per_token=32,
        embedding_size=128, hidden_size=256, n_heads=4, d_ff=1024,
        gen_layers=12, disc_layers=12, max_patches=args.max_patches,
        gen_loss_weight=50.0,
    ).to(device)
    info = model.load_codec_pretrained(args.codec_ckpt, map_location=device)
    print(f"[Codec load] missing={len(info['missing'])}, unexpected={len(info['unexpected'])}")

    # 모드별 codec param 설정
    if args.mode == "freeze":
        recon_w = 0.0  # freeze 면 recon 의미 없음
    elif args.mode == "unfreeze":
        model.enable_codec_cotrain()
        recon_w = args.recon_weight
    elif args.mode == "encoder_only":
        # 직접 분기: encoder 부분만 unfreeze, decoder 는 freeze 유지
        model._codec_frozen = False
        for p in model.codec.embedding.parameters(): p.requires_grad = True
        for p in model.codec.enc_pos.parameters(): p.requires_grad = True
        for p in model.codec.enc_layers.parameters(): p.requires_grad = True
        for p in model.codec.enc_pool_proj.parameters(): p.requires_grad = True
        # decoder 명시적 freeze (이미 False 일 것)
        for p in model.codec.dec_upsample.parameters(): p.requires_grad = False
        for p in model.codec.dec_pos.parameters(): p.requires_grad = False
        for p in model.codec.dec_layers.parameters(): p.requires_grad = False
        for p in model.codec.head.parameters(): p.requires_grad = False
        recon_w = 0.0  # decoder freeze 라 recon 불필요

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    codec_trainable = sum(p.numel() for p in model.codec.parameters() if p.requires_grad)
    print(f"[Params] trainable={trainable/1e6:.2f}M (codec_trainable={codec_trainable/1e6:.2f}M)")
    print(f"[Loss] recon_weight={recon_w}")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-4, betas=(0.9, 0.999), eps=1e-8,
    )

    # 데이터
    ds = BBPETokenDataset(
        file_paths=[args.corpus],
        bbpe_tokenizer=bbpe, jamo_tokenizer=jamo,
        max_patches=args.max_patches, max_jamo_per_token=32,
    )
    ds._prewarm_cache(verbose=False)
    loader = DataLoader(ds, batch_size=args.batch_size, num_workers=0,
                        collate_fn=collate_batch)
    data_iter = iter(loader)

    def step_fn():
        batch = next(data_iter)
        jamo_ids = batch["jamo_ids"].to(device, non_blocking=True)
        jamo_mask = batch["jamo_mask"].to(device, non_blocking=True)
        token_pad_mask = batch["token_pad_mask"].to(device, non_blocking=True)
        special_token_mask = batch["special_token_mask"].to(device, non_blocking=True)
        n_tokens = batch["n_tokens"].to(device, non_blocking=True)

        with record_function("data_mask"):
            masked_patch_mask = make_patch_mask(
                n_tokens, max_patches=args.max_patches, mask_ratio=0.20,
                special_patch_mask=special_token_mask,
            )
            masked_jamo_ids, masked_jamo_mask, per_jamo_mask = apply_mask(
                jamo_ids, jamo_mask, masked_patch_mask,
            )

        with torch.amp.autocast("cuda", dtype=amp_dtype):
            with record_function("forward_full"):
                out = model(jamo_ids, jamo_mask, token_pad_mask,
                            masked_jamo_ids, masked_jamo_mask,
                            masked_patch_mask, per_jamo_mask,
                            recon_weight=recon_w)
            loss = out["total_loss"]
        with record_function("backward"):
            loss.backward()
        with record_function("optimizer"):
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

    # Warmup (jit cache, allocator)
    print("[Warmup] 3 steps")
    model.train()
    for _ in range(3):
        step_fn()
    torch.cuda.synchronize()

    # 단순 timing 먼저
    print("[Timing] 5 steps")
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(args.n_active):
        step_fn()
    torch.cuda.synchronize()
    dt = (time.time() - t0) / args.n_active
    mem = torch.cuda.max_memory_allocated(device) / 1024**3
    print(f"  {dt*1000:.1f} ms/step, {1/dt:.2f} step/s, peak mem {mem:.2f} GB")

    # Profiler
    print("[Profile] 5 steps with profiler")
    os.makedirs(args.out_dir, exist_ok=True)
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
    ) as prof:
        for _ in range(args.n_active):
            step_fn()
        torch.cuda.synchronize()

    # 상위 시간 항목
    print("\n=== Top by CUDA time ===")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
    print("\n=== record_function 단위 (forward/backward 등) ===")
    print(prof.key_averages(group_by_input_shape=False).table(
        sort_by="cuda_time_total", row_limit=30,
        max_name_column_width=50))


if __name__ == "__main__":
    main()
