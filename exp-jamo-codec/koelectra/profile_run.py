"""KoELECTRA 학습 파이프라인 프로파일링.

목적: codec_encoder / transformer / codec_decoder / masking 등 각 구간의
CUDA 시간 비중 측정. DataLoader 병목 여부도 확인.

사용:
    python -m exp-jamo-codec.koelectra.profile_run \
        --codec_ckpt exp-jamo-codec/checkpoints/composition_6L_step600000.pt \
        --parquet corpus/jamo-codec-v3/val.parquet \
        --max_seq_len 1536 --max_patches 512 \
        --batch_size 8 --n_steps 10
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

_THIS = os.path.abspath(os.path.dirname(__file__))
_EXP_ROOT = os.path.abspath(os.path.join(_THIS, ".."))
_PROJECT_ROOT = os.path.abspath(os.path.join(_EXP_ROOT, ".."))
for p in (_EXP_ROOT, _PROJECT_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from data.bbpe_jamo_dataset import BBPEJamoDataset, load_bbpe_tokenizer  # noqa: E402
from tok.jamo_tokenizer import JamoTokenizer  # noqa: E402

from koelectra.model.electra import JamoKoElectra  # noqa: E402
from koelectra.data.masking import make_patch_mask, apply_mask  # noqa: E402


def collate(samples):
    return {
        "jamo_ids": torch.stack([s["jamo_ids"] for s in samples]),
        "jamo_mask": torch.stack([s["jamo_mask"] for s in samples]),
        "segment_ids": torch.stack([s["segment_ids"] for s in samples]),
        "n_segments": torch.tensor([s["n_segments"] for s in samples], dtype=torch.long),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--codec_ckpt", type=str,
                    default="exp-jamo-codec/checkpoints/composition_6L_step600000.pt")
    ap.add_argument("--parquet", type=str, default="corpus/jamo-codec-v3/val.parquet")
    ap.add_argument("--max_seq_len", type=int, default=1536)
    ap.add_argument("--max_patches", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--n_steps", type=int, default=8)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--mask_ratio", type=float, default=0.20)
    ap.add_argument("--trace_dir", type=str, default="/tmp/ko_profile")
    ap.add_argument("--no_profiler", action="store_true",
                    help="profiler 없이 단순 timer만")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    print(f"[Setup] device={device}, dtype={dtype}")
    print(f"[Config] max_seq_len={args.max_seq_len}, max_patches={args.max_patches}, "
          f"batch={args.batch_size}, workers={args.num_workers}")

    # Tokenizers
    bbpe = load_bbpe_tokenizer()
    jamo = JamoTokenizer()

    ds = BBPEJamoDataset(
        file_paths=[args.parquet], bbpe_tokenizer=bbpe, jamo_tokenizer=jamo,
        max_seq_len=args.max_seq_len, max_patches=args.max_patches,
        text_key="text", rank=0, world_size=1,
    )
    loader = DataLoader(ds, batch_size=args.batch_size, num_workers=args.num_workers,
                        collate_fn=collate, pin_memory=(device.type == "cuda"),
                        persistent_workers=(args.num_workers > 0))

    model = JamoKoElectra(
        codec_d_model=256, codec_n_layers=6, codec_kernel_size=7,
        max_jamo_per_token=32, embedding_size=128, hidden_size=256,
        n_heads=4, d_ff=1024, gen_layers=12, disc_layers=12,
        max_patches=args.max_patches, gen_loss_weight=50.0,
    ).to(device)
    model.load_codec_pretrained(args.codec_ckpt, map_location=device)
    model.train()

    opt = torch.optim.AdamW(model.parameters(), lr=5e-4)

    # ──────────────────────────────────────────────────────────────
    # 1단계: 수동 타이머로 구간별 측정 (CUDA events)
    # ──────────────────────────────────────────────────────────────
    print("\n=== 구간별 CUDA event 타이밍 (ms, avg over active steps) ===")

    ev_names = [
        "data_load", "masking",
        "codec_enc_masked", "emb+proj_gen", "generator_tf", "codec_decoder",
        "argmax+corrupt", "codec_enc_corrupt", "emb+proj_disc",
        "discriminator_tf", "disc_head+loss", "backward", "optimizer",
    ]
    sums = {k: 0.0 for k in ev_names}
    total_sum = 0.0
    count = 0

    data_iter = iter(loader)
    for step in range(args.warmup + args.n_steps):
        # Data load 타이밍
        t_data_start = time.perf_counter()
        batch = next(data_iter)
        data_time = (time.perf_counter() - t_data_start) * 1000.0

        jamo_ids = batch["jamo_ids"].to(device, non_blocking=True)
        jamo_mask = batch["jamo_mask"].to(device, non_blocking=True)
        segment_ids = batch["segment_ids"].to(device, non_blocking=True)
        n_segments = batch["n_segments"].to(device, non_blocking=True)

        # CUDA events
        evs = {k: (torch.cuda.Event(enable_timing=True),
                   torch.cuda.Event(enable_timing=True)) for k in ev_names}

        evs["data_load"][0].record()
        evs["data_load"][1].record()  # data_load는 CPU 측정이라 0 처리

        torch.cuda.synchronize()
        t_step_start = time.perf_counter()

        with torch.amp.autocast("cuda", dtype=dtype, enabled=True):
            # ── Masking ──
            evs["masking"][0].record()
            masked_patch_mask = make_patch_mask(n_segments, max_patches=args.max_patches,
                                                mask_ratio=args.mask_ratio)
            masked_jamo_ids, per_jamo_mask = apply_mask(
                jamo_ids, segment_ids, jamo_mask, masked_patch_mask
            )
            evs["masking"][1].record()

            # ── Gen: codec encoder on masked ──
            evs["codec_enc_masked"][0].record()
            z_masked = model.codec_encoder(masked_jamo_ids, jamo_mask, segment_ids, n_segments)
            from koelectra.model.electra import _pad_patches_to
            z_masked = _pad_patches_to(z_masked, args.max_patches)
            evs["codec_enc_masked"][1].record()

            # ── Gen: emb + proj ──
            evs["emb+proj_gen"][0].record()
            B = jamo_ids.size(0)
            P = args.max_patches
            pos = torch.arange(P, device=device).unsqueeze(0).expand(B, -1)
            patch_pad_mask = pos < n_segments.unsqueeze(-1)
            e_masked = model._embed(z_masked, patch_pad_mask)
            h_gen = model.gen_hidden_proj(e_masked)
            evs["emb+proj_gen"][1].record()

            # ── Generator transformer ──
            evs["generator_tf"][0].record()
            h_gen = model.generator(h_gen, patch_pad_mask)
            evs["generator_tf"][1].record()

            # ── Codec decoder (gen head) ──
            evs["codec_decoder"][0].record()
            L = jamo_ids.size(1)
            jamo_logits = model.codec_decoder(h_gen, segment_ids, L, jamo_mask)
            evs["codec_decoder"][1].record()

            # Gen loss + corrupted 재구성
            evs["argmax+corrupt"][0].record()
            V = model.jamo_vocab
            ce = F.cross_entropy(jamo_logits.reshape(-1, V),
                                 jamo_ids.reshape(-1), reduction="none").reshape(B, L)
            denom = per_jamo_mask.sum().clamp(min=1)
            gen_loss = (ce * per_jamo_mask.float()).sum() / denom

            with torch.no_grad():
                sampled = jamo_logits.argmax(-1)
                jamo_corrupted = torch.where(per_jamo_mask, sampled, jamo_ids)
                diff = (sampled != jamo_ids) & per_jamo_mask
                from koelectra.data.masking import scatter_any_per_patch
                replaced = scatter_any_per_patch(diff, segment_ids, P)
            evs["argmax+corrupt"][1].record()

            # ── Disc: codec encoder on corrupted ──
            evs["codec_enc_corrupt"][0].record()
            z_corrupt = model.codec_encoder(jamo_corrupted, jamo_mask, segment_ids, n_segments)
            z_corrupt = _pad_patches_to(z_corrupt, args.max_patches)
            evs["codec_enc_corrupt"][1].record()

            # ── Disc: emb + proj ──
            evs["emb+proj_disc"][0].record()
            e_corrupt = model._embed(z_corrupt, patch_pad_mask)
            h_disc = model.disc_hidden_proj(e_corrupt)
            evs["emb+proj_disc"][1].record()

            # ── Discriminator transformer ──
            evs["discriminator_tf"][0].record()
            h_disc = model.discriminator(h_disc, patch_pad_mask)
            evs["discriminator_tf"][1].record()

            # ── Disc head + loss ──
            evs["disc_head+loss"][0].record()
            disc_logits = model.disc_head(h_disc).squeeze(-1)
            disc_loss = F.binary_cross_entropy_with_logits(
                disc_logits[patch_pad_mask], replaced[patch_pad_mask].float()
            )
            total_loss = disc_loss + 50.0 * gen_loss
            evs["disc_head+loss"][1].record()

        # ── Backward ──
        evs["backward"][0].record()
        total_loss.backward()
        evs["backward"][1].record()

        # ── Optimizer ──
        evs["optimizer"][0].record()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        evs["optimizer"][1].record()

        torch.cuda.synchronize()
        step_ms = (time.perf_counter() - t_step_start) * 1000.0

        if step >= args.warmup:
            for k in ev_names[1:]:  # data_load는 별도
                ms = evs[k][0].elapsed_time(evs[k][1])
                sums[k] += ms
            sums["data_load"] += data_time
            total_sum += step_ms
            count += 1

    print(f"\n{'구간':<22} {'평균(ms)':>10} {'비중(%)':>8}")
    print("-" * 46)
    total_ev = sum(sums.values())
    for k in ev_names:
        avg = sums[k] / count
        pct = (sums[k] / total_ev) * 100.0 if total_ev > 0 else 0
        print(f"{k:<22} {avg:>10.2f} {pct:>7.1f}%")
    print("-" * 46)
    print(f"{'구간 합':<22} {total_ev/count:>10.2f}")
    print(f"{'wall-clock step':<22} {total_sum/count:>10.2f}")
    print(f"{'step/s':<22} {1000.0/(total_sum/count):>10.2f}")
    gap = total_sum/count - total_ev/count
    print(f"{'측정 외 gap':<22} {gap:>10.2f}  (sync overhead / data wait)")

    # ──────────────────────────────────────────────────────────────
    # 2단계: PyTorch profiler (top-N CUDA kernel)
    # ──────────────────────────────────────────────────────────────
    if not args.no_profiler:
        print("\n=== PyTorch profiler: top CUDA 커널 (forward+backward) ===")
        os.makedirs(args.trace_dir, exist_ok=True)

        from torch.profiler import profile, ProfilerActivity, schedule, tensorboard_trace_handler
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=schedule(wait=1, warmup=1, active=3, repeat=1),
            on_trace_ready=tensorboard_trace_handler(args.trace_dir),
            record_shapes=False,
            with_stack=False,
        ) as prof:
            for step in range(6):
                batch = next(data_iter)
                jamo_ids = batch["jamo_ids"].to(device, non_blocking=True)
                jamo_mask = batch["jamo_mask"].to(device, non_blocking=True)
                segment_ids = batch["segment_ids"].to(device, non_blocking=True)
                n_segments = batch["n_segments"].to(device, non_blocking=True)
                masked_patch_mask = make_patch_mask(n_segments, max_patches=args.max_patches,
                                                    mask_ratio=args.mask_ratio)
                masked_jamo_ids, per_jamo_mask = apply_mask(jamo_ids, segment_ids, jamo_mask, masked_patch_mask)

                with torch.amp.autocast("cuda", dtype=dtype, enabled=True):
                    out = model(jamo_ids, jamo_mask, segment_ids, n_segments,
                                masked_jamo_ids, per_jamo_mask, masked_patch_mask)
                out["total_loss"].backward()
                opt.step()
                opt.zero_grad(set_to_none=True)
                prof.step()

        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=25))
        print(f"\nChrome trace: {args.trace_dir}")


if __name__ == "__main__":
    main()
