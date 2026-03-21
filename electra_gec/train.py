"""KoELECTRA-Base-v3 + Two-head GECToR 학습 스크립트

에포크 기반 학습 + progressive unfreezing + BF16 AMP + DDP.

Usage:
    # 단일 GPU
    python -m electra_gec.train \
        --corpus corpus/val_50k.jsonl --text_key text \
        --max_epochs 10 --batch_size 32 --noise_preset realistic

    # DDP 멀티 GPU
    torchrun --nproc_per_node=4 -m electra_gec.train \
        --corpus corpus/sample_full.jsonl --text_key text \
        --max_epochs 10 --batch_size 32 --noise_preset realistic
"""
import argparse
import math
import os
import sys
import time
from contextlib import nullcontext

import torch
import torch._inductor.config
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from electra_gec.model import KoELECTRAGECToR, ACTION_KEEP
from electra_gec.dataset import WordPieceGECDataset, IGNORE
from training.noising import DenoisingNoiser, NoiseConfig
from training.upload_gdrive import upload_and_cleanup


# ── 유틸 ──

def fmt_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    m, s = divmod(int(seconds), 60)
    if m < 60:
        return f"{m}m{s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m"


def fmt_bytes(n: float) -> str:
    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.2f}GB"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}MB"
    if n >= 1_000:
        return f"{n / 1_000:.1f}KB"
    return f"{n:.0f}B"


def compute_f(p: float, r: float, beta: float = 0.5) -> float:
    b2 = beta * beta
    if p + r == 0:
        return 0.0
    return (1 + b2) * p * r / (b2 * p + r)


def collate_dynamic_pad(batch: list[dict]) -> dict:
    """배치 내 최대 길이로 동적 패딩"""
    max_len = max(b["attention_mask"].sum().item() for b in batch)
    bytes_read = max(b.get("_bytes_read", 0) for b in batch)
    total_bytes = max(b.get("_total_bytes", 0) for b in batch)
    return {
        "input_ids": torch.stack([b["input_ids"][:max_len] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"][:max_len] for b in batch]),
        "action_tags": torch.stack([b["action_tags"][:max_len] for b in batch]),
        "content_tags": torch.stack([b["content_tags"][:max_len] for b in batch]),
        "_bytes_read": bytes_read,
        "_total_bytes": total_bytes,
    }


# ── 검증 ──

@torch.no_grad()
def validate(model, val_loader, device, n_batches=50, use_amp=False):
    model.eval()

    total_act_loss = 0.0
    total_cont_loss = 0.0
    total_tokens = 0
    total_act_correct = 0
    total_cont_correct = 0
    total_cont_tokens = 0
    edit_tp = edit_fp = edit_fn = 0

    act_criterion = nn.CrossEntropyLoss(ignore_index=IGNORE)
    cont_criterion = nn.CrossEntropyLoss(ignore_index=IGNORE)

    val_iter = iter(val_loader)
    for _ in range(n_batches):
        try:
            batch = next(val_iter)
        except StopIteration:
            break

        input_ids = batch["input_ids"].to(device)
        attn_mask = batch["attention_mask"].to(device)
        action_tags = batch["action_tags"].to(device)
        content_tags = batch["content_tags"].to(device)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            act_logits, cont_logits = model(input_ids, attn_mask)

        V = cont_logits.size(-1)
        act_loss = act_criterion(act_logits.float().view(-1, 4), action_tags.view(-1))
        cont_loss = cont_criterion(cont_logits.float().view(-1, V), content_tags.view(-1))

        valid = (action_tags != IGNORE)
        n_tok = valid.sum().item()
        total_act_loss += act_loss.item() * n_tok
        total_tokens += n_tok

        act_preds = act_logits.argmax(dim=-1)
        total_act_correct += (act_preds[valid] == action_tags[valid]).sum().item()

        edit_pos = (content_tags != IGNORE)
        n_cont = edit_pos.sum().item()
        if n_cont > 0:
            cont_preds = cont_logits.argmax(dim=-1)
            total_cont_correct += (cont_preds[edit_pos] == content_tags[edit_pos]).sum().item()
            total_cont_tokens += n_cont
            total_cont_loss += cont_loss.item() * n_cont

        pred_edit = act_preds[valid] != ACTION_KEEP
        true_edit = action_tags[valid] != ACTION_KEEP
        edit_tp += (pred_edit & true_edit).sum().item()
        edit_fp += (pred_edit & ~true_edit).sum().item()
        edit_fn += (~pred_edit & true_edit).sum().item()

    model.train()
    if total_tokens == 0:
        return {}

    edit_p = edit_tp / max(edit_tp + edit_fp, 1)
    edit_r = edit_tp / max(edit_tp + edit_fn, 1)
    return {
        "val_act_loss": total_act_loss / total_tokens,
        "val_cont_loss": total_cont_loss / max(total_cont_tokens, 1),
        "val_loss": total_act_loss / total_tokens + 0.5 * total_cont_loss / max(total_cont_tokens, 1),
        "act_acc": total_act_correct / total_tokens,
        "cont_acc": total_cont_correct / max(total_cont_tokens, 1),
        "edit_p": edit_p,
        "edit_r": edit_r,
        "edit_f05": compute_f(edit_p, edit_r, 0.5),
    }


# ── 학습 ──

def train(args):
    # ── DDP ──
    is_ddp = int(os.environ.get("WORLD_SIZE", 1)) > 1
    if is_ddp:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        dist.init_process_group(backend="nccl")
        global_rank = dist.get_rank()
        world_size = dist.get_world_size()
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        local_rank = 0
        global_rank = 0
        world_size = 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    is_main = (global_rank == 0)
    use_amp = args.bf16 and device.type == "cuda"

    # TF32 활성화 (BF16 AMP와 함께 사용, matmul 2x 가속)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    if is_main:
        print(f"Device: {device} | World: {world_size} | AMP: {'BF16' if use_amp else 'off'} | TF32: on")

    # ── 모델 ──
    if is_main:
        print(f"\n모델 로드: {args.model_name}")
    model = KoELECTRAGECToR(args.model_name, dropout=args.dropout).to(device)
    raw_model = model  # DDP 래핑 전 참조 (state_dict, freeze/unfreeze용)
    total_params = sum(p.numel() for p in model.parameters())
    if is_main:
        print(f"  총 파라미터: {total_params:,}")

    # torch.compile — raw_model은 원본 유지, compiled는 forward용
    if args.compile:
        mode = args.compile_mode
        # grad accum > 1이면 CUDA graph 비활성 (micro step 간 텐서 덮어쓰기 충돌)
        if args.grad_accum_steps > 1 and mode in ("reduce-overhead", "max-autotune"):
            torch._inductor.config.triton.cudagraphs = False
            if is_main:
                print(f"  torch.compile (mode={mode}, cudagraphs=off — grad_accum={args.grad_accum_steps})")
        else:
            if is_main:
                print(f"  torch.compile (mode={mode})")
        compiled_model = torch.compile(raw_model, mode=mode)
    else:
        compiled_model = raw_model

    # ── 노이즈 ──
    # DDP: rank 0이 먼저 noiser 초기화 (nltk/g2pk 코퍼스 다운로드 race condition 방지)
    from keyboard_tokenizer.keyboard_wrapper import KeyboardTokenizer
    kb_tok_path = os.path.join(
        os.path.dirname(__file__), "..", "keyboard_tokenizer", "keyboard_tokenizer.json"
    )
    kb_tok = KeyboardTokenizer(kb_tok_path)
    noise_cfg = NoiseConfig(
        token_mask_ratio=0.0, token_delete_ratio=0.0, text_infill_ratio=0.0,
        korean_error_prob=args.error_prob,
        korean_error_count=args.error_count,
        weight_preset=args.noise_preset,
    )
    if is_ddp:
        if is_main:
            # rank 0이 먼저 초기화 — nltk 코퍼스 다운로드/unzip 완료
            _noiser_init = DenoisingNoiser(kb_tok, noise_cfg, seed=0, use_korean_errors=True)
            del _noiser_init
        dist.barrier()  # 다른 rank 대기 후 진행
    noiser = DenoisingNoiser(kb_tok, noise_cfg, seed=args.seed + global_rank, use_korean_errors=True)
    if is_main:
        print(f"  noise: preset={args.noise_preset}, error_prob={args.error_prob}, count={args.error_count}")

    # ── 데이터셋 ──
    train_dataset = WordPieceGECDataset(
        args.corpus, noiser,
        tokenizer_name=args.model_name,
        max_seq_len=args.max_seq_len,
        text_key=args.text_key,
        seed=args.seed,
        rank=global_rank,
        world_size=world_size,
    )
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
        collate_fn=collate_dynamic_pad,
        prefetch_factor=4 if args.num_workers > 0 else None,
        persistent_workers=args.num_workers > 0,
    )

    val_loader = None
    if args.val_corpus:
        val_noiser = DenoisingNoiser(kb_tok, noise_cfg, seed=args.seed + 1, use_korean_errors=True)
        val_dataset = WordPieceGECDataset(
            args.val_corpus, val_noiser,
            tokenizer_name=args.model_name,
            max_seq_len=args.max_seq_len,
            text_key=args.text_key,
            seed=args.seed + 1,
            rank=global_rank,
            world_size=world_size,
        )
        val_loader = DataLoader(
            val_dataset, batch_size=args.batch_size,
            num_workers=max(args.num_workers, 1), pin_memory=True, drop_last=True,
            collate_fn=collate_dynamic_pad,
            prefetch_factor=4, persistent_workers=True,
        )
        if is_main:
            print(f"  검증: {args.val_corpus}")

    # ── Loss ──
    act_weight = torch.ones(4, device=device)
    act_weight[1:] = args.edit_loss_weight
    act_criterion = nn.CrossEntropyLoss(
        weight=act_weight, ignore_index=IGNORE, label_smoothing=args.label_smoothing,
    )
    cont_criterion = nn.CrossEntropyLoss(
        ignore_index=IGNORE, label_smoothing=args.label_smoothing,
    )

    # ── Progressive Unfreezing ──
    stages = [
        {"name": "heads_only", "epochs": 1, "lr": args.stage1_lr,
         "setup": lambda: raw_model.freeze_encoder()},
        {"name": "top6_unfreeze", "epochs": 2, "lr": args.stage2_lr,
         "setup": lambda: raw_model.unfreeze_top_layers(args.unfreeze_layers)},
        {"name": "full_finetune", "epochs": max(args.max_epochs - 3, 1), "lr": args.stage3_lr,
         "setup": lambda: raw_model.unfreeze_all()},
    ]

    # ── 체크포인트 ──
    ckpt_dir = os.path.join(os.path.dirname(__file__), "checkpoints")
    if is_main:
        os.makedirs(ckpt_dir, exist_ok=True)

    # ── Resume ──
    start_epoch = 0
    global_step = 0
    best_val_loss = float("inf")
    no_improve = 0
    resume_stage = None

    if args.resume:
        ckpt_path = args.resume
        if os.path.isdir(ckpt_path):
            # epoch_*.pt와 step_*.pt 모두 검색, 가장 최근 수정 파일 선택
            import glob
            candidates = sorted(
                glob.glob(os.path.join(ckpt_path, "epoch_*.pt"))
                + glob.glob(os.path.join(ckpt_path, "step_*.pt")),
                key=os.path.getmtime,
            )
            ckpt_path = candidates[-1] if candidates else None

        if ckpt_path and os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            raw_model.load_state_dict(ckpt["model_state_dict"])
            start_epoch = ckpt["epoch"]
            global_step = ckpt.get("global_step", 0)
            best_val_loss = ckpt.get("best_val_loss", float("inf"))
            no_improve = ckpt.get("no_improve", 0)
            resume_stage = ckpt.get("stage")

            # 데이터셋 + 노이저 RNG 상태 복원
            data_state = ckpt.get("data_state")
            if data_state:
                if "noiser_state" in data_state:
                    noiser.load_state_dict(data_state["noiser_state"])
                if "dataset_state" in data_state:
                    train_dataset.load_state_dict(data_state["dataset_state"])

            if is_main:
                print(f"\n  Resume: {ckpt_path}")
                print(f"  epoch={start_epoch}, step={global_step}, stage={resume_stage}")
                if data_state:
                    print(f"  data state 복원: noiser={'noiser_state' in data_state}, dataset={'dataset_state' in data_state}")

    def _make_data_state():
        """데이터셋 + 노이저 RNG 상태 스냅샷"""
        return {
            "noiser_state": noiser.state_dict(),
            "dataset_state": train_dataset.state_dict(),
        }

    train_start = time.time()

    if is_main:
        eff_batch = args.batch_size * world_size * args.grad_accum_steps
        print(f"\n학습 시작 (max_epochs={args.max_epochs}, batch={args.batch_size}x{world_size}x{args.grad_accum_steps}={eff_batch}, seq≤{args.max_seq_len})")
        print(f"  α={args.content_loss_weight}, edit_weight={args.edit_loss_weight}")

    epoch = 0
    ddp_model = None  # stage마다 재래핑
    for stage in stages:
        if stage["epochs"] <= 0:
            continue

        stage["setup"]()
        trainable = [p for p in raw_model.parameters() if p.requires_grad]

        # DDP: stage마다 재래핑 (frozen/unfrozen 파라미터 셋 변경에 대응)
        if is_ddp:
            if ddp_model is not None:
                del ddp_model
            ddp_model = DDP(compiled_model, device_ids=[local_rank], gradient_as_bucket_view=True)
            train_model = ddp_model
        else:
            train_model = compiled_model
        optimizer = torch.optim.AdamW(
            trainable, lr=stage["lr"],
            betas=(0.9, 0.999), weight_decay=0.01,
            fused=device.type == "cuda",
        )
        cur_lr = stage["lr"]

        # Resume: optimizer state 복원
        if resume_stage == stage["name"] and args.resume:
            ckpt_path_opt = args.resume
            if os.path.isdir(ckpt_path_opt):
                candidates = sorted(
                    [f for f in os.listdir(ckpt_path_opt) if f.startswith("epoch_") and f.endswith(".pt")],
                    key=lambda f: int(f.split("_")[1].split(".")[0]),
                )
                if candidates:
                    ckpt_path_opt = os.path.join(ckpt_path_opt, candidates[-1])
            if os.path.exists(ckpt_path_opt):
                ckpt_opt = torch.load(ckpt_path_opt, map_location=device, weights_only=False)
                if "optimizer_state_dict" in ckpt_opt:
                    try:
                        optimizer.load_state_dict(ckpt_opt["optimizer_state_dict"])
                        if is_main:
                            print(f"  optimizer state 복원 완료")
                    except Exception as e:
                        if is_main:
                            print(f"  optimizer state 복원 실패 (무시): {e}")
                del ckpt_opt

        if is_main:
            print(f"\n{'='*60}")
            print(f"Stage: {stage['name']} | LR={stage['lr']} | trainable={raw_model.count_trainable():,}")
            print(f"{'='*60}")

        for _ in range(stage["epochs"]):
            epoch += 1
            if epoch > args.max_epochs:
                break

            if epoch <= start_epoch:
                if is_main:
                    print(f"  Epoch {epoch}/{args.max_epochs}: skip (resumed)")
                continue

            train_dataset.set_epoch(epoch)
            ep_loss = 0.0
            ep_tokens = 0
            t0 = time.time()

            accum = args.grad_accum_steps
            micro_step = 0
            # micro batch 누적용 버퍼
            _acc_act = 0.0
            _acc_cont = 0.0
            _acc_tok = 0

            for batch in train_loader:
                micro_step += 1

                input_ids = batch["input_ids"].to(device)
                attn_mask = batch["attention_mask"].to(device)
                action_tags = batch["action_tags"].to(device)
                content_tags = batch["content_tags"].to(device)

                # DDP: 마지막 micro step에서만 grad sync (나머지는 no_sync)
                no_sync = is_ddp and (micro_step % accum != 0)
                ctx = ddp_model.no_sync() if no_sync else nullcontext()

                with ctx:
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                        act_logits, cont_logits = train_model(input_ids, attn_mask)
                        V = cont_logits.size(-1)
                        act_loss = act_criterion(act_logits.view(-1, 4), action_tags.view(-1))
                        cont_loss = cont_criterion(cont_logits.view(-1, V), content_tags.view(-1))
                        loss = (act_loss + args.content_loss_weight * cont_loss) / accum
                    loss.backward()

                # micro batch 통계 누적 (unscaled)
                _acc_act += act_loss.item()
                _acc_cont += cont_loss.item()
                _acc_tok += (action_tags != IGNORE).sum().item()

                if micro_step % accum != 0:
                    continue  # gradient 누적 중 — optimizer step 건너뜀

                torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                # 누적된 micro batch 평균
                step_act_loss = _acc_act / accum
                step_cont_loss = _acc_cont / accum
                step_loss = step_act_loss + args.content_loss_weight * step_cont_loss
                step_tok = _acc_tok

                ep_loss += step_loss * step_tok
                ep_tokens += step_tok

                # 버퍼 리셋
                _acc_act = 0.0
                _acc_cont = 0.0
                _acc_tok = 0

                if is_main and global_step % args.log_interval == 0:
                    elapsed = time.time() - train_start
                    ep_elapsed = time.time() - t0
                    tps = ep_tokens * world_size / max(ep_elapsed, 0.001)
                    avg = ep_loss / max(ep_tokens, 1)
                    seq_len = input_ids.size(1)

                    br = batch.get("_bytes_read", 0)
                    tb = batch.get("_total_bytes", 0)
                    progress = ""
                    if tb > 0 and br > 0:
                        pct = br / tb * 100
                        eta_ep = ep_elapsed * (tb - br) / br
                        remaining_epochs = args.max_epochs - epoch
                        eta_total = eta_ep + remaining_epochs * (ep_elapsed / max(pct / 100, 1e-6))
                        progress = f" {fmt_bytes(br)}/{fmt_bytes(tb)} ({pct:.1f}% ETA {fmt_time(eta_total)})"

                    print(
                        f"  [{fmt_time(elapsed)}] ep{epoch}/{args.max_epochs} s{global_step}{progress} | "
                        f"loss={step_loss:.4f} avg={avg:.4f} | "
                        f"act={step_act_loss:.4f} cont={step_cont_loss:.4f} | "
                        f"lr={cur_lr:.1e} {tps:.0f} tok/s seq={seq_len}"
                    )

                if val_loader is not None and global_step % args.val_every == 0:
                    m = validate(raw_model, val_loader, device, n_batches=args.val_batches, use_amp=use_amp)
                    if is_main and m:
                        print(
                            f"  [VAL] loss={m['val_loss']:.4f} act_acc={m['act_acc']:.3f} "
                            f"cont_acc={m['cont_acc']:.3f} | "
                            f"P={m['edit_p']:.3f} R={m['edit_r']:.3f} F0.5={m['edit_f05']:.3f}"
                        )

                # step 단위 체크포인트
                if is_main and args.save_interval and global_step % args.save_interval == 0:
                    ckpt_path = os.path.join(ckpt_dir, f"step_{global_step}.pt")
                    torch.save({
                        "epoch": epoch, "global_step": global_step,
                        "model_state_dict": raw_model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "stage": stage["name"],
                        "best_val_loss": best_val_loss,
                        "no_improve": no_improve,
                        "data_state": _make_data_state(),
                    }, ckpt_path)
                    print(f"  [SAVE] {ckpt_path}")
                    if args.gdrive_remote:
                        upload_and_cleanup(ckpt_path, args.log_file, args.gdrive_remote, keep_latest_n=1)

            # 에포크 종료
            dt = time.time() - t0
            total_elapsed = time.time() - train_start
            avg = ep_loss / max(ep_tokens, 1)
            remaining_epochs = args.max_epochs - epoch
            eta_total = remaining_epochs * dt if dt > 0 else 0
            if is_main:
                print(
                    f"\n  Epoch {epoch}/{args.max_epochs}: avg_loss={avg:.4f} | "
                    f"{fmt_time(dt)} ({ep_tokens * world_size:,} tok) | "
                    f"경과 {fmt_time(total_elapsed)} / 잔여 ~{fmt_time(eta_total)}"
                )

            # 에포크 검증
            if val_loader is not None:
                m = validate(raw_model, val_loader, device, n_batches=args.val_batches * 2, use_amp=use_amp)
                if is_main and m:
                    vl = m["val_loss"]
                    print(
                        f"  [EPOCH VAL] loss={vl:.4f} act_acc={m['act_acc']:.3f} "
                        f"cont_acc={m['cont_acc']:.3f} | "
                        f"P={m['edit_p']:.3f} R={m['edit_r']:.3f} F0.5={m['edit_f05']:.3f}"
                    )

                    if vl < best_val_loss:
                        best_val_loss = vl
                        no_improve = 0
                        if is_main:
                            best_path = os.path.join(ckpt_dir, "best.pt")
                            torch.save({
                                "epoch": epoch, "global_step": global_step,
                                "model_state_dict": raw_model.state_dict(),
                                "optimizer_state_dict": optimizer.state_dict(),
                                "val_loss": vl, "metrics": m,
                                "stage": stage["name"],
                                "best_val_loss": best_val_loss,
                                "no_improve": no_improve,
                                "data_state": _make_data_state(),
                            }, best_path)
                            print(f"  -> best 저장 (val_loss={vl:.4f})")
                            if args.gdrive_remote:
                                upload_and_cleanup(best_path, args.log_file, args.gdrive_remote, keep_latest_n=1)
                    else:
                        no_improve += 1
                        if no_improve >= args.patience:
                            if is_main:
                                print(f"\n  Early stop: {args.patience}ep 개선 없음")
                            break

            # 에포크 체크포인트
            if is_main:
                ep_ckpt_path = os.path.join(ckpt_dir, f"epoch_{epoch}.pt")
                torch.save({
                    "epoch": epoch, "global_step": global_step,
                    "model_state_dict": raw_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "stage": stage["name"],
                    "best_val_loss": best_val_loss,
                    "no_improve": no_improve,
                    "data_state": _make_data_state(),
                }, ep_ckpt_path)
                print(f"  [SAVE] {ep_ckpt_path}")
                if args.gdrive_remote:
                    upload_and_cleanup(ep_ckpt_path, args.log_file, args.gdrive_remote, keep_latest_n=1)

            if is_ddp:
                dist.barrier()

        if epoch > args.max_epochs or no_improve >= args.patience:
            break

    total_time = time.time() - train_start
    if is_main:
        print(f"\n완료! {epoch}ep, {global_step} steps, {fmt_time(total_time)}")
        if best_val_loss < float("inf"):
            print(f"Best val_loss: {best_val_loss:.4f}")

    if is_ddp:
        dist.destroy_process_group()


def main():
    p = argparse.ArgumentParser(description="KoELECTRA + Two-head GECToR")
    p.add_argument("--corpus", required=True)
    p.add_argument("--val_corpus", default=None)
    p.add_argument("--text_key", default=None)
    p.add_argument("--model_name", default="monologg/koelectra-base-v3-discriminator")
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--max_seq_len", type=int, default=512)
    p.add_argument("--max_epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--stage1_lr", type=float, default=5e-4)
    p.add_argument("--stage2_lr", type=float, default=2e-5)
    p.add_argument("--stage3_lr", type=float, default=1e-5)
    p.add_argument("--unfreeze_layers", type=int, default=6)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--grad_accum_steps", type=int, default=1, help="gradient accumulation steps")
    p.add_argument("--content_loss_weight", type=float, default=0.5)
    p.add_argument("--edit_loss_weight", type=float, default=2.0)
    p.add_argument("--label_smoothing", type=float, default=0.1)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--noise_preset", default="realistic")
    p.add_argument("--error_prob", type=float, default=0.5)
    p.add_argument("--error_count", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--log_interval", type=int, default=100)
    p.add_argument("--val_every", type=int, default=500)
    p.add_argument("--val_batches", type=int, default=50)
    p.add_argument("--bf16", action="store_true", default=True, help="BF16 AMP (기본 활성)")
    p.add_argument("--no_bf16", dest="bf16", action="store_false")
    p.add_argument("--compile", action="store_true", help="torch.compile 적용")
    p.add_argument("--compile_mode", default="reduce-overhead", choices=["default", "reduce-overhead", "max-autotune"])
    p.add_argument("--resume", default=None, help="체크포인트 경로 또는 디렉토리")
    p.add_argument("--save_interval", type=int, default=0, help="step 단위 체크포인트 주기 (0=에포크만)")
    p.add_argument("--gdrive_remote", default=None, help="rclone 대상 (예: 'gdrive:electra-gec-ckpts/')")
    p.add_argument("--log_file", default=None, help="동기화할 로그 파일")
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
