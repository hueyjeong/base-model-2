"""CompositionCodec 학습 — BBPE + Conv 자모 composition (concat 방식)

K-EXAONE 153K BBPE로 경계 결정 → 자모 분해 → concat → Conv 인코더/디코더 학습.
DDP 지원 (torchrun --nproc_per_node=N).
"""
import argparse
import math
import os
import sys
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from codec.composition_codec import CompositionCodec
from data.bbpe_jamo_dataset import BBPEJamoDataset, load_bbpe_tokenizer
from tok.jamo_tokenizer import JamoTokenizer


def _unwrap_state_dict(model):
    """DDP/compile 래핑을 벗긴 state_dict 반환"""
    m = model
    if hasattr(m, "module"):
        m = m.module
    sd = m.state_dict()
    prefix = "_orig_mod."
    return {k[len(prefix):] if k.startswith(prefix) else k: v for k, v in sd.items()}


@torch.no_grad()
def validate(codec, val_loader, device, max_samples=1000):
    """검증 세트 복원 정확도 측정"""
    codec.eval()
    total_correct = 0
    total_jamo = 0
    total_loss = 0.0
    n_batches = 0
    n_samples = 0

    raw_codec = codec.module if hasattr(codec, "module") else codec
    raw_codec = raw_codec._orig_mod if hasattr(raw_codec, "_orig_mod") else raw_codec
    is_parallel = getattr(raw_codec, "parallel_decoder", False)

    for batch in val_loader:
        jamo_ids = batch["jamo_ids"].to(device)
        jamo_mask = batch["jamo_mask"].to(device)
        segment_ids = batch["segment_ids"].to(device)
        n_segments = batch["n_segments"].to(device)

        out = codec(jamo_ids, jamo_mask, segment_ids, n_segments)
        if is_parallel:
            # logits: [B, P, S, V], target_slot: [B, P, S], slot_loss_mask: [B, P, S]
            pred = out["logits"].argmax(dim=-1)  # [B, P, S]
            target_slot = out["target_slot"]
            slot_mask = out["slot_loss_mask"]
            total_correct += ((pred == target_slot) & slot_mask).sum().item()
            total_jamo += slot_mask.sum().item()
        else:
            pred = out["logits"].argmax(dim=-1)
            valid = jamo_mask
            total_correct += ((pred == jamo_ids) & valid).sum().item()
            total_jamo += valid.sum().item()
        total_loss += out["loss"].item()
        n_batches += 1
        n_samples += jamo_ids.size(0)

        if n_samples >= max_samples:
            break

    codec.train()
    acc = total_correct / max(total_jamo, 1) * 100
    avg_loss = total_loss / max(n_batches, 1)
    return {"val_loss": avg_loss, "val_acc": acc, "val_samples": n_samples}


def train(args):
    # TF32 허용 (Ampere+ GPU에서 matmul 속도 향상; bf16 autocast와 별개로 f32 경로 가속)
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # DDP 초기화
    is_distributed = "RANK" in os.environ
    if is_distributed:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(device)
    else:
        rank = 0
        world_size = 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if rank == 0:
        print(f"Device: {device}" + (f" (DDP {world_size} GPUs)" if is_distributed else ""))

    # 토크나이저
    bbpe = load_bbpe_tokenizer()
    jamo = JamoTokenizer()
    if rank == 0:
        print(f"BBPE: {bbpe.vocab_size:,} vocab, Jamo: {jamo.vocab_size} vocab")

    # 모델
    codec = CompositionCodec(
        jamo_vocab=jamo.vocab_size,
        d_model=args.d_model,
        n_layers=args.n_layers,
        kernel_size=args.kernel_size,
        dropout=args.dropout,
        segment_masked=args.segment_masked,
        max_jamo_per_token=args.max_jamo_per_token,
        parallel_decoder=args.parallel_decoder,
        decoder_layers=args.decoder_layers,
        decoder_heads=args.decoder_heads,
    ).to(device)

    n_params = sum(p.numel() for p in codec.parameters())
    if rank == 0:
        print(f"CompositionCodec (concat): d={args.d_model}, L={args.n_layers}, "
              f"k={args.kernel_size}, params={n_params/1e6:.2f}M")

    # torch.compile
    if args.compile:
        if rank == 0:
            print("torch.compile 적용 중...")
        codec = torch.compile(codec)
        if rank == 0:
            print("torch.compile 완료")

    if is_distributed:
        codec = DDP(codec, device_ids=[rank])

    # 데이터
    dataset = BBPEJamoDataset(
        file_paths=args.corpus,
        bbpe_tokenizer=bbpe,
        jamo_tokenizer=jamo,
        max_seq_len=args.max_seq_len,
        max_jamo_per_token=args.max_jamo_per_token,
        text_key=args.text_key,
        rank=rank,
        world_size=world_size,
        append_pad_slot=args.append_pad_slot,
        fixed_slot=args.fixed_slot,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Validation 데이터
    val_loader = None
    if args.val_corpus and rank == 0:
        val_dataset = BBPEJamoDataset(
            file_paths=args.val_corpus,
            bbpe_tokenizer=bbpe,
            jamo_tokenizer=jamo,
            max_seq_len=args.max_seq_len,
            max_jamo_per_token=args.max_jamo_per_token,
            text_key=args.text_key,
            append_pad_slot=args.append_pad_slot,
            fixed_slot=args.fixed_slot,
        )
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=0)

    # Optimizer
    optimizer = torch.optim.AdamW(
        codec.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # LR schedule: warmup → cosine decay
    def lr_lambda(step):
        if step < args.warmup_steps:
            return step / max(args.warmup_steps, 1)
        progress = (step - args.warmup_steps) / max(args.max_steps - args.warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Init from checkpoint (가중치만 로드, 옵티마이저/스케줄러/데이터 상태 초기화)
    global_step = 0
    if args.init_from:
        if rank == 0:
            print(f"가중치 초기화: {args.init_from}")
        ckpt = torch.load(args.init_from, map_location=device, weights_only=False)
        sd = ckpt["model"]
        prefix = "_orig_mod."
        if any(k.startswith(prefix) for k in sd):
            sd = {k[len(prefix):] if k.startswith(prefix) else k: v for k, v in sd.items()}
        raw = codec.module if hasattr(codec, "module") else codec
        if hasattr(raw, "_orig_mod"):
            missing, unexpected = raw._orig_mod.load_state_dict(sd, strict=False)
        else:
            missing, unexpected = raw.load_state_dict(sd, strict=False)
        if rank == 0 and missing:
            print(f"  [init_from] 새로 초기화된 파라미터: {missing}")
        if rank == 0 and unexpected:
            print(f"  [init_from] 무시된 파라미터: {unexpected}")
        if rank == 0:
            print(f"  가중치 로드 완료 (step 0부터 재학습)")

    # Resume from checkpoint (전체 상태 복원)
    if args.resume:
        if rank == 0:
            print(f"체크포인트 복원: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        sd = ckpt["model"]
        prefix = "_orig_mod."
        if any(k.startswith(prefix) for k in sd):
            sd = {k[len(prefix):] if k.startswith(prefix) else k: v for k, v in sd.items()}
        # DDP/compile 래핑 전 모델에 로드
        # strict=False: 새 파라미터(intra_pos_emb 등) 추가 시에도 resume 가능
        raw = codec.module if hasattr(codec, "module") else codec
        if hasattr(raw, "_orig_mod"):
            missing, unexpected = raw._orig_mod.load_state_dict(sd, strict=False)
        else:
            missing, unexpected = raw.load_state_dict(sd, strict=False)
        if rank == 0 and missing:
            print(f"  [resume] 새로 초기화된 파라미터: {missing}")
        if rank == 0 and unexpected:
            print(f"  [resume] 무시된 파라미터: {unexpected}")
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        global_step = ckpt.get("step", 0)
        # scheduler를 복원된 step까지 진행
        if "scheduler" not in ckpt:
            for _ in range(global_step):
                scheduler.step()
        # 데이터셋 위치 복원
        data_state = ckpt.get("data_state")
        if isinstance(data_state, dict):
            dataset.load_state_dict(data_state)
            if rank == 0:
                print(f"  데이터 복원: line {data_state.get('line_counter', 0):,}")
        if rank == 0:
            print(f"  step {global_step}부터 재개")

    # BF16
    use_amp = args.bf16 and device.type == "cuda"

    # 학습
    codec.train()
    accum_loss = 0.0
    accum_correct = 0
    accum_total = 0
    max_line_counter = 0
    t_start = time.time()

    grad_accum = args.grad_accum_steps
    if rank == 0:
        eff_batch = args.batch_size * grad_accum * world_size
        batch_desc = f"batch={args.batch_size}"
        if grad_accum > 1 or world_size > 1:
            parts = [str(args.batch_size)]
            if grad_accum > 1:
                parts.append(f"accum{grad_accum}")
            if world_size > 1:
                parts.append(f"{world_size}gpu")
            batch_desc = f"batch={'×'.join(parts)}={eff_batch}"
        print(f"\n학습 시작: max_steps={args.max_steps}, {batch_desc}"
              + f", seq_len={args.max_seq_len}")
        print(f"{'step':>8} {'loss':>8} {'acc':>8} {'lr':>10} {'tok/s':>8}")
        print("-" * 50)

    micro_step = 0
    for batch in loader:
        if global_step >= args.max_steps:
            break

        jamo_ids = batch["jamo_ids"].to(device)
        jamo_mask = batch["jamo_mask"].to(device)
        segment_ids = batch["segment_ids"].to(device)
        n_segments = batch["n_segments"].to(device)
        if "_line_counter" in batch:
            max_line_counter = max(max_line_counter, batch["_line_counter"].max().item())

        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            out = codec(jamo_ids, jamo_mask, segment_ids, n_segments)
            loss = out["loss"] / grad_accum

        loss.backward()

        # 통계
        with torch.no_grad():
            pred = out["logits"].argmax(dim=-1)  # conv decoder: [B,L] / parallel: [B,P,S]
            if "target_slot" in out:
                target_slot = out["target_slot"]
                slot_mask = out["slot_loss_mask"]
                correct = ((pred == target_slot) & slot_mask).sum().item()
                total = slot_mask.sum().item()
            else:
                valid = jamo_mask
                correct = ((pred == jamo_ids) & valid).sum().item()
                total = valid.sum().item()

        accum_loss += loss.item() * grad_accum
        accum_correct += correct
        accum_total += total
        micro_step += 1

        if micro_step % grad_accum != 0:
            continue

        if args.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(codec.parameters(), args.max_grad_norm)

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        global_step += 1

        # 로깅
        if global_step % args.log_every == 0 and rank == 0:
            dt = time.time() - t_start
            avg_loss = accum_loss / args.log_every
            avg_acc = accum_correct / max(accum_total, 1) * 100
            tok_s = accum_total / max(dt, 1e-6)
            if is_distributed:
                tok_s *= world_size
            lr = scheduler.get_last_lr()[0]

            progress = global_step / args.max_steps * 100
            print(f"{global_step:8d} {avg_loss:8.4f} {avg_acc:7.2f}% {lr:10.2e} {tok_s:8.0f}  {progress:.1f}% L{max_line_counter:,}")

            accum_loss = 0.0
            accum_correct = 0
            accum_total = 0
            t_start = time.time()

        # Validation
        if val_loader is not None and args.val_every > 0 and global_step % args.val_every == 0 and rank == 0:
            val_metrics = validate(codec, val_loader, device, args.val_samples)
            print(f"  [VAL] loss={val_metrics['val_loss']:.4f}, "
                  f"acc={val_metrics['val_acc']:.4f}%, "
                  f"samples={val_metrics['val_samples']}")

        # 체크포인트
        if args.save_every > 0 and global_step % args.save_every == 0 and rank == 0:
            model_sd = _unwrap_state_dict(codec)
            tag = f"composition_{args.n_layers}L"
            save_path = os.path.join(args.out_dir, f"{tag}_step{global_step}.pt")
            os.makedirs(args.out_dir, exist_ok=True)
            torch.save({
                "model": model_sd,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "step": global_step,
                "data_state": {"line_counter": int(max_line_counter)},
                "args": vars(args),
            }, save_path)
            print(f"  → 체크포인트 저장: {save_path}")
            # 구글 드라이브 업로드 (GDRIVE 환경변수: rclone 원격지)
            gdrive = os.environ.get("GDRIVE")
            if gdrive:
                from training.upload_gdrive import upload_and_cleanup
                log_path = os.path.join(
                    os.path.dirname(args.out_dir), "composition_train_log.txt",
                )
                upload_and_cleanup(save_path, log_path, gdrive)
                print(f"  → GDrive 업로드 시작 (백그라운드): {gdrive}")

    if rank == 0:
        print(f"\n학습 완료: {global_step} steps")

    # 최종 저장
    if args.out_dir and rank == 0:
        model_sd = _unwrap_state_dict(codec)
        tag = f"composition_{args.n_layers}L"
        save_path = os.path.join(args.out_dir, f"{tag}_final.pt")
        os.makedirs(args.out_dir, exist_ok=True)
        torch.save({
            "model": model_sd,
            "step": global_step,
            "args": vars(args),
        }, save_path)
        print(f"최종 저장: {save_path}")
        gdrive = os.environ.get("GDRIVE")
        if gdrive:
            import subprocess, shlex
            log_path = os.path.join(
                os.path.dirname(args.out_dir), "composition_train_log.txt",
            )
            print(f"  → GDrive 최종 업로드 중... ({gdrive})")
            try:
                subprocess.run(
                    f"rclone copy {shlex.quote(save_path)} {shlex.quote(gdrive)}",
                    shell=True, check=True,
                )
                if os.path.exists(log_path):
                    subprocess.run(
                        f"rclone copy {shlex.quote(log_path)} {shlex.quote(gdrive)}",
                        shell=True, check=True,
                    )
                print(f"  → GDrive 업로드 완료")
            except Exception as e:
                print(f"  → GDrive 업로드 실패: {e}")

    if is_distributed:
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description="CompositionCodec 학습 (concat)")

    # 데이터
    parser.add_argument("--corpus", nargs="+", required=True)
    parser.add_argument("--text_key", default="text")
    parser.add_argument("--max_seq_len", type=int, default=512)

    # 모델
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--n_layers", type=int, default=5)
    parser.add_argument("--kernel_size", type=int, default=7)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--segment_masked", action="store_true",
                        help="Conv가 토큰 경계를 못 넘도록 segment mask 적용 (리크 0)")
    parser.add_argument("--append_pad_slot", action="store_true",
                        help="각 segment 끝에 PAD 슬롯 1개 추가 (decoder가 PAD 출력 학습)")
    parser.add_argument("--fixed_slot", action="store_true",
                        help="모든 segment를 max_jamo_per_token 슬롯으로 고정 (decode_from_vec 완벽 대응)")
    parser.add_argument("--max_jamo_per_token", type=int, default=32,
                        help="토큰당 자모 슬롯 수 상한 (fixed_slot/parallel_decoder 시 정확히 이 값)")
    parser.add_argument("--parallel_decoder", action="store_true",
                        help="Decoder를 ParallelSlotDecoder(self-attention 2L)로 교체. "
                             "Encoder는 가변 입력 유지, decoder만 max_jamo_per_token 슬롯 학습. "
                             "Downstream encoder-only 경로 영향 없음")
    parser.add_argument("--decoder_layers", type=int, default=2,
                        help="ParallelSlotDecoder transformer layer 수")
    parser.add_argument("--decoder_heads", type=int, default=4,
                        help="ParallelSlotDecoder transformer head 수")

    # 학습
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--max_steps", type=int, default=10000)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--num_workers", type=int, default=2)

    # 로깅/저장/재개
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--save_every", type=int, default=0)
    parser.add_argument("--val_corpus", nargs="+", default=None, help="검증 코퍼스")
    parser.add_argument("--val_every", type=int, default=5000, help="검증 주기 (steps)")
    parser.add_argument("--val_samples", type=int, default=1000, help="검증 샘플 수")
    parser.add_argument("--out_dir", default="exp-jamo-codec/checkpoints")
    parser.add_argument("--resume", default=None, help="체크포인트에서 재개 (모델+옵티마이저+스케줄러+데이터 상태 복원)")
    parser.add_argument("--init_from", default=None, help="체크포인트 가중치로 초기화만 (옵티마이저/스케줄러/데이터 상태는 초기화)")

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
