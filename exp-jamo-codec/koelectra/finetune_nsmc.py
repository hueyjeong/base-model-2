"""JamoKoElectra NSMC fine-tune (binary sentiment classification).

30k 체크포인트 등 중간 단계 사전학습 모델의 표현력을 빠르게 검증하기 위한
스크립트. Discriminator transformer만 재사용하고 Generator/Decoder는 버림.

사용 예:
    python -m exp-jamo-codec.koelectra.finetune_nsmc \\
        --pretrained exp-jamo-codec/koelectra/checkpoints/electra_step_30000.pt \\
        --epochs 2 --batch_size 64 --lr 2e-5

비교 기준 (KoELECTRA Small v3 공개 수치): NSMC test acc ≈ 89.6%.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import urllib.request

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

from data.bbpe_jamo_dataset import (  # noqa: E402
    JAMO_BOS, JAMO_EOS, JAMO_PAD, decompose_token, load_bbpe_tokenizer,
)
from tok.jamo_tokenizer import JamoTokenizer  # noqa: E402
from koelectra.model.electra import JamoKoElectra  # noqa: E402


NSMC_TRAIN_URL = "https://raw.githubusercontent.com/e9t/nsmc/master/ratings_train.txt"
NSMC_TEST_URL = "https://raw.githubusercontent.com/e9t/nsmc/master/ratings_test.txt"


# ──────────────────────────────────────────────────────────────────────────
# Data
# ──────────────────────────────────────────────────────────────────────────
def download_nsmc(cache_dir: str = "corpus/nsmc"):
    os.makedirs(cache_dir, exist_ok=True)
    paths = {}
    for name, url in [("train", NSMC_TRAIN_URL), ("test", NSMC_TEST_URL)]:
        dst = os.path.join(cache_dir, f"ratings_{name}.txt")
        if not os.path.exists(dst):
            print(f"[NSMC] downloading {url} → {dst}")
            urllib.request.urlretrieve(url, dst)
        paths[name] = dst
    return paths["train"], paths["test"]


class NSMCDataset(Dataset):
    """NSMC TSV → (jamo_ids, mask, segment_ids, n_segments, label) 텐서.

    각 리뷰를 [BOS] + BBPE → jamo 분해 토큰들 + [EOS]로 구성.
    길이 초과분은 truncation (앞쪽 유지). max_jamo/max_patches 상한 내.
    """

    def __init__(self, tsv_path: str, bbpe, jamo: JamoTokenizer,
                 max_patches: int = 128, max_jamo: int = 640):
        self.rows = []
        with open(tsv_path, "r", encoding="utf-8") as f:
            header = next(f)
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) != 3:
                    continue
                _, text, label = parts
                text = text.strip()
                if not text:
                    continue
                try:
                    self.rows.append((text, int(label)))
                except ValueError:
                    continue
        self.bbpe = bbpe
        self.jamo = jamo
        self.max_patches = max_patches
        self.max_jamo = max_jamo

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        text, label = self.rows[idx]
        bbpe_ids = self.bbpe.encode(text, add_special_tokens=False)

        # [BOS] + tokens + [EOS]
        seqs = [[JAMO_BOS]]
        for tid in bbpe_ids:
            s = self.bbpe.decode([tid])
            j = decompose_token(s, self.jamo)
            if 0 < len(j) <= 32:
                seqs.append(j)
        seqs.append([JAMO_EOS])

        all_jamo, seg_ids, seg_idx = [], [], 0
        for seq in seqs:
            if seg_idx >= self.max_patches:
                break
            if len(all_jamo) + len(seq) > self.max_jamo:
                break
            all_jamo.extend(seq)
            seg_ids.extend([seg_idx] * len(seq))
            seg_idx += 1

        L = len(all_jamo)
        pad = self.max_jamo - L
        jamo_ids = torch.tensor(all_jamo + [JAMO_PAD] * pad, dtype=torch.long)
        jamo_mask = torch.tensor([True] * L + [False] * pad, dtype=torch.bool)
        segment_ids = torch.tensor(seg_ids + [0] * pad, dtype=torch.long)

        return {
            "jamo_ids": jamo_ids,
            "jamo_mask": jamo_mask,
            "segment_ids": segment_ids,
            "n_segments": seg_idx,
            "label": label,
        }


def collate(batch):
    return {
        "jamo_ids": torch.stack([b["jamo_ids"] for b in batch]),
        "jamo_mask": torch.stack([b["jamo_mask"] for b in batch]),
        "segment_ids": torch.stack([b["segment_ids"] for b in batch]),
        "n_segments": torch.tensor([b["n_segments"] for b in batch], dtype=torch.long),
        "labels": torch.tensor([b["label"] for b in batch], dtype=torch.long),
    }


# ──────────────────────────────────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────────────────────────────────
class ElectraClassifier(nn.Module):
    """Disc 경로 그대로 쓰고 pooled output에 binary head.

    Pooling:
        - 'bos': 첫 patch([BOS]) hidden 사용
        - 'mean': valid patch hidden의 평균
    """

    def __init__(self, backbone: JamoKoElectra, num_classes: int = 2,
                 pooling: str = "bos", dropout: float = 0.1):
        super().__init__()
        self.backbone = backbone
        self.max_patches = backbone.max_patches
        self.pooling = pooling
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(backbone.hidden_size, num_classes)

    def forward(self, jamo_ids, jamo_mask, segment_ids, n_segments):
        B = jamo_ids.size(0)
        P = self.max_patches
        device = jamo_ids.device
        pos = torch.arange(P, device=device).unsqueeze(0).expand(B, -1)
        patch_pad_mask = pos < n_segments.unsqueeze(-1)  # [B, P]

        # Disc forward (Gen/Decoder 우회)
        z = self.backbone.codec_encoder(jamo_ids, jamo_mask, segment_ids, n_segments)
        e = self.backbone._embed(z, patch_pad_mask)
        h = self.backbone.disc_hidden_proj(e)
        h = self.backbone.discriminator(h, patch_pad_mask)

        # Pooling
        if self.pooling == "bos":
            pooled = h[:, 0]
        else:  # mean over valid patches
            mask = patch_pad_mask.unsqueeze(-1).to(h.dtype)
            pooled = (h * mask).sum(1) / mask.sum(1).clamp(min=1)

        return self.classifier(self.dropout(pooled))


def load_pretrained_backbone(ckpt_path: str, device: str = "cpu") -> JamoKoElectra:
    """체크포인트에서 JamoKoElectra 구조 복원. args dict 사용."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    args = ckpt.get("args", {})

    backbone = JamoKoElectra(
        codec_d_model=args.get("codec_d_model", 256),
        codec_n_layers=args.get("codec_n_layers", 6),
        codec_kernel_size=args.get("codec_kernel_size", 7),
        max_jamo_per_token=args.get("max_jamo_per_token", 32),
        embedding_size=args.get("embedding_size", 128),
        hidden_size=args.get("hidden_size", 256),
        n_heads=args.get("n_heads", 4),
        d_ff=args.get("d_ff", 1024),
        gen_layers=args.get("gen_layers", 12),
        disc_layers=args.get("disc_layers", 12),
        dropout=args.get("dropout", 0.1),
        max_patches=args.get("max_patches", 512),
        gen_loss_weight=args.get("gen_loss_weight", 50.0),
    )
    missing, unexpected = backbone.load_state_dict(ckpt["model"], strict=True)
    print(f"[Load] missing={len(missing)}, unexpected={len(unexpected)}, "
          f"step={ckpt.get('step', '?')}")
    return backbone


# ──────────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────────
def run_epoch(model, loader, optimizer, scheduler, device, amp_dtype, train: bool):
    if train:
        model.train()
    else:
        model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    ctx = torch.no_grad() if not train else torch.enable_grad()
    with ctx:
        for batch in loader:
            jids = batch["jamo_ids"].to(device, non_blocking=True)
            jmask = batch["jamo_mask"].to(device, non_blocking=True)
            sids = batch["segment_ids"].to(device, non_blocking=True)
            nseg = batch["n_segments"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=(device.type == "cuda")):
                logits = model(jids, jmask, sids, nseg)
                loss = F.cross_entropy(logits, labels)

            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            total_loss += loss.item() * labels.size(0)
            preds = logits.argmax(-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return total_loss / max(total, 1), correct / max(total, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pretrained", type=str, required=True,
                    help="JamoKoElectra 체크포인트 (electra_step_*.pt)")
    ap.add_argument("--nsmc_dir", type=str, default="corpus/nsmc")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--eval_batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--warmup_ratio", type=float, default=0.05)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--max_patches", type=int, default=128,
                    help="NSMC 리뷰는 짧음 — 128이면 99%+ 커버")
    ap.add_argument("--max_jamo", type=int, default=640)
    ap.add_argument("--pooling", choices=["bos", "mean"], default="bos")
    ap.add_argument("--bf16", action="store_true", default=True)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.bfloat16 if args.bf16 else torch.float32
    print(f"[Setup] device={device}, bf16={args.bf16}")

    # TF32
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # ── NSMC 데이터 ──
    train_tsv, test_tsv = download_nsmc(args.nsmc_dir)
    bbpe = load_bbpe_tokenizer()
    jamo = JamoTokenizer()

    train_ds = NSMCDataset(train_tsv, bbpe, jamo,
                           max_patches=args.max_patches, max_jamo=args.max_jamo)
    test_ds = NSMCDataset(test_tsv, bbpe, jamo,
                          max_patches=args.max_patches, max_jamo=args.max_jamo)
    print(f"[Data] train={len(train_ds)}, test={len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, collate_fn=collate,
                               pin_memory=(device.type == "cuda"))
    test_loader = DataLoader(test_ds, batch_size=args.eval_batch_size, shuffle=False,
                              num_workers=args.num_workers, collate_fn=collate,
                              pin_memory=(device.type == "cuda"))

    # ── 모델 ──
    # NSMC는 짧아 max_patches 작게 써도 되지만 pos_emb shape 호환 위해
    # backbone.max_patches는 pretrained 설정 그대로 유지.
    # classifier.forward에서 실제 shape [B, max_patches_pre, D]로 돌리되
    # 입력 텐서는 args.max_patches 길이로만 채움 → 뒤쪽은 padding mask로 제외.
    # 즉 backbone을 그대로 쓰고 데이터만 짧게.
    backbone = load_pretrained_backbone(args.pretrained, device="cpu")

    # 데이터 shape이 max_patches_pretrain과 달라도 pos_emb/segment는 동적 길이 OK.
    # 다만 우리 electra.py는 self.max_patches를 forward에서 P로 쓴다.
    # 간단히 backbone.max_patches를 args.max_patches로 덮어쓰고 pos_emb slice.
    if args.max_patches < backbone.max_patches:
        print(f"[Trim] backbone.max_patches {backbone.max_patches} → {args.max_patches}")
        # CompositionEncoder.fixed_output_len도 맞춰 변경
        backbone.codec_encoder.fixed_output_len = args.max_patches
        # pos_emb slice: 앞쪽 args.max_patches slot만 사용
        old = backbone.pos_emb.weight.data
        new_pos = nn.Embedding(args.max_patches, old.size(1))
        new_pos.weight.data.copy_(old[:args.max_patches])
        backbone.pos_emb = new_pos
        backbone.max_patches = args.max_patches

    model = ElectraClassifier(backbone, num_classes=2, pooling=args.pooling).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[Model] total params={total_params/1e6:.2f}M (backbone 전체 포함, "
          f"실제 fwd는 Disc 경로만)")

    # ── Optimizer + linear warmup/decay ──
    no_decay = ("bias", "LayerNorm.weight", "norm1.weight", "norm2.weight",
                "emb_layer_norm.weight")
    decay_params, no_decay_params = [], []
    for n, p in model.named_parameters():
        if any(nd in n for nd in no_decay):
            no_decay_params.append(p)
        else:
            decay_params.append(p)
    optimizer = torch.optim.AdamW(
        [{"params": decay_params, "weight_decay": args.weight_decay},
         {"params": no_decay_params, "weight_decay": 0.0}],
        lr=args.lr, betas=(0.9, 0.999), eps=1e-6,
    )

    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 1.0 - progress)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ── 학습 루프 ──
    print(f"\n[Train] total_steps={total_steps}, warmup={warmup_steps}")
    best_acc = 0.0
    for epoch in range(args.epochs):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(model, train_loader, optimizer, scheduler,
                                     device, amp_dtype, train=True)
        t_train = time.time() - t0
        t0 = time.time()
        te_loss, te_acc = run_epoch(model, test_loader, optimizer, scheduler,
                                     device, amp_dtype, train=False)
        t_eval = time.time() - t0
        print(f"[Epoch {epoch+1}/{args.epochs}] "
              f"train loss={tr_loss:.4f} acc={tr_acc:.4f} ({t_train:.0f}s) | "
              f"test loss={te_loss:.4f} acc={te_acc:.4f} ({t_eval:.0f}s)")
        if te_acc > best_acc:
            best_acc = te_acc

    print(f"\n=== NSMC Best Test Acc: {best_acc:.4f} ===")
    print(f"참고 baseline:")
    print(f"  KoELECTRA Small v3: 89.6%")
    print(f"  mBERT:              87.0%")
    print(f"  Random:             50.0%")


if __name__ == "__main__":
    main()
