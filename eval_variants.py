import torch
import torch.nn as nn
from model.seq2seq import BitMambaSeq2Seq
from model.config import BitMambaSeq2SeqConfig
from training.pretrain import load_tokenizer
from torch.utils.data import DataLoader
from training.dataset import StreamingPackedDataset
from training.noising import DenoisingNoiser, NoiseConfig
import time

def evaluate_zero_shot_variants(num_batches=20):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = load_tokenizer("keyboard")
    print(f"Device: {device}, Tokenizer: keyboard")

    noiser = DenoisingNoiser(tokenizer, NoiseConfig(), use_korean_errors=True)
    val_dataset = StreamingPackedDataset(
        "corpus/val_50k.jsonl", tokenizer, noiser,
        pack_size=1024, text_key="text", seed=42
    )

    def collate_packed(batch):
        from torch.nn.utils.rnn import pad_sequence
        src_list = [b["src_ids"] for b in batch]
        tgt_list = [b["tgt_ids"] for b in batch]
        n_chars = sum(b["n_chars"] for b in batch)
        src_ids = pad_sequence(src_list, batch_first=True, padding_value=0)
        tgt_ids = pad_sequence(tgt_list, batch_first=True, padding_value=0)
        src_mask = src_ids != 0
        return {"src_ids": src_ids, "tgt_ids": tgt_ids, "src_mask": src_mask, "n_chars": n_chars}

    val_loader = DataLoader(val_dataset, batch_size=2, num_workers=0, collate_fn=collate_packed)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id)

    # 평가할 설정 목록
    configs = [
        {"name": "Baseline", "bias": 0.0, "copy_gate": False},
        {"name": "Trial A (Bias 0.5)", "bias": 0.5, "copy_gate": False},
        {"name": "Trial A (Bias 1.0)", "bias": 1.0, "copy_gate": False},
        {"name": "Trial B (Copy Gate)", "bias": 0.0, "copy_gate": True},
        {"name": "Trial A+B (Bias 0.5 + Gate)", "bias": 0.5, "copy_gate": True},
    ]

    # 공통 데이터 수집 (동일 배치로 공정한 비교)
    print(f"\nCollecting {num_batches} batches...")
    batches = []
    val_iter = iter(val_loader)
    for _ in range(num_batches):
        batches.append(next(val_iter))
    print(f"Collected {len(batches)} batches.\n")

    results = []

    for cfg in configs:
        name = cfg["name"]
        bias = cfg["bias"]
        use_copy = cfg["copy_gate"]

        model_config = BitMambaSeq2SeqConfig(
            vocab_size=tokenizer.vocab_size,
            d_model=288, d_inner=576, d_ff=544,
            n_encoder_layers=3, n_decoder_layers=5,
            n_heads=8, n_kv_heads=4, dt_rank=18,
            use_copy_gate=use_copy
        )
        model = BitMambaSeq2Seq(model_config).to(device)
        model.eval()

        total_loss = 0.0
        total_tokens = 0
        start = time.time()

        with torch.no_grad():
            for batch in batches:
                src = batch["src_ids"].to(device)
                tgt = batch["tgt_ids"].to(device)
                mask = batch["src_mask"].to(device)

                tgt_in = tgt[:, :-1]
                tgt_out = tgt[:, 1:]

                logits = model(src, tgt_in, mask, source_bias=bias)
                loss = criterion(logits.reshape(-1, tokenizer.vocab_size), tgt_out.reshape(-1))

                n_tok = (tgt_out != tokenizer.pad_id).sum().item()
                total_loss += loss.item() * n_tok
                total_tokens += n_tok

        avg_loss = total_loss / total_tokens if total_tokens > 0 else float('nan')
        elapsed = time.time() - start
        
        print(f"[{name}] Loss: {avg_loss:.4f} | Time: {elapsed:.2f}s")
        results.append((name, avg_loss))

if __name__ == "__main__":
    evaluate_zero_shot_variants(num_batches=20)
