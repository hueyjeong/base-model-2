import torch
import torch.nn as nn
from model.seq2seq import BitMambaSeq2Seq
from model.config import BitMambaSeq2SeqConfig
from training.pretrain import load_tokenizer
from torch.utils.data import DataLoader
from training.dataset import StreamingPackedDataset
from training.noising import DenoisingNoiser, NoiseConfig

def eval_model(name, source_bias=0.0, use_copy_gate=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = load_tokenizer("keyboard")
    
    # 임의의 초기화된 8M 모델로 BPC 평가 (학습하지 않고 복사 능력만 확인)
    config = BitMambaSeq2SeqConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=288, d_inner=576, d_ff=544,
        n_encoder_layers=3, n_decoder_layers=5,
        n_heads=8, n_kv_heads=4, dt_rank=18,
        use_copy_gate=use_copy_gate
    )
    model = BitMambaSeq2Seq(config).to(device)
    model.eval()

    noiser = DenoisingNoiser(tokenizer, NoiseConfig(), use_korean_errors=True)
    val_dataset = StreamingPackedDataset(
        "corpus/val_50k.jsonl", tokenizer, noiser,
        pack_size=2048, text_key="text", seed=42
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

    val_loader = DataLoader(val_dataset, batch_size=4, collate_fn=collate_packed)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id)

    total_loss = 0.0
    total_tokens = 0
    import time
    
    print(f"\n--- Evaluating {name} ---")
    start = time.time()
    
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= 10: # 10배치만 빠르게 검증
                break
                
            src_ids = batch["src_ids"].to(device)
            tgt_ids = batch["tgt_ids"].to(device)
            src_mask = batch["src_mask"].to(device)

            tgt_input = tgt_ids[:, :-1]
            tgt_target = tgt_ids[:, 1:]

            logits = model(src_ids, tgt_input, src_mask, source_bias=source_bias)
            loss = criterion(logits.view(-1, config.vocab_size), tgt_target.reshape(-1))

            n_tok = (tgt_target != tokenizer.pad_id).sum().item()
            total_loss += loss.item() * n_tok
            total_tokens += n_tok

    avg_loss = total_loss / total_tokens
    print(f"Result for {name}: Loss = {avg_loss:.4f}, Time = {time.time() - start:.2f}s")
    return avg_loss

if __name__ == "__main__":
    eval_model("Baseline", source_bias=0.0, use_copy_gate=False)
    eval_model("Trial A (Bias 0.5)", source_bias=0.5, use_copy_gate=False)
    eval_model("Trial A (Bias 2.0)", source_bias=2.0, use_copy_gate=False)
    eval_model("Trial B (Copy Gate)", source_bias=0.0, use_copy_gate=True)
