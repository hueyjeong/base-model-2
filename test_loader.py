from training.pretrain import load_tokenizer
from training.dataset import StreamingPackedDataset
from training.noising import DenoisingNoiser, NoiseConfig
from torch.utils.data import DataLoader
import time

print("Loading tokenizer...")
tokenizer = load_tokenizer("keyboard")
print("Tokenizer loaded.")

noiser = DenoisingNoiser(tokenizer, NoiseConfig(), seed=42, use_korean_errors=True)
dataset = StreamingPackedDataset(
    "corpus/sample_10g.jsonl", tokenizer, noiser,
    pack_size=2048, text_key="text", seed=42
)

def collate_packed(batch):
    from torch.nn.utils.rnn import pad_sequence
    src_list = [b["src_ids"] for b in batch]
    tgt_list = [b["tgt_ids"] for b in batch]
    src_ids = pad_sequence(src_list, batch_first=True, padding_value=0)
    tgt_ids = pad_sequence(tgt_list, batch_first=True, padding_value=0)
    return {"src_ids": src_ids, "tgt_ids": tgt_ids}

loader = DataLoader(dataset, batch_size=4, num_workers=0, collate_fn=collate_packed)

print("Starting loader iter...")
data_iter = iter(loader)

for i in range(20):
    start = time.time()
    batch = next(data_iter)
    print(f"Batch {i+1} received! shape: {batch['src_ids'].shape} in {time.time()-start:.3f}s")
