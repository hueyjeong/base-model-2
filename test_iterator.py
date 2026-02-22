import time
from bbpe_tokenizer.bbpe_wrapper import BBPETokenizer
from training.noising import DenoisingNoiser, NoiseConfig
from training.dataset import StreamingPackedDataset

tok = BBPETokenizer("bbpe_tokenizer/bbpe.json")
noiser = DenoisingNoiser(tok, NoiseConfig(), seed=42)

dataset = StreamingPackedDataset(
    "corpus/sample_10g.jsonl", tok, noiser,
    pack_size=2048, text_key="text", seed=42
)

iterator = iter(dataset)
for i in range(10):
    start = time.time()
    batch = next(iterator)
    print(f"Sample {i}: src={batch['src_ids'].shape}, time={time.time()-start:.2f}s")
