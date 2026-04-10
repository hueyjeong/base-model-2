import time
import torch
from data.bbpe_jamo_dataset import BBPEJamoDataset, load_bbpe_tokenizer
from tok.jamo_tokenizer import JamoTokenizer

def test_resume():
    print("=== Dataset Resume Efficiency Test ===")

    bbpe = load_bbpe_tokenizer()
    jamo = JamoTokenizer()
    corpus = "corpus/train.parquet"

    # 1. 처음부터 읽기 (기준점 확보)
    ds_full = BBPEJamoDataset(corpus, bbpe, jamo)
    it_full = iter(ds_full)

    target_step = 1000
    print(f"Reading first {target_step} samples for reference...")
    ref_sample = None
    for i in range(target_step):
        ref_sample = next(it_full)

    ref_line = ref_sample["_line_counter"]
    print(f"Reference line at step {target_step}: {ref_line}")

    # 2. Resume 테스트
    print(f"\nResuming from line {ref_line - 1}...")
    ds_resume = BBPEJamoDataset(corpus, bbpe, jamo)
    ds_resume.load_state_dict({"line_counter": ref_line - 1})

    it_resume = iter(ds_resume)
    start_time = time.time()
    resume_sample = next(it_resume)
    end_time = time.time()

    print(f"Resume took: {end_time - start_time:.4f}s")

    # 데이터 일치 확인
    match = torch.equal(ref_sample["jamo_ids"], resume_sample["jamo_ids"])
    print(f"Data match: {'SUCCESS' if match else 'FAIL'}")
    print(f"Resume sample line: {resume_sample['_line_counter']}")

    # 3. 대규모 Skip 테스트 (속도 측정)
    large_skip = 500000
    print(f"\nTesting large skip ({large_skip} lines)...")
    ds_large = BBPEJamoDataset(corpus, bbpe, jamo)
    ds_large.load_state_dict({"line_counter": large_skip})

    start_time = time.time()
    it_large = iter(ds_large)
    large_sample = next(it_large)
    end_time = time.time()

    print(f"Large skip (500k) took: {end_time - start_time:.4f}s")
    print(f"First sample after skip line: {large_sample['_line_counter']}")

    if (end_time - start_time) < 1.0:
        print("RESULT: Fast-forward is working efficiently! (under 1s)")
    else:
        print("RESULT: Fast-forward is slow (linear scan suspected).")

if __name__ == "__main__":
    test_resume()
