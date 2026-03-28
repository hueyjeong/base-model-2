"""NIKL PARA 데이터를 train/val/test로 분리

3개 JSON 파일에서 (original_form, corrected_form) 쌍을 추출하여
document 단위로 70/15/15 분리.

Usage:
    python corpus/split_nikl_para.py
"""
import json
import os
import random

PARA_DIR = os.path.join(os.path.dirname(__file__), "PARA")
OUT_DIR = os.path.join(os.path.dirname(__file__), "nikl_para")

SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
# TEST_RATIO = 0.15 (나머지)

FILES = [
    ("EXEC2102112091.json", "paragraph"),   # SNS/웹
    ("MXEC2102112091.json", "utterance"),    # 메신저 2021
    ("MXEC2202210100.json", "utterance"),    # 메신저 2022
]


def extract_documents(filepath, unit_key):
    """JSON 파일에서 document별 (orig, cor) 쌍 리스트를 추출"""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    documents = []
    for doc in data["document"]:
        pairs = []
        for item in doc.get(unit_key, []):
            orig = item.get("original_form", "").replace("\r\n", " ").replace("\r", " ").replace("\n", " ").strip()
            cor = item.get("corrected_form", "").replace("\r\n", " ").replace("\r", " ").replace("\n", " ").strip()
            if orig and cor:
                pairs.append((orig, cor))
        if pairs:
            documents.append({
                "doc_id": doc["id"],
                "source": os.path.basename(filepath),
                "pairs": pairs,
            })
    return documents


def write_split(docs, split_name, out_dir):
    """split별 orig.txt, cor.txt, jsonl 저장"""
    orig_path = os.path.join(out_dir, f"{split_name}_orig.txt")
    cor_path = os.path.join(out_dir, f"{split_name}_cor.txt")
    jsonl_path = os.path.join(out_dir, f"{split_name}.jsonl")

    n_pairs = 0
    with open(orig_path, "w") as fo, \
         open(cor_path, "w") as fc, \
         open(jsonl_path, "w") as fj:
        for doc in docs:
            for orig, cor in doc["pairs"]:
                fo.write(orig + "\n")
                fc.write(cor + "\n")
                fj.write(json.dumps({"text": cor}, ensure_ascii=False) + "\n")
                n_pairs += 1
    return n_pairs


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # 전체 document 수집
    all_docs = []
    for fname, unit_key in FILES:
        fpath = os.path.join(PARA_DIR, fname)
        docs = extract_documents(fpath, unit_key)
        print(f"  {fname}: {len(docs)} docs, {sum(len(d['pairs']) for d in docs)} pairs")
        all_docs.extend(docs)

    print(f"\n전체: {len(all_docs)} docs, {sum(len(d['pairs']) for d in all_docs)} pairs")

    # document 단위 셔플 & 분리
    random.seed(SEED)
    random.shuffle(all_docs)

    n = len(all_docs)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)

    train_docs = all_docs[:n_train]
    val_docs = all_docs[n_train:n_train + n_val]
    test_docs = all_docs[n_train + n_val:]

    # 저장
    for split_name, docs in [("train", train_docs), ("val", val_docs), ("test", test_docs)]:
        n_pairs = write_split(docs, split_name, OUT_DIR)
        print(f"  {split_name}: {len(docs)} docs, {n_pairs} pairs")

    print(f"\n저장 위치: {OUT_DIR}/")


if __name__ == "__main__":
    main()
