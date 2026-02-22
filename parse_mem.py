import pickle
with open("mem_snapshot.pickle", "rb") as f:
    snapshot = pickle.load(f)

max_alloc = 0
for segment in snapshot["segments"]:
    for block in segment["blocks"]:
        max_alloc += block["size"]

print(f"Max allocated bytes in snapshot segments: {max_alloc / 1e9:.2f} GB")
