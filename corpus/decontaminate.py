#!/usr/bin/env python3
"""Remove val_50k.jsonl lines from sample_1g.jsonl and sample_10g.jsonl."""

import os
import sys
import tempfile
import shutil

VAL_FILE = "val_50k.jsonl"
TARGETS = ["sample_1g.jsonl", "sample_10g.jsonl"]

def main():
    # 1. Load validation lines into a set (raw line strings for exact match)
    print(f"Loading {VAL_FILE} into memory...")
    val_lines = set()
    with open(VAL_FILE, "r", encoding="utf-8") as f:
        for line in f:
            val_lines.add(line.rstrip("\n"))
    print(f"  Loaded {len(val_lines):,} validation lines.")

    # 2. Filter each target file
    for target in TARGETS:
        if not os.path.exists(target):
            print(f"  Skipping {target} (not found)")
            continue

        print(f"\nProcessing {target}...")
        original_count = 0
        kept_count = 0
        removed_count = 0

        # Write to a temp file in the same directory (for atomic rename)
        tmp_fd, tmp_path = tempfile.mkstemp(
            dir=".", prefix=f".{target}.tmp_", suffix=".jsonl"
        )
        try:
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as out_f, \
                 open(target, "r", encoding="utf-8") as in_f:
                for line in in_f:
                    original_count += 1
                    stripped = line.rstrip("\n")
                    if stripped in val_lines:
                        removed_count += 1
                    else:
                        out_f.write(line)
                        kept_count += 1

                    if original_count % 1_000_000 == 0:
                        print(f"  ... processed {original_count:,} lines "
                              f"(removed {removed_count:,} so far)")

            # Replace original with filtered version
            shutil.move(tmp_path, target)
            print(f"  Done: {original_count:,} -> {kept_count:,} lines "
                  f"(removed {removed_count:,})")

        except Exception:
            # Clean up temp file on error
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise

if __name__ == "__main__":
    main()
