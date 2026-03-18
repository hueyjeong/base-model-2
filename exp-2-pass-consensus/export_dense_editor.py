"""DenseEditor (BiMamba2) 체크포인트를 BMMQ 포맷으로 export

DenseEditor의 state_dict 키 패턴:
  BitLinear (packed2bit):
    layers.{i}.mixing.{fwd,bwd}.mamba2.{in_proj,out_proj}.weight
    layers.{i}.ffn.{gate_up_proj,down_proj}.weight
    tag_head.weight
  F32 (그 외 전부):
    embedding, norms, conv1d, SSM params (A_log, D, dt_bias)

사용법:
    python exp-2-pass-consensus/export_dense_editor.py \
        exp-2-pass-consensus/dense_mamba2_d640_step_50000.pt \
        -o exp-2-pass-consensus/exported/
"""
import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from export_model import (
    quantize_weights_158, pack_ternary_2bit, unpack_2bit_to_i8,
    write_bmmq, copy_tokenizer_files,
    DTYPE_F32, DTYPE_PACKED2BIT,
)


# 모든 가중치를 f32로 export — INT8 양자화 오차가 15 레이어 누적되어 발산
# BitLinear weight도 f32로 저장 (Rust에서 FP32 matmul)
def _is_bitlinear(key: str) -> bool:
    return False  # 모든 weight를 f32로 export


def export_dense_editor(ckpt_path: str, output_dir: str):
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    project_root = Path(__file__).resolve().parent.parent

    print(f"체크포인트 로드: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    step = ckpt.get("step", "unknown")
    config = ckpt["config"]
    state_dict = ckpt["model"]

    n_params = sum(v.numel() for v in state_dict.values())
    print(f"  step: {step}, params: {n_params:,}")

    # Config 저장
    config_out = dict(config)
    config_path = out_dir / "config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config_out, f, indent=2, ensure_ascii=False)
    print(f"config 저장: {config_path}")

    # BMMQ export
    entries = []
    stats = {"packed2bit": 0, "f32": 0}

    for key in sorted(state_dict.keys()):
        tensor = state_dict[key]

        if _is_bitlinear(key):
            # ternary 1.58-bit 양자화 → 2-bit 패킹
            w_quant_i8, gamma = quantize_weights_158(tensor)
            w_np = w_quant_i8.numpy()
            rows, cols = w_np.shape
            row_sums = w_np.astype(np.int32).sum(axis=1).astype(np.int32)
            packed, packed_stride = pack_ternary_2bit(w_np)

            # 검증
            unpacked = unpack_2bit_to_i8(packed, rows, cols, packed_stride)
            assert np.array_equal(w_np, unpacked), f"2-bit pack 검증 실패: {key}"

            data = packed.tobytes()
            extra = struct.pack("<f", gamma.item()) + row_sums.tobytes()

            entries.append({
                "name": key, "dtype": DTYPE_PACKED2BIT,
                "shape": (rows, cols), "data": data, "extra": extra,
            })
            stats["packed2bit"] += 1
        else:
            # f32 그대로
            t_f32 = tensor.float().contiguous()
            # conv1d weight: (channels, 1, kernel) → flatten
            data = t_f32.numpy().tobytes()

            entries.append({
                "name": key, "dtype": DTYPE_F32,
                "shape": tuple(t_f32.shape), "data": data,
            })
            stats["f32"] += 1

    model_path = out_dir / "model.bmmq"
    write_bmmq(model_path, entries)
    file_mb = model_path.stat().st_size / (1024 * 1024)

    print(f"\nBMMQ 저장: {model_path} ({file_mb:.1f} MB)")
    print(f"  packed2bit (BitLinear): {stats['packed2bit']}개")
    print(f"  f32 (기타): {stats['f32']}개")

    # 토크나이저 복사
    print("토크나이저 복사 (keyboard)...")
    copied = copy_tokenizer_files("keyboard", project_root, out_dir)
    for f in copied:
        print(f"  {f}")

    # 메타데이터
    metadata = {
        "model_type": "DenseEditor",
        "mixing_type": config.get("mixing_type", "mamba2"),
        "source_checkpoint": str(ckpt_path),
        "step": step,
        "format": "bmmq",
        "param_count": n_params,
    }
    meta_path = out_dir / "metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"\n완료! export: {out_dir}/")
    return out_dir


def main():
    parser = argparse.ArgumentParser(description="DenseEditor BMMQ export")
    parser.add_argument("checkpoint", help="체크포인트 경로")
    parser.add_argument("-o", "--output", default="exp-2-pass-consensus/exported")
    args = parser.parse_args()
    export_dense_editor(args.checkpoint, args.output)


if __name__ == "__main__":
    main()
