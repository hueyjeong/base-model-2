"""BitEditor 체크포인트를 추론용 BMMQ 포맷으로 export

사용법:
    python export_editor.py editor_128M_step50000.pt -o exported_editor/

텐서 분류:
    BitLinear (2-bit packed): bi_rwkv.*.{r,k,v,o}_proj.weight, moe_ffn.experts.*.{gate,up,down}_proj.weight
    Linear (i8): shared_attn.*_proj.weight, tag_head.weight, g_proj.weight, router.weight, LoRA weights
    F32: embedding.weight, w_base, norm weights, bias
"""
import argparse
import json
import os
import shutil
import struct
from pathlib import Path

import numpy as np
import torch

from export_model import (
    pack_ternary_2bit, unpack_2bit_to_i8,
    quantize_weights_158, quantize_linear_i8,
    write_bmmq, DTYPE_F32, DTYPE_I8, DTYPE_PACKED2BIT,
    copy_tokenizer_files,
)


def is_bitlinear_weight(key: str) -> bool:
    """BitLinear으로 처리되는 weight 키 (RWKV proj + MoE expert FFN)"""
    # RWKV projections
    if any(p in key for p in [".r_proj.", ".k_proj.", ".v_proj.", ".o_proj."]):
        if ".bi_rwkv." in key and key.endswith(".weight"):
            return True
    # MoE expert FFN
    if ".moe_ffn.experts." in key and ".weight" in key:
        if any(p in key for p in [".gate_proj.", ".up_proj.", ".down_proj."]):
            return True
    return False


def is_linear_weight(key: str) -> bool:
    """일반 Linear (per-row i8 양자화 대상)"""
    if is_bitlinear_weight(key):
        return False
    if not key.endswith(".weight"):
        return False
    # Shared attention projections
    if ".shared_attn." in key and "_proj." in key:
        return True
    # LoRA weights
    if ".lora_" in key:
        return True
    # Tag head
    if key == "tag_head.weight":
        return True
    # RWKV gate projection
    if ".g_proj." in key:
        return True
    # MoE router
    if ".router." in key:
        return True
    # w_lora_down/up
    if ".w_lora_" in key:
        return True
    return False


def export_editor_bmmq(state_dict: dict, out_path: Path):
    """BitEditor 모델을 BMMQ 포맷으로 export"""
    entries = []
    stats = {"packed2bit": 0, "i8": 0, "f32": 0}

    for key in sorted(state_dict.keys()):
        tensor = state_dict[key]

        if is_bitlinear_weight(key):
            # BitLinear → ternary 양자화 → 2-bit 패킹
            w_quant_i8, gamma = quantize_weights_158(tensor)
            w_np = w_quant_i8.numpy()
            rows, cols = w_np.shape
            row_sums = w_np.astype(np.int32).sum(axis=1).astype(np.int32)
            packed, packed_stride = pack_ternary_2bit(w_np)

            # 검증
            unpacked = unpack_2bit_to_i8(packed, rows, cols, packed_stride)
            assert np.array_equal(w_np, unpacked), f"2-bit pack/unpack 검증 실패: {key}"

            data = packed.tobytes()
            extra = struct.pack("<f", gamma.item()) + row_sums.tobytes()

            entries.append({
                "name": key, "dtype": DTYPE_PACKED2BIT,
                "shape": (rows, cols), "data": data, "extra": extra,
            })
            stats["packed2bit"] += 1

        elif is_linear_weight(key):
            # Linear → per-row i8 양자화
            w_i8, row_scales, row_sums = quantize_linear_i8(tensor)
            rows, cols = w_i8.shape
            data = w_i8.tobytes()
            extra = row_scales.tobytes() + row_sums.tobytes()

            entries.append({
                "name": key, "dtype": DTYPE_I8,
                "shape": (rows, cols), "data": data, "extra": extra,
            })
            stats["i8"] += 1

        else:
            # f32 그대로
            t_f32 = tensor.float().contiguous()
            data = t_f32.numpy().tobytes()

            entries.append({
                "name": key, "dtype": DTYPE_F32,
                "shape": tuple(t_f32.shape), "data": data,
            })
            stats["f32"] += 1

    write_bmmq(out_path, entries)
    file_mb = out_path.stat().st_size / (1024 * 1024)

    print(f"\nBMMQ 저장: {out_path} ({file_mb:.1f} MB)")
    print(f"  packed2bit (BitLinear): {stats['packed2bit']}개")
    print(f"  i8 (Linear): {stats['i8']}개")
    print(f"  f32 (기타): {stats['f32']}개")
    return file_mb


def export(checkpoint_path: str, output_dir: str):
    ckpt_path = Path(checkpoint_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    project_root = Path(__file__).parent

    print(f"체크포인트 로드: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    step = ckpt.get("step", "unknown")
    config = ckpt["config"]
    state_dict = ckpt["model"]

    print(f"  step: {step}")
    print(f"  파라미터 수: {sum(v.numel() for v in state_dict.values()):,}")

    # Config 저장
    config_path = out_dir / "config.json"
    # attn_insertion_points: tuple → list for JSON
    config_out = dict(config)
    if "attn_insertion_points" in config_out:
        config_out["attn_insertion_points"] = list(config_out["attn_insertion_points"])
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config_out, f, indent=2, ensure_ascii=False)
    print(f"config 저장: {config_path}")

    # BMMQ export
    model_path = out_dir / "model.bmmq"
    file_mb = export_editor_bmmq(state_dict, model_path)

    # 토크나이저 복사
    print(f"\n토크나이저 복사 (keyboard)...")
    copied = copy_tokenizer_files("keyboard", project_root, out_dir)
    for f in copied:
        print(f"  {f}")

    # 메타데이터
    metadata = {
        "model_type": "BitEditor",
        "source_checkpoint": str(ckpt_path),
        "step": step,
        "format": "bmmq",
        "param_count": sum(v.numel() for v in state_dict.values()),
    }
    meta_path = out_dir / "metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"\n완료! export 디렉토리: {out_dir}/")


def main():
    parser = argparse.ArgumentParser(description="BitEditor 체크포인트 export")
    parser.add_argument("checkpoint", help="학습 체크포인트 경로")
    parser.add_argument("-o", "--output", default="exported_editor")
    args = parser.parse_args()
    export(args.checkpoint, args.output)


if __name__ == "__main__":
    main()
