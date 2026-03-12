"""학습 체크포인트를 추론용 포맷으로 export

사용법:
    python export_model.py step_106000.pt -o exported/
    python export_model.py step_106000.pt -o exported/ --quantize   # BitLinear 사전 양자화
    python export_model.py step_106000.pt -o exported/ --format bmmq  # BMMQ 사전양자화 바이너리

출력 파일:
    exported/
    ├── config.json              # 모델 설정 (weight shape 기반 보정 포함)
    ├── model.safetensors        # 모델 가중치 (safetensors 포맷)
    ├── model.bmmq               # BMMQ 사전양자화 바이너리 (--format bmmq)
    ├── tokenizer_config.json    # 토크나이저 메타 (종류, 특수 토큰 ID 등)
    ├── keyboard_tokenizer.json  # 토크나이저 vocab (keyboard 토크나이저인 경우)
    ├── jamo_token_map.json      # 자모 직접 매핑 (keyboard 토크나이저인 경우)
    └── metadata.json            # 학습 정보 (step, total_chars 등)
"""

import argparse
import json
import os
import struct
import shutil
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file


# ──────────────────────────────────────────────
#  BitLinear weight 양자화 (추론용 사전 양자화)
# ──────────────────────────────────────────────

def quantize_weights_158(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """1.58-bit ternary {-1, 0, +1} 양자화 (model/bitlinear.py와 동일)"""
    gamma = w.abs().mean().clamp(min=1e-5)
    w_scaled = w / gamma
    w_clipped = w_scaled.clamp(-1.0, 1.0)
    w_quant = w_clipped.round()  # {-1, 0, +1}
    return w_quant.to(torch.int8), gamma


def is_bitlinear_weight(key: str) -> bool:
    """BitLinear으로 처리되는 weight 키인지 판별

    BitLinear 대상:
    - encoder/decoder FFN의 gate_proj, up_proj, down_proj
    NOT BitLinear:
    - Mamba2 내부 (in_proj, out_proj) — 라이브러리 자체 Linear
    - cross_attn (q/k/v/o_proj) — FP16/BF16 Linear
    - embedding, lm_head, copy_gate — 일반 Linear
    """
    return ".ffn." in key and key.endswith(".weight")


def is_linear_weight(key: str) -> bool:
    """일반 Linear (per-row i8 양자화 대상) weight 키인지 판별

    대상: Mamba in_proj/out_proj, cross_attn q/k/v/o_proj, lm_head
    """
    if is_bitlinear_weight(key):
        return False
    return key.endswith("_proj.weight") or key == "lm_head.weight"


# ──────────────────────────────────────────────
#  2-bit ternary 패킹 (BMMQ 포맷용)
# ──────────────────────────────────────────────

# 인코딩: 00=0, 01=+1, 11=-1 (2-bit, 4값/byte, MSB-first)
TERNARY_TO_2BIT = {0: 0b00, 1: 0b01, -1: 0b11}


def pack_ternary_2bit(w_i8: np.ndarray) -> tuple[np.ndarray, int]:
    """ternary {-1,0,+1} i8 배열을 2-bit 패킹

    Args:
        w_i8: shape (rows, cols)의 int8 배열

    Returns:
        (packed, packed_stride) — packed shape (rows, packed_stride)
    """
    rows, cols = w_i8.shape
    packed_stride = (cols + 3) // 4  # 4값/byte
    packed = np.zeros((rows, packed_stride), dtype=np.uint8)

    for r in range(rows):
        for c in range(cols):
            byte_idx = c // 4
            bit_pos = (3 - (c % 4)) * 2  # MSB-first: c%4=0→bit6, 1→4, 2→2, 3→0
            val = int(w_i8[r, c])
            code = TERNARY_TO_2BIT[val]
            packed[r, byte_idx] |= (code << bit_pos)

    return packed, packed_stride


def unpack_2bit_to_i8(packed: np.ndarray, rows: int, cols: int, packed_stride: int) -> np.ndarray:
    """2-bit 패킹을 i8 ternary로 언팩 (검증용)"""
    BIT2_TO_TERNARY = {0b00: 0, 0b01: 1, 0b11: -1, 0b10: 0}  # 10은 unused, 안전하게 0

    out = np.zeros((rows, cols), dtype=np.int8)
    for r in range(rows):
        for c in range(cols):
            byte_idx = c // 4
            bit_pos = (3 - (c % 4)) * 2
            code = (int(packed[r, byte_idx]) >> bit_pos) & 0x03
            out[r, c] = BIT2_TO_TERNARY[code]
    return out


def quantize_linear_i8(w: torch.Tensor) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """per-row int8 양자화 (Linear weight용)

    Returns:
        (w_i8, row_scales, row_sums) — numpy 배열
    """
    w_np = w.float().numpy()
    rows, cols = w_np.shape

    row_scales = np.zeros(rows, dtype=np.float32)
    row_sums = np.zeros(rows, dtype=np.int32)
    w_i8 = np.zeros((rows, cols), dtype=np.int8)

    for r in range(rows):
        max_abs = np.abs(w_np[r]).max()
        if max_abs < 1e-10:
            max_abs = 1e-10
        scale = max_abs / 127.0
        row_scales[r] = scale
        row_i8 = np.clip(np.round(w_np[r] / scale), -127, 127).astype(np.int8)
        w_i8[r] = row_i8
        row_sums[r] = row_i8.astype(np.int32).sum()

    return w_i8, row_scales, row_sums


# ──────────────────────────────────────────────
#  BMMQ 바이너리 포맷 writer
# ──────────────────────────────────────────────

BMMQ_MAGIC = b"BMMQ"
BMMQ_VERSION = 1

# dtype 코드
DTYPE_F32 = 0
DTYPE_I8 = 1
DTYPE_PACKED2BIT = 2


def write_bmmq(path: Path, tensor_entries: list[dict]):
    """BMMQ 바이너리 파일 작성

    tensor_entries: list of dict:
        - name: str
        - dtype: 0 (f32), 1 (i8), 2 (packed2bit)
        - shape: tuple
        - data: bytes (raw data)
        - extra: bytes (row_scales+row_sums for i8; gamma+row_sums for packed2bit)
    """
    with open(path, "wb") as f:
        # 헤더
        f.write(BMMQ_MAGIC)
        f.write(struct.pack("<H", BMMQ_VERSION))
        f.write(struct.pack("<I", len(tensor_entries)))

        for entry in tensor_entries:
            name_bytes = entry["name"].encode("utf-8")
            # name
            f.write(struct.pack("<H", len(name_bytes)))
            f.write(name_bytes)
            # dtype, ndim
            f.write(struct.pack("<B", entry["dtype"]))
            ndim = len(entry["shape"])
            f.write(struct.pack("<B", ndim))
            # shape
            for s in entry["shape"]:
                f.write(struct.pack("<I", s))
            # data_len (data + extra)
            total_len = len(entry["data"]) + len(entry.get("extra", b""))
            f.write(struct.pack("<Q", total_len))
            # data
            f.write(entry["data"])
            if entry.get("extra"):
                f.write(entry["extra"])


def export_bmmq(state_dict: dict, tied_keys: set, out_path: Path):
    """BMMQ 포맷으로 사전양자화 export

    텐서 분류:
    - BitLinear (.ffn.*.weight) → packed 2-bit ternary
    - Linear (*_proj.weight, lm_head.weight) → i8 per-row 양자화
    - 나머지 → f32
    """
    entries = []
    stats = {"packed2bit": 0, "i8": 0, "f32": 0}
    total_bytes = 0

    for key in sorted(state_dict.keys()):
        if key in tied_keys:
            continue

        tensor = state_dict[key]

        if is_bitlinear_weight(key):
            # BitLinear → ternary 양자화 → 2-bit 패킹
            w_quant_i8, gamma = quantize_weights_158(tensor)
            w_np = w_quant_i8.numpy()
            rows, cols = w_np.shape

            # row_sums 계산
            row_sums = w_np.astype(np.int32).sum(axis=1).astype(np.int32)

            # 2-bit 패킹
            packed, packed_stride = pack_ternary_2bit(w_np)

            # 검증: unpack → compare
            unpacked = unpack_2bit_to_i8(packed, rows, cols, packed_stride)
            if not np.array_equal(w_np, unpacked):
                raise ValueError(f"2-bit pack/unpack 검증 실패: {key}")

            data = packed.tobytes()
            extra = struct.pack("<f", gamma.item()) + row_sums.tobytes()

            entries.append({
                "name": key,
                "dtype": DTYPE_PACKED2BIT,
                "shape": (rows, cols),
                "data": data,
                "extra": extra,
            })
            total_bytes += len(data) + len(extra)
            stats["packed2bit"] += 1

        elif is_linear_weight(key):
            # Linear → per-row i8 양자화
            w_i8, row_scales, row_sums = quantize_linear_i8(tensor)
            rows, cols = w_i8.shape

            data = w_i8.tobytes()
            extra = row_scales.tobytes() + row_sums.tobytes()

            entries.append({
                "name": key,
                "dtype": DTYPE_I8,
                "shape": (rows, cols),
                "data": data,
                "extra": extra,
            })
            total_bytes += len(data) + len(extra)
            stats["i8"] += 1

        else:
            # f32 그대로
            t_f32 = tensor.float().contiguous()
            data = t_f32.numpy().tobytes()

            entries.append({
                "name": key,
                "dtype": DTYPE_F32,
                "shape": tuple(t_f32.shape),
                "data": data,
            })
            total_bytes += len(data)
            stats["f32"] += 1

    write_bmmq(out_path, entries)
    file_mb = out_path.stat().st_size / (1024 * 1024)

    print(f"\nBMMQ 저장: {out_path} ({file_mb:.1f} MB)")
    print(f"  packed2bit (BitLinear): {stats['packed2bit']}개")
    print(f"  i8 (Linear): {stats['i8']}개")
    print(f"  f32 (기타): {stats['f32']}개")

    return file_mb


# ──────────────────────────────────────────────
#  Config 보정: weight shape 기반
# ──────────────────────────────────────────────

def correct_config(config: dict, state_dict: dict) -> dict:
    """state_dict의 실제 weight shape를 기반으로 config 값 보정"""
    corrected = dict(config)
    corrections = []

    d_model = config["d_model"]
    n_heads = config["n_heads"]
    d_head = d_model // n_heads

    # n_kv_heads 보정: k_proj weight shape에서 역산
    k_proj_key = "decoder.layers.0.cross_attn.k_proj.weight"
    if k_proj_key in state_dict:
        k_proj_shape = state_dict[k_proj_key].shape
        real_n_kv_heads = k_proj_shape[0] // d_head
        if real_n_kv_heads != config.get("n_kv_heads"):
            corrections.append(
                f"  n_kv_heads: {config.get('n_kv_heads')} → {real_n_kv_heads} "
                f"(k_proj shape {list(k_proj_shape)} 기반)"
            )
            corrected["n_kv_heads"] = real_n_kv_heads

    # tie_embeddings 보정: encoder/decoder embedding 동일 여부
    enc_emb = state_dict.get("encoder_embedding.weight")
    dec_emb = state_dict.get("decoder_embedding.weight")
    if enc_emb is not None and dec_emb is not None:
        tied = torch.equal(enc_emb, dec_emb)
        if "tie_embeddings" not in corrected:
            corrected["tie_embeddings"] = tied
            corrections.append(f"  tie_embeddings: (missing) → {tied}")

    # tie_lm_head 보정: lm_head == decoder_embedding 여부
    lm_head = state_dict.get("lm_head.weight")
    if dec_emb is not None and lm_head is not None:
        tied = torch.equal(dec_emb, lm_head)
        if "tie_lm_head" not in corrected:
            corrected["tie_lm_head"] = tied
            corrections.append(f"  tie_lm_head: (missing) → {tied}")

    # 기본값 보충 (checkpoint config에 없을 수 있는 필드들)
    defaults = {
        "max_seq_len": 512,
        "dropout": 0.1,
        "rms_norm_eps": 1e-6,
        "pad_id": 0,
        "tie_embeddings": True,
        "tie_lm_head": True,
    }
    for k, v in defaults.items():
        if k not in corrected:
            corrected[k] = v
            corrections.append(f"  {k}: (missing) → {v} (기본값)")

    if corrections:
        print("Config 보정:")
        for c in corrections:
            print(c)

    return corrected


# ──────────────────────────────────────────────
#  토크나이저 파일 복사
# ──────────────────────────────────────────────

TOKENIZER_FILES = {
    "keyboard": [
        "keyboard_tokenizer/keyboard_tokenizer.json",
        "keyboard_tokenizer/jamo_token_map.json",
    ],
    "nfd": [
        "nfd_tokenizer/nfd_tokenizer.json",
    ],
    "char": [
        "char_tokenizer/char_vocab.json",
    ],
    "bbpe": [
        "bbpe_tokenizer/tokenizer.json",
    ],
    "mecab_bbpe": [
        "mecab_bbpe_tokenizer/tokenizer.json",
    ],
}


def copy_tokenizer_files(tokenizer_name: str, src_root: Path, dst_dir: Path):
    """토크나이저 관련 파일들을 export 디렉토리로 복사"""
    files = TOKENIZER_FILES.get(tokenizer_name, [])
    copied = []
    for rel_path in files:
        src = src_root / rel_path
        if src.exists():
            dst = dst_dir / os.path.basename(rel_path)
            shutil.copy2(src, dst)
            copied.append(os.path.basename(rel_path))
        else:
            print(f"  경고: 토크나이저 파일 없음 — {src}")
    return copied


# ──────────────────────────────────────────────
#  메인 export
# ──────────────────────────────────────────────

def export(checkpoint_path: str, output_dir: str, quantize: bool = False,
           fmt: str = "safetensors"):
    ckpt_path = Path(checkpoint_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    project_root = Path(__file__).parent

    print(f"체크포인트 로드: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    step = ckpt.get("step", "unknown")
    total_chars = ckpt.get("total_chars", 0)
    config_raw = ckpt["config"]
    args = ckpt.get("args", {})
    state_dict = ckpt["model"]
    tokenizer_name = args.get("tokenizer", "keyboard")

    print(f"  step: {step}")
    print(f"  total_chars: {total_chars:,}")
    print(f"  tokenizer: {tokenizer_name}")
    print(f"  파라미터 수: {sum(v.numel() for v in state_dict.values()):,}")
    print()

    # ── 1. Config 보정 및 저장 ──
    config = correct_config(config_raw, state_dict)
    config_path = out_dir / "config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"\nconfig 저장: {config_path}")

    # ── 2. 가중치 저장 ──
    # ── Tied weight 처리 ──
    tied_keys = set()
    enc_emb = state_dict.get("encoder_embedding.weight")
    dec_emb = state_dict.get("decoder_embedding.weight")
    lm_head = state_dict.get("lm_head.weight")

    if enc_emb is not None and dec_emb is not None and enc_emb.data_ptr() == dec_emb.data_ptr():
        tied_keys.add("decoder_embedding.weight")
        print("  tie_embeddings 감지 → decoder_embedding 생략 (encoder_embedding 참조)")
    if dec_emb is not None and lm_head is not None and dec_emb.data_ptr() == lm_head.data_ptr():
        tied_keys.add("lm_head.weight")
        print("  tie_lm_head 감지 → lm_head 생략 (encoder_embedding 참조)")

    if fmt == "bmmq":
        # ── BMMQ 사전양자화 바이너리 포맷 ──
        print(f"\nBMMQ 포맷으로 export (2-bit BitLinear + i8 Linear + f32 기타)...")
        model_path = out_dir / "model.bmmq"
        file_size_mb = export_bmmq(state_dict, tied_keys, model_path)
        model_format = "bmmq"

    elif quantize:
        print("\nBitLinear weight 사전 양자화 적용...")
        tensors = {}
        quantized_keys = []
        for key, tensor in state_dict.items():
            if key in tied_keys:
                continue
            if is_bitlinear_weight(key):
                w_quant, gamma = quantize_weights_158(tensor)
                tensors[key] = w_quant
                scale_key = key.replace(".weight", ".weight_scale")
                tensors[scale_key] = gamma.unsqueeze(0)
                quantized_keys.append(key)
            else:
                tensors[key] = tensor.to(torch.float32)
        print(f"  양자화된 레이어: {len(quantized_keys)}개")
        for k in quantized_keys[:5]:
            print(f"    {k}")
        if len(quantized_keys) > 5:
            print(f"    ... 외 {len(quantized_keys) - 5}개")

        model_path = out_dir / "model.safetensors"
        save_file(tensors, str(model_path))
        file_size_mb = model_path.stat().st_size / (1024 * 1024)
        print(f"모델 저장: {model_path} ({file_size_mb:.1f} MB)")
        model_format = "safetensors"

    else:
        tensors = {
            k: v.to(torch.float32)
            for k, v in state_dict.items()
            if k not in tied_keys
        }
        model_path = out_dir / "model.safetensors"
        save_file(tensors, str(model_path))
        file_size_mb = model_path.stat().st_size / (1024 * 1024)
        print(f"모델 저장: {model_path} ({file_size_mb:.1f} MB)")
        model_format = "safetensors"

    # ── 3. 토크나이저 파일 복사 ──
    print(f"\n토크나이저 복사 ({tokenizer_name})...")
    copied = copy_tokenizer_files(tokenizer_name, project_root, out_dir)
    for f in copied:
        print(f"  {f}")

    tok_config = {
        "type": tokenizer_name,
        "vocab_size": config["vocab_size"],
        "pad_id": config.get("pad_id", 0),
        "bos_id": config.get("bos_id", 2),
        "eos_id": 3,
        "unk_id": 1,
        "files": copied,
    }
    tok_config_path = out_dir / "tokenizer_config.json"
    with open(tok_config_path, "w", encoding="utf-8") as f:
        json.dump(tok_config, f, indent=2, ensure_ascii=False)
    print(f"토크나이저 설정: {tok_config_path}")

    # ── 4. 메타데이터 저장 ──
    metadata = {
        "source_checkpoint": str(ckpt_path),
        "step": step,
        "total_chars": total_chars,
        "tokenizer": tokenizer_name,
        "model_size": args.get("size", "unknown"),
        "quantized_bitlinear": quantize or fmt == "bmmq",
        "format": model_format,
        "param_count": sum(v.numel() for v in state_dict.values()),
    }
    meta_path = out_dir / "metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"메타데이터: {meta_path}")

    print(f"\n완료! export 디렉토리: {out_dir}/")
    print(f"  config.json            — 모델 설정")
    print(f"  {model_path.name:<24} — 가중치 ({file_size_mb:.1f} MB)")
    print(f"  tokenizer_config.json  — 토크나이저 설정")
    for f in copied:
        print(f"  {f:<24} — 토크나이저 데이터")
    print(f"  metadata.json          — 학습 정보")


def main():
    parser = argparse.ArgumentParser(
        description="학습 체크포인트를 추론용 포맷으로 export"
    )
    parser.add_argument(
        "checkpoint",
        help="학습 체크포인트 경로 (예: step_106000.pt)",
    )
    parser.add_argument(
        "-o", "--output",
        default="exported",
        help="출력 디렉토리 (기본: exported/)",
    )
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="BitLinear weight를 ternary {-1,0,+1}로 사전 양자화 (safetensors)",
    )
    parser.add_argument(
        "--format",
        choices=["safetensors", "bmmq"],
        default="safetensors",
        dest="fmt",
        help="출력 포맷: safetensors (기본) 또는 bmmq (사전양자화 바이너리)",
    )
    args = parser.parse_args()
    export(args.checkpoint, args.output, args.quantize, args.fmt)


if __name__ == "__main__":
    main()
