"""학습 체크포인트를 추론용 포맷으로 export

사용법:
    python export_model.py step_106000.pt -o exported/
    python export_model.py step_106000.pt -o exported/ --quantize   # BitLinear 사전 양자화

출력 파일:
    exported/
    ├── config.json              # 모델 설정 (weight shape 기반 보정 포함)
    ├── model.safetensors        # 모델 가중치 (safetensors 포맷)
    ├── tokenizer_config.json    # 토크나이저 메타 (종류, 특수 토큰 ID 등)
    ├── keyboard_tokenizer.json  # 토크나이저 vocab (keyboard 토크나이저인 경우)
    ├── jamo_token_map.json      # 자모 직접 매핑 (keyboard 토크나이저인 경우)
    └── metadata.json            # 학습 정보 (step, total_chars 등)
"""

import argparse
import json
import os
import shutil
from pathlib import Path

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

def export(checkpoint_path: str, output_dir: str, quantize: bool = False):
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
    # encoder_embedding, decoder_embedding, lm_head가 메모리 공유하는 경우
    # 하나만 저장하고 config에 tying 정보 기록
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

    if quantize:
        print("\nBitLinear weight 사전 양자화 적용...")
        tensors = {}
        quantized_keys = []
        for key, tensor in state_dict.items():
            if key in tied_keys:
                continue
            if is_bitlinear_weight(key):
                w_quant, gamma = quantize_weights_158(tensor)
                # ternary weight → int8, scale → float32
                tensors[key] = w_quant
                scale_key = key.replace(".weight", ".weight_scale")
                tensors[scale_key] = gamma.unsqueeze(0)  # (1,) scalar
                quantized_keys.append(key)
            else:
                tensors[key] = tensor.to(torch.float32)
        print(f"  양자화된 레이어: {len(quantized_keys)}개")
        for k in quantized_keys[:5]:
            print(f"    {k}")
        if len(quantized_keys) > 5:
            print(f"    ... 외 {len(quantized_keys) - 5}개")
    else:
        # FP32로 통일하여 저장 (tied weight 제외)
        tensors = {
            k: v.to(torch.float32)
            for k, v in state_dict.items()
            if k not in tied_keys
        }

    model_path = out_dir / "model.safetensors"
    save_file(tensors, str(model_path))
    file_size_mb = model_path.stat().st_size / (1024 * 1024)
    print(f"모델 저장: {model_path} ({file_size_mb:.1f} MB)")

    # ── 3. 토크나이저 파일 복사 ──
    print(f"\n토크나이저 복사 ({tokenizer_name})...")
    copied = copy_tokenizer_files(tokenizer_name, project_root, out_dir)
    for f in copied:
        print(f"  {f}")

    # 토크나이저 메타 저장
    # 특수 토큰 ID를 기록 (Rust에서 하드코딩 불필요하도록)
    tok_config = {
        "type": tokenizer_name,
        "vocab_size": config["vocab_size"],
        "pad_id": config.get("pad_id", 0),
        "bos_id": config.get("bos_id", 2),
        "eos_id": 3,  # 모든 토크나이저 공통
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
        "quantized_bitlinear": quantize,
        "format": "safetensors",
        "param_count": sum(v.numel() for v in state_dict.values()),
    }
    meta_path = out_dir / "metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"메타데이터: {meta_path}")

    print(f"\n완료! export 디렉토리: {out_dir}/")
    print(f"  config.json            — 모델 설정")
    print(f"  model.safetensors      — 가중치 ({file_size_mb:.1f} MB)")
    print(f"  tokenizer_config.json  — 토크나이저 설정")
    for f in copied:
        print(f"  {f:<24} — 토크나이저 데이터")
    print(f"  metadata.json          — 학습 정보")


def main():
    parser = argparse.ArgumentParser(
        description="학습 체크포인트를 추론용 safetensors 포맷으로 export"
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
        help="BitLinear weight를 ternary {-1,0,+1}로 사전 양자화",
    )
    args = parser.parse_args()
    export(args.checkpoint, args.output, args.quantize)


if __name__ == "__main__":
    main()
