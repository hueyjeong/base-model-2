"""합의 기반 2단계 반복 교정 실험

50k 체크포인트 DenseEditor에 대해 MC Dropout 기반 stochastic inference로
V1(single-pass), V2(2-pass), V3(consensus-2), V4(2-stage consensus) 비교.

최적화:
  - 패킹: Mamba-2 BOS state reset 활용, 여러 문장을 하나의 시퀀스로 묶어 처리
  - INT8 CUDA BitLinear: 학습과 동일한 fused INT8 matmul 가속
  - BF16 AMP: mixed precision 추론
  - Paired prediction: V3/V4에서 2회 stochastic run을 1회 forward pass로
  - Length-sorted packing: 비슷한 길이끼리 패킹하여 패딩 최소화

사용법:
    python exp-2-pass-consensus/run_experiment.py \
        --ckpt exp-2-pass-consensus/dense_mamba2_d640_step_50000.pt \
        --corpus corpus/val_50k.jsonl \
        --n_samples 5000 --n_repeats 5
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

# INT8 CUDA BitLinear 환경 변수 (fused 커널)
os.environ.setdefault("BITLINEAR_CUDA_FUSED_ACT", "1")
os.environ.setdefault("BITLINEAR_CUDA_FUSED_WEIGHT", "1")

# 프로젝트 루트를 path에 추가
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from model.dense_editor import DenseEditor
from model.dense_editor_config import DenseEditorConfig
from model.edit_tags import apply_edit_tags, TAG_KEEP
from keyboard_tokenizer.keyboard_wrapper import KeyboardTokenizer
from training.noising import DenoisingNoiser, NoiseConfig

# C++ 가속 Levenshtein (Python 대비 50-100x)
try:
    from training.editor_dataset import compute_edit_tags
except ImportError:
    from model.edit_tags import compute_edit_tags


# ── 추론 최적화 ────────────────────────────────────────────────────────

def prepare_model_for_inference(model: DenseEditor):
    """추론 전용 최적화 — 가중치 불변, 학습 호환성 유지

    1. BitLinearCuda: 가중치 사전 양자화 (매 forward 재양자화 제거, 31모듈 × 6 element-wise 삭제)
    2. BiMamba2: bwd_reset 캐싱 (매 레이어 flip+clone 제거, 14회 → 0회)
    """
    import types

    n_bitlinear = 0
    n_bimamba = 0

    # --- BitLinearCuda 가중치 사전 양자화 ---
    try:
        from model.cuda_bitlinear import (
            BitLinearCuda, _quantize_weights, _quantize_activations,
            _pad_int8_for_int_mm,
        )

        for module in model.modules():
            if not isinstance(module, BitLinearCuda):
                continue

            # 가중치를 한 번만 양자화하여 버퍼로 저장
            with torch.no_grad():
                w_q, w_scale = _quantize_weights(module.weight.float())
                w_int8 = w_q.to(torch.int8).contiguous()
                w_scale_1 = w_scale.reshape(1).contiguous().float()
                # 패딩된 버전도 사전 계산 (차원이 8배수가 아닌 경우)
                N, K = w_int8.shape
                from model.cuda_bitlinear import _ceil_to_multiple_of_8
                K8 = _ceil_to_multiple_of_8(K)
                N8 = _ceil_to_multiple_of_8(N)
                if N8 != N or K8 != K:
                    w_int8_pad = torch.zeros((N8, K8), device=w_int8.device, dtype=torch.int8)
                    w_int8_pad[:N, :K] = w_int8
                else:
                    w_int8_pad = w_int8
                module._inf_w_int8_pad = w_int8_pad
                module._inf_w_scale = w_scale_1
                module._inf_N = N
                module._inf_K8 = K8

            # forward를 사전 양자화 버전으로 교체
            def _fast_forward(self, x):
                x_norm = self.norm(x)
                if torch.is_autocast_enabled('cuda'):
                    out_dtype = torch.get_autocast_dtype('cuda')
                else:
                    out_dtype = x_norm.dtype

                x_q, x_scale = _quantize_activations(x_norm.float())
                batch_shape = x_norm.shape[:-1]
                M = x_q.reshape(-1, x_q.shape[-1]).shape[0]
                x_int8 = x_q.reshape(M, -1).to(torch.int8).contiguous()
                x_scale_2d = x_scale.reshape(M, 1).contiguous().float()

                # 입력만 패딩 (가중치는 사전 패딩 완료)
                from model.cuda_bitlinear import _ceil_to_multiple_of_8
                M8 = _ceil_to_multiple_of_8(M)
                if M8 != M or x_int8.shape[1] != self._inf_K8:
                    x_pad = torch.zeros((M8, self._inf_K8),
                                        device=x_int8.device, dtype=torch.int8)
                    x_pad[:M, :x_int8.shape[1]] = x_int8
                else:
                    x_pad = x_int8

                out_i32 = torch._int_mm(x_pad, self._inf_w_int8_pad.t().contiguous())
                out_2d = out_i32[:M, :self._inf_N].float() * (x_scale_2d * self._inf_w_scale)

                if self.bias is not None:
                    out_2d = out_2d + self.bias.float()

                return out_2d.to(out_dtype).reshape(*batch_shape, -1)

            module.forward = types.MethodType(_fast_forward, module)
            n_bitlinear += 1

    except ImportError:
        pass

    # --- BiMamba2 bwd_reset 캐싱 ---
    from model.mixing.bi_mamba2 import BiMamba2Mixing

    for module in model.modules():
        if not isinstance(module, BiMamba2Mixing):
            continue

        module._cached_bwd_reset = None
        module._cached_reset_ptr = -1

        def _cached_forward(self, x, pad_mask=None, reset_mask=None):
            fwd_out = self.fwd(x, reset_mask=reset_mask)
            if reset_mask is not None:
                ptr = reset_mask.data_ptr()
                if ptr != self._cached_reset_ptr:
                    self._cached_bwd_reset = reset_mask.flip(1).clone()
                    self._cached_bwd_reset[:, 0] = True
                    self._cached_reset_ptr = ptr
                bwd_reset = self._cached_bwd_reset
            else:
                bwd_reset = None
            bwd_out = self.bwd(x.flip(1), reset_mask=bwd_reset).flip(1)
            out = fwd_out + bwd_out
            if pad_mask is not None:
                out = out * pad_mask.unsqueeze(-1).to(out.dtype)
            return out

        module.forward = types.MethodType(_cached_forward, module)
        n_bimamba += 1

    print(f"  추론 최적화: BitLinearCuda {n_bitlinear}개 가중치 사전양자화, "
          f"BiMamba2 {n_bimamba}개 bwd_reset 캐싱")


# ── 모델 로딩 ──────────────────────────────────────────────────────────

def load_model(
    ckpt_path: str, device: str, use_int8: bool = True,
) -> tuple[DenseEditor, DenseEditorConfig]:
    """체크포인트에서 DenseEditor 모델 로드 (MC Dropout 활성 상태)"""
    print(f"체크포인트 로드: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    config = DenseEditorConfig(**ckpt["config"])
    model = DenseEditor(config)
    model.load_state_dict(ckpt["model"])

    # INT8 CUDA BitLinear (옵션 — 대형 GPU에서만 이득, 소형 GPU는 BF16이 더 빠름)
    if use_int8 and device == "cuda":
        try:
            from model.cuda_bitlinear import replace_bitlinear_with_cuda
            model = replace_bitlinear_with_cuda(model)
            print("  INT8 CUDA BitLinear 활성화")
        except Exception as e:
            print(f"  INT8 CUDA 불가: {e}")
    elif not use_int8:
        print("  BF16 matmul (INT8 비활성화 — 소형 GPU에서 더 빠름)")

    model.to(device)

    # MC Dropout: train 모드 유지 → dropout(0.1) 활성
    model.train()

    # 추론 전용 최적화 (가중치 사전 양자화 + bwd_reset 캐싱)
    if device == "cuda":
        prepare_model_for_inference(model)

    step = ckpt.get("step", "unknown")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  step: {step}, params: {n_params:,}, device: {device}")
    print(f"  MC Dropout 활성 (dropout={config.dropout})")
    return model, config


# ── 평가 데이터 준비 ───────────────────────────────────────────────────

def prepare_eval_data(
    corpus_path: str,
    tokenizer: KeyboardTokenizer,
    n_samples: int,
    noise_seed: int,
) -> list[tuple[list[int], list[int]]]:
    """val JSONL에서 (noised_ids, clean_ids) 쌍 생성. BOS/EOS 포함."""
    noise_cfg = NoiseConfig(
        korean_error_prob=0.5,
        korean_error_count=3,
        token_mask_ratio=0.0,
        token_delete_ratio=0.0,
        text_infill_ratio=0.0,
    )
    noiser = DenoisingNoiser(tokenizer, noise_cfg, seed=noise_seed)

    pairs = []
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            if len(pairs) >= n_samples:
                break
            line = line.strip()
            if not line:
                continue
            try:
                text = json.loads(line).get("text", "")
            except json.JSONDecodeError:
                text = line

            if len(text) < 10:
                continue

            noised_ids, clean_ids, _ = noiser(text)
            if len(noised_ids) > 1024 or len(clean_ids) > 1024:
                continue

            pairs.append((noised_ids, clean_ids))

    print(f"평가 데이터: {len(pairs)}개 문장 ({corpus_path})")
    return pairs


# ── 패킹 기반 추론 ────────────────────────────────────────────────────

def _pack_sentences(
    all_ids: list[list[int]],
    max_seq_len: int = 2048,
) -> tuple[list[list[int]], list[list[tuple[int, int, int]]]]:
    """여러 문장을 max_seq_len까지 패킹. 길이 정렬로 패딩 최소화.

    Returns:
        packed_seqs: 패킹된 토큰 시퀀스 리스트
        boundaries: [(원본_인덱스, 시작위치, 끝위치), ...] per packed seq
    """
    # 길이 순 정렬 (비슷한 길이끼리 패킹 → 배치 내 패딩 최소화)
    indexed = sorted(enumerate(all_ids), key=lambda x: len(x[1]))

    packed_seqs: list[list[int]] = []
    boundaries: list[list[tuple[int, int, int]]] = []

    current_seq: list[int] = []
    current_bounds: list[tuple[int, int, int]] = []

    for orig_idx, ids in indexed:
        slen = len(ids)

        if slen > max_seq_len:
            if current_seq:
                packed_seqs.append(current_seq)
                boundaries.append(current_bounds)
                current_seq = []
                current_bounds = []
            packed_seqs.append(ids[:max_seq_len])
            boundaries.append([(orig_idx, 0, min(slen, max_seq_len))])
            continue

        if len(current_seq) + slen > max_seq_len:
            packed_seqs.append(current_seq)
            boundaries.append(current_bounds)
            current_seq = []
            current_bounds = []

        start = len(current_seq)
        current_seq.extend(ids)
        current_bounds.append((orig_idx, start, start + slen))

    if current_seq:
        packed_seqs.append(current_seq)
        boundaries.append(current_bounds)

    return packed_seqs, boundaries


def _pack_to_batched_tensors(
    all_ids: list[list[int]],
    batch_size: int,
    device: str,
    max_seq_len: int = 2048,
) -> tuple[list[tuple[torch.Tensor, torch.Tensor, list]], int]:
    """패킹 → 배치별 pre-padded GPU 텐서로 사전 변환 (1회만 실행)

    핫 루프에서 torch.tensor(Python list) 호출 제거 — 최대 병목 해소.

    Returns:
        batches: [(input_ids, pad_mask, bounds_list), ...] — GPU 텐서
        n_sents: 총 문장 수
    """
    packed_seqs, boundaries = _pack_sentences(all_ids, max_seq_len)
    batches = []

    for i in range(0, len(packed_seqs), batch_size):
        batch = packed_seqs[i:i + batch_size]
        bounds = boundaries[i:i + batch_size]
        max_len = max(len(s) for s in batch)

        # numpy → torch.from_numpy (zero-copy) → GPU
        # Python list → torch.tensor 대비 10-50x 빠름
        arr = np.zeros((len(batch), max_len), dtype=np.int64)
        for j, seq in enumerate(batch):
            arr[j, :len(seq)] = seq

        input_ids = torch.from_numpy(arr).to(device, non_blocking=True)
        pad_mask = input_ids != 0
        batches.append((input_ids, pad_mask, bounds))

    return batches, len(all_ids)


def _forward_batch(
    model: DenseEditor, input_ids: torch.Tensor, pad_mask: torch.Tensor,
    use_amp: bool, device: str,
) -> torch.Tensor:
    """단일 배치 forward pass → argmax tag 텐서 (GPU)"""
    if use_amp and device == "cuda":
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = model(input_ids, pad_mask)
    else:
        with torch.no_grad():
            logits = model(input_ids, pad_mask)
    return logits.argmax(dim=-1)


def _unpack_tags(
    pred_cpu: torch.Tensor,
    bounds_list: list[list[tuple[int, int, int]]],
    out: list,
):
    """CPU 텐서에서 각 문장의 tags를 추출하여 out에 저장"""
    for seq_idx, bounds in enumerate(bounds_list):
        for sent_idx, start, end in bounds:
            out[sent_idx] = pred_cpu[seq_idx, start:end].tolist()


def _predict_from_batches(
    model: DenseEditor,
    batches: list[tuple[torch.Tensor, torch.Tensor, list]],
    n_sents: int,
    device: str,
    use_amp: bool = True,
) -> list[list[int]]:
    """사전 변환된 배치 텐서로 추론 — 핫 루프에 텐서 생성 없음"""
    all_tags: list[list[int] | None] = [None] * n_sents

    # GPU에서 모든 배치 실행, 배치 단위 CPU 전송
    for input_ids, pad_mask, bounds_list in batches:
        pred = _forward_batch(model, input_ids, pad_mask, use_amp, device)
        pred_cpu = pred.cpu()
        _unpack_tags(pred_cpu, bounds_list, all_tags)

    return all_tags


def _predict_paired_from_batches(
    model: DenseEditor,
    batches: list[tuple[torch.Tensor, torch.Tensor, list]],
    n_sents: int,
    device: str,
    use_amp: bool = True,
) -> tuple[list[list[int]], list[list[int]]]:
    """사전 변환된 배치에서 2회 stochastic 추론을 1회 forward로

    input_ids.repeat(2,1)로 GPU 내 복제 → dropout이 독립 적용.
    """
    all_tags_a: list[list[int] | None] = [None] * n_sents
    all_tags_b: list[list[int] | None] = [None] * n_sents

    for input_ids, pad_mask, bounds_list in batches:
        B = input_ids.shape[0]
        doubled_ids = input_ids.repeat(2, 1)
        doubled_mask = pad_mask.repeat(2, 1)

        pred = _forward_batch(model, doubled_ids, doubled_mask, use_amp, device)
        pred_cpu = pred.cpu()

        for seq_idx, bounds in enumerate(bounds_list):
            for sent_idx, start, end in bounds:
                all_tags_a[sent_idx] = pred_cpu[seq_idx, start:end].tolist()
                all_tags_b[sent_idx] = pred_cpu[seq_idx + B, start:end].tolist()

    return all_tags_a, all_tags_b


def _predict_dynamic(
    model: DenseEditor,
    all_ids: list[list[int]],
    batch_size: int,
    device: str,
    use_amp: bool = True,
    max_seq_len: int = 2048,
) -> list[list[int]]:
    """동적 입력용 추론 — V2/V4 중간 결과 처리 (numpy 기반 빠른 텐서 생성)"""
    packed_seqs, boundaries = _pack_sentences(all_ids, max_seq_len)
    n_sents = len(all_ids)
    all_tags: list[list[int] | None] = [None] * n_sents

    for i in range(0, len(packed_seqs), batch_size):
        batch = packed_seqs[i:i + batch_size]
        bounds = boundaries[i:i + batch_size]
        max_len = max(len(s) for s in batch)

        arr = np.zeros((len(batch), max_len), dtype=np.int64)
        for j, seq in enumerate(batch):
            arr[j, :len(seq)] = seq

        input_ids = torch.from_numpy(arr).to(device, non_blocking=True)
        pad_mask = input_ids != 0

        pred = _forward_batch(model, input_ids, pad_mask, use_amp, device)
        pred_cpu = pred.cpu()
        _unpack_tags(pred_cpu, bounds, all_tags)

    return all_tags


def _predict_paired_dynamic(
    model: DenseEditor,
    all_ids: list[list[int]],
    batch_size: int,
    device: str,
    use_amp: bool = True,
    max_seq_len: int = 2048,
) -> tuple[list[list[int]], list[list[int]]]:
    """동적 입력용 paired 추론 — V4 stage 2"""
    packed_seqs, boundaries = _pack_sentences(all_ids, max_seq_len)
    n_sents = len(all_ids)
    all_tags_a: list[list[int] | None] = [None] * n_sents
    all_tags_b: list[list[int] | None] = [None] * n_sents

    for i in range(0, len(packed_seqs), batch_size):
        batch = packed_seqs[i:i + batch_size]
        bounds = boundaries[i:i + batch_size]
        max_len = max(len(s) for s in batch)

        arr = np.zeros((len(batch), max_len), dtype=np.int64)
        for j, seq in enumerate(batch):
            arr[j, :len(seq)] = seq

        input_ids = torch.from_numpy(arr).to(device, non_blocking=True)
        pad_mask = input_ids != 0
        B = input_ids.shape[0]
        doubled_ids = input_ids.repeat(2, 1)
        doubled_mask = pad_mask.repeat(2, 1)

        pred = _forward_batch(model, doubled_ids, doubled_mask, use_amp, device)
        pred_cpu = pred.cpu()

        for seq_idx, blist in enumerate(bounds):
            for sent_idx, start, end in blist:
                all_tags_a[sent_idx] = pred_cpu[seq_idx, start:end].tolist()
                all_tags_b[sent_idx] = pred_cpu[seq_idx + B, start:end].tolist()

    return all_tags_a, all_tags_b


# ── Consensus ──────────────────────────────────────────────────────────

def consensus_tags(tags_a: list[int], tags_b: list[int]) -> list[int]:
    """두 tag 시퀀스의 합의: 동일한 tag만 유지, 불일치 → TAG_KEEP"""
    assert len(tags_a) == len(tags_b), \
        f"consensus 길이 불일치: {len(tags_a)} vs {len(tags_b)}"
    return [a if a == b else TAG_KEEP for a, b in zip(tags_a, tags_b)]


# ── Variation 실행기 ───────────────────────────────────────────────────

def _apply_tags_all(
    all_ids: list[list[int]],
    all_tags: list[list[int]],
    vocab_size: int,
) -> list[list[int]]:
    return [
        apply_edit_tags(ids, tags, vocab_size)
        for ids, tags in zip(all_ids, all_tags)
    ]


def run_variation_batch(
    variation: str,
    model: DenseEditor,
    eval_data: list[tuple[list[int], list[int]]],
    noised_batches: list[tuple[torch.Tensor, torch.Tensor, list]],
    n_sents: int,
    vocab_size: int,
    device: str,
    batch_size: int,
) -> list[list[int]]:
    """noised_batches: 사전 변환된 GPU 텐서 배치 (첫 pass에 재사용)"""
    all_noised = [pair[0] for pair in eval_data]

    if variation == "v1":
        tags = _predict_from_batches(model, noised_batches, n_sents, device)
        return _apply_tags_all(all_noised, tags, vocab_size)

    elif variation == "v2":
        tags_1 = _predict_from_batches(model, noised_batches, n_sents, device)
        intermediates = _apply_tags_all(all_noised, tags_1, vocab_size)
        tags_2 = _predict_dynamic(model, intermediates, batch_size, device)
        return _apply_tags_all(intermediates, tags_2, vocab_size)

    elif variation == "v3":
        tags_a, tags_b = _predict_paired_from_batches(
            model, noised_batches, n_sents, device)
        cons = [consensus_tags(a, b) for a, b in zip(tags_a, tags_b)]
        return _apply_tags_all(all_noised, cons, vocab_size)

    elif variation == "v4":
        # Stage 1: paired consensus on x (사전 변환 텐서)
        tags_a, tags_b = _predict_paired_from_batches(
            model, noised_batches, n_sents, device)
        cons_1 = [consensus_tags(a, b) for a, b in zip(tags_a, tags_b)]
        y_list = _apply_tags_all(all_noised, cons_1, vocab_size)
        # Stage 2: paired consensus on y (동적 텐서)
        tags_a2, tags_b2 = _predict_paired_dynamic(
            model, y_list, batch_size, device)
        cons_2 = [consensus_tags(a, b) for a, b in zip(tags_a2, tags_b2)]
        return _apply_tags_all(y_list, cons_2, vocab_size)

    raise ValueError(f"알 수 없는 variation: {variation}")


# ── 평가 메트릭 ────────────────────────────────────────────────────────

def precompute_gold_tags(
    eval_data: list[tuple[list[int], list[int]]],
    vocab_size: int,
) -> list[list[int]]:
    """gold_tags를 사전 계산 (모든 repeat에서 동일하므로 한 번만)"""
    return [
        compute_edit_tags(noised_ids, clean_ids, vocab_size)
        for noised_ids, clean_ids in eval_data
    ]


def evaluate_all(
    eval_data: list[tuple[list[int], list[int]]],
    finals: list[list[int]],
    vocab_size: int,
    gold_tags_cache: list[list[int]] | None = None,
) -> dict:
    tp_exact = 0
    fp = 0
    fn = 0
    tp_detect = 0

    total_pred_edits = 0
    total_gold_edits = 0
    total_sentences = len(eval_data)
    changed_sentences = 0

    for i, ((noised_ids, clean_ids), final_ids) in enumerate(zip(eval_data, finals)):
        gold_tags = gold_tags_cache[i] if gold_tags_cache else \
            compute_edit_tags(noised_ids, clean_ids, vocab_size)
        pred_tags = compute_edit_tags(noised_ids, final_ids, vocab_size)

        n_pred = sum(1 for t in pred_tags if t != TAG_KEEP)
        n_gold = sum(1 for t in gold_tags if t != TAG_KEEP)
        total_pred_edits += n_pred
        total_gold_edits += n_gold
        if n_pred > 0:
            changed_sentences += 1

        for g, p in zip(gold_tags, pred_tags):
            g_edit = (g != TAG_KEEP)
            p_edit = (p != TAG_KEEP)

            if g_edit and p_edit:
                tp_detect += 1
                if g == p:
                    tp_exact += 1
                else:
                    fp += 1
                    fn += 1
            elif p_edit and not g_edit:
                fp += 1
            elif g_edit and not p_edit:
                fn += 1

    p_exact = tp_exact / max(tp_exact + fp, 1)
    r_exact = tp_exact / max(tp_exact + fn, 1)
    f05_exact = _f_beta(p_exact, r_exact, beta=0.5)
    f1_exact = _f_beta(p_exact, r_exact, beta=1.0)

    p_detect = tp_detect / max(tp_detect + fp, 1)
    r_detect = tp_detect / max(tp_detect + fn, 1)
    f05_detect = _f_beta(p_detect, r_detect, beta=0.5)
    f1_detect = _f_beta(p_detect, r_detect, beta=1.0)

    return {
        "precision_exact": p_exact,
        "recall_exact": r_exact,
        "f05_exact": f05_exact,
        "f1_exact": f1_exact,
        "tp_exact": tp_exact,
        "precision_detect": p_detect,
        "recall_detect": r_detect,
        "f05_detect": f05_detect,
        "f1_detect": f1_detect,
        "tp_detect": tp_detect,
        "fp": fp,
        "fn": fn,
        "total_pred_edits": total_pred_edits,
        "total_gold_edits": total_gold_edits,
        "avg_edits_per_sent": total_pred_edits / max(total_sentences, 1),
        "changed_sent_ratio": changed_sentences / max(total_sentences, 1),
        "n_sentences": total_sentences,
    }


def _f_beta(precision: float, recall: float, beta: float) -> float:
    b2 = beta * beta
    denom = b2 * precision + recall
    if denom == 0:
        return 0.0
    return (1 + b2) * precision * recall / denom


# ── 결과 포맷팅 ────────────────────────────────────────────────────────

VARIATION_NAMES = {
    "v1": "V1 single-pass",
    "v2": "V2 2-pass",
    "v3": "V3 consensus-2",
    "v4": "V4 2-stage consensus",
}


def print_summary(all_results: dict[str, list[dict]]):
    import statistics

    header = (
        f"{'Variation':<25} | {'P_exact':>13} | {'R_exact':>13} | "
        f"{'F0.5_exact':>13} | {'F1_exact':>13} | {'Avg Edits':>13} | "
        f"{'Changed%':>10}"
    )
    sep = "-" * len(header)

    print(f"\n{sep}")
    print("Summary (mean ± std)")
    print(sep)
    print(header)
    print(sep)

    for var_key in sorted(all_results.keys()):
        runs = all_results[var_key]
        name = VARIATION_NAMES.get(var_key, var_key)
        print(_format_summary_row(name, runs))

    print(sep)
    _print_interpretation(all_results)


def _format_summary_row(name: str, runs: list[dict]) -> str:
    import statistics

    def _mean_std(key):
        vals = [r[key] for r in runs]
        m = statistics.mean(vals)
        s = statistics.stdev(vals) if len(vals) > 1 else 0.0
        return f"{m:.4f}±{s:.4f}"

    return (
        f"{name:<25} | {_mean_std('precision_exact'):>13} | "
        f"{_mean_std('recall_exact'):>13} | {_mean_std('f05_exact'):>13} | "
        f"{_mean_std('f1_exact'):>13} | {_mean_std('avg_edits_per_sent'):>13} | "
        f"{_mean_std('changed_sent_ratio'):>10}"
    )


def _print_interpretation(all_results: dict[str, list[dict]]):
    import statistics

    print("\n[ 해석 요약 ]")

    best_var = None
    best_f05 = -1
    for var_key, runs in all_results.items():
        mean_f05 = statistics.mean([r["f05_exact"] for r in runs])
        if mean_f05 > best_f05:
            best_f05 = mean_f05
            best_var = var_key

    print(f"  F0.5 기준 최고: {VARIATION_NAMES.get(best_var, best_var)} "
          f"(F0.5={best_f05:.4f})")

    if "v1" in all_results:
        v1_p = statistics.mean([r["precision_exact"] for r in all_results["v1"]])
        v1_r = statistics.mean([r["recall_exact"] for r in all_results["v1"]])
        v1_f05 = statistics.mean([r["f05_exact"] for r in all_results["v1"]])

        for var_key in sorted(all_results.keys()):
            if var_key == "v1":
                continue
            runs = all_results[var_key]
            name = VARIATION_NAMES.get(var_key, var_key)
            p = statistics.mean([r["precision_exact"] for r in runs])
            r = statistics.mean([r["recall_exact"] for r in runs])
            f05 = statistics.mean([r["f05_exact"] for r in runs])

            dp = p - v1_p
            dr = r - v1_r
            df = f05 - v1_f05
            print(f"  {name} vs V1: P {dp:+.4f}, R {dr:+.4f}, F0.5 {df:+.4f}")

    for var_key, runs in all_results.items():
        if len(runs) < 2:
            continue
        f05_std = statistics.stdev([r["f05_exact"] for r in runs])
        name = VARIATION_NAMES.get(var_key, var_key)
        if f05_std > 0.02:
            print(f"  주의: {name}의 F0.5 std={f05_std:.4f} — 분산이 큼")


# ── 메인 실험 루프 ─────────────────────────────────────────────────────

def run_experiment(args):
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA 불가 → CPU 사용")
        device = "cpu"

    # cuDNN 벤치마크 모드 (최적 커널 선택)
    torch.backends.cudnn.benchmark = True

    model, config = load_model(args.ckpt, device, use_int8=args.int8)
    vocab_size = config.vocab_size

    tokenizer = KeyboardTokenizer()

    eval_data = prepare_eval_data(
        args.corpus, tokenizer, args.n_samples, args.seed,
    )

    # 패킹 → GPU 텐서 사전 변환 (핫 루프에서 텐서 생성 제거)
    all_noised = [pair[0] for pair in eval_data]
    total_tokens = sum(len(ids) for ids in all_noised)

    print("패킹 + GPU 텐서 사전 변환...")
    t_pack = time.time()
    noised_batches, n_sents = _pack_to_batched_tensors(
        all_noised, args.batch_size, device)
    n_batches = len(noised_batches)
    print(f"  {len(eval_data)}문장 → {n_batches}배치 "
          f"(총 {total_tokens:,} tokens, 평균 {total_tokens/len(eval_data):.0f}tok/sent, "
          f"{time.time() - t_pack:.1f}s)")

    # Gold tags 사전 계산 (모든 repeat에서 동일, C++ 가속)
    print("Gold tags 사전 계산...")
    t_gold = time.time()
    gold_tags_cache = precompute_gold_tags(eval_data, vocab_size)
    print(f"  완료 ({time.time() - t_gold:.1f}s)")

    # CUDA warmup (사전 변환된 첫 배치로)
    if device == "cuda" and noised_batches:
        print("CUDA warmup...")
        torch.manual_seed(0)
        input_ids, pad_mask, _ = noised_batches[0]
        _forward_batch(model, input_ids, pad_mask, True, device)
        torch.cuda.synchronize()

    all_results: dict[str, list[dict]] = {}

    for var in args.variations:
        print(f"\n{'='*60}")
        print(f"Variation: {VARIATION_NAMES.get(var, var)}")
        print(f"{'='*60}")

        var_runs = []

        for repeat_idx in range(args.n_repeats):
            dropout_seed = (repeat_idx + 1) * 1000
            torch.manual_seed(dropout_seed)
            if device == "cuda":
                torch.cuda.manual_seed(dropout_seed)

            t0 = time.time()
            finals = run_variation_batch(
                var, model, eval_data, noised_batches, n_sents,
                vocab_size, device, args.batch_size,
            )
            if device == "cuda":
                torch.cuda.synchronize()
            elapsed = time.time() - t0

            t_eval = time.time()
            metrics = evaluate_all(eval_data, finals, vocab_size, gold_tags_cache)
            eval_elapsed = time.time() - t_eval

            metrics["repeat"] = repeat_idx
            metrics["seed"] = dropout_seed
            metrics["elapsed_sec"] = round(elapsed, 2)
            metrics["eval_sec"] = round(eval_elapsed, 2)

            var_runs.append(metrics)

            print(
                f"  repeat {repeat_idx}: "
                f"P={metrics['precision_exact']:.4f} "
                f"R={metrics['recall_exact']:.4f} "
                f"F0.5={metrics['f05_exact']:.4f} "
                f"F1={metrics['f1_exact']:.4f} "
                f"edits/sent={metrics['avg_edits_per_sent']:.2f} "
                f"(infer {elapsed:.1f}s, eval {eval_elapsed:.1f}s)"
            )

        all_results[var] = var_runs

    print_summary(all_results)

    # JSON 저장
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_path = out_dir / "raw_results.json"
    output = {
        "config": {
            "ckpt": args.ckpt,
            "corpus": args.corpus,
            "n_samples": len(eval_data),
            "n_repeats": args.n_repeats,
            "batch_size": args.batch_size,
            "device": device,
            "seed": args.seed,
            "variations": args.variations,
            "stochasticity": "mc_dropout",
            "dropout": config.dropout,
            "int8": args.int8,
        },
        "results": all_results,
    }
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n결과 저장: {raw_path}")

    summary_path = out_dir / "summary.txt"
    _save_summary_to_file(all_results, output["config"], summary_path)
    print(f"요약 저장: {summary_path}")


def _save_summary_to_file(
    all_results: dict[str, list[dict]], config: dict, path: Path,
):
    import statistics

    lines = [
        "합의 기반 2단계 반복 교정 실험 결과",
        "=" * 60,
        f"체크포인트: {config['ckpt']}",
        f"데이터: {config['corpus']} ({config['n_samples']}개)",
        f"반복: {config['n_repeats']}회",
        f"Stochasticity: MC Dropout (p={config['dropout']})",
        f"INT8 CUDA: {config.get('int8', 'N/A')}",
        "",
    ]

    header = (
        f"{'Variation':<25} | {'P_exact':>13} | {'R_exact':>13} | "
        f"{'F0.5_exact':>13} | {'F1_exact':>13} | {'Avg Edits':>13}"
    )
    lines.append(header)
    lines.append("-" * len(header))

    for var_key in sorted(all_results.keys()):
        runs = all_results[var_key]
        name = VARIATION_NAMES.get(var_key, var_key)

        def ms(key, _runs=runs):
            vals = [r[key] for r in _runs]
            m = statistics.mean(vals)
            s = statistics.stdev(vals) if len(vals) > 1 else 0.0
            return f"{m:.4f}±{s:.4f}"

        lines.append(
            f"{name:<25} | {ms('precision_exact'):>13} | "
            f"{ms('recall_exact'):>13} | {ms('f05_exact'):>13} | "
            f"{ms('f1_exact'):>13} | {ms('avg_edits_per_sent'):>13}"
        )

    lines.append("")
    lines.append("Raw per-run 결과:")

    for var_key in sorted(all_results.keys()):
        name = VARIATION_NAMES.get(var_key, var_key)
        lines.append(f"\n  {name}:")
        for run in all_results[var_key]:
            lines.append(
                f"    repeat {run['repeat']}: "
                f"P={run['precision_exact']:.4f} R={run['recall_exact']:.4f} "
                f"F0.5={run['f05_exact']:.4f} F1={run['f1_exact']:.4f} "
                f"edits/sent={run['avg_edits_per_sent']:.2f} "
                f"(infer {run['elapsed_sec']}s, eval {run.get('eval_sec', '?')}s)"
            )

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# ── CLI ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="합의 기반 2단계 반복 교정 실험",
    )
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--corpus", default="corpus/val_50k.jsonl")
    parser.add_argument("--n_samples", type=int, default=5000)
    parser.add_argument("--n_repeats", type=int, default=5)
    parser.add_argument(
        "--variations", nargs="+", default=["v1", "v2", "v3", "v4"],
        choices=["v1", "v2", "v3", "v4"],
    )
    parser.add_argument("--batch_size", type=int, default=32,
                        help="패킹된 시퀀스 배치 크기")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output_dir", default="exp-2-pass-consensus/results")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--int8", action="store_true",
                        help="INT8 CUDA BitLinear 활성화 (대형 GPU용, 기본=BF16)")

    args = parser.parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
