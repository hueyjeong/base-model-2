"""
Doc-Isolated Linear Cross-Attention 전후 비교 벤치마크

측정 항목:
  1. 정확도: CUDA 커널 출력 vs PyTorch loop fallback (최대 절대 오차)
  2. Forward 속도 (ms)
  3. Forward+Backward 속도 (ms)
  4. Peak GPU 메모리 (MB)

실행:
  /workspace/base-model-2/.venv/bin/python3 bench_doc_linear_attn.py
  /workspace/base-model-2/.venv/bin/python3 bench_doc_linear_attn.py --B 2 --src 2048 --tgt 2048 --docs 40
"""
import argparse
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── 인자 파싱 ──────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--B",    type=int, default=2,   help="배치 크기")
parser.add_argument("--H",    type=int, default=12,  help="헤드 수")
parser.add_argument("--src",  type=int, default=1536, help="소스 시퀀스 길이")
parser.add_argument("--tgt",  type=int, default=1536, help="타겟 시퀀스 길이")
parser.add_argument("--d",    type=int, default=64,  help="d_head")
parser.add_argument("--docs", type=int, default=30,  help="팩당 평균 문서 수 (D)")
parser.add_argument("--warmup", type=int, default=5,  help="워밍업 반복수")
parser.add_argument("--iters",  type=int, default=20, help="측정 반복수")
parser.add_argument("--no_bwd", action="store_true",  help="backward 스킵")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type != "cuda":
    print("⚠️  CUDA 없음 — CPU에서는 CUDA 커널 비교 불가능. PyTorch 경로만 실행됩니다.")

B, H, src_len, tgt_len, d = args.B, args.H, args.src, args.tgt, args.d
D = args.docs

print(f"\n{'='*60}")
print(f"  설정: B={B}  H={H}  src={src_len}  tgt={tgt_len}  d={d}  D(문서수)={D}")
print(f"{'='*60}\n")


# ── 입력 생성 ──────────────────────────────────────────────────────────
def make_inputs(requires_grad=True):
    """재현 가능한 랜덤 입력 생성"""
    torch.manual_seed(42)
    q = (F.elu(torch.randn(B, H, tgt_len, d, device=device)) + 1.0)
    k = (F.elu(torch.randn(B, H, src_len, d, device=device)) + 1.0)
    v = torch.randn(B, H, src_len, d, device=device)

    if requires_grad:
        q = q.float().requires_grad_(True)
        k = k.float().requires_grad_(True)
        v = v.float().requires_grad_(True)
    else:
        q = q.float()
        k = k.float()
        v = v.float()

    # 균등 분할: 각 위치가 0~D-1 중 하나의 doc_id를 가짐
    src_doc_ids = torch.arange(src_len, device=device).unsqueeze(0).expand(B, -1)
    src_doc_ids = (src_doc_ids * D // src_len).int()
    tgt_doc_ids = torch.arange(tgt_len, device=device).unsqueeze(0).expand(B, -1)
    tgt_doc_ids = (tgt_doc_ids * D // tgt_len).int()

    return q, k, v, src_doc_ids, tgt_doc_ids


# ── PyTorch loop 구현 (기존 방식) ─────────────────────────────────────
def pytorch_loop(q, k, v, src_doc_ids, tgt_doc_ids, eps=1e-5):
    out = torch.zeros_like(q)
    max_doc = int(max(src_doc_ids.max().item(), tgt_doc_ids.max().item())) + 1
    for doc in range(max_doc):
        src_mask = (src_doc_ids == doc).unsqueeze(1).unsqueeze(-1).to(k.dtype)
        k_d = k * src_mask
        v_d = v * src_mask
        ctx_d = torch.matmul(k_d.transpose(-1, -2), v_d)
        z_d   = k_d.sum(dim=-2)
        num_d = torch.matmul(q, ctx_d)
        den_d = torch.einsum("bhld,bhd->bhl", q, z_d)
        out_d = num_d / (den_d.unsqueeze(-1) + eps)
        tgt_mask = (tgt_doc_ids == doc).unsqueeze(1).unsqueeze(-1).to(out_d.dtype)
        out = out + out_d * tgt_mask
    return out


# ── CUDA scatter/gather 구현 (신규) ───────────────────────────────────
cuda_fn = None
if device.type == "cuda":
    try:
        import sys, os
        sys.path.insert(0, os.path.dirname(__file__))
        from model.linear_attention import cuda_doc_linear_attn, _load_cuda_doc_linear_attn
        _load_cuda_doc_linear_attn()   # JIT compile
        cuda_fn = cuda_doc_linear_attn
        print("✔  CUDA doc-isolated 커널 로드 완료\n")
    except Exception as e:
        print(f"⚠️  CUDA 커널 로드 실패: {e}")
        print("   PyTorch loop 방식만 측정합니다.\n")


# ── 정확도 검증 ────────────────────────────────────────────────────────
if cuda_fn is not None and device.type == "cuda":
    print("[ 정확도 검증 ]")
    q, k, v, src_ids, tgt_ids = make_inputs(requires_grad=False)
    max_docs = D

    out_pt   = pytorch_loop(q, k, v, src_ids, tgt_ids)
    out_cuda = cuda_fn(q, k, v, src_ids, tgt_ids, max_docs)

    abs_err  = (out_pt - out_cuda).abs()
    max_err  = abs_err.max().item()
    mean_err = abs_err.mean().item()
    rel_err  = (abs_err / (out_pt.abs() + 1e-8)).mean().item()

    status = "✔ PASS" if max_err < 1e-3 else "✗ FAIL"
    print(f"  {status}  최대 절대 오차: {max_err:.2e}  평균: {mean_err:.2e}  상대: {rel_err:.2e}\n")


# ── 측정 헬퍼 ──────────────────────────────────────────────────────────
def measure(fn, warmup, iters, backward=True):
    """시간(ms)과 peak 메모리(MB)를 측정"""
    # 워밍업
    for _ in range(warmup):
        q, k, v, src_ids, tgt_ids = make_inputs(requires_grad=backward)
        out = fn(q, k, v, src_ids, tgt_ids)
        if backward and not args.no_bwd:
            out.sum().backward()
        if device.type == "cuda":
            torch.cuda.synchronize()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(iters):
        q, k, v, src_ids, tgt_ids = make_inputs(requires_grad=backward)
        out = fn(q, k, v, src_ids, tgt_ids)
        if backward and not args.no_bwd:
            out.sum().backward()
        if device.type == "cuda":
            torch.cuda.synchronize()
    elapsed = (time.perf_counter() - t0) / iters * 1000  # ms per iter

    peak_mb = 0.0
    if device.type == "cuda":
        peak_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

    return elapsed, peak_mb


# ── 벤치마크 실행 ──────────────────────────────────────────────────────
bwd_label = "fwd+bwd" if not args.no_bwd else "fwd only"

print(f"[ 속도 벤치마크: {bwd_label}, warmup={args.warmup}, iters={args.iters} ]")

# PyTorch loop
def loop_fn(q, k, v, src_ids, tgt_ids):
    return pytorch_loop(q, k, v, src_ids, tgt_ids)

ms_pt, mem_pt = measure(loop_fn, args.warmup, args.iters, backward=(not args.no_bwd))
print(f"  [기존]  PyTorch loop    : {ms_pt:8.2f} ms  peak GPU: {mem_pt:7.1f} MB")

if cuda_fn is not None and device.type == "cuda":
    def cuda_wrapper(q, k, v, src_ids, tgt_ids):
        max_docs = D
        return cuda_doc_linear_attn(q, k, v, src_ids, tgt_ids, max_docs)

    ms_cu, mem_cu = measure(cuda_wrapper, args.warmup, args.iters, backward=(not args.no_bwd))
    speedup  = ms_pt / ms_cu
    mem_save = mem_pt - mem_cu
    print(f"  [신규]  CUDA scatter/gather: {ms_cu:8.2f} ms  peak GPU: {mem_cu:7.1f} MB")
    print()
    print(f"  → 속도 {speedup:.2f}x  |  메모리 절감 {mem_save:+.1f} MB")

print()

# ── Forward only 별도 측정 ─────────────────────────────────────────────
if not args.no_bwd:
    print(f"[ Forward only ]")
    ms_pt_f,  _ = measure(loop_fn,  args.warmup, args.iters, backward=False)
    print(f"  [기존]  PyTorch loop    : {ms_pt_f:8.2f} ms")

    if cuda_fn is not None and device.type == "cuda":
        ms_cu_f,  _ = measure(cuda_wrapper, args.warmup, args.iters, backward=False)
        speedup_f = ms_pt_f / ms_cu_f
        print(f"  [신규]  CUDA scatter/gather: {ms_cu_f:8.2f} ms")
        print(f"  → Forward 속도 {speedup_f:.2f}x")
    print()
