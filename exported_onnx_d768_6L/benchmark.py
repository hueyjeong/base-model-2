"""DenseEditor ONNX 벤치마크 — Windows/Linux/Mac 공통

사용법:
    pip install onnxruntime                    # CPU 기본
    pip install onnxruntime-openvino           # Intel CPU/iGPU
    pip install onnxruntime-directml           # Windows DirectX12 GPU
    pip install onnxruntime-gpu                # NVIDIA CUDA GPU

    python benchmark.py                        # 자동 감지 + 벤치마크
    python benchmark.py --ep openvino-gpu      # 특정 EP 지정
    python benchmark.py --ep directml          # DirectML (Windows)
    python benchmark.py --infer                # 추론 모드 (stdin JSON)
"""
import argparse
import json
import time
import sys
from pathlib import Path

import numpy as np


def find_model():
    """모델 파일 자동 탐색 (가중치 임베드된 파일 우선)"""
    base = Path(__file__).parent
    for name in ["model_fp16.onnx", "model_f32.onnx", "model.onnx"]:
        p = base / name
        if p.exists():
            return str(p), name
    raise FileNotFoundError("ONNX 모델 파일을 찾을 수 없음")


def get_session(model_path, ep="auto"):
    """EP별 ONNX Runtime 세션 생성"""
    import onnxruntime as ort

    available = ort.get_available_providers()

    ep_map = {
        "openvino-gpu": [("OpenVINOExecutionProvider", {"device_type": "GPU"}), "CPUExecutionProvider"],
        "openvino-cpu": [("OpenVINOExecutionProvider", {"device_type": "CPU"}), "CPUExecutionProvider"],
        "directml":     [("DmlExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"],
        "directml:0":   [("DmlExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"],
        "directml:1":   [("DmlExecutionProvider", {"device_id": 1}), "CPUExecutionProvider"],
        "directml:2":   [("DmlExecutionProvider", {"device_id": 2}), "CPUExecutionProvider"],
        "cuda":         ["CUDAExecutionProvider", "CPUExecutionProvider"],
        "tensorrt":     ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"],
        "cpu":          ["CPUExecutionProvider"],
    }

    if ep == "auto":
        # 우선순위: CUDA > OpenVINO GPU > DirectML > OpenVINO CPU > CPU
        if "CUDAExecutionProvider" in available:
            ep = "cuda"
        elif "OpenVINOExecutionProvider" in available:
            ep = "openvino-gpu"  # GPU 시도, 실패하면 CPU 폴백
        elif "DmlExecutionProvider" in available:
            ep = "directml"
        else:
            ep = "cpu"

    providers = ep_map.get(ep, ["CPUExecutionProvider"])
    print(f"요청 EP: {ep}")

    try:
        sess = ort.InferenceSession(model_path, providers=providers)
    except Exception as e:
        print(f"  {ep} 실패: {e}")
        if ep == "openvino-gpu":
            print("  → OpenVINO CPU로 폴백")
            return get_session(model_path, "openvino-cpu")
        print("  → CPU로 폴백")
        sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

    active = sess.get_providers()
    print(f"활성 EP: {active[0]}")
    return sess


def benchmark(sess, seq_lens=(256, 512, 1024, 2048, 4096), n_runs=None):
    """다양한 seq_len으로 벤치마크"""
    # config에서 bos_id 로드
    config_path = Path(__file__).parent / "config.json"
    bos_id = 2
    if config_path.exists():
        with open(config_path) as f:
            bos_id = json.load(f).get("bos_id", 2)

    # 작은 입력으로 동작 확인
    print("워밍업...", end="", flush=True)
    try:
        test_ids = np.array([[bos_id] + [42] * 255], dtype=np.int64)
        sess.run(None, {"input_ids": test_ids})
        print(" OK")
    except Exception as e:
        print(f" 실패: {e}")
        return

    print(f"\n{'T':>6s}  {'Avg(ms)':>10s}  {'Med(ms)':>10s}  {'N':>4s}", flush=True)
    print("-" * 38, flush=True)

    for T in seq_lens:
        ids = np.random.randint(3, 300, (1, T)).astype(np.int64)
        ids[0, 0] = bos_id

        try:
            sys.stdout.write(f"{T:>6d}  ")
            sys.stdout.flush()

            # warmup
            for _ in range(2):
                sess.run(None, {"input_ids": ids})

            N = n_runs or max(5, 50 // max(1, T // 256))
            times = []
            for _ in range(N):
                t0 = time.perf_counter()
                sess.run(None, {"input_ids": ids})
                times.append((time.perf_counter() - t0) * 1000)

            avg = sum(times) / len(times)
            med = sorted(times)[len(times) // 2]
            print(f"{avg:>10.1f}  {med:>10.1f}  {N:>4d}", flush=True)
        except Exception as e:
            print(f"ERROR: {e}", flush=True)


def infer(sess):
    """추론 모드: stdin으로 token IDs 입력 → stdout으로 tag IDs 출력"""
    print("추론 모드 (JSON Lines: {\"ids\": [2, 42, 43, ...]})", file=sys.stderr)
    print("Ctrl+C로 종료", file=sys.stderr)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        data = json.loads(line)
        ids = np.array([data["ids"]], dtype=np.int64)

        t0 = time.perf_counter()
        out = sess.run(None, {"input_ids": ids})[0]
        ms = (time.perf_counter() - t0) * 1000

        tags = out[0].argmax(axis=-1).tolist()
        print(json.dumps({"tags": tags}), flush=True)
        print(f"  {len(data['ids'])} tokens → {ms:.1f}ms", file=sys.stderr)


def scan_all_eps(model_path):
    """설치된 모든 EP로 벤치마크"""
    import onnxruntime as ort
    available = ort.get_available_providers()
    print(f"설치된 EP: {available}\n")

    eps_to_try = []
    if "CUDAExecutionProvider" in available:
        eps_to_try.append("cuda")
    if "OpenVINOExecutionProvider" in available:
        eps_to_try.extend(["openvino-gpu", "openvino-cpu"])
    if "DmlExecutionProvider" in available:
        eps_to_try.append("directml")
    eps_to_try.append("cpu")

    for ep in eps_to_try:
        print(f"\n{'='*40}")
        try:
            sess = get_session(model_path, ep)
            benchmark(sess, seq_lens=[256, 1024, 4096])
        except Exception as e:
            print(f"  에러: {e}")
        print()


def main():
    parser = argparse.ArgumentParser(description="DenseEditor ONNX 벤치마크")
    parser.add_argument("--model", help="ONNX 모델 경로 (자동 탐색)")
    parser.add_argument("--ep", default="auto",
                        help="Execution Provider (auto, cuda, directml, directml:0, directml:1, "
                             "openvino-gpu, openvino-cpu, cpu)")
    parser.add_argument("--infer", action="store_true", help="추론 모드")
    parser.add_argument("--scan-all", action="store_true", help="모든 EP 벤치마크")
    parser.add_argument("--list-gpus", action="store_true", help="DirectML GPU 목록 출력")
    parser.add_argument("--seq-lens", default="256,512,1024,2048,4096",
                        help="벤치마크 시퀀스 길이 (쉼표 구분)")
    parser.add_argument("--n-runs", type=int, help="벤치마크 반복 횟수")
    args = parser.parse_args()

    if args.model:
        model_path = args.model
        model_name = Path(model_path).name
    else:
        model_path, model_name = find_model()

    print(f"모델: {model_name} ({Path(model_path).stat().st_size / 1024 / 1024:.1f}MB)")

    if args.list_gpus:
        try:
            import onnxruntime as ort
            print("DirectML GPU 탐색:")
            for i in range(8):
                try:
                    sess = ort.InferenceSession(
                        model_path,
                        providers=[("DmlExecutionProvider", {"device_id": i}), "CPUExecutionProvider"])
                    if "DmlExecutionProvider" in sess.get_providers():
                        # 간단한 추론으로 동작 확인
                        ids = np.array([[2] + [42] * 255], dtype=np.int64)
                        sess.run(None, {"input_ids": ids})
                        print(f"  device_id={i}: OK (--ep directml:{i})")
                    del sess
                except Exception:
                    break
        except ImportError:
            print("onnxruntime-directml 미설치")
        return

    if args.scan_all:
        scan_all_eps(model_path)
        return

    sess = get_session(model_path, args.ep)

    if args.infer:
        infer(sess)
    else:
        seq_lens = [int(x) for x in args.seq_lens.split(",")]
        benchmark(sess, seq_lens=seq_lens, n_runs=args.n_runs)


if __name__ == "__main__":
    main()
