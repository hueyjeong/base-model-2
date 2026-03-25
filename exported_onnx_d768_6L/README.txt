DenseEditor ONNX 추론 패키지
============================

모델: d768 6L (58M params, BiMamba-2 + BitNet)

파일 구성:
  model.onnx        - FP32 모델 (graph + external data)
  model.onnx.data   - FP32 가중치 (223MB)
  model_fp16.onnx   - FP16 모델 (113MB, GPU 추천)
  config.json       - 모델 설정
  benchmark.py      - 벤치마크/추론 스크립트
  *_tokenizer.json  - 토크나이저

설치 (택 1):
  pip install onnxruntime                   # CPU 기본
  pip install onnxruntime-openvino          # Intel CPU/iGPU (추천)
  pip install onnxruntime-directml          # Windows GPU (Intel/AMD/NVIDIA)
  pip install numpy                         # 공통 필수

벤치마크:
  python benchmark.py                       # 자동 EP 선택
  python benchmark.py --scan-all            # 모든 EP 비교
  python benchmark.py --ep openvino-gpu     # Intel iGPU
  python benchmark.py --ep directml         # DirectML
  python benchmark.py --ep cpu              # CPU만

추론:
  echo {"ids": [2, 42, 43, 44]} | python benchmark.py --infer

참고 성능 (RTX 5060 Ti, d768 6L):
  ORT FP16 CUDA:   T=256 → 8ms,  T=4096 → 38ms
  ORT FP32 CUDA:   T=256 → 9ms,  T=4096 → 69ms
  OpenVINO CPU:    T=256 → 71ms, T=4096 → 974ms
  ORT CPU:         T=256 → 169ms, T=4096 → 2445ms
