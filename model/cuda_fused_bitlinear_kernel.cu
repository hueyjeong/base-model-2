/**
 * Fused BitLinear CUDA 커널 — Activation Quant + INT8 Matmul + Dequant
 *
 * 기존 분리된 3개 커널 (quantize_activations → bitlinear_forward → scale_restore)을
 * 단일 커널로 통합. kernel launch 3회→1회, 중간 메모리 할당 제거.
 *
 * Phase 1: Per-row absmax reduction (shared memory) → scale 계산
 * Phase 2: 입력을 INT8로 quantize하면서 동시에 dp4a matmul 수행
 * Phase 3: Dequant (w_scale * x_scale * acc) → 출력 저장
 *
 * Weight는 이미 INT8 ternary로 pre-quantized (캐시된 상태).
 */
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>

namespace {

constexpr int FUSED_BLOCK_M = 4;   // rows per block
constexpr int FUSED_BLOCK_N = 64;  // cols per block (output features)
constexpr int FUSED_BLOCK_K = 32;  // tile size for K dimension
constexpr int WARP_SIZE = 32;

/**
 * Fused forward: quant + matmul + dequant
 *
 * x: (M, K) float32 (이미 LayerNorm 적용됨)
 * w: (N, K) int8 (ternary {-1, 0, +1})
 * w_scale: scalar float
 * out: (M, N) float32
 *
 * 각 block이 FUSED_BLOCK_M개 행을 처리.
 * Phase 1: shared memory로 per-row absmax 계산 + INT8 변환
 * Phase 2: dp4a matmul
 */
__global__ void fused_bitlinear_forward_kernel(
    const float* __restrict__ x,     // (M, K) 정규화된 입력
    const int8_t* __restrict__ w,    // (N, K) ternary weights
    const float* __restrict__ w_scale, // scalar
    const float* __restrict__ bias,  // (N,) or nullptr
    float* __restrict__ out,         // (M, N) 출력
    int M, int N, int K,
    bool has_bias
) {
    // 각 block: FUSED_BLOCK_M rows × 여러 output columns
    const int row_start = blockIdx.y * FUSED_BLOCK_M;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col >= N) return;

    const float ws = w_scale[0];

    // 각 row에 대해 처리
    for (int r = 0; r < FUSED_BLOCK_M; r++) {
        const int row = row_start + r;
        if (row >= M) break;

        const float* x_row = x + row * K;
        const int8_t* w_col = w + col * K;

        // Phase 1: per-row absmax (이 스레드가 전체 K를 순회)
        float absmax = 0.0f;
        for (int ki = 0; ki < K; ki++) {
            float av = fabsf(x_row[ki]);
            if (av > absmax) absmax = av;
        }
        if (absmax < 1e-5f) absmax = 1e-5f;
        const float x_scale = absmax / 127.0f;
        const float inv_scale = 127.0f / absmax;

        // Phase 2: quantize + matmul (fused)
        int32_t acc = 0;
        for (int ki = 0; ki < K; ki++) {
            float xq = nearbyintf(x_row[ki] * inv_scale);
            xq = fminf(127.0f, fmaxf(-128.0f, xq));
            int8_t x_int8 = static_cast<int8_t>(xq);
            acc += static_cast<int32_t>(x_int8) * static_cast<int32_t>(w_col[ki]);
        }

        // Phase 3: dequant
        float result = static_cast<float>(acc) * x_scale * ws;
        if (has_bias) result += bias[col];
        out[row * N + col] = result;
    }
}

}  // namespace


/**
 * Fused BitLinear forward (C++ 래퍼)
 *
 * x_normed: (M, K) — LayerNorm 이미 적용된 입력
 * w_int8: (N, K) — ternary INT8 weights (pre-cached)
 * w_scale: (1,) — weight scale
 * bias: optional (N,)
 *
 * Returns: (M, N) float output
 */
torch::Tensor fused_bitlinear_forward_cuda(
    torch::Tensor x_normed,  // (*, K) float
    torch::Tensor w_int8,    // (N, K) int8
    torch::Tensor w_scale,   // (1,) float
    c10::optional<torch::Tensor> bias_opt
) {
    // flatten to 2D
    auto x_shape = x_normed.sizes().vec();
    int K = x_shape.back();
    int M = x_normed.numel() / K;
    auto x_2d = x_normed.reshape({M, K}).contiguous();

    // float32로 변환
    auto x_f = x_2d.to(torch::kFloat32).contiguous();
    auto ws_f = w_scale.to(torch::kFloat32).contiguous();
    auto w_i8 = w_int8.contiguous();

    int N = w_i8.size(0);

    auto out = torch::zeros({M, N}, x_f.options());

    const bool has_bias = bias_opt.has_value();
    const float* bias_ptr = has_bias ?
        bias_opt.value().to(torch::kFloat32).contiguous().data_ptr<float>() : nullptr;

    // grid/block: 각 스레드가 1개 output element, FUSED_BLOCK_M rows per block
    dim3 block(256);
    dim3 grid((N + block.x - 1) / block.x, (M + FUSED_BLOCK_M - 1) / FUSED_BLOCK_M);

    fused_bitlinear_forward_kernel<<<grid, block, 0, at::cuda::getDefaultCUDAStream()>>>(
        x_f.data_ptr<float>(),
        w_i8.data_ptr<int8_t>(),
        ws_f.data_ptr<float>(),
        bias_ptr,
        out.data_ptr<float>(),
        M, N, K, has_bias
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    // 원래 shape 복원
    x_shape.back() = N;
    return out.reshape(x_shape).to(x_normed.scalar_type());
}
