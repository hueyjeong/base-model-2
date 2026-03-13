/**
 * Fused per-row absmax + STE quantize 커널 (float 출력)
 *
 * quantize_activations_8bit의 5+ PyTorch 커널을 단일 CUDA 커널로 fusion.
 * 기존 quantize_activations_int8_kernel과 동일한 로직이지만,
 * INT8 대신 float 출력 → autograd STE와 직접 호환.
 *
 * Phase 1: Per-row absmax reduction (shared memory)
 * Phase 2: Fused scale → round → clamp → float 출력
 */
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>

namespace {

constexpr int QA_THREADS = 256;

__global__ void fused_quant_act_kernel(
    const float* __restrict__ x,
    float* __restrict__ x_quant,
    float* __restrict__ x_scale,
    int M, int K
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    if (row >= M) return;

    __shared__ float smax[QA_THREADS];

    // Phase 1: per-row absmax — shared memory tree reduction
    float local_max = 0.0f;
    const int row_off = row * K;
    for (int col = tid; col < K; col += blockDim.x) {
        float v = fabsf(x[row_off + col]);
        if (v > local_max) local_max = v;
    }
    smax[tid] = local_max;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (smax[tid + s] > smax[tid]) smax[tid] = smax[tid + s];
        }
        __syncthreads();
    }

    float absmax = smax[0];
    if (absmax < 1e-5f) absmax = 1e-5f;
    const float inv_scale = 127.0f / absmax;
    const float scale = absmax / 127.0f;

    if (tid == 0) {
        x_scale[row] = scale;
    }

    // Phase 2: fused quantize (STE round, float 출력)
    for (int col = tid; col < K; col += blockDim.x) {
        float q = nearbyintf(x[row_off + col] * inv_scale);
        q = fminf(127.0f, fmaxf(-128.0f, q));
        x_quant[row_off + col] = q;
    }
}

}  // namespace


std::vector<torch::Tensor> fused_quant_act_forward_cuda(torch::Tensor x) {
    const auto M = static_cast<int>(x.size(0));
    const auto K = static_cast<int>(x.size(1));

    auto x_quant = torch::empty({M, K}, x.options());
    auto x_scale = torch::empty({M, 1}, x.options());

    if (M > 0) {
        fused_quant_act_kernel<<<M, QA_THREADS, 0, at::cuda::getDefaultCUDAStream()>>>(
            x.data_ptr<float>(),
            x_quant.data_ptr<float>(),
            x_scale.data_ptr<float>(),
            M, K
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    return {x_quant, x_scale};
}
