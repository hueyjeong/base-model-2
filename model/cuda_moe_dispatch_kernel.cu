/**
 * MoE Fused Dispatch CUDA 커널
 *
 * GPU-CPU sync 완전 제거. 고정 capacity 버퍼로 expert별 토큰 할당.
 *
 * 1. moe_scatter: 토큰을 expert 버퍼에 분배 (atomic counter)
 * 2. moe_gather: expert 출력을 원래 위치로 모음
 */
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>

namespace {

/**
 * Scatter: 토큰을 expert 버퍼에 분배
 *
 * expert_idx: (N,) int — 각 토큰이 할당된 expert
 * x: (N, D) float — 입력 토큰
 * expert_w: (N,) float — routing weight
 *
 * buffers: (n_experts, capacity, D) float — expert별 입력 버퍼
 * weights: (n_experts, capacity) float — expert별 가중치
 * counts: (n_experts,) int — expert별 할당된 토큰 수
 * token_pos: (N,) int — 각 토큰의 버퍼 내 위치 (gather 시 사용)
 */
__global__ void moe_scatter_kernel(
    const int64_t* __restrict__ expert_idx,
    const float* __restrict__ x,
    const float* __restrict__ expert_w,
    float* __restrict__ buffers,
    float* __restrict__ weights,
    int* __restrict__ counts,
    int* __restrict__ token_pos,
    int N, int D, int n_experts, int capacity
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;

    const int eidx = static_cast<int>(expert_idx[tid]);
    if (eidx < 0 || eidx >= n_experts) return;

    // atomic으로 expert별 position 할당
    int pos = atomicAdd(&counts[eidx], 1);
    if (pos >= capacity) {
        token_pos[tid] = -1;  // overflow → drop
        return;
    }

    token_pos[tid] = pos;

    // 입력 복사
    const float* src = x + tid * D;
    float* dst = buffers + (eidx * capacity + pos) * D;
    for (int d = threadIdx.y; d < D; d += blockDim.y) {
        dst[d] = src[d];
    }

    weights[eidx * capacity + pos] = expert_w[tid];
}


/**
 * Gather: expert 출력을 원래 위치에 weighted scatter-add
 *
 * expert_out: (n_experts, capacity, D) — expert별 출력
 * weights: (n_experts, capacity) — routing weights
 * expert_idx: (N,) — 각 토큰의 expert 인덱스
 * token_pos: (N,) — 각 토큰의 버퍼 위치
 * out: (N, D) — 최종 출력 (scatter-add)
 */
__global__ void moe_gather_kernel(
    const float* __restrict__ expert_out,
    const float* __restrict__ weights,
    const int64_t* __restrict__ expert_idx,
    const int* __restrict__ token_pos,
    float* __restrict__ out,
    int N, int D, int capacity
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;

    const int pos = token_pos[tid];
    if (pos < 0) return;  // overflow → skip

    const int eidx = static_cast<int>(expert_idx[tid]);
    const float w = weights[eidx * capacity + pos];

    const float* src = expert_out + (eidx * capacity + pos) * D;
    float* dst = out + tid * D;

    for (int d = 0; d < D; d++) {
        dst[d] += src[d] * w;
    }
}

}  // namespace


std::vector<torch::Tensor> moe_scatter_cuda(
    torch::Tensor expert_idx,  // (N,) int64
    torch::Tensor x,           // (N, D) float
    torch::Tensor expert_w,    // (N,) float
    int n_experts,
    int capacity
) {
    int N = x.size(0);
    int D = x.size(1);

    auto buffers = torch::zeros({n_experts, capacity, D}, x.options());
    auto weights = torch::zeros({n_experts, capacity}, x.options());
    auto counts = torch::zeros({n_experts}, x.options().dtype(torch::kInt32));
    auto token_pos = torch::full({N}, -1, x.options().dtype(torch::kInt32));

    auto x_f = x.to(torch::kFloat32).contiguous();
    auto w_f = expert_w.to(torch::kFloat32).contiguous();
    auto idx_c = expert_idx.contiguous();

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    moe_scatter_kernel<<<blocks, threads, 0, at::cuda::getDefaultCUDAStream()>>>(
        idx_c.data_ptr<int64_t>(),
        x_f.data_ptr<float>(),
        w_f.data_ptr<float>(),
        buffers.data_ptr<float>(),
        weights.data_ptr<float>(),
        counts.data_ptr<int>(),
        token_pos.data_ptr<int>(),
        N, D, n_experts, capacity
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return {buffers, weights, counts, token_pos};
}


torch::Tensor moe_gather_cuda(
    torch::Tensor expert_out,   // (n_experts, capacity, D)
    torch::Tensor weights,      // (n_experts, capacity)
    torch::Tensor expert_idx,   // (N,)
    torch::Tensor token_pos,    // (N,)
    int N, int D
) {
    auto out = torch::zeros({N, D}, expert_out.options());
    int capacity = expert_out.size(1);

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    moe_gather_kernel<<<blocks, threads, 0, at::cuda::getDefaultCUDAStream()>>>(
        expert_out.data_ptr<float>(),
        weights.data_ptr<float>(),
        expert_idx.data_ptr<int64_t>(),
        token_pos.data_ptr<int>(),
        out.data_ptr<float>(),
        N, D, capacity
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return out;
}
