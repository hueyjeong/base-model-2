/**
 * Linear Cross-Attention Fused CUDA Kernel
 * 
 * K^T V context matrix 계산 + Q·(K^T V) 출력을 하나의 커널로 fusion.
 * d_head=64 기준 context matrix(64x64=16KB)를 shared memory에 적재.
 * 
 * 구조:
 *   Phase 1: src_len 순회하며 context = K^T·V, z = sum(K) 누적
 *   Phase 2: tgt_len 순회하며 out[t] = (Q[t]·context) / (Q[t]·z + eps)
 * 
 * Grid: (B * n_heads, 1, 1)
 * Block: (d_head, 1, 1) = (64, 1, 1)
 */

#include <torch/extension.h>
#include <cuda_runtime.h>

#define MAX_D_HEAD 64

__global__ void linear_cross_attn_fwd_kernel(
    const float* __restrict__ Q,       // [BH, tgt_len, D]
    const float* __restrict__ K,       // [BH, src_len, D]
    const float* __restrict__ V,       // [BH, src_len, D]
    const bool*  __restrict__ mask,    // [B, src_len] or nullptr
    float* __restrict__ Out,           // [BH, tgt_len, D]
    const int B,
    const int n_heads,
    const int src_len,
    const int tgt_len,
    const int D,
    const float eps
) {
    const int bh = blockIdx.x;                   // batch * head index
    const int b  = bh / n_heads;                 // batch index (for mask lookup)
    const int tid = threadIdx.x;                 // 0..D-1

    if (tid >= D) return;

    // Shared memory layout
    extern __shared__ float smem[];
    float* context = smem;                       // [D][D] = D*D floats
    float* z_shared = smem + D * D;              // [D] floats
    float* v_shared = z_shared + D;              // [D] floats (temp for V loading)
    float* q_shared = v_shared + D;              // [D] floats (temp for Q loading)

    // ── Phase 0: Initialize shared memory ──
    z_shared[tid] = 0.0f;
    for (int j = 0; j < D; j++) {
        context[tid * D + j] = 0.0f;
    }
    __syncthreads();

    // ── Phase 1: Accumulate context = K^T·V and z = sum(K) ──
    const float* K_bh = K + bh * src_len * D;
    const float* V_bh = V + bh * src_len * D;
    const bool* mask_b = mask ? (mask + b * src_len) : nullptr;

    for (int s = 0; s < src_len; s++) {
        // Coalesced load: each thread loads its component
        float k_i = K_bh[s * D + tid];
        v_shared[tid] = V_bh[s * D + tid];
        __syncthreads();

        // Check mask (mask=true means valid)
        bool valid = (mask_b == nullptr) || mask_b[s];

        if (valid) {
            z_shared[tid] += k_i;
            for (int j = 0; j < D; j++) {
                context[tid * D + j] += k_i * v_shared[j];
            }
        }
        __syncthreads();
    }

    // ── Phase 2: Compute output for each target position ──
    const float* Q_bh = Q + bh * tgt_len * D;
    float* Out_bh = Out + bh * tgt_len * D;

    for (int t = 0; t < tgt_len; t++) {
        // Coalesced load of Q[t, :]
        q_shared[tid] = Q_bh[t * D + tid];
        __syncthreads();

        // numerator: out[tid] = sum_i q[i] * context[i][tid]
        float num_val = 0.0f;
        float den_val = 0.0f;
        for (int i = 0; i < D; i++) {
            float q_i = q_shared[i];
            num_val += q_i * context[i * D + tid];
            // denominator is the same for all output dims, 
            // but we compute it in each thread (redundant but avoids extra sync)
            den_val += q_i * z_shared[i];
        }

        // Write output (coalesced)
        Out_bh[t * D + tid] = num_val / (den_val + eps);
        __syncthreads();
    }
}


torch::Tensor linear_cross_attn_fwd_cuda(
    torch::Tensor Q,      // [B, n_heads, tgt_len, D]
    torch::Tensor K,      // [B, n_heads, src_len, D]
    torch::Tensor V,      // [B, n_heads, src_len, D]
    torch::Tensor mask,   // [B, src_len] bool, or empty
    float eps
) {
    const int B = Q.size(0);
    const int n_heads = Q.size(1);
    const int tgt_len = Q.size(2);
    const int D = Q.size(3);
    const int src_len = K.size(2);

    TORCH_CHECK(D <= MAX_D_HEAD, "d_head must be <= ", MAX_D_HEAD);

    // Ensure contiguous FP32
    auto Q_f = Q.contiguous().to(torch::kFloat32);
    auto K_f = K.contiguous().to(torch::kFloat32);
    auto V_f = V.contiguous().to(torch::kFloat32);

    // Reshape to [B*n_heads, seq_len, D]
    auto Q_2d = Q_f.reshape({B * n_heads, tgt_len, D});
    auto K_2d = K_f.reshape({B * n_heads, src_len, D});
    auto V_2d = V_f.reshape({B * n_heads, src_len, D});

    auto Out = torch::empty({B * n_heads, tgt_len, D}, Q_f.options());

    const bool* mask_ptr = nullptr;
    torch::Tensor mask_bool;
    if (mask.numel() > 0) {
        mask_bool = mask.contiguous().to(torch::kBool);
        mask_ptr = mask_bool.data_ptr<bool>();
    }

    const int BH = B * n_heads;
    const int threads = D;
    // Shared memory: context[D*D] + z[D] + v_temp[D] + q_temp[D]
    const int smem_size = (D * D + 3 * D) * sizeof(float);

    linear_cross_attn_fwd_kernel<<<BH, threads, smem_size>>>(
        Q_2d.data_ptr<float>(),
        K_2d.data_ptr<float>(),
        V_2d.data_ptr<float>(),
        mask_ptr,
        Out.data_ptr<float>(),
        B, n_heads, src_len, tgt_len, D, eps
    );

    return Out.reshape({B, n_heads, tgt_len, D}).to(Q.dtype());
}
