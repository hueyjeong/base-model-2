/**
 * Document-Isolated Linear Cross-Attention CUDA Kernels
 *
 * 개요:
 *   문서별로 context matrix (K^T·V) 를 분리해 구성하고,
 *   각 target 위치가 자신의 document에 해당하는 context만 사용.
 *
 * 기존 Python loop 방식 문제:
 *   - for d in range(D): num_d = q @ context_d  → D배 full-seq 연산
 *   - 각 iter마다 (B,H,tgt_len,d_head) 텐서 생성 후 1/D만 사용
 *   - autograd가 D세트 중간 텐서 전부 보관
 *
 * 개선:
 *   Phase 1 (scatter): context[b,h,doc,i,j] += k[s,i]*v[s,j]
 *                      z[b,h,doc,i]          += k[s,i]
 *     - Grid=(B*H,), Block=(d_head,): 각 블록이 독점 → atomic 불필요
 *     - src 위치마다 doc ID 보고 해당 슬라이스에 직접 누적
 *
 *   Phase 2 (gather): out[b,h,t,j] = q[t]·context[d_t,:,j] / (q[t]·z[d_t,:]+eps)
 *     - Grid=(B*H,), Block=(d_head,): tgt 전체를 한 번에 처리
 *     - Python loop 없음, 중간 full-seq 텐서 없음
 *
 * Backward:
 *   동일한 scatter/gather 커널을 (q/den, grad_out) 입력으로 재활용.
 *   grad_context, grad_z를 먼저 구성한 뒤 grad_k, grad_v, grad_q를 별도 커널로 계산.
 */

#include <torch/extension.h>
#include <cuda_runtime.h>

// =========================================================================
// Kernel 1: Phase 1 — per-doc scatter (K^T V 누적)
//   Grid:  (B * H, 1, 1)
//   Block: (d_head, 1, 1)
//
//   각 블록이 하나의 (b, h) 슬라이스를 독점하므로 atomic 없이 누적 가능.
//   context 출력: (B*H, max_docs, d_head, d_head) — row-major [doc][i][j]
//   z      출력: (B*H, max_docs, d_head)
// =========================================================================
__global__ void doc_scatter_kv_kernel(
    const float* __restrict__ K,          // [B*H, src_len, d]
    const float* __restrict__ V,          // [B*H, src_len, d]
    const int*   __restrict__ doc_ids,    // [B, src_len]
    float*       __restrict__ context,    // [B*H, max_docs, d, d]  pre-zeroed
    float*       __restrict__ z,          // [B*H, max_docs, d]     pre-zeroed
    int B, int H, int src_len, int d, int max_docs
) {
    const int bh  = blockIdx.x;
    const int b   = bh / H;
    const int i   = threadIdx.x;          // row index (= k dimension)

    if (i >= d) return;

    // 각 블록이 독점하는 포인터
    const float* K_bh  = K + (long)bh * src_len * d;
    const float* V_bh  = V + (long)bh * src_len * d;
    float*       ctx   = context + (long)bh * max_docs * d * d;
    float*       z_bh  = z       + (long)bh * max_docs * d;
    const int*   doc_b = doc_ids + (long)b * src_len;

    // V의 한 열을 shared memory에 캐시
    extern __shared__ float v_s[];    // [d] floats

    for (int s = 0; s < src_len; s++) {
        // V[s,:] 를 shared memory 로 로드 (coalesced)
        v_s[i] = V_bh[(long)s * d + i];
        __syncthreads();

        float k_i = K_bh[(long)s * d + i];
        int   doc = doc_b[s];

        // context[doc, i, :] += k_i * V[s, :]
        float* ctx_row = ctx + ((long)doc * d + i) * d;
        for (int j = 0; j < d; j++) {
            ctx_row[j] += k_i * v_s[j];
        }

        // z[doc, i] += k_i
        z_bh[(long)doc * d + i] += k_i;

        __syncthreads();
    }
}


// =========================================================================
// Kernel 2: Phase 2 — per-tgt gather & normalize
//   Grid:  (B * H, 1, 1)
//   Block: (d_head, 1, 1)
//
//   out[t, j] = q[t,:] · context[d_t, :, j]  /  (q[t,:] · z[d_t,:] + eps)
//   q[t,:]    → 공유 메모리에 올려서 재사용
//   context[d_t, :, j] → j=tid 위치만 읽음 (stride-d 접근이지만 context가 작아서 캐시 hit)
// =========================================================================
__global__ void doc_gather_query_kernel(
    const float* __restrict__ Q,          // [B*H, tgt_len, d]
    const float* __restrict__ context,    // [B*H, max_docs, d, d]
    const float* __restrict__ z,          // [B*H, max_docs, d]
    const int*   __restrict__ tgt_doc_ids,// [B, tgt_len]
    float*       __restrict__ out,        // [B*H, tgt_len, d]
    float*       __restrict__ den_out,    // [B*H, tgt_len]   denominator 저장 (backward 재활용)
    int B, int H, int tgt_len, int d, int max_docs, float eps
) {
    const int bh  = blockIdx.x;
    const int b   = bh / H;
    const int j   = threadIdx.x;          // output dimension

    if (j >= d) return;

    const float* Q_bh   = Q       + (long)bh * tgt_len * d;
    const float* ctx    = context  + (long)bh * max_docs * d * d;
    const float* z_bh   = z        + (long)bh * max_docs * d;
    float*       out_bh = out      + (long)bh * tgt_len * d;
    float*       den_bh = den_out  + (long)bh * tgt_len;
    const int*   doc_t  = tgt_doc_ids + (long)b * tgt_len;

    extern __shared__ float q_s[];    // [d] floats

    for (int t = 0; t < tgt_len; t++) {
        // Q[t,:] → shared memory (coalesced)
        q_s[j] = Q_bh[(long)t * d + j];
        __syncthreads();

        int doc = doc_t[t];
        const float* ctx_d = ctx + (long)doc * d * d;    // context[d_t, :, :]
        const float* z_d   = z_bh + (long)doc * d;       // z[d_t, :]

        // num = Σ_i q[i] * context[d_t, i, j]
        // den = Σ_i q[i] * z[d_t, i]  (모든 j 스레드가 동일하게 계산)
        float num = 0.0f;
        float den = 0.0f;
        for (int i = 0; i < d; i++) {
            float q_i = q_s[i];
            num += q_i * ctx_d[(long)i * d + j];  // context[d_t, i, j]
            den += q_i * z_d[i];
        }

        float den_eps = den + eps;
        out_bh[(long)t * d + j] = num / den_eps;

        // den 저장: j==0 스레드가 대표로 기록 (모두 동일값)
        if (j == 0) {
            den_bh[t] = den_eps;
        }
        __syncthreads();
    }
}


// =========================================================================
// Kernel 3: Backward — grad_k, grad_v 계산
//   각 src 위치 s에 대해:
//     grad_k[s,i] = Σ_j grad_ctx[d_s,i,j] * v[s,j]  +  grad_z[d_s,i]
//     grad_v[s,j] = Σ_i grad_ctx[d_s,i,j] * k[s,i]
//                 = Σ_i grad_ctx[d_s,i,j] * k[s,i]   (= grad_ctx[d_s,:,j]·k[s,:])
//   Grid:  (B * H,)
//   Block: (d_head,)
// =========================================================================
__global__ void doc_backward_kv_kernel(
    const float* __restrict__ K,          // [B*H, src_len, d]
    const float* __restrict__ V,          // [B*H, src_len, d]
    const float* __restrict__ grad_ctx,   // [B*H, max_docs, d, d]
    const float* __restrict__ grad_zz,    // [B*H, max_docs, d]
    const int*   __restrict__ src_doc_ids,// [B, src_len]
    float*       __restrict__ grad_k,     // [B*H, src_len, d]
    float*       __restrict__ grad_v,     // [B*H, src_len, d]
    int B, int H, int src_len, int d, int max_docs
) {
    const int bh  = blockIdx.x;
    const int b   = bh / H;
    const int tid = threadIdx.x;         // dimension index

    if (tid >= d) return;

    const float* K_bh   = K         + (long)bh * src_len * d;
    const float* V_bh   = V         + (long)bh * src_len * d;
    const float* gctx   = grad_ctx  + (long)bh * max_docs * d * d;
    const float* gz_bh  = grad_zz   + (long)bh * max_docs * d;
    float*       gk_bh  = grad_k    + (long)bh * src_len * d;
    float*       gv_bh  = grad_v    + (long)bh * src_len * d;
    const int*   doc_b  = src_doc_ids + (long)b * src_len;

    extern __shared__ float kv_s[];    // [2*d] floats: k_s[d], v_s[d]
    float* k_s = kv_s;
    float* v_s = kv_s + d;

    for (int s = 0; s < src_len; s++) {
        k_s[tid] = K_bh[(long)s * d + tid];
        v_s[tid] = V_bh[(long)s * d + tid];
        __syncthreads();

        int doc = doc_b[s];
        const float* gctx_d = gctx + (long)doc * d * d;   // grad_ctx[d_s,:,:]
        const float* gz_d   = gz_bh + (long)doc * d;       // grad_z[d_s,:]

        // grad_k[s, tid]:
        //   = Σ_j grad_ctx[d_s, tid, j] * v_s[j]   (row tid of grad_ctx dot v)
        //   + grad_z[d_s, tid]
        float gk_val = gz_d[tid];
        const float* gctx_row = gctx_d + (long)tid * d;   // grad_ctx[d_s, tid, :]
        for (int j = 0; j < d; j++) {
            gk_val += gctx_row[j] * v_s[j];
        }
        gk_bh[(long)s * d + tid] = gk_val;

        // grad_v[s, tid]:
        //   = Σ_i grad_ctx[d_s, i, tid] * k_s[i]   (column tid of grad_ctx dot k)
        float gv_val = 0.0f;
        for (int i = 0; i < d; i++) {
            gv_val += gctx_d[(long)i * d + tid] * k_s[i];
        }
        gv_bh[(long)s * d + tid] = gv_val;

        __syncthreads();
    }
}


// =========================================================================
// Kernel 4: Backward — grad_q 계산
//   grad_q[t, i] = (1/den[t]) * (Σ_j go[t,j]*ctx[d_t,i,j]  - alpha[t]*z[d_t,i])
//   Grid:  (B * H,)
//   Block: (d_head,)
// =========================================================================
__global__ void doc_backward_q_kernel(
    const float* __restrict__ Q,              // [B*H, tgt_len, d]
    const float* __restrict__ context,        // [B*H, max_docs, d, d]
    const float* __restrict__ z,              // [B*H, max_docs, d]
    const float* __restrict__ out,            // [B*H, tgt_len, d]
    const float* __restrict__ den,            // [B*H, tgt_len]
    const float* __restrict__ grad_out,       // [B*H, tgt_len, d]
    const int*   __restrict__ tgt_doc_ids,    // [B, tgt_len]
    float*       __restrict__ grad_q,         // [B*H, tgt_len, d]
    int B, int H, int tgt_len, int d, int max_docs
) {
    const int bh  = blockIdx.x;
    const int b   = bh / H;
    const int i   = threadIdx.x;         // output dimension (grad_q의 d 차원)

    if (i >= d) return;

    const float* ctx   = context   + (long)bh * max_docs * d * d;
    const float* z_bh  = z         + (long)bh * max_docs * d;
    const float* out_bh     = out      + (long)bh * tgt_len * d;
    const float* den_bh     = den      + (long)bh * tgt_len;
    const float* go_bh      = grad_out + (long)bh * tgt_len * d;
    float*       gq_bh      = grad_q   + (long)bh * tgt_len * d;
    const int*   doc_t      = tgt_doc_ids + (long)b * tgt_len;

    extern __shared__ float go_s[];    // [d] floats

    for (int t = 0; t < tgt_len; t++) {
        // grad_out[t,:] → shared memory
        go_s[i] = go_bh[(long)t * d + i];
        __syncthreads();

        int doc = doc_t[t];
        const float* ctx_d = ctx  + (long)doc * d * d;
        const float* z_d   = z_bh + (long)doc * d;
        float inv_den = 1.0f / den_bh[t];          // den 이미 + eps 포함

        // alpha[t] = Σ_j go[t,j] * out[t,j]  (모든 스레드 동일하게 계산)
        // (d_head개 mul-add → 각 스레드가 따로 계산, 소규모 중복은 sync 대신 수용)
        float alpha = 0.0f;
        const float* out_t = out_bh + (long)t * d;
        for (int j = 0; j < d; j++) {
            alpha += go_s[j] * out_t[j];
        }

        // Σ_j go[t,j] * ctx[d_t, i, j]   (row i of context, dot grad_out)
        float ctx_go = 0.0f;
        const float* ctx_row = ctx_d + (long)i * d;  // context[d_t, i, :]
        for (int j = 0; j < d; j++) {
            ctx_go += ctx_row[j] * go_s[j];
        }

        gq_bh[(long)t * d + i] = inv_den * (ctx_go - alpha * z_d[i]);
        __syncthreads();
    }
}


// =========================================================================
// C++ launcher functions
// =========================================================================

// Phase 1: scatter KV → context, z
// Returns: (context [B*H, max_docs, d, d], z [B*H, max_docs, d])
std::pair<torch::Tensor, torch::Tensor> doc_scatter_kv_fwd(
    torch::Tensor K,          // [B, H, src_len, d]
    torch::Tensor V,          // [B, H, src_len, d]
    torch::Tensor src_doc_ids,// [B, src_len] int32
    int max_docs
) {
    TORCH_CHECK(K.is_cuda(), "K must be on CUDA");
    TORCH_CHECK(K.dtype() == torch::kFloat32, "K must be float32");

    const int B       = K.size(0);
    const int H       = K.size(1);
    const int src_len = K.size(2);
    const int d       = K.size(3);
    const int BH      = B * H;

    auto K_  = K.contiguous().view({BH, src_len, d});
    auto V_  = V.contiguous().view({BH, src_len, d});
    auto doc = src_doc_ids.contiguous().to(torch::kInt32);

    auto opts    = K_.options();
    auto context = torch::zeros({BH, max_docs, d, d}, opts);
    auto z       = torch::zeros({BH, max_docs, d},    opts);

    dim3 grid(BH);
    dim3 block(d);
    size_t smem = d * sizeof(float);    // v_s

    doc_scatter_kv_kernel<<<grid, block, smem>>>(
        K_.data_ptr<float>(), V_.data_ptr<float>(),
        doc.data_ptr<int>(),
        context.data_ptr<float>(), z.data_ptr<float>(),
        B, H, src_len, d, max_docs
    );
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "doc_scatter_kv_kernel launch failed");

    return {context, z};
}


// Phase 2: gather & normalize
// Returns: (out [B, H, tgt_len, d], den [B, H, tgt_len])
std::pair<torch::Tensor, torch::Tensor> doc_gather_query_fwd(
    torch::Tensor Q,              // [B, H, tgt_len, d]
    torch::Tensor context,        // [B*H, max_docs, d, d]
    torch::Tensor z,              // [B*H, max_docs, d]
    torch::Tensor tgt_doc_ids,    // [B, tgt_len] int32
    float eps
) {
    TORCH_CHECK(Q.is_cuda(), "Q must be on CUDA");
    TORCH_CHECK(Q.dtype() == torch::kFloat32, "Q must be float32");

    const int B       = Q.size(0);
    const int H       = Q.size(1);
    const int tgt_len = Q.size(2);
    const int d       = Q.size(3);
    const int BH      = B * H;

    auto Q_  = Q.contiguous().view({BH, tgt_len, d});
    auto doc = tgt_doc_ids.contiguous().to(torch::kInt32);

    auto out     = torch::empty({BH, tgt_len, d}, Q_.options());
    auto den_out = torch::empty({BH, tgt_len},    Q_.options());

    dim3 grid(BH);
    dim3 block(d);
    size_t smem = d * sizeof(float);    // q_s

    doc_gather_query_kernel<<<grid, block, smem>>>(
        Q_.data_ptr<float>(),
        context.data_ptr<float>(), z.data_ptr<float>(),
        doc.data_ptr<int>(),
        out.data_ptr<float>(), den_out.data_ptr<float>(),
        B, H, tgt_len, d, context.size(1), eps
    );
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "doc_gather_query_kernel launch failed");

    return {out.view({B, H, tgt_len, d}), den_out.view({B, H, tgt_len})};
}


// Backward: grad_k, grad_v
// Returns: (grad_k [B,H,src_len,d], grad_v [B,H,src_len,d])
std::pair<torch::Tensor, torch::Tensor> doc_backward_kv(
    torch::Tensor K,           // [B, H, src_len, d]
    torch::Tensor V,           // [B, H, src_len, d]
    torch::Tensor grad_ctx,    // [B*H, max_docs, d, d]
    torch::Tensor grad_zz,     // [B*H, max_docs, d]
    torch::Tensor src_doc_ids  // [B, src_len] int32
) {
    const int B       = K.size(0);
    const int H       = K.size(1);
    const int src_len = K.size(2);
    const int d       = K.size(3);
    const int BH      = B * H;
    const int max_docs = grad_ctx.size(1);

    auto K_  = K.contiguous().view({BH, src_len, d});
    auto V_  = V.contiguous().view({BH, src_len, d});
    auto doc = src_doc_ids.contiguous().to(torch::kInt32);

    auto gk = torch::empty_like(K_);
    auto gv = torch::empty_like(V_);

    dim3 grid(BH);
    dim3 block(d);
    size_t smem = 2 * d * sizeof(float);   // k_s + v_s

    doc_backward_kv_kernel<<<grid, block, smem>>>(
        K_.data_ptr<float>(), V_.data_ptr<float>(),
        grad_ctx.data_ptr<float>(), grad_zz.data_ptr<float>(),
        doc.data_ptr<int>(),
        gk.data_ptr<float>(), gv.data_ptr<float>(),
        B, H, src_len, d, max_docs
    );
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "doc_backward_kv_kernel launch failed");

    return {gk.view({B, H, src_len, d}), gv.view({B, H, src_len, d})};
}


// Backward: grad_q
torch::Tensor doc_backward_q(
    torch::Tensor Q,           // [B, H, tgt_len, d]
    torch::Tensor context,     // [B*H, max_docs, d, d]
    torch::Tensor z,           // [B*H, max_docs, d]
    torch::Tensor out,         // [B, H, tgt_len, d]
    torch::Tensor den,         // [B, H, tgt_len]
    torch::Tensor grad_out_t,  // [B, H, tgt_len, d]
    torch::Tensor tgt_doc_ids  // [B, tgt_len] int32
) {
    const int B       = Q.size(0);
    const int H       = Q.size(1);
    const int tgt_len = Q.size(2);
    const int d       = Q.size(3);
    const int BH      = B * H;
    const int max_docs = context.size(1);

    auto Q_   = Q.contiguous().view({BH, tgt_len, d});
    auto out_ = out.contiguous().view({BH, tgt_len, d});
    auto den_ = den.contiguous().view({BH, tgt_len});
    auto go_  = grad_out_t.contiguous().view({BH, tgt_len, d});
    auto doc  = tgt_doc_ids.contiguous().to(torch::kInt32);

    auto gq = torch::empty_like(Q_);

    dim3 grid(BH);
    dim3 block(d);
    size_t smem = d * sizeof(float);    // go_s

    doc_backward_q_kernel<<<grid, block, smem>>>(
        Q_.data_ptr<float>(),
        context.data_ptr<float>(), z.data_ptr<float>(),
        out_.data_ptr<float>(), den_.data_ptr<float>(),
        go_.data_ptr<float>(),
        doc.data_ptr<int>(),
        gq.data_ptr<float>(),
        B, H, tgt_len, d, max_docs
    );
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "doc_backward_q_kernel launch failed");

    return gq.view({B, H, tgt_len, d});
}
