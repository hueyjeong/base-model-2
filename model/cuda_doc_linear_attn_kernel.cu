/**
 * Document-Isolated Linear Cross-Attention CUDA Kernels — 최적화 버전
 *
 * 주요 최적화:
 *
 * [Scatter kernel v2]
 *   - 기존: src 위치마다 __syncthreads() 2회 = 2 * src_len = 3072회 sync
 *   - 개선: SCATTER_TILE=4 rows를 smem에 묶어 로드 → sync를 2*(src_len/4)=768회로 감소
 *   - __ldg() 로 K/V 읽기 (read-only L1 texture cache)
 *   - 내부 j 루프 #pragma unroll 4
 *
 * [Gather kernel v2]
 *   - 기존: context[d_t, i, j] 를 stride-d 글로벌 접근 → 비연속 read, 캐시 미스
 *   - 개선: doc가 바뀔 때만 context[d_t] (d×d=16KB)를 smem에 통째로 coalesced 로드
 *           이후 smem 에서 읽으므로 bandwidth 문제 해소
 *   - den 계산: warp shuffle reduce → 중복 연산 제거
 *
 * [Backward KV kernel v2]
 *   - Scatter 와 동일한 4x tiling 적용 (sync 횟수 75% 감소)
 *
 * [Backward Q kernel v2]
 *   - alpha = Σ_j go[j]*out[j]: 기존 64 스레드가 각자 d회 루프 중복 계산
 *     → warp shuffle reduce + 2-warp smem 합산으로 교체
 *   - context/z/go/out smem 캐시로 글로벌 메모리 접근 최소화
 */

#include <torch/extension.h>
#include <cuda_runtime.h>

// V-tile 크기: 한 sync 사이클에 처리하는 src 위치 수
#define SCATTER_TILE 4

// Tgt-tile: Phase 2 backward 에서 한 번에 처리하는 tgt 위치 수
// smem 추가: (TGT_TILE-1)*d*2 floats. TGT_TILE=4 → 추가 1.5KB (총 35KB, <48KB)
#define TGT_TILE 4

// warp-level reduce sum (single warp, 32 lanes)
__device__ __forceinline__ float warp_reduce_sum(float v) {
    v += __shfl_down_sync(0xffffffff, v, 16);
    v += __shfl_down_sync(0xffffffff, v, 8);
    v += __shfl_down_sync(0xffffffff, v, 4);
    v += __shfl_down_sync(0xffffffff, v, 2);
    v += __shfl_down_sync(0xffffffff, v, 1);
    return v;
}

// =========================================================================
// Kernel 1v2: Phase 1 — per-doc scatter (K^T V 누적), SCATTER_TILE 최적화
//
//   Grid:  (B * H,)
//   Block: (d_head,)
//
//   SCATTER_TILE개의 V 행을 smem에 묶어 로드한 뒤 일괄 처리.
//   → __syncthreads 횟수: 2 * ceil(src_len / SCATTER_TILE)  (기존: 2 * src_len)
//   smem: SCATTER_TILE * d floats = 4 * 64 * 4 = 1 KB
// =========================================================================
__global__ void doc_scatter_kv_kernel(
    const float* __restrict__ K,       // [B*H, src_len, d]
    const float* __restrict__ V,       // [B*H, src_len, d]
    const int*   __restrict__ doc_ids, // [B, src_len]
    float*       __restrict__ context, // [B*H, max_docs, d, d]  pre-zeroed
    float*       __restrict__ z,       // [B*H, max_docs, d]     pre-zeroed
    int B, int H, int src_len, int d, int max_docs
) {
    const int bh  = blockIdx.x;
    const int b   = bh / H;
    const int i   = threadIdx.x;

    if (i >= d) return;

    const float* K_bh  = K       + (long)bh * src_len * d;
    const float* V_bh  = V       + (long)bh * src_len * d;
    float*       ctx   = context + (long)bh * max_docs * d * d;
    float*       z_bh  = z       + (long)bh * max_docs * d;
    const int*   doc_b = doc_ids + (long)b  * src_len;

    // smem: SCATTER_TILE 행의 V 벡터  — v_tile[tt * d + j]
    extern __shared__ float smem_s[];

    for (int s = 0; s < src_len; s += SCATTER_TILE) {
        const int tile_end = min(s + SCATTER_TILE, src_len);
        const int tile_sz  = tile_end - s;

        // ── load V[s..s+tile_sz-1, i] into smem (coalesced) ──────────────
        for (int tt = 0; tt < tile_sz; tt++) {
            smem_s[tt * d + i] = __ldg(&V_bh[(long)(s + tt) * d + i]);
        }
        __syncthreads();   // sync 1: V tile 로드 완료

        // ── SCATTER_TILE 위치 일괄 처리 ──────────────────────────────────
        for (int tt = 0; tt < tile_sz; tt++) {
            float k_i   = __ldg(&K_bh[(long)(s + tt) * d + i]);
            int   doc   = doc_b[s + tt];
            float* ctx_row = ctx + ((long)doc * d + i) * d;

            #pragma unroll 4
            for (int j = 0; j < d; j++) {
                ctx_row[j] += k_i * smem_s[tt * d + j];
            }
            z_bh[(long)doc * d + i] += k_i;
        }
        __syncthreads();   // sync 2: 다음 tile 로드 전 smem 완료
    }
}


// =========================================================================
// Kernel 2v2: Phase 2 — per-tgt gather & normalize, smem context 캐시
//
//   Grid:  (B * H,)
//   Block: (d_head,)
//
//   개선:
//   - doc가 바뀔 때만 context[doc] (d×d=16KB)를 smem에 coalesced 로드
//     thread i가 row i 를 통째로 로드 (i*d ~ i*d+d-1: 연속 접근)
//   - 이후 context 접근은 smem → 글로벌 stride-d 캐시 미스 없음
//   den: 각 스레드가 r=0..d-1 전체를 루프하므로 full-sum 직접 확보 (warp reduce 불필요)
//   smem: d*d + d + d = d*(d+2) floats = 64*66*4 = 16.9 KB
// =========================================================================
__global__ void doc_gather_query_kernel(
    const float* __restrict__ Q,           // [B*H, tgt_len, d]
    const float* __restrict__ context,     // [B*H, max_docs, d, d]
    const float* __restrict__ z,           // [B*H, max_docs, d]
    const int*   __restrict__ tgt_doc_ids, // [B, tgt_len]
    float*       __restrict__ out,         // [B*H, tgt_len, d]
    float*       __restrict__ den_out,     // [B*H, tgt_len]
    int B, int H, int tgt_len, int d, int max_docs, float eps
) {
    const int bh  = blockIdx.x;
    const int b   = bh / H;
    const int i   = threadIdx.x;

    if (i >= d) return;

    const float* Q_bh   = Q       + (long)bh * tgt_len * d;
    const float* ctx    = context + (long)bh * max_docs * d * d;
    const float* z_bh   = z       + (long)bh * max_docs * d;
    float*       out_bh = out     + (long)bh * tgt_len * d;
    float*       den_bh = den_out + (long)bh * tgt_len;
    const int*   doc_t  = tgt_doc_ids + (long)b * tgt_len;

    // smem: ctx_sm[d*d] | z_sm[d] | q_sm[d]
    extern __shared__ float smem_g[];
    float* ctx_sm = smem_g;
    float* z_sm   = smem_g + d * d;
    float* q_sm   = smem_g + d * d + d;

    int cached_doc = -1;

    for (int t = 0; t < tgt_len; t++) {
        const int doc = doc_t[t];

        // ── doc 변경 시 context + z 를 smem 으로 coalesced 로드 ───────────
        if (doc != cached_doc) {
            const float* ctx_d = ctx  + (long)doc * d * d;
            const float* z_d   = z_bh + (long)doc * d;
            // thread i 가 row i 를 통째로 로드 (연속 읽기, coalesced)
            for (int jj = 0; jj < d; jj++) {
                ctx_sm[i * d + jj] = ctx_d[i * d + jj];
            }
            z_sm[i] = z_d[i];
            cached_doc = doc;
            __syncthreads();
        }

        // ── Q[t] 로드 ────────────────────────────────────────────────────
        q_sm[i] = __ldg(&Q_bh[(long)t * d + i]);
        __syncthreads();

        // ── numerator: column i of context (smem), denominator ──────────────
        // 각 스레드가 r=0..d-1 전체를 루프하므로 den은 full-sum으로 직접 확보
        float num = 0.0f;
        float den = 0.0f;
        #pragma unroll 4
        for (int r = 0; r < d; r++) {
            float q_r = q_sm[r];
            num += q_r * ctx_sm[r * d + i];  // smem, column i
            den += q_r * z_sm[r];
        }
        float den_eps = den + eps;

        out_bh[(long)t * d + i] = num / den_eps;
        if (i == 0) den_bh[t] = den_eps;
        __syncthreads();
    }
}


// =========================================================================
// Kernel 3v2: Backward — grad_k, grad_v, SCATTER_TILE 최적화
//
//   Scatter kernel 과 동일한 tiling 구조 적용.
//   smem: [K tile: TILE*d] + [V tile: TILE*d] = 2 * SCATTER_TILE * d floats
// =========================================================================
__global__ void doc_backward_kv_kernel(
    const float* __restrict__ K,           // [B*H, src_len, d]
    const float* __restrict__ V,           // [B*H, src_len, d]
    const float* __restrict__ grad_ctx,    // [B*H, max_docs, d, d]
    const float* __restrict__ grad_zz,     // [B*H, max_docs, d]
    const int*   __restrict__ src_doc_ids, // [B, src_len]
    float*       __restrict__ grad_k,      // [B*H, src_len, d]
    float*       __restrict__ grad_v,      // [B*H, src_len, d]
    int B, int H, int src_len, int d, int max_docs
) {
    const int bh  = blockIdx.x;
    const int b   = bh / H;
    const int tid = threadIdx.x;

    if (tid >= d) return;

    const float* K_bh   = K        + (long)bh * src_len * d;
    const float* V_bh   = V        + (long)bh * src_len * d;
    const float* gctx   = grad_ctx + (long)bh * max_docs * d * d;
    const float* gz_bh  = grad_zz  + (long)bh * max_docs * d;
    float*       gk_bh  = grad_k   + (long)bh * src_len * d;
    float*       gv_bh  = grad_v   + (long)bh * src_len * d;
    const int*   doc_b  = src_doc_ids + (long)b * src_len;

    // smem: k_tile[SCATTER_TILE*d] | v_tile[SCATTER_TILE*d]
    extern __shared__ float smem_bkv[];
    float* k_tile = smem_bkv;
    float* v_tile = smem_bkv + SCATTER_TILE * d;

    for (int s = 0; s < src_len; s += SCATTER_TILE) {
        const int tile_end = min(s + SCATTER_TILE, src_len);
        const int tile_sz  = tile_end - s;

        for (int tt = 0; tt < tile_sz; tt++) {
            k_tile[tt * d + tid] = __ldg(&K_bh[(long)(s + tt) * d + tid]);
            v_tile[tt * d + tid] = __ldg(&V_bh[(long)(s + tt) * d + tid]);
        }
        __syncthreads();

        for (int tt = 0; tt < tile_sz; tt++) {
            const int   doc      = doc_b[s + tt];
            const float* gctx_d  = gctx  + (long)doc * d * d;
            const float* gz_d    = gz_bh + (long)doc * d;

            // grad_k[s+tt, tid] = Σ_j gctx[doc, tid, j] * v[s+tt, j] + gz[doc, tid]
            float gk_val = gz_d[tid];
            const float* gctx_row = gctx_d + (long)tid * d;
            #pragma unroll 4
            for (int j = 0; j < d; j++) {
                gk_val += gctx_row[j] * v_tile[tt * d + j];
            }
            gk_bh[(long)(s + tt) * d + tid] = gk_val;

            // grad_v[s+tt, tid] = Σ_r gctx[doc, r, tid] * k[s+tt, r]
            float gv_val = 0.0f;
            #pragma unroll 4
            for (int r = 0; r < d; r++) {
                gv_val += gctx_d[(long)r * d + tid] * k_tile[tt * d + r];
            }
            gv_bh[(long)(s + tt) * d + tid] = gv_val;
        }
        __syncthreads();
    }
}


// =========================================================================
// Kernel 4v2: Backward — grad_q, smem context 캐시 + warp-reduce alpha
//
//   개선:
//   - context/z smem 캐시 (doc 변경 시만 로드)
//   - alpha = Σ_j go[j]*out[j]: 기존 64스레드 각자 d회 루프 중복
//     → warp shuffle reduce + 2-warp smem 합산
//   smem: d*d + d + d + d + n_warps floats ≈ 16.8 KB
// =========================================================================
__global__ void doc_backward_q_kernel(
    const float* __restrict__ Q,           // [B*H, tgt_len, d]
    const float* __restrict__ context,     // [B*H, max_docs, d, d]
    const float* __restrict__ z,           // [B*H, max_docs, d]
    const float* __restrict__ out,         // [B*H, tgt_len, d]
    const float* __restrict__ den,         // [B*H, tgt_len]
    const float* __restrict__ grad_out,    // [B*H, tgt_len, d]
    const int*   __restrict__ tgt_doc_ids, // [B, tgt_len]
    float*       __restrict__ grad_q,      // [B*H, tgt_len, d]
    int B, int H, int tgt_len, int d, int max_docs
) {
    const int bh  = blockIdx.x;
    const int b   = bh / H;
    const int i   = threadIdx.x;

    if (i >= d) return;

    const float* ctx  = context  + (long)bh * max_docs * d * d;
    const float* z_bh = z        + (long)bh * max_docs * d;
    const float* out_ = out      + (long)bh * tgt_len * d;
    const float* den_ = den      + (long)bh * tgt_len;
    const float* go_  = grad_out + (long)bh * tgt_len * d;
    float*       gq_  = grad_q   + (long)bh * tgt_len * d;
    const int*   doc_t = tgt_doc_ids + (long)b * tgt_len;

    // smem: ctx_sm[d*d] | z_sm[d] | go_sm[d] | out_sm[d] | warp_buf[n_warps]
    const int n_warps = (d + 31) / 32;
    extern __shared__ float smem_bq[];
    float* ctx_sm   = smem_bq;
    float* z_sm     = smem_bq + d * d;
    float* go_sm    = smem_bq + d * d + d;
    float* out_sm   = smem_bq + d * d + d + d;
    float* warp_buf = smem_bq + d * d + d + d + d;

    int cached_doc = -1;

    for (int t = 0; t < tgt_len; t++) {
        const int doc = doc_t[t];

        // ── context/z smem 캐시 ───────────────────────────────────────────
        if (doc != cached_doc) {
            const float* ctx_d = ctx  + (long)doc * d * d;
            const float* z_d   = z_bh + (long)doc * d;
            for (int jj = 0; jj < d; jj++) {
                ctx_sm[i * d + jj] = ctx_d[i * d + jj];
            }
            z_sm[i] = z_d[i];
            cached_doc = doc;
            __syncthreads();
        }

        // ── grad_out[t], out[t] 로드 ─────────────────────────────────────
        go_sm[i]  = __ldg(&go_[(long)t * d + i]);
        out_sm[i] = __ldg(&out_[(long)t * d + i]);
        __syncthreads();

        float inv_den = 1.0f / den_[(long)t];

        // ── alpha = Σ_j go[j]*out[j]: warp reduce ────────────────────────
        float alpha_partial = go_sm[i] * out_sm[i];
        float alpha_warp    = warp_reduce_sum(alpha_partial);
        const int warp_id   = i >> 5;
        const int lane_id   = i & 31;
        if (lane_id == 0) warp_buf[warp_id] = alpha_warp;
        __syncthreads();

        float alpha = 0.0f;
        for (int w = 0; w < n_warps; w++) alpha += warp_buf[w];

        // ── Σ_j go[j] * ctx[doc, i, j] (row i from smem) ─────────────────
        float ctx_go = 0.0f;
        const float* ctx_row = ctx_sm + (long)i * d;
        #pragma unroll 4
        for (int jj = 0; jj < d; jj++) {
            ctx_go += ctx_row[jj] * go_sm[jj];
        }

        gq_[(long)t * d + i] = inv_den * (ctx_go - alpha * z_sm[i]);
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
    size_t smem = (size_t)SCATTER_TILE * d * sizeof(float);    // V tile

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
    // ctx_sm[d*d] + z_sm[d] + q_sm[d]
    size_t smem = ((size_t)d * d + d + d) * sizeof(float);

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
    // k_tile[TILE*d] + v_tile[TILE*d]
    size_t smem = (size_t)2 * SCATTER_TILE * d * sizeof(float);

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

    const int n_warps_bq = (d + 31) / 32;
    dim3 grid(BH);
    dim3 block(d);
    // ctx_sm[d*d] + z_sm[d] + go_sm[d] + out_sm[d] + warp_buf[n_warps]
    size_t smem = ((size_t)d * d + d + d + d + n_warps_bq) * sizeof(float);

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


// #########################################################################
//  FUSED KERNELS — Grid=(B*H*max_docs,): 문서별 병렬, context smem 전용
//
//  핵심 개선:
//    1. 블록 720개 (기존 24개 → 30x 병렬) → SM 점유율 대폭 증가 → GPU watt ↑
//    2. context가 smem에만 존재 → 글로벌 메모리 11.3MB write+read 제거
//    3. Autograd 저장 텐서 감소 (context/z 저장 불필요 → 11.5MB/layer 절약)
//    4. Backward도 단일 커널: 3개 phase를 한 블록에서 순차 실행
// #########################################################################

// Binary search helpers for sorted doc_ids
__device__ __forceinline__ int d_lower_bound(const int* arr, int lo, int hi, int val) {
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        if (__ldg(&arr[mid]) < val) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}

__device__ __forceinline__ int d_upper_bound(const int* arr, int lo, int hi, int val) {
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        if (__ldg(&arr[mid]) <= val) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}

// =========================================================================
// Fused Forward Kernel
//   Grid:  (B * H * max_docs,)
//   Block: (d,)
//   각 블록이 하나의 (b, h, doc)을 전담.
//   Phase 1: scatter K,V → smem ctx/z
//   Phase 2: gather from smem ctx → out, den
//   context/z가 글로벌 메모리에 전혀 안 씀.
//   smem: d*d + d + SCATTER_TILE*d floats
// =========================================================================
__global__ void doc_fused_fwd_kernel(
    const float* __restrict__ Q,           // [B*H, tgt_len, d]
    const float* __restrict__ K,           // [B*H, src_len, d]
    const float* __restrict__ V,           // [B*H, src_len, d]
    const int*   __restrict__ src_doc_ids, // [B, src_len]  sorted
    const int*   __restrict__ tgt_doc_ids, // [B, tgt_len]  sorted
    float*       __restrict__ out,         // [B*H, tgt_len, d]
    float*       __restrict__ den_out,     // [B*H, tgt_len]
    int B, int H, int src_len, int tgt_len, int d, int max_docs, float eps
) {
    const int block_id = blockIdx.x;          // 0 .. B*H*max_docs - 1
    const int doc = block_id % max_docs;
    const int bh  = block_id / max_docs;
    const int b   = bh / H;
    const int i   = threadIdx.x;

    if (i >= d) return;

    const int* src_doc_b = src_doc_ids + (long)b * src_len;
    const int* tgt_doc_b = tgt_doc_ids + (long)b * tgt_len;

    // 이진 탐색으로 이 doc의 src/tgt 범위 결정 (all threads uniform → no divergence)
    const int src_s = d_lower_bound(src_doc_b, 0, src_len, doc);
    const int src_e = d_upper_bound(src_doc_b, src_s, src_len, doc);
    const int tgt_s = d_lower_bound(tgt_doc_b, 0, tgt_len, doc);
    const int tgt_e = d_upper_bound(tgt_doc_b, tgt_s, tgt_len, doc);

    if (src_s >= src_e && tgt_s >= tgt_e) return;

    // smem layout: ctx_sm[d*d] | z_sm[d] | vtile[SCATTER_TILE*d]
    extern __shared__ float smem_fwd[];
    float* ctx_sm = smem_fwd;
    float* z_sm   = smem_fwd + d * d;
    float* vtile  = smem_fwd + d * d + d;   // reused as q_sm in phase 2

    const float* K_bh = K + (long)bh * src_len * d;
    const float* V_bh = V + (long)bh * src_len * d;

    // ── Phase 1: scatter K^T·V → smem ctx ────────────────────────────
    #pragma unroll 4
    for (int j = 0; j < d; j++) ctx_sm[i * d + j] = 0.0f;
    z_sm[i] = 0.0f;
    __syncthreads();

    for (int s = src_s; s < src_e; s += SCATTER_TILE) {
        const int tile_end = min(s + SCATTER_TILE, src_e);
        const int tile_sz  = tile_end - s;

        for (int tt = 0; tt < tile_sz; tt++)
            vtile[tt * d + i] = __ldg(&V_bh[(long)(s + tt) * d + i]);
        __syncthreads();

        for (int tt = 0; tt < tile_sz; tt++) {
            float k_i = __ldg(&K_bh[(long)(s + tt) * d + i]);
            float* ctx_row = ctx_sm + i * d;
            #pragma unroll 4
            for (int j = 0; j < d; j++)
                ctx_row[j] += k_i * vtile[tt * d + j];
            z_sm[i] += k_i;
        }
        __syncthreads();
    }

    // ── Phase 2: gather from smem ctx ────────────────────────────────
    const float* Q_bh   = Q       + (long)bh * tgt_len * d;
    float*       out_bh = out     + (long)bh * tgt_len * d;
    float*       den_bh = den_out + (long)bh * tgt_len;

    for (int t = tgt_s; t < tgt_e; t++) {
        vtile[i] = __ldg(&Q_bh[(long)t * d + i]);   // q_sm
        __syncthreads();

        float num = 0.0f, den = 0.0f;
        #pragma unroll 4
        for (int r = 0; r < d; r++) {
            float q_r = vtile[r];
            num += q_r * ctx_sm[r * d + i];
            den += q_r * z_sm[r];
        }
        float den_eps = den + eps;
        out_bh[(long)t * d + i] = num / den_eps;
        if (i == 0) den_bh[t] = den_eps;
        __syncthreads();
    }
}

// =========================================================================
// Fused Backward Kernel
//   Grid:  (B * H * max_docs,)
//   Block: (d,)
//   Phase 1: re-scatter K,V → smem ctx/z
//   Phase 2: tgt 위치 TGT_TILE개씩 일괄 처리 ─ grad_q 계산 + gctx/gz 누적
//            → __syncthreads 횟수: 2*ceil(N/TGT_TILE) (기존: 2*N)
//   Phase 3: src 위치 순회 — grad_k, grad_v 계산
//   smem: 2*d*d + (2+TGT_TILE*2)*d + TGT_TILE + n_warps*TGT_TILE floats
// =========================================================================
__global__ void doc_fused_bwd_kernel(
    const float* __restrict__ Q,           // [B*H, tgt_len, d]
    const float* __restrict__ K,           // [B*H, src_len, d]
    const float* __restrict__ V,           // [B*H, src_len, d]
    const float* __restrict__ fwd_out,     // [B*H, tgt_len, d]
    const float* __restrict__ fwd_den,     // [B*H, tgt_len]
    const float* __restrict__ grad_out,    // [B*H, tgt_len, d]
    const int*   __restrict__ src_doc_ids, // [B, src_len]
    const int*   __restrict__ tgt_doc_ids, // [B, tgt_len]
    float*       __restrict__ grad_q,      // [B*H, tgt_len, d]
    float*       __restrict__ grad_k,      // [B*H, src_len, d]
    float*       __restrict__ grad_v,      // [B*H, src_len, d]
    int B, int H, int src_len, int tgt_len, int d, int max_docs
) {
    const int block_id = blockIdx.x;
    const int doc = block_id % max_docs;
    const int bh  = block_id / max_docs;
    const int b   = bh / H;
    const int i   = threadIdx.x;

    if (i >= d) return;

    const int* src_doc_b = src_doc_ids + (long)b * src_len;
    const int* tgt_doc_b = tgt_doc_ids + (long)b * tgt_len;

    const int src_s = d_lower_bound(src_doc_b, 0, src_len, doc);
    const int src_e = d_upper_bound(src_doc_b, src_s, src_len, doc);
    const int tgt_s = d_lower_bound(tgt_doc_b, 0, tgt_len, doc);
    const int tgt_e = d_upper_bound(tgt_doc_b, tgt_s, tgt_len, doc);

    if (src_s >= src_e && tgt_s >= tgt_e) return;

    // smem layout:
    //   ctx_sm [d*d]  | z_sm [d]  | gctx_sm [d*d] | gz_sm [d]
    //   | go_tile [TGT_TILE*d] | out_tile [TGT_TILE*d]
    //   | alpha_buf [TGT_TILE] | warp_buf [n_warps*TGT_TILE]
    const int n_warps = (d + 31) / 32;
    extern __shared__ float smem_bwd[];
    float* ctx_sm    = smem_bwd;
    float* z_sm      = smem_bwd + d * d;
    float* gctx_sm   = smem_bwd + d * d + d;
    float* gz_sm     = smem_bwd + 2 * d * d + d;
    float* go_tile   = smem_bwd + 2 * d * d + 2 * d;
    float* out_tile  = smem_bwd + 2 * d * d + 2 * d + TGT_TILE * d;
    float* alpha_buf = smem_bwd + 2 * d * d + 2 * d + 2 * TGT_TILE * d;
    float* warp_buf  = smem_bwd + 2 * d * d + 2 * d + 2 * TGT_TILE * d + TGT_TILE;

    const float* K_bh  = K   + (long)bh * src_len * d;
    const float* V_bh  = V   + (long)bh * src_len * d;

    // ═══════════ Phase 1: Re-scatter ctx, z to smem ═══════════════════
    #pragma unroll 4
    for (int j = 0; j < d; j++) ctx_sm[i * d + j] = 0.0f;
    z_sm[i] = 0.0f;
    __syncthreads();

    // V tile은 gctx_sm 영역을 임시 사용 (phase 2에서 초기화됨)
    float* v_tile_p1 = gctx_sm;

    for (int s = src_s; s < src_e; s += SCATTER_TILE) {
        const int tile_end = min(s + SCATTER_TILE, src_e);
        const int tile_sz  = tile_end - s;

        for (int tt = 0; tt < tile_sz; tt++)
            v_tile_p1[tt * d + i] = __ldg(&V_bh[(long)(s + tt) * d + i]);
        __syncthreads();

        for (int tt = 0; tt < tile_sz; tt++) {
            float k_i = __ldg(&K_bh[(long)(s + tt) * d + i]);
            float* ctx_row = ctx_sm + i * d;
            #pragma unroll 4
            for (int j = 0; j < d; j++)
                ctx_row[j] += k_i * v_tile_p1[tt * d + j];
            z_sm[i] += k_i;
        }
        __syncthreads();
    }

    // ═══════════ Phase 2: grad_q + accumulate gctx, gz (TGT_TILE tiling) ═
    #pragma unroll 4
    for (int j = 0; j < d; j++) gctx_sm[i * d + j] = 0.0f;
    gz_sm[i] = 0.0f;
    __syncthreads();

    const float* Q_bh   = Q        + (long)bh * tgt_len * d;
    const float* out_bh = fwd_out  + (long)bh * tgt_len * d;
    const float* den_bh = fwd_den  + (long)bh * tgt_len;
    const float* go_bh  = grad_out + (long)bh * tgt_len * d;
    float*       gq_bh  = grad_q   + (long)bh * tgt_len * d;

    for (int t = tgt_s; t < tgt_e; t += TGT_TILE) {
        const int tile_end = min(t + TGT_TILE, tgt_e);
        const int tile_sz  = tile_end - t;

        // ── Load TGT_TILE go and out vectors into smem ──
        for (int tt = 0; tt < tile_sz; tt++) {
            go_tile[tt * d + i]  = __ldg(&go_bh[(long)(t + tt) * d + i]);
            out_tile[tt * d + i] = __ldg(&out_bh[(long)(t + tt) * d + i]);
        }
        __syncthreads();

        // ── Process each tgt position in the tile (no sync between positions) ──
        for (int tt = 0; tt < tile_sz; tt++) {
            float inv_den = 1.0f / __ldg(&den_bh[t + tt]);
            float q_i     = __ldg(&Q_bh[(long)(t + tt) * d + i]);

            // alpha = Σ_j go[j]*out[j] via warp reduce
            float alpha_part = go_tile[tt * d + i] * out_tile[tt * d + i];
            float alpha_warp = warp_reduce_sum(alpha_part);
            int warp_id = i >> 5;
            int lane_id = i & 31;
            if (lane_id == 0) warp_buf[tt * n_warps + warp_id] = alpha_warp;
        }
        __syncthreads();

        for (int tt = 0; tt < tile_sz; tt++) {
            float alpha = 0.0f;
            for (int w = 0; w < n_warps; w++) alpha += warp_buf[tt * n_warps + w];

            float inv_den = 1.0f / __ldg(&den_bh[t + tt]);
            float q_i     = __ldg(&Q_bh[(long)(t + tt) * d + i]);

            // grad_q[t+tt, i]
            float ctx_go = 0.0f;
            const float* ctx_row = ctx_sm + i * d;
            #pragma unroll 4
            for (int j = 0; j < d; j++)
                ctx_go += ctx_row[j] * go_tile[tt * d + j];
            gq_bh[(long)(t + tt) * d + i] = inv_den * (ctx_go - alpha * z_sm[i]);

            // gctx[i,j] += q_bar_i * go[j]
            float q_bar_i = q_i * inv_den;
            float* gctx_row = gctx_sm + i * d;
            #pragma unroll 4
            for (int j = 0; j < d; j++)
                gctx_row[j] += q_bar_i * go_tile[tt * d + j];

            // gz[i] += -q_bar_i * alpha
            gz_sm[i] += -q_bar_i * alpha;
        }
        __syncthreads();
    }

    // ═══════════ Phase 3: grad_k, grad_v from gctx, gz ═══════════════
    // ctx_sm 영역을 K/V tile 용도로 재활용 (더 이상 사용 안 함)
    float* k_tile_p3 = ctx_sm;
    float* v_tile_p3 = ctx_sm + SCATTER_TILE * d;

    float* gk_bh = grad_k + (long)bh * src_len * d;
    float* gv_bh = grad_v + (long)bh * src_len * d;

    for (int s = src_s; s < src_e; s += SCATTER_TILE) {
        const int tile_end = min(s + SCATTER_TILE, src_e);
        const int tile_sz  = tile_end - s;

        for (int tt = 0; tt < tile_sz; tt++) {
            k_tile_p3[tt * d + i] = __ldg(&K_bh[(long)(s + tt) * d + i]);
            v_tile_p3[tt * d + i] = __ldg(&V_bh[(long)(s + tt) * d + i]);
        }
        __syncthreads();

        for (int tt = 0; tt < tile_sz; tt++) {
            // grad_k[s+tt, i] = Σ_j gctx[i,j]*v[s+tt,j] + gz[i]
            float gk_val = gz_sm[i];
            const float* gctx_row = gctx_sm + i * d;
            #pragma unroll 4
            for (int j = 0; j < d; j++)
                gk_val += gctx_row[j] * v_tile_p3[tt * d + j];
            gk_bh[(long)(s + tt) * d + i] = gk_val;

            // grad_v[s+tt, i] = Σ_r gctx[r,i]*k[s+tt,r]
            float gv_val = 0.0f;
            #pragma unroll 4
            for (int r = 0; r < d; r++)
                gv_val += gctx_sm[(long)r * d + i] * k_tile_p3[tt * d + r];
            gv_bh[(long)(s + tt) * d + i] = gv_val;
        }
        __syncthreads();
    }
}


// =========================================================================
// Fused C++ launchers
// =========================================================================

std::pair<torch::Tensor, torch::Tensor> doc_fused_forward(
    torch::Tensor Q,            // [B, H, tgt_len, d]  (any float dtype)
    torch::Tensor K,            // [B, H, src_len, d]  (any float dtype)
    torch::Tensor V,            // [B, H, src_len, d]  (any float dtype)
    torch::Tensor src_doc_ids,  // [B, src_len] int32
    torch::Tensor tgt_doc_ids,  // [B, tgt_len] int32
    int max_docs,
    float eps
) {
    TORCH_CHECK(Q.is_cuda(), "Q must be on CUDA");

    const int B       = Q.size(0);
    const int H       = Q.size(1);
    const int tgt_len = Q.size(2);
    const int d       = Q.size(3);
    const int src_len = K.size(2);
    const int BH      = B * H;

    // 내부 float32 변환 — Python 측 .float() 제거, autograd에 원본 dtype 저장 허용
    auto Q_ = Q.to(torch::kFloat32).contiguous().view({BH, tgt_len, d});
    auto K_ = K.to(torch::kFloat32).contiguous().view({BH, src_len, d});
    auto V_ = V.to(torch::kFloat32).contiguous().view({BH, src_len, d});
    auto src_doc = src_doc_ids.contiguous().to(torch::kInt32);
    auto tgt_doc = tgt_doc_ids.contiguous().to(torch::kInt32);

    auto out     = torch::empty({BH, tgt_len, d}, Q_.options());
    auto den_out = torch::empty({BH, tgt_len},    Q_.options());

    const int num_blocks = BH * max_docs;
    dim3 grid(num_blocks);
    dim3 block(d);
    // ctx_sm[d*d] + z_sm[d] + vtile[SCATTER_TILE*d]
    size_t smem = ((size_t)d * d + d + SCATTER_TILE * d) * sizeof(float);

    doc_fused_fwd_kernel<<<grid, block, smem>>>(
        Q_.data_ptr<float>(), K_.data_ptr<float>(), V_.data_ptr<float>(),
        src_doc.data_ptr<int>(), tgt_doc.data_ptr<int>(),
        out.data_ptr<float>(), den_out.data_ptr<float>(),
        B, H, src_len, tgt_len, d, max_docs, eps
    );
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "doc_fused_fwd_kernel launch failed");

    return {out.view({B, H, tgt_len, d}), den_out.view({B, H, tgt_len})};
}


std::vector<torch::Tensor> doc_fused_backward(
    torch::Tensor Q,            // [B, H, tgt_len, d]  (any float dtype)
    torch::Tensor K,            // [B, H, src_len, d]  (any float dtype)
    torch::Tensor V,            // [B, H, src_len, d]  (any float dtype)
    torch::Tensor fwd_out,      // [B, H, tgt_len, d]
    torch::Tensor fwd_den,      // [B, H, tgt_len]
    torch::Tensor grad_out_t,   // [B, H, tgt_len, d]
    torch::Tensor src_doc_ids,  // [B, src_len] int32
    torch::Tensor tgt_doc_ids,  // [B, tgt_len] int32
    int max_docs
) {
    const int B       = Q.size(0);
    const int H       = Q.size(1);
    const int tgt_len = Q.size(2);
    const int d       = Q.size(3);
    const int src_len = K.size(2);
    const int BH      = B * H;

    // 내부 float32 변환 — backward에서도 bf16 입력 허용
    auto Q_   = Q.to(torch::kFloat32).contiguous().view({BH, tgt_len, d});
    auto K_   = K.to(torch::kFloat32).contiguous().view({BH, src_len, d});
    auto V_   = V.to(torch::kFloat32).contiguous().view({BH, src_len, d});
    auto out_ = fwd_out.to(torch::kFloat32).contiguous().view({BH, tgt_len, d});
    auto den_ = fwd_den.to(torch::kFloat32).contiguous().view({BH, tgt_len});
    auto go_  = grad_out_t.to(torch::kFloat32).contiguous().view({BH, tgt_len, d});
    auto src_doc = src_doc_ids.contiguous().to(torch::kInt32);
    auto tgt_doc = tgt_doc_ids.contiguous().to(torch::kInt32);

    auto grad_q = torch::empty_like(Q_);
    auto grad_k = torch::empty_like(K_);
    auto grad_v = torch::empty_like(V_);

    const int num_blocks = BH * max_docs;
    const int n_warps = (d + 31) / 32;
    dim3 grid(num_blocks);
    dim3 block(d);
    // ctx[d*d] + z[d] + gctx[d*d] + gz[d] + go_tile[TGT_TILE*d] + out_tile[TGT_TILE*d] + alpha_buf[TGT_TILE] + warp_buf[n_warps*TGT_TILE]
    size_t smem = ((size_t)2 * d * d + 2 * d + 2 * TGT_TILE * d + TGT_TILE + n_warps * TGT_TILE) * sizeof(float);

    doc_fused_bwd_kernel<<<grid, block, smem>>>(
        Q_.data_ptr<float>(), K_.data_ptr<float>(), V_.data_ptr<float>(),
        out_.data_ptr<float>(), den_.data_ptr<float>(),
        go_.data_ptr<float>(),
        src_doc.data_ptr<int>(), tgt_doc.data_ptr<int>(),
        grad_q.data_ptr<float>(), grad_k.data_ptr<float>(), grad_v.data_ptr<float>(),
        B, H, src_len, tgt_len, d, max_docs
    );
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "doc_fused_bwd_kernel launch failed");

    return {grad_q.view({B, H, tgt_len, d}),
            grad_k.view({B, H, src_len, d}),
            grad_v.view({B, H, src_len, d})};
}
