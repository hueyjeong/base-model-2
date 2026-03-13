/**
 * RWKV-6 Recurrent Scan CUDA 커널 — Forward + Backward
 *
 * fla 라이브러리 대체: BF16 직접 입력, 내부 FP32 연산, BF16 출력.
 * 모든 dtype 변환/transpose 제거.
 *
 * 알고리즘 (per head):
 *   Forward:
 *     state[K,V] = 0
 *     for t in 0..T:
 *       kv = k[t] ⊗ v[t]
 *       o[t,v] = sum_k (state[k,v] + u[k]*kv[k,v]) * r[t,k]
 *       state[k,v] = exp(w[t,k]) * state[k,v] + kv[k,v]
 *
 *   Backward (reverse scan):
 *     grad_state[K,V] = 0
 *     for t in T-1..0:
 *       grad_r[t,k] = sum_v (state[k,v] + u[k]*kv[k,v]) * grad_o[t,v]
 *       grad_u[k] += sum_v kv[k,v] * r[t,k] * grad_o[t,v]
 *       grad_kv[k,v] = u[k] * r[t,k] * grad_o[t,v] + grad_state[k,v]
 *       grad_k[t,k] = sum_v grad_kv[k,v] * v[t,v]
 *       grad_v[t,v] = sum_k grad_kv[k,v] * k[t,k]
 *       grad_w[t,k] = sum_v exp(w[t,k]) * state[k,v] * grad_state[k,v]
 *       grad_state = exp(w[t,k]) * (grad_state + r[t,k] * grad_o[t])
 *
 * CUDA 설계:
 *   Grid: (B, H) — 배치 × 헤드 완전 병렬화
 *   Block: (K, V_TILE) — state matrix 타일 처리
 *   Shared memory: state[K][V] (K=32, V=32 → 4KB float)
 */
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

namespace {

// 128M config: K=32, V=32 (headdim=32)
constexpr int MAX_DIM = 64;  // 최대 headdim 지원

/**
 * Forward 커널: 1 block = 1 (batch, head)
 *
 * Input layout: (B, T, H, D) — BF16 또는 FP32
 * Output layout: (B, T, H, D) — 입력과 동일 dtype
 * u: (H, D) — FP32
 *
 * shared memory에 state[K][V] 유지 (float32)
 */
template <int BLOCK_K>
__global__ void rwkv6_forward_kernel(
    const float* __restrict__ r,   // (B, T, H, K)
    const float* __restrict__ k,   // (B, T, H, K)
    const float* __restrict__ v,   // (B, T, H, K)
    const float* __restrict__ w,   // (B, T, H, K)
    const float* __restrict__ u,   // (H, K)
    float* __restrict__ out,       // (B, T, H, K)
    float* __restrict__ state_out, // (B, H, K, K) — backward용 state 저장 (T+1 frames)
    int B, int T, int H, int K
) {
    const int b = blockIdx.x;
    const int h = blockIdx.y;
    if (b >= B || h >= H) return;

    const int tid = threadIdx.x;  // 0..K-1: 각 스레드가 state의 1행(k=tid) 담당

    if (tid >= K) return;

    // state[tid][v] — 각 스레드가 K_dim의 한 행 전체(V_dim) 담당
    // register에 state 유지 (K=32일 때 32 floats = 128B per thread)
    float s[MAX_DIM];
    #pragma unroll
    for (int vi = 0; vi < K; vi++) s[vi] = 0.0f;

    // u[h, tid] — 이 head의 이 k-dim에 대한 u 파라미터
    const float u_val = u[h * K + tid];

    // state_out 초기 상태 저장 (t=0 이전)
    // state_out: (B, T+1, H, K, K) — 각 시점의 state 보관
    const int state_stride = H * K * K;
    float* state_base = state_out + (b * (T + 1) * state_stride) + h * K * K;

    // t=0 이전 상태 (zeros)
    #pragma unroll
    for (int vi = 0; vi < K; vi++) {
        state_base[tid * K + vi] = 0.0f;
    }

    // 입력 stride: (B, T, H, K) → offset = b*T*H*K + t*H*K + h*K + k
    const int bhk_stride = T * H * K;
    const int hk_stride = H * K;
    const int base_bh = b * bhk_stride + h * K;

    for (int t = 0; t < T; t++) {
        const int t_off = base_bh + t * hk_stride;

        // 이 스레드의 r, k, w 값 로드
        const float r_val = r[t_off + tid];
        const float k_val = k[t_off + tid];
        const float w_val = w[t_off + tid];
        const float ew = expf(w_val);  // exp(w) ∈ (0, 1)

        // v[t]는 모든 스레드가 필요 → shared memory로 broadcast
        __shared__ float v_shared[MAX_DIM];
        __shared__ float r_shared[MAX_DIM];
        if (tid < K) {
            v_shared[tid] = v[t_off + tid];
            r_shared[tid] = r[t_off + tid];
        }
        __syncthreads();

        // kv = k_val * v[vi] (outer product의 한 행)
        // o[t, vi] = sum_k (s[k][vi] + u[k]*k[k]*v[vi]) * r[k]
        // → 각 스레드는 자기 행(k=tid)의 기여분을 계산, 모든 k에 대해 reduce 필요

        // 전략: 각 스레드가 (s[tid][vi] + u*k*v[vi]) * r[tid]를 계산
        //        → vi 축 전체에 대해 → 그 결과를 k 축으로 warp reduce

        // output 계산: o[vi] = sum_k (s[k][vi] + u[k]*k[k]*v[vi]) * r[k]
        // 각 스레드(k=tid)가 v_dim 전체에 대해 기여분 계산
        float contrib[MAX_DIM];
        #pragma unroll
        for (int vi = 0; vi < K; vi++) {
            contrib[vi] = (s[vi] + u_val * k_val * v_shared[vi]) * r_val;
        }

        // warp-level reduce (k 축 합산)
        // K=32이면 1 warp. K>32이면 shared memory reduce 필요.
        // K ≤ 32: warp shuffle로 충분
        __shared__ float out_accum[MAX_DIM];
        if (tid == 0) {
            #pragma unroll
            for (int vi = 0; vi < K; vi++) out_accum[vi] = 0.0f;
        }
        __syncthreads();

        // atomicAdd로 k 축 합산 (K=32 → 32 threads atomic, 충돌 낮음)
        for (int vi = 0; vi < K; vi++) {
            atomicAdd(&out_accum[vi], contrib[vi]);
        }
        __syncthreads();

        // 출력 저장 (tid=0..K-1 분담)
        if (tid < K) {
            out[t_off + tid] = out_accum[tid];
        }

        // state 업데이트: s[vi] = ew * s[vi] + k_val * v[vi]
        #pragma unroll
        for (int vi = 0; vi < K; vi++) {
            s[vi] = ew * s[vi] + k_val * v_shared[vi];
        }

        // state 저장 (backward에서 사용)
        float* state_t = state_base + (t + 1) * state_stride;
        #pragma unroll
        for (int vi = 0; vi < K; vi++) {
            state_t[tid * K + vi] = s[vi];
        }

        __syncthreads();
    }
}


/**
 * Backward 커널: reverse scan으로 gradient 계산
 *
 * 입력: grad_output (B, T, H, K), 저장된 states, 원본 r,k,v,w,u
 * 출력: grad_r, grad_k, grad_v, grad_w (B, T, H, K), grad_u (H, K)
 */
template <int BLOCK_K>
__global__ void rwkv6_backward_kernel(
    const float* __restrict__ r,
    const float* __restrict__ k,
    const float* __restrict__ v,
    const float* __restrict__ w,
    const float* __restrict__ u,
    const float* __restrict__ grad_out,    // (B, T, H, K)
    const float* __restrict__ state_saved, // (B, T+1, H, K, K)
    float* __restrict__ grad_r,   // (B, T, H, K)
    float* __restrict__ grad_k,   // (B, T, H, K)
    float* __restrict__ grad_v,   // (B, T, H, K)
    float* __restrict__ grad_w,   // (B, T, H, K)
    float* __restrict__ grad_u,   // (H, K) — atomicAdd across B
    int B, int T, int H, int K
) {
    const int b = blockIdx.x;
    const int h = blockIdx.y;
    if (b >= B || h >= H) return;

    const int tid = threadIdx.x;
    if (tid >= K) return;

    const float u_val = u[h * K + tid];

    // grad_state[tid][vi] — register에 유지
    float gs[MAX_DIM];
    #pragma unroll
    for (int vi = 0; vi < K; vi++) gs[vi] = 0.0f;

    const int bhk_stride = T * H * K;
    const int hk_stride = H * K;
    const int base_bh = b * bhk_stride + h * K;
    const int state_stride = H * K * K;
    const float* state_base = state_saved + b * (T + 1) * state_stride + h * K * K;

    float grad_u_local = 0.0f;

    for (int t = T - 1; t >= 0; t--) {
        const int t_off = base_bh + t * hk_stride;

        const float r_val = r[t_off + tid];
        const float k_val = k[t_off + tid];
        const float w_val = w[t_off + tid];
        const float ew = expf(w_val);

        // shared memory에 v[t], grad_out[t] broadcast
        __shared__ float v_sh[MAX_DIM];
        __shared__ float go_sh[MAX_DIM];
        __shared__ float r_sh[MAX_DIM];
        __shared__ float k_sh[MAX_DIM];
        if (tid < K) {
            v_sh[tid] = v[t_off + tid];
            go_sh[tid] = grad_out[t_off + tid];
            r_sh[tid] = r[t_off + tid];
            k_sh[tid] = k[t_off + tid];
        }
        __syncthreads();

        // 저장된 state[t] (t 시점 이전 상태)
        const float* state_t = state_base + t * state_stride;

        // grad_r[t, tid] = sum_v (state[tid][vi] + u*k*v[vi]) * go[vi]
        float gr = 0.0f;
        #pragma unroll
        for (int vi = 0; vi < K; vi++) {
            float s_tv = state_t[tid * K + vi];
            gr += (s_tv + u_val * k_val * v_sh[vi]) * go_sh[vi];
        }
        grad_r[t_off + tid] = gr;

        // grad_u[tid] += sum_v kv[tid][vi] * r[tid] * go[vi]
        float gu = 0.0f;
        #pragma unroll
        for (int vi = 0; vi < K; vi++) {
            gu += k_val * v_sh[vi] * r_val * go_sh[vi];
        }
        grad_u_local += gu;

        // grad_kv[tid][vi] = u[tid] * r[tid] * go[vi] + gs[vi]
        // grad_k[t, tid] = sum_v grad_kv[tid][vi] * v[vi]
        // grad_v[t, vi] = sum_k grad_kv[k][vi] * k[k]  (needs reduce over k)
        float gk = 0.0f;
        float grad_kv[MAX_DIM];
        #pragma unroll
        for (int vi = 0; vi < K; vi++) {
            grad_kv[vi] = u_val * r_val * go_sh[vi] + gs[vi];
            gk += grad_kv[vi] * v_sh[vi];
        }
        grad_k[t_off + tid] = gk;

        // grad_v: sum_k grad_kv[k][vi] * k[k] → k 축 reduce
        __shared__ float gv_accum[MAX_DIM];
        if (tid == 0) {
            #pragma unroll
            for (int vi = 0; vi < K; vi++) gv_accum[vi] = 0.0f;
        }
        __syncthreads();

        for (int vi = 0; vi < K; vi++) {
            atomicAdd(&gv_accum[vi], grad_kv[vi] * k_val);
        }
        __syncthreads();

        if (tid < K) {
            grad_v[t_off + tid] = gv_accum[tid];
        }

        // grad_w[t, tid] = sum_v ew * state[tid][vi] * gs[vi]
        float gw = 0.0f;
        #pragma unroll
        for (int vi = 0; vi < K; vi++) {
            gw += ew * state_t[tid * K + vi] * gs[vi];
        }
        grad_w[t_off + tid] = gw;

        // gs 업데이트: total_gs_t = exp(w_t) * total_gs_{t+1} + r_t * go_t
        // gs는 현재 total_gs_{t+1}, 업데이트 후 total_gs_t
        #pragma unroll
        for (int vi = 0; vi < K; vi++) {
            gs[vi] = ew * gs[vi] + r_val * go_sh[vi];
        }

        __syncthreads();
    }

    // grad_u: batch 간 atomicAdd
    atomicAdd(&grad_u[h * K + tid], grad_u_local);
}

}  // namespace


// C++ 래퍼: BF16/FP32 입력 처리 + float32 변환

std::vector<torch::Tensor> rwkv6_cuda_forward(
    torch::Tensor r,  // (B, T, H, K)
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor w,  // (B, T, H, K), 음수 log-space decay
    torch::Tensor u,  // (H, K)
    bool save_state    // backward용 state 저장 여부
) {
    TORCH_CHECK(r.is_cuda(), "r must be CUDA tensor");
    auto orig_dtype = r.scalar_type();

    // float32로 변환 (커널은 float32 전용)
    auto r_f = r.to(torch::kFloat32).contiguous();
    auto k_f = k.to(torch::kFloat32).contiguous();
    auto v_f = v.to(torch::kFloat32).contiguous();
    auto w_f = w.to(torch::kFloat32).contiguous();
    auto u_f = u.to(torch::kFloat32).contiguous();

    int B = r_f.size(0);
    int T = r_f.size(1);
    int H = r_f.size(2);
    int K = r_f.size(3);

    TORCH_CHECK(K <= MAX_DIM, "headdim must be <= ", MAX_DIM);

    auto out = torch::zeros({B, T, H, K}, r_f.options());
    auto state_out = save_state ?
        torch::zeros({B, T + 1, H, K, K}, r_f.options()) :
        torch::zeros({1}, r_f.options());  // dummy

    dim3 grid(B, H);
    dim3 block(K);

    rwkv6_forward_kernel<64><<<grid, block, 0, at::cuda::getDefaultCUDAStream()>>>(
        r_f.data_ptr<float>(),
        k_f.data_ptr<float>(),
        v_f.data_ptr<float>(),
        w_f.data_ptr<float>(),
        u_f.data_ptr<float>(),
        out.data_ptr<float>(),
        state_out.data_ptr<float>(),
        B, T, H, K
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    // 원래 dtype으로 복원
    auto out_cast = out.to(orig_dtype);

    return {out_cast, state_out};
}


std::vector<torch::Tensor> rwkv6_cuda_backward(
    torch::Tensor r,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor w,
    torch::Tensor u,
    torch::Tensor grad_out,
    torch::Tensor state_saved  // forward에서 저장한 state
) {
    auto r_f = r.to(torch::kFloat32).contiguous();
    auto k_f = k.to(torch::kFloat32).contiguous();
    auto v_f = v.to(torch::kFloat32).contiguous();
    auto w_f = w.to(torch::kFloat32).contiguous();
    auto u_f = u.to(torch::kFloat32).contiguous();
    auto go_f = grad_out.to(torch::kFloat32).contiguous();
    auto st_f = state_saved.contiguous();  // 이미 float32

    int B = r_f.size(0);
    int T = r_f.size(1);
    int H = r_f.size(2);
    int K = r_f.size(3);

    auto grad_r = torch::zeros_like(r_f);
    auto grad_k = torch::zeros_like(k_f);
    auto grad_v = torch::zeros_like(v_f);
    auto grad_w = torch::zeros_like(w_f);
    auto grad_u = torch::zeros({H, K}, u_f.options());

    dim3 grid(B, H);
    dim3 block(K);

    rwkv6_backward_kernel<64><<<grid, block, 0, at::cuda::getDefaultCUDAStream()>>>(
        r_f.data_ptr<float>(),
        k_f.data_ptr<float>(),
        v_f.data_ptr<float>(),
        w_f.data_ptr<float>(),
        u_f.data_ptr<float>(),
        go_f.data_ptr<float>(),
        st_f.data_ptr<float>(),
        grad_r.data_ptr<float>(),
        grad_k.data_ptr<float>(),
        grad_v.data_ptr<float>(),
        grad_w.data_ptr<float>(),
        grad_u.data_ptr<float>(),
        B, T, H, K
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    auto orig_dtype = r.scalar_type();
    return {
        grad_r.to(orig_dtype),
        grad_k.to(orig_dtype),
        grad_v.to(orig_dtype),
        grad_w.to(orig_dtype),
        grad_u,  // u는 항상 float32 파라미터
    };
}
