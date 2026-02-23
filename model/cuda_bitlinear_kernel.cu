#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublasLt.h>
#include <sm_61_intrinsics.h>
#include <mutex>
#include <unordered_map>

namespace {

constexpr int BLOCK_M = 16;
constexpr int BLOCK_N = 16;
constexpr int BLOCK_K = 32;
constexpr int Q_THREADS = 256;

struct LtAlgoCacheEntry {
    cublasLtMatmulAlgo_t algo;
    bool valid;
};

std::mutex g_lt_mutex;
cublasLtHandle_t g_lt_handle = nullptr;
std::unordered_map<unsigned long long, LtAlgoCacheEntry> g_lt_algo_cache;

unsigned long long make_lt_key(int device, int M, int N, int K) {
    // 16bit packing for dims (충분히 큰 shape는 희귀; 충돌 우려 매우 낮음)
    unsigned long long key = 0;
    key |= (static_cast<unsigned long long>(device & 0xFF) << 56);
    key |= (static_cast<unsigned long long>(M & 0xFFFF) << 40);
    key |= (static_cast<unsigned long long>(N & 0xFFFF) << 24);
    key |= (static_cast<unsigned long long>(K & 0xFFFF) << 8);
    return key;
}

cublasLtHandle_t get_lt_handle() {
    std::lock_guard<std::mutex> lock(g_lt_mutex);
    if (g_lt_handle == nullptr) {
        if (cublasLtCreate(&g_lt_handle) != CUBLAS_STATUS_SUCCESS) {
            return nullptr;
        }
    }
    return g_lt_handle;
}

__global__ void bitlinear_int8_forward_kernel(
    const int8_t* __restrict__ x,
    const int8_t* __restrict__ w,
    const float* __restrict__ x_scale,
    const float* __restrict__ w_scale,
    const float* __restrict__ bias,
    float* __restrict__ out,
    int M,
    int N,
    int K,
    bool has_bias) {
    const int row = blockIdx.y * BLOCK_M + threadIdx.y;
    const int col = blockIdx.x * BLOCK_N + threadIdx.x;

    if (row >= M || col >= N) {
        return;
    }

    __shared__ __align__(16) int8_t x_tile[BLOCK_M][BLOCK_K];
    __shared__ __align__(16) int8_t w_tile[BLOCK_N][BLOCK_K];

    int32_t acc = 0;

    for (int k0 = 0; k0 < K; k0 += BLOCK_K) {
        for (int kk = threadIdx.x; kk < BLOCK_K; kk += BLOCK_N) {
            const int k = k0 + kk;
            x_tile[threadIdx.y][kk] = (k < K) ? x[row * K + k] : static_cast<int8_t>(0);
        }

        for (int kk = threadIdx.y; kk < BLOCK_K; kk += BLOCK_M) {
            const int k = k0 + kk;
            w_tile[threadIdx.x][kk] = (k < K) ? w[col * K + k] : static_cast<int8_t>(0);
        }

        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < BLOCK_K; kk += 4) {
            const int x_pack = *reinterpret_cast<const int*>(&x_tile[threadIdx.y][kk]);
            const int w_pack = *reinterpret_cast<const int*>(&w_tile[threadIdx.x][kk]);
            acc = __dp4a(x_pack, w_pack, acc);
        }

        __syncthreads();
    }

    float y = static_cast<float>(acc) * x_scale[row] * w_scale[0];
    if (has_bias) {
        y += bias[col];
    }

    out[row * N + col] = y;
}

__global__ void bitlinear_int8_backward_input_kernel(
    const int8_t* __restrict__ go,
    const int8_t* __restrict__ w,
    const float* __restrict__ go_scale,
    const float* __restrict__ w_scale,
    float* __restrict__ grad_x,
    int M,
    int N,
    int K) {
    const int row = blockIdx.y * BLOCK_M + threadIdx.y;
    const int col = blockIdx.x * BLOCK_N + threadIdx.x;

    if (row >= M || col >= K) {
        return;
    }

    int32_t acc = 0;
    for (int n = 0; n < N; ++n) {
        acc += static_cast<int32_t>(go[row * N + n]) *
               static_cast<int32_t>(w[n * K + col]);
    }

    grad_x[row * K + col] = static_cast<float>(acc) * go_scale[row] * w_scale[0];
}

__global__ void quantize_activations_int8_kernel(
    const float* __restrict__ x,
    int8_t* __restrict__ x_q,
    float* __restrict__ x_scale,
    int M,
    int K) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    if (row >= M) return;

    __shared__ float smax[Q_THREADS];
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

    for (int col = tid; col < K; col += blockDim.x) {
        float q = nearbyintf(x[row_off + col] * inv_scale);
        q = fminf(127.0f, fmaxf(-128.0f, q));
        x_q[row_off + col] = static_cast<int8_t>(q);
    }
}

}  // namespace

torch::Tensor bitlinear_int8_forward_cuda(
    torch::Tensor x_int8,
    torch::Tensor w_int8,
    torch::Tensor x_scale,
    torch::Tensor w_scale,
    c10::optional<torch::Tensor> bias_opt) {
    const auto M = static_cast<int>(x_int8.size(0));
    const auto K = static_cast<int>(x_int8.size(1));
    const auto N = static_cast<int>(w_int8.size(0));

    auto out = torch::zeros({M, N}, x_scale.options().dtype(torch::kFloat));

    const dim3 block(BLOCK_N, BLOCK_M);
    const dim3 grid((N + BLOCK_N - 1) / BLOCK_N, (M + BLOCK_M - 1) / BLOCK_M);

    const bool has_bias = bias_opt.has_value();
    const float* bias_ptr = has_bias ? bias_opt.value().data_ptr<float>() : nullptr;

    bitlinear_int8_forward_kernel<<<grid, block, 0, at::cuda::getDefaultCUDAStream()>>>(
        x_int8.data_ptr<int8_t>(),
        w_int8.data_ptr<int8_t>(),
        x_scale.data_ptr<float>(),
        w_scale.data_ptr<float>(),
        bias_ptr,
        out.data_ptr<float>(),
        M,
        N,
        K,
        has_bias);

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

torch::Tensor bitlinear_int8_backward_input_cuda(
    torch::Tensor go_int8,
    torch::Tensor w_int8,
    torch::Tensor go_scale,
    torch::Tensor w_scale) {
    const auto M = static_cast<int>(go_int8.size(0));
    const auto N = static_cast<int>(go_int8.size(1));
    const auto K = static_cast<int>(w_int8.size(1));

    auto grad_x = torch::zeros({M, K}, go_scale.options().dtype(torch::kFloat));

    const dim3 block(BLOCK_N, BLOCK_M);
    const dim3 grid((K + BLOCK_N - 1) / BLOCK_N, (M + BLOCK_M - 1) / BLOCK_M);

    bitlinear_int8_backward_input_kernel<<<grid, block, 0, at::cuda::getDefaultCUDAStream()>>>(
        go_int8.data_ptr<int8_t>(),
        w_int8.data_ptr<int8_t>(),
        go_scale.data_ptr<float>(),
        w_scale.data_ptr<float>(),
        grad_x.data_ptr<float>(),
        M,
        N,
        K);

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return grad_x;
}

std::vector<torch::Tensor> quantize_activations_int8_cuda(torch::Tensor x_fp32) {
    const auto M = static_cast<int>(x_fp32.size(0));
    const auto K = static_cast<int>(x_fp32.size(1));

    auto x_q = torch::zeros({M, K}, x_fp32.options().dtype(torch::kChar));
    auto x_scale = torch::zeros({M, 1}, x_fp32.options().dtype(torch::kFloat));

    quantize_activations_int8_kernel<<<M, Q_THREADS, 0, at::cuda::getDefaultCUDAStream()>>>(
        x_fp32.data_ptr<float>(),
        reinterpret_cast<int8_t*>(x_q.data_ptr<int8_t>()),
        x_scale.data_ptr<float>(),
        M,
        K);

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {x_q, x_scale};
}

std::vector<torch::Tensor> quantize_weights_ternary_cuda(torch::Tensor w_fp32) {
    // gamma = mean(abs(w)), w_q = round(clamp(w / gamma, -1, 1))
    auto gamma = at::mean(at::abs(w_fp32)).clamp_min(1e-5f);
    auto w_scaled = w_fp32 / gamma;
    auto w_q_f = at::round(at::clamp(w_scaled, -1.0f, 1.0f));
    auto w_q_i8 = w_q_f.to(torch::kChar).contiguous();
    auto gamma_1 = gamma.reshape({1}).to(torch::kFloat).contiguous();
    return {w_q_i8, gamma_1};
}

torch::Tensor bitlinear_grad_weight_cuda(torch::Tensor go_2d, torch::Tensor x_2d) {
    // grad_w = go^T @ x  ==> (N, M) @ (M, K) = (N, K)
    // 우선 cublasLt(FP32/TF32) 경로를 시도하고 실패 시 ATen matmul로 fallback.
    TORCH_CHECK(go_2d.is_cuda() && x_2d.is_cuda(), "grad_weight tensors must be CUDA");
    TORCH_CHECK(go_2d.dim() == 2 && x_2d.dim() == 2, "grad_weight tensors must be 2D");
    TORCH_CHECK(go_2d.size(0) == x_2d.size(0), "M dim mismatch in grad_weight");

    const auto M = static_cast<int>(go_2d.size(0));
    const auto N = static_cast<int>(go_2d.size(1));
    const auto K = static_cast<int>(x_2d.size(1));

    if (go_2d.scalar_type() != torch::kFloat || x_2d.scalar_type() != torch::kFloat) {
        auto go_t = go_2d.transpose(0, 1).contiguous();
        return at::matmul(go_t, x_2d);
    }

    auto go = go_2d.contiguous();
    auto x = x_2d.contiguous();
    auto out = torch::zeros({N, K}, go.options().dtype(torch::kFloat));

    cublasLtHandle_t ltHandle = get_lt_handle();
    cublasLtMatmulDesc_t opDesc = nullptr;
    cublasLtMatrixLayout_t aLayout = nullptr;
    cublasLtMatrixLayout_t bLayout = nullptr;
    cublasLtMatrixLayout_t cLayout = nullptr;
    cublasLtMatmulPreference_t pref = nullptr;
    cublasLtMatmulHeuristicResult_t heuristicResult{};
    size_t workspaceSize = 1 << 20;  // 1MB
    auto workspace = torch::empty({static_cast<long>(workspaceSize)}, go.options().dtype(torch::kUInt8));

    bool ok = (ltHandle != nullptr);

    if (ok) {
        ok = ok && (cublasLtMatmulDescCreate(&opDesc, CUBLAS_COMPUTE_32F_FAST_TF32, CUDA_R_32F) == CUBLAS_STATUS_SUCCESS);
    }

    cublasOperation_t transA = CUBLAS_OP_T;
    cublasOperation_t transB = CUBLAS_OP_N;
    if (ok) {
        ok = ok && (cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transA, sizeof(transA)) == CUBLAS_STATUS_SUCCESS);
        ok = ok && (cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transB, sizeof(transB)) == CUBLAS_STATUS_SUCCESS);
    }

    // row-major layout 지정
    cublasLtOrder_t order = CUBLASLT_ORDER_ROW;

    // A=go(M,N), op(A)=go^T(N,M)
    if (ok) {
        ok = ok && (cublasLtMatrixLayoutCreate(&aLayout, CUDA_R_32F, M, N, N) == CUBLAS_STATUS_SUCCESS);
        ok = ok && (cublasLtMatrixLayoutSetAttribute(aLayout, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)) == CUBLAS_STATUS_SUCCESS);
    }
    // B=x(M,K)
    if (ok) {
        ok = ok && (cublasLtMatrixLayoutCreate(&bLayout, CUDA_R_32F, M, K, K) == CUBLAS_STATUS_SUCCESS);
        ok = ok && (cublasLtMatrixLayoutSetAttribute(bLayout, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)) == CUBLAS_STATUS_SUCCESS);
    }
    // C=out(N,K)
    if (ok) {
        ok = ok && (cublasLtMatrixLayoutCreate(&cLayout, CUDA_R_32F, N, K, K) == CUBLAS_STATUS_SUCCESS);
        ok = ok && (cublasLtMatrixLayoutSetAttribute(cLayout, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)) == CUBLAS_STATUS_SUCCESS);
    }

    if (ok) {
        ok = ok && (cublasLtMatmulPreferenceCreate(&pref) == CUBLAS_STATUS_SUCCESS);
    }
    if (ok) {
        ok = ok && (cublasLtMatmulPreferenceSetAttribute(
            pref,
            CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
            &workspaceSize,
            sizeof(workspaceSize)
        ) == CUBLAS_STATUS_SUCCESS);
    }

    int returnedResults = 0;
    int device_index = go.get_device();
    auto cache_key = make_lt_key(device_index, M, N, K);
    bool cache_hit = false;
    if (ok) {
        std::lock_guard<std::mutex> lock(g_lt_mutex);
        auto it = g_lt_algo_cache.find(cache_key);
        if (it != g_lt_algo_cache.end() && it->second.valid) {
            heuristicResult.algo = it->second.algo;
            cache_hit = true;
        }
    }

    if (ok && !cache_hit) {
        ok = ok && (cublasLtMatmulAlgoGetHeuristic(
            ltHandle,
            opDesc,
            aLayout,
            bLayout,
            cLayout,
            cLayout,
            pref,
            1,
            &heuristicResult,
            &returnedResults
        ) == CUBLAS_STATUS_SUCCESS);
        ok = ok && (returnedResults > 0);
        if (ok) {
            std::lock_guard<std::mutex> lock(g_lt_mutex);
            g_lt_algo_cache[cache_key] = LtAlgoCacheEntry{heuristicResult.algo, true};
        }
    }

    const float alpha = 1.0f;
    const float beta = 0.0f;
    if (ok) {
        auto stream = at::cuda::getDefaultCUDAStream();
        ok = ok && (cublasLtMatmul(
            ltHandle,
            opDesc,
            &alpha,
            go.data_ptr<float>(),
            aLayout,
            x.data_ptr<float>(),
            bLayout,
            &beta,
            out.data_ptr<float>(),
            cLayout,
            out.data_ptr<float>(),
            cLayout,
            &heuristicResult.algo,
            workspace.data_ptr(),
            workspaceSize,
            stream.stream()
        ) == CUBLAS_STATUS_SUCCESS);
    }

    if (pref) cublasLtMatmulPreferenceDestroy(pref);
    if (cLayout) cublasLtMatrixLayoutDestroy(cLayout);
    if (bLayout) cublasLtMatrixLayoutDestroy(bLayout);
    if (aLayout) cublasLtMatrixLayoutDestroy(aLayout);
    if (opDesc) cublasLtMatmulDescDestroy(opDesc);

    if (!ok) {
        auto go_t = go_2d.transpose(0, 1).contiguous();
        return at::matmul(go_t, x_2d);
    }

    return out;
}
