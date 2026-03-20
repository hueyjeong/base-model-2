/**
 * RWKV-6 CUDA 확장 pybind 래퍼
 */
#include <torch/extension.h>
#include <vector>

std::vector<torch::Tensor> rwkv6_cuda_forward(
    torch::Tensor r, torch::Tensor k, torch::Tensor v,
    torch::Tensor w, torch::Tensor u, bool save_state);

std::vector<torch::Tensor> rwkv6_cuda_backward(
    torch::Tensor r, torch::Tensor k, torch::Tensor v,
    torch::Tensor w, torch::Tensor u,
    torch::Tensor grad_out, torch::Tensor state_saved);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &rwkv6_cuda_forward,
          "RWKV-6 recurrent scan forward (CUDA)");
    m.def("backward", &rwkv6_cuda_backward,
          "RWKV-6 recurrent scan backward (CUDA)");
}
