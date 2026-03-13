/**
 * MoE Fused Dispatch CUDA 확장 pybind 래퍼
 */
#include <torch/extension.h>

std::vector<torch::Tensor> moe_scatter_cuda(
    torch::Tensor expert_idx, torch::Tensor x, torch::Tensor expert_w,
    int n_experts, int capacity);

torch::Tensor moe_gather_cuda(
    torch::Tensor expert_out, torch::Tensor weights,
    torch::Tensor expert_idx, torch::Tensor token_pos,
    int N, int D);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("scatter", &moe_scatter_cuda, "MoE scatter: tokens → expert buffers (CUDA)");
    m.def("gather", &moe_gather_cuda, "MoE gather: expert outputs → tokens (CUDA)");
}
