/**
 * Fused BitLinear CUDA 확장 pybind 래퍼
 */
#include <torch/extension.h>

torch::Tensor fused_bitlinear_forward_cuda(
    torch::Tensor x_normed,
    torch::Tensor w_int8,
    torch::Tensor w_scale,
    c10::optional<torch::Tensor> bias_opt);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &fused_bitlinear_forward_cuda,
          "Fused BitLinear forward: quant + matmul + dequant (CUDA)");
}
