#include <torch/extension.h>
#include <vector>

std::vector<torch::Tensor> fused_quant_act_forward_cuda(torch::Tensor x);

std::vector<torch::Tensor> fused_quant_act_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::kFloat, "x must be float32");
    TORCH_CHECK(x.dim() == 2, "x must be 2D [M, K]");
    return fused_quant_act_forward_cuda(x.contiguous());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &fused_quant_act_forward,
          "Fused per-row absmax activation quantization (STE float output)");
}
