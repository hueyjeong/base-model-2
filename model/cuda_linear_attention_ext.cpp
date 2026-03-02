/**
 * Linear Cross-Attention CUDA Extension - PyBind11 바인딩
 */

#include <torch/extension.h>

torch::Tensor linear_cross_attn_fwd_cuda(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    torch::Tensor mask,
    float eps
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("linear_cross_attn_fwd", &linear_cross_attn_fwd_cuda,
          "Linear Cross-Attention Forward (CUDA)",
          py::arg("Q"), py::arg("K"), py::arg("V"),
          py::arg("mask"), py::arg("eps") = 1e-5f);
}
