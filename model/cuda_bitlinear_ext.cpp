#include <torch/extension.h>

#include <vector>

torch::Tensor bitlinear_int8_forward_cuda(
    torch::Tensor x_int8,
    torch::Tensor w_int8,
    torch::Tensor x_scale,
    torch::Tensor w_scale,
    c10::optional<torch::Tensor> bias_opt);

torch::Tensor bitlinear_int8_backward_input_cuda(
    torch::Tensor go_int8,
    torch::Tensor w_int8,
    torch::Tensor go_scale,
    torch::Tensor w_scale);

std::vector<torch::Tensor> quantize_activations_int8_cuda(torch::Tensor x_fp32);
std::vector<torch::Tensor> quantize_weights_ternary_cuda(torch::Tensor w_fp32);

torch::Tensor bitlinear_grad_weight_cuda(torch::Tensor go_2d, torch::Tensor x_2d);

torch::Tensor bitlinear_int8_forward(
    torch::Tensor x_int8,
    torch::Tensor w_int8,
    torch::Tensor x_scale,
    torch::Tensor w_scale,
    c10::optional<torch::Tensor> bias_opt) {
    TORCH_CHECK(x_int8.is_cuda(), "x_int8 must be CUDA tensor");
    TORCH_CHECK(w_int8.is_cuda(), "w_int8 must be CUDA tensor");
    TORCH_CHECK(x_scale.is_cuda(), "x_scale must be CUDA tensor");
    TORCH_CHECK(w_scale.is_cuda(), "w_scale must be CUDA tensor");

    TORCH_CHECK(x_int8.scalar_type() == torch::kInt8, "x_int8 must be int8");
    TORCH_CHECK(w_int8.scalar_type() == torch::kInt8, "w_int8 must be int8");

    TORCH_CHECK(x_int8.dim() == 2, "x_int8 must be 2D [M, K]");
    TORCH_CHECK(w_int8.dim() == 2, "w_int8 must be 2D [N, K]");
    TORCH_CHECK(x_int8.size(1) == w_int8.size(1), "K mismatch between x_int8 and w_int8");

    TORCH_CHECK(x_scale.dim() == 2 || x_scale.dim() == 1, "x_scale must be [M,1] or [M]");
    if (x_scale.dim() == 2) {
        TORCH_CHECK(x_scale.size(1) == 1, "x_scale second dim must be 1");
        TORCH_CHECK(x_scale.size(0) == x_int8.size(0), "x_scale first dim must match M");
    } else {
        TORCH_CHECK(x_scale.size(0) == x_int8.size(0), "x_scale size must match M");
    }

    TORCH_CHECK(w_scale.numel() == 1, "w_scale must be scalar tensor");
    if (bias_opt.has_value()) {
        auto bias = bias_opt.value();
        TORCH_CHECK(bias.is_cuda(), "bias must be CUDA tensor");
        TORCH_CHECK(bias.dim() == 1, "bias must be 1D [N]");
        TORCH_CHECK(bias.size(0) == w_int8.size(0), "bias size must match N");
    }

    return bitlinear_int8_forward_cuda(
        x_int8.contiguous(),
        w_int8.contiguous(),
        x_scale.contiguous(),
        w_scale.contiguous(),
        bias_opt.has_value() ? c10::optional<torch::Tensor>(bias_opt.value().contiguous()) : c10::nullopt);
}

torch::Tensor bitlinear_int8_backward_input(
    torch::Tensor go_int8,
    torch::Tensor w_int8,
    torch::Tensor go_scale,
    torch::Tensor w_scale) {
    TORCH_CHECK(go_int8.is_cuda(), "go_int8 must be CUDA tensor");
    TORCH_CHECK(w_int8.is_cuda(), "w_int8 must be CUDA tensor");
    TORCH_CHECK(go_scale.is_cuda(), "go_scale must be CUDA tensor");
    TORCH_CHECK(w_scale.is_cuda(), "w_scale must be CUDA tensor");

    TORCH_CHECK(go_int8.scalar_type() == torch::kInt8, "go_int8 must be int8");
    TORCH_CHECK(w_int8.scalar_type() == torch::kInt8, "w_int8 must be int8");

    TORCH_CHECK(go_int8.dim() == 2, "go_int8 must be 2D [M, N]");
    TORCH_CHECK(w_int8.dim() == 2, "w_int8 must be 2D [N, K]");
    TORCH_CHECK(go_int8.size(1) == w_int8.size(0), "N mismatch between go_int8 and w_int8");

    TORCH_CHECK(go_scale.dim() == 2 || go_scale.dim() == 1, "go_scale must be [M,1] or [M]");
    if (go_scale.dim() == 2) {
        TORCH_CHECK(go_scale.size(1) == 1, "go_scale second dim must be 1");
        TORCH_CHECK(go_scale.size(0) == go_int8.size(0), "go_scale first dim must match M");
    } else {
        TORCH_CHECK(go_scale.size(0) == go_int8.size(0), "go_scale size must match M");
    }

    TORCH_CHECK(w_scale.numel() == 1, "w_scale must be scalar tensor");

    return bitlinear_int8_backward_input_cuda(
        go_int8.contiguous(),
        w_int8.contiguous(),
        go_scale.contiguous(),
        w_scale.contiguous());
}

std::vector<torch::Tensor> quantize_activations_int8(torch::Tensor x_fp32) {
    TORCH_CHECK(x_fp32.is_cuda(), "x_fp32 must be CUDA tensor");
    TORCH_CHECK(x_fp32.scalar_type() == torch::kFloat, "x_fp32 must be float32");
    TORCH_CHECK(x_fp32.dim() == 2, "x_fp32 must be 2D [M, K]");
    return quantize_activations_int8_cuda(x_fp32.contiguous());
}

std::vector<torch::Tensor> quantize_weights_ternary(torch::Tensor w_fp32) {
    TORCH_CHECK(w_fp32.is_cuda(), "w_fp32 must be CUDA tensor");
    TORCH_CHECK(w_fp32.scalar_type() == torch::kFloat, "w_fp32 must be float32");
    TORCH_CHECK(w_fp32.dim() == 2, "w_fp32 must be 2D [N, K]");
    return quantize_weights_ternary_cuda(w_fp32.contiguous());
}

torch::Tensor bitlinear_grad_weight(torch::Tensor go_2d, torch::Tensor x_2d) {
    TORCH_CHECK(go_2d.is_cuda(), "go_2d must be CUDA tensor");
    TORCH_CHECK(x_2d.is_cuda(), "x_2d must be CUDA tensor");
    TORCH_CHECK(go_2d.dim() == 2, "go_2d must be 2D [M, N]");
    TORCH_CHECK(x_2d.dim() == 2, "x_2d must be 2D [M, K]");
    TORCH_CHECK(go_2d.size(0) == x_2d.size(0), "M mismatch between go_2d and x_2d");
    return bitlinear_grad_weight_cuda(go_2d.contiguous(), x_2d.contiguous());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("bitlinear_int8_forward", &bitlinear_int8_forward,
          "BitLinear int8 forward (CUDA)");
    m.def("bitlinear_int8_backward_input", &bitlinear_int8_backward_input,
          "BitLinear int8 backward input grad (CUDA)");
    m.def("quantize_activations_int8", &quantize_activations_int8,
          "Fused per-row absmax activation quantization (CUDA)");
        m.def("quantize_weights_ternary", &quantize_weights_ternary,
            "Fused absmean ternary weight quantization (CUDA)");
        m.def("bitlinear_grad_weight", &bitlinear_grad_weight,
            "BitLinear grad_weight GEMM path (CUDA)");
}
