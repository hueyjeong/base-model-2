/**
 * Document-Isolated Linear Cross-Attention CUDA Extension — PyBind11 바인딩
 */

#include <torch/extension.h>
#include <utility>
#include <vector>

// Forward declarations — v2 separate kernels (kernel.cu)
std::pair<torch::Tensor, torch::Tensor> doc_scatter_kv_fwd(
    torch::Tensor K,
    torch::Tensor V,
    torch::Tensor src_doc_ids,
    int max_docs
);

std::pair<torch::Tensor, torch::Tensor> doc_gather_query_fwd(
    torch::Tensor Q,
    torch::Tensor context,
    torch::Tensor z,
    torch::Tensor tgt_doc_ids,
    float eps
);

std::pair<torch::Tensor, torch::Tensor> doc_backward_kv(
    torch::Tensor K,
    torch::Tensor V,
    torch::Tensor grad_ctx,
    torch::Tensor grad_zz,
    torch::Tensor src_doc_ids
);

torch::Tensor doc_backward_q(
    torch::Tensor Q,
    torch::Tensor context,
    torch::Tensor z,
    torch::Tensor out,
    torch::Tensor den,
    torch::Tensor grad_out,
    torch::Tensor tgt_doc_ids
);

// Forward declarations — fused kernels (kernel.cu)
std::pair<torch::Tensor, torch::Tensor> doc_fused_forward(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    torch::Tensor src_doc_ids,
    torch::Tensor tgt_doc_ids,
    int max_docs,
    float eps
);

std::vector<torch::Tensor> doc_fused_backward(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    torch::Tensor fwd_out,
    torch::Tensor fwd_den,
    torch::Tensor grad_out,
    torch::Tensor src_doc_ids,
    torch::Tensor tgt_doc_ids,
    int max_docs
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // v2 separate kernels
    m.def("doc_scatter_kv_fwd", &doc_scatter_kv_fwd,
          "Phase 1: scatter K^T·V per document (forward)",
          py::arg("K"), py::arg("V"),
          py::arg("src_doc_ids"), py::arg("max_docs"));

    m.def("doc_gather_query_fwd", &doc_gather_query_fwd,
          "Phase 2: gather per-doc context and normalize (forward)",
          py::arg("Q"), py::arg("context"), py::arg("z"),
          py::arg("tgt_doc_ids"), py::arg("eps") = 1e-5f);

    m.def("doc_backward_kv", &doc_backward_kv,
          "Backward: grad_k, grad_v from grad_context and grad_z",
          py::arg("K"), py::arg("V"),
          py::arg("grad_ctx"), py::arg("grad_zz"),
          py::arg("src_doc_ids"));

    m.def("doc_backward_q", &doc_backward_q,
          "Backward: grad_q",
          py::arg("Q"), py::arg("context"), py::arg("z"),
          py::arg("out"), py::arg("den"), py::arg("grad_out"),
          py::arg("tgt_doc_ids"));

    // Fused kernels (Grid=B*H*D, context in smem only)
    m.def("doc_fused_forward", &doc_fused_forward,
          "Fused scatter+gather forward (context stays in smem)",
          py::arg("Q"), py::arg("K"), py::arg("V"),
          py::arg("src_doc_ids"), py::arg("tgt_doc_ids"),
          py::arg("max_docs"), py::arg("eps") = 1e-5f);

    m.def("doc_fused_backward", &doc_fused_backward,
          "Fused backward (ctx recomputed in smem, single kernel)",
          py::arg("Q"), py::arg("K"), py::arg("V"),
          py::arg("fwd_out"), py::arg("fwd_den"), py::arg("grad_out"),
          py::arg("src_doc_ids"), py::arg("tgt_doc_ids"),
          py::arg("max_docs"));
}
