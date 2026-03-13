/**
 * 배치 Levenshtein DP + Edit Tag 생성 — C++ with OpenMP
 *
 * Python 순수 구현 대비 ~50-100x 속도 향상 (C++ 루프 + OpenMP 배치 병렬화).
 * GPU→CPU→GPU 전환 비용은 동일하나, DP 계산 자체가 병목이므로 효과적.
 */
#include <torch/extension.h>
#include <vector>
#include <algorithm>

// 태그 상수 (edit_tags.py와 동일)
constexpr int TAG_KEEP = 0;
constexpr int TAG_DELETE = 1;

inline int tag_replace(int token_id) { return 2 + token_id; }
inline int tag_insert(int token_id, int vocab_size) { return 2 + vocab_size + token_id; }

/**
 * 단일 시퀀스 쌍에 대한 Levenshtein DP + backtrace + 태그 생성
 */
static std::vector<int> compute_edit_tags_single(
    const int* source, int n,
    const int* target, int m,
    int vocab_size
) {
    // DP 테이블 (1D 압축)
    std::vector<int> dp((n + 1) * (m + 1));
    auto idx = [&](int i, int j) { return i * (m + 1) + j; };

    for (int i = 0; i <= n; i++) dp[idx(i, 0)] = i;
    for (int j = 0; j <= m; j++) dp[idx(0, j)] = j;

    for (int i = 1; i <= n; i++) {
        for (int j = 1; j <= m; j++) {
            if (source[i - 1] == target[j - 1]) {
                dp[idx(i, j)] = dp[idx(i - 1, j - 1)];
            } else {
                dp[idx(i, j)] = 1 + std::min({
                    dp[idx(i - 1, j)],      // delete
                    dp[idx(i, j - 1)],       // insert
                    dp[idx(i - 1, j - 1)]    // replace
                });
            }
        }
    }

    // Backtrace → 태그 생성
    std::vector<int> tags(n, TAG_KEEP);

    // ops: (op_type, src_idx, tgt_token)
    // 0=match, 1=sub, 2=ins, 3=del
    struct Op { int type; int src_idx; int tgt_token; };
    std::vector<Op> ops;
    ops.reserve(n + m);

    int i = n, j = m;
    while (i > 0 || j > 0) {
        if (i > 0 && j > 0 && source[i - 1] == target[j - 1]) {
            ops.push_back({0, i - 1, -1});  // match
            i--; j--;
        } else if (i > 0 && j > 0 && dp[idx(i, j)] == dp[idx(i - 1, j - 1)] + 1) {
            ops.push_back({1, i - 1, target[j - 1]});  // sub
            i--; j--;
        } else if (j > 0 && dp[idx(i, j)] == dp[idx(i, j - 1)] + 1) {
            ops.push_back({2, i, target[j - 1]});  // ins
            j--;
        } else if (i > 0 && dp[idx(i, j)] == dp[idx(i - 1, j)] + 1) {
            ops.push_back({3, i - 1, -1});  // del
            i--;
        } else {
            break;
        }
    }
    std::reverse(ops.begin(), ops.end());

    // ops → tags
    std::vector<bool> insert_used(n, false);
    for (const auto& op : ops) {
        if (op.type == 0) {
            tags[op.src_idx] = TAG_KEEP;
        } else if (op.type == 1) {
            tags[op.src_idx] = tag_replace(op.tgt_token);
        } else if (op.type == 3) {
            tags[op.src_idx] = TAG_DELETE;
        } else if (op.type == 2) {
            int ins_at = std::max(0, op.src_idx - 1);
            if (!insert_used[ins_at] && tags[ins_at] == TAG_KEEP) {
                tags[ins_at] = tag_insert(op.tgt_token, vocab_size);
                insert_used[ins_at] = true;
            }
        }
    }
    return tags;
}

/**
 * 단일 시퀀스에 대한 apply_edit_tags
 */
static std::vector<int> apply_edit_tags_single(
    const int* source, const int* tag_ids, int n,
    int vocab_size
) {
    std::vector<int> result;
    result.reserve(n * 2);  // 최대 삽입 시 2배

    for (int i = 0; i < n; i++) {
        int tag = tag_ids[i];
        if (tag == TAG_KEEP) {
            result.push_back(source[i]);
        } else if (tag == TAG_DELETE) {
            // skip
        } else if (tag < 2 + vocab_size) {
            // REPLACE
            result.push_back(tag - 2);
        } else {
            // INSERT: 원본 유지 + 삽입
            result.push_back(source[i]);
            result.push_back(tag - 2 - vocab_size);
        }
    }
    return result;
}

/**
 * 배치 iterative refinement step (Python 루프 대체)
 *
 * 입력: current_ids (B, T), pred_tags (B, T), original_ids (B, T), pad_mask (B, T)
 * 출력: new_current_ids (B, T), new_edit_tags (B, T), new_pad_mask (B, T)
 */
std::vector<torch::Tensor> batch_refinement_step(
    torch::Tensor current_ids,    // (B, T) int64
    torch::Tensor pred_tags,      // (B, T) int64
    torch::Tensor original_ids,   // (B, T) int64
    torch::Tensor pad_mask,       // (B, T) bool
    int64_t vocab_size,
    int64_t pad_id,
    int64_t max_seq_len
) {
    // CPU로 이동 (GPU tensor 대응)
    auto cur_cpu = current_ids.cpu().to(torch::kInt32).contiguous();
    auto pred_cpu = pred_tags.cpu().to(torch::kInt32).contiguous();
    auto orig_cpu = original_ids.cpu().to(torch::kInt32).contiguous();
    auto mask_cpu = pad_mask.cpu().contiguous();

    int B = cur_cpu.size(0);
    int T = cur_cpu.size(1);

    auto new_ids = torch::full({B, max_seq_len}, (int)pad_id, torch::kInt64);
    auto new_tags = torch::full({B, max_seq_len}, TAG_KEEP, torch::kInt64);
    auto new_mask = torch::zeros({B, max_seq_len}, torch::kBool);

    auto cur_ptr = cur_cpu.data_ptr<int>();
    auto pred_ptr = pred_cpu.data_ptr<int>();
    auto orig_ptr = orig_cpu.data_ptr<int>();
    auto mask_ptr = mask_cpu.data_ptr<bool>();

    auto nid_ptr = new_ids.data_ptr<int64_t>();
    auto ntag_ptr = new_tags.data_ptr<int64_t>();
    auto nmask_ptr = new_mask.data_ptr<bool>();

    #pragma omp parallel for schedule(dynamic)
    for (int b = 0; b < B; b++) {
        // 유효 source/tags 추출
        std::vector<int> src, tags_b, orig;
        for (int t = 0; t < T; t++) {
            if (mask_ptr[b * T + t]) {
                src.push_back(cur_ptr[b * T + t]);
                tags_b.push_back(pred_ptr[b * T + t]);
            }
        }
        // 유효 original 추출
        for (int t = 0; t < T; t++) {
            int tok = orig_ptr[b * T + t];
            if (tok != (int)pad_id) orig.push_back(tok);
        }

        // apply_edit_tags
        auto modified = apply_edit_tags_single(
            src.data(), tags_b.data(), (int)src.size(), (int)vocab_size);

        // truncate
        if ((int)modified.size() > (int)max_seq_len)
            modified.resize(max_seq_len);

        // compute_edit_tags
        auto new_edit_tags = compute_edit_tags_single(
            modified.data(), (int)modified.size(),
            orig.data(), (int)orig.size(),
            (int)vocab_size);

        // 출력 기록
        int len = (int)modified.size();
        for (int t = 0; t < len && t < (int)max_seq_len; t++) {
            nid_ptr[b * max_seq_len + t] = modified[t];
            ntag_ptr[b * max_seq_len + t] = new_edit_tags[t];
            nmask_ptr[b * max_seq_len + t] = true;
        }
    }

    // 원래 device로 이동
    auto device = current_ids.device();
    return {
        new_ids.to(device),
        new_tags.to(device),
        new_mask.to(device),
    };
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("batch_refinement_step", &batch_refinement_step,
          "Batched edit tag refinement step (C++ OpenMP)");
}
