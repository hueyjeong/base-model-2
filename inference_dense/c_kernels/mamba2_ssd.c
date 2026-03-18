/*
 * Mamba-2 SSD chunk-parallel forward — mamba_ssm CUDA 커널과 수치적 호환
 *
 * sequential scan과 달리 chunk 내 matmul 기반 누적으로
 * 학습 시 사용된 CUDA chunk-parallel 커널과 동일한 수치 결과 제공.
 *
 * 인터페이스:
 *   mamba2_ssd_fwd(x, B, C, dt, A, D, y, chunk_size, seq_len, nheads, headdim, d_state, ngroups)
 *
 * 입력:
 *   x[seq_len, nheads*headdim]  — SiLU 적용된 입력
 *   B[seq_len, ngroups*d_state] — SiLU 적용된 B projection
 *   C[seq_len, ngroups*d_state] — SiLU 적용된 C projection
 *   dt[seq_len, nheads]         — softplus(raw_dt + dt_bias)
 *   A[nheads]                   — negative (e.g., -exp(A_log))
 *   D[nheads]                   — skip connection scale
 *
 * 출력:
 *   y[seq_len, nheads*headdim]  — scan output (norm/gate 전)
 */

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <immintrin.h>
#include <omp.h>

void mamba2_ssd_fwd(
    const float* x,       /* [seq_len, nheads * headdim] */
    const float* B,       /* [seq_len, ngroups * d_state] */
    const float* C,       /* [seq_len, ngroups * d_state] */
    const float* dt,      /* [seq_len, nheads] */
    const float* A,       /* [nheads] — negative values */
    const float* D,       /* [nheads] — skip connection */
    float* y,             /* [seq_len, nheads * headdim] — output */
    int chunk_size,
    int seq_len, int nheads, int headdim, int d_state, int ngroups
) {
    int d_inner = nheads * headdim;
    int heads_per_group = nheads / ngroups;
    int nchunks = (seq_len + chunk_size - 1) / chunk_size;

    /* 1. dA_cumsum: cumsum(A[h] * dt[t,h]) per chunk
     *    Shape: [nchunks, nheads, chunk_size] */
    float* dA_cumsum = (float*)calloc(nchunks * nheads * chunk_size, sizeof(float));

    for (int c = 0; c < nchunks; c++) {
        for (int h = 0; h < nheads; h++) {
            float cumsum = 0.0f;
            for (int l = 0; l < chunk_size; l++) {
                int t = c * chunk_size + l;
                float dt_val = (t < seq_len) ? dt[t * nheads + h] : 0.0f;
                cumsum += A[h] * dt_val;
                dA_cumsum[(c * nheads + h) * chunk_size + l] = cumsum;
            }
        }
    }

    /* 2. chunk_state: per-chunk accumulated state
     *    state[c, h, p, n] = Σ_l exp(dA_cumsum[c,h,L-1] - dA_cumsum[c,h,l]) * dt[c,l,h] * B[c,l,g,n] * x[c,l,h,p]
     *    Shape: [nchunks, nheads, headdim, d_state] */
    float* chunk_states = (float*)calloc(nchunks * nheads * headdim * d_state, sizeof(float));

    for (int c = 0; c < nchunks; c++) {
        for (int h = 0; h < nheads; h++) {
            int g = h / heads_per_group;
            float dA_last = dA_cumsum[(c * nheads + h) * chunk_size + chunk_size - 1];

            for (int l = 0; l < chunk_size; l++) {
                int t = c * chunk_size + l;
                if (t >= seq_len) break;

                float dA_l = dA_cumsum[(c * nheads + h) * chunk_size + l];
                float decay = expf(dA_last - dA_l);
                float dt_val = dt[t * nheads + h];
                float scale = decay * dt_val;

                for (int p = 0; p < headdim; p++) {
                    float x_val = x[t * d_inner + h * headdim + p] * scale;
                    for (int n = 0; n < d_state; n++) {
                        chunk_states[((c * nheads + h) * headdim + p) * d_state + n]
                            += B[t * ngroups * d_state + g * d_state + n] * x_val;
                    }
                }
            }
        }
    }

    /* 3. state_passing: inter-chunk state propagation
     *    prev_states[c, h, p, n] = Σ_{c'<c} (product of inter-chunk decays) * chunk_states[c', h, p, n]
     *    Shape: [nchunks, nheads, headdim, d_state] */
    float* prev_states = (float*)calloc(nchunks * nheads * headdim * d_state, sizeof(float));

    if (nchunks > 1) {
        /* dA_chunk_cumsum[h] = cumulative sum of dA_cumsum[c,h,-1] over chunks */
        float* running_state = (float*)calloc(nheads * headdim * d_state, sizeof(float));

        for (int c = 0; c < nchunks; c++) {
            /* prev_states[c] = running_state (accumulated from chunks 0..c-1) */
            int base_prev = c * nheads * headdim * d_state;
            memcpy(&prev_states[base_prev], running_state,
                   nheads * headdim * d_state * sizeof(float));

            /* Update running_state: decay by chunk c's total dA, then add chunk_states[c] */
            for (int h = 0; h < nheads; h++) {
                float dA_total = dA_cumsum[(c * nheads + h) * chunk_size + chunk_size - 1];
                float inter_decay = expf(dA_total);

                for (int p = 0; p < headdim; p++) {
                    for (int n = 0; n < d_state; n++) {
                        int idx = (h * headdim + p) * d_state + n;
                        running_state[idx] = inter_decay * running_state[idx]
                            + chunk_states[((c * nheads + h) * headdim + p) * d_state + n];
                    }
                }
            }
        }
        free(running_state);
    }

    /* 4. chunk_scan: compute output per timestep
     *
     * For each (c, l, h):
     *   intra = Σ_{s<=l} exp(dA[l] - dA[s]) * dt[s] * (C[l] · B[s]) * x[s,h,:]  (within-chunk)
     *   inter = C[l] @ (exp(dA[l]) * prev_states[c,h,:,:])                        (from previous chunks)
     *   y[t,h,:] = intra + inter + D[h] * x[t,h,:]
     */
    #pragma omp parallel for schedule(static) if(nheads >= 4)
    for (int h = 0; h < nheads; h++) {
        int g = h / heads_per_group;

        for (int c = 0; c < nchunks; c++) {
            /* Precompute CB scores: CB[l,s] = C[c*L+l] · B[c*L+s] (dot product over d_state) */
            float* CB = (float*)calloc(chunk_size * chunk_size, sizeof(float));

            for (int l = 0; l < chunk_size; l++) {
                int t_l = c * chunk_size + l;
                if (t_l >= seq_len) break;
                for (int s = 0; s <= l; s++) {
                    int t_s = c * chunk_size + s;
                    if (t_s >= seq_len) break;
                    float dot = 0.0f;
                    for (int n = 0; n < d_state; n++) {
                        dot += C[t_l * ngroups * d_state + g * d_state + n]
                             * B[t_s * ngroups * d_state + g * d_state + n];
                    }
                    CB[l * chunk_size + s] = dot;
                }
            }

            /* Compute output for each timestep in this chunk */
            for (int l = 0; l < chunk_size; l++) {
                int t = c * chunk_size + l;
                if (t >= seq_len) break;

                float dA_l = dA_cumsum[(c * nheads + h) * chunk_size + l];

                /* Intra-chunk contribution (double precision 누적 — Triton matmul 오차 초월) */
                for (int p = 0; p < headdim; p++) {
                    double intra = 0.0;
                    for (int s = 0; s <= l; s++) {
                        int t_s = c * chunk_size + s;
                        float dA_s = dA_cumsum[(c * nheads + h) * chunk_size + s];
                        double decay_ls = exp((double)(dA_l - dA_s));
                        double dt_s = (double)dt[t_s * nheads + h];
                        double score = (double)CB[l * chunk_size + s] * decay_ls * dt_s;
                        intra += score * (double)x[t_s * d_inner + h * headdim + p];
                    }

                    /* Inter-chunk contribution */
                    double inter = 0.0;
                    double state_decay = exp((double)dA_l);
                    int ps_base = ((c * nheads + h) * headdim + p) * d_state;
                    for (int n = 0; n < d_state; n++) {
                        inter += (double)C[t * ngroups * d_state + g * d_state + n]
                               * (double)prev_states[ps_base + n] * state_decay;
                    }

                    /* Skip connection */
                    float skip = D[h] * x[t * d_inner + h * headdim + p];

                    y[t * d_inner + h * headdim + p] = (float)(intra + inter) + skip;
                }
            }

            free(CB);
        }
    }

    free(dA_cumsum);
    free(chunk_states);
    free(prev_states);
}

/* ── FP32 batch sgemm (AVX2 + FMA) ──────────────────────────
 * y[n,m] = w[m,k] @ x[n,k]^T
 * 즉 y[t,j] = Σ_i w[j,i] * x[t,i]
 * w: [m, k] row-major, x: [n, k] row-major, y: [n, m] row-major
 */
void f32_sgemm_avx2(
    const float* w,  /* [m, k] */
    const float* x,  /* [n, k] */
    float* y,        /* [n, m] */
    int m, int n, int k
) {
    /* row(m) 방향 병렬화 — 스레드당 weight 부분 행만 읽어 L2 캐시 적합 */
    #pragma omp parallel for schedule(static) if(m >= 64)
    for (int j = 0; j < m; j++) {
        const float* w_row = w + j * k;
        for (int t = 0; t < n; t++) {
            const float* x_t = x + t * k;
            __m256 vsum = _mm256_setzero_ps();
            int i = 0;
            for (; i + 8 <= k; i += 8) {
                __m256 vw = _mm256_loadu_ps(w_row + i);
                __m256 vx = _mm256_loadu_ps(x_t + i);
                vsum = _mm256_fmadd_ps(vw, vx, vsum);
            }
            /* horizontal sum */
            __m128 hi = _mm256_extractf128_ps(vsum, 1);
            __m128 lo = _mm256_castps256_ps128(vsum);
            __m128 s4 = _mm_add_ps(lo, hi);
            __m128 s2 = _mm_add_ps(s4, _mm_movehl_ps(s4, s4));
            __m128 s1 = _mm_add_ss(s2, _mm_shuffle_ps(s2, s2, 1));
            float sum = _mm_cvtss_f32(s1);
            /* scalar tail */
            for (; i < k; i++) {
                sum += w_row[i] * x_t[i];
            }
            y[t * m + j] = sum;
        }
    }
}

/* ── Ternary matmul (AVX2) ─────────────────────────────────
 * y[n,m] = gamma * (w_i8[m,k] @ x[n,k]^T)
 * w_i8[j,i] ∈ {-1, 0, +1} — i8→i32→f32 변환 후 FMA
 * 메모리 대역폭 4x 절약 (f32 weight 대비 i8)
 */
void ternary_f32_sgemm_avx2(
    const int8_t* w,  /* [m, k] — ternary {-1,0,+1} */
    const float* x,   /* [n, k] */
    float* y,         /* [n, m] */
    float gamma,
    int m, int n, int k
) {
    __m256 v_gamma = _mm256_set1_ps(gamma);

    #pragma omp parallel for schedule(static) if(m >= 64)
    for (int j = 0; j < m; j++) {
        const int8_t* w_row = w + j * k;
        for (int t = 0; t < n; t++) {
            const float* x_t = x + t * k;
            __m256 vsum = _mm256_setzero_ps();
            int i = 0;
            /* 8 i8 → 8 i32 → 8 f32, then FMA with x */
            for (; i + 8 <= k; i += 8) {
                /* _mm_loadl_epi64: 8 bytes (i8) → __m128i low 64-bit */
                __m128i w8 = _mm_loadl_epi64((const __m128i*)(w_row + i));
                __m256i w32 = _mm256_cvtepi8_epi32(w8);
                __m256 wf = _mm256_cvtepi32_ps(w32);
                __m256 vx = _mm256_loadu_ps(x_t + i);
                vsum = _mm256_fmadd_ps(wf, vx, vsum);
            }
            /* horizontal sum */
            __m128 hi = _mm256_extractf128_ps(vsum, 1);
            __m128 lo = _mm256_castps256_ps128(vsum);
            __m128 s4 = _mm_add_ps(lo, hi);
            __m128 s2 = _mm_add_ps(s4, _mm_movehl_ps(s4, s4));
            __m128 s1 = _mm_add_ss(s2, _mm_shuffle_ps(s2, s2, 1));
            float sum = _mm_cvtss_f32(s1);
            /* scalar tail */
            for (; i < k; i++) {
                sum += (float)w_row[i] * x_t[i];
            }
            y[t * m + j] = gamma * sum;
        }
    }
}
