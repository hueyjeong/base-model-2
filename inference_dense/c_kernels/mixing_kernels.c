/**
 * Mixing Layer C 커널 — DenseEditor CPU 추론 엔진용 (최적화 v2)
 *
 * AVX2 FMA 벡터화 scan/conv 커널:
 *   1. mamba_scan_avx2       — Mamba selective scan (fast-exp)
 *   2. retention_scan_avx2   — RetNet retention scan
 *   3. slstm_scan_avx2       — sLSTM recurrent scan (vectorized activations)
 *   4. mlstm_scan_avx2       — mLSTM matrix scan (NEW)
 *   5. depthwise_conv1d_avx2 — dilated depthwise 1D conv (transposed weight)
 *
 * wkv6_scan_avx2는 i8_kernel_*.c에 정의 (중복 방지)
 */

#include <immintrin.h>
#include <stdint.h>
#include <math.h>
#include <string.h>

/* ── AVX2 fast exp 근사 (정밀도 ~1e-4, Schraudolph 방식 개선) ── */

static inline __m256 fast_exp_avx2(__m256 x) {
    /* exp(x) ≈ 2^(x / ln2) = 2^(n + f) where n=floor, f=frac
     * 2^f ≈ polynomial approximation */
    const __m256 log2e = _mm256_set1_ps(1.44269504089f);  /* 1/ln2 */
    const __m256 c0 = _mm256_set1_ps(1.0f);
    const __m256 c1 = _mm256_set1_ps(0.6931472f);    /* ln2 */
    const __m256 c2 = _mm256_set1_ps(0.2402265f);    /* ln2^2/2 */
    const __m256 c3 = _mm256_set1_ps(0.0558011f);    /* ln2^3/6 */
    const __m256 shift = _mm256_set1_ps(127.0f);
    const __m256 clamp_lo = _mm256_set1_ps(-87.0f);
    const __m256 clamp_hi = _mm256_set1_ps(88.0f);

    x = _mm256_max_ps(x, clamp_lo);
    x = _mm256_min_ps(x, clamp_hi);

    __m256 t = _mm256_mul_ps(x, log2e);
    __m256 n = _mm256_round_ps(t, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    __m256 f = _mm256_sub_ps(t, n);   /* fractional part */

    /* 2^n via integer bit manipulation */
    __m256i ni = _mm256_cvtps_epi32(_mm256_add_ps(n, shift));
    __m256i pow2n = _mm256_slli_epi32(ni, 23);
    __m256 exp2n = _mm256_castsi256_ps(pow2n);

    /* 2^f ≈ 1 + f*ln2 + f²*ln2²/2 + f³*ln2³/6 (3rd order) */
    __m256 f_ln2 = _mm256_mul_ps(f, c1);
    __m256 poly = _mm256_fmadd_ps(c3, f, c2);
    poly = _mm256_fmadd_ps(poly, f, c1);
    poly = _mm256_fmadd_ps(poly, f, c0);

    return _mm256_mul_ps(exp2n, poly);
}

static inline __m256 fast_sigmoid_avx2(__m256 x) {
    /* sigmoid(x) = 1 / (1 + exp(-x)) */
    __m256 neg_x = _mm256_sub_ps(_mm256_setzero_ps(), x);
    __m256 exp_neg = fast_exp_avx2(neg_x);
    __m256 one = _mm256_set1_ps(1.0f);
    return _mm256_div_ps(one, _mm256_add_ps(one, exp_neg));
}

static inline __m256 fast_tanh_avx2(__m256 x) {
    /* tanh(x) = 2*sigmoid(2x) - 1 */
    __m256 two = _mm256_set1_ps(2.0f);
    __m256 one = _mm256_set1_ps(1.0f);
    __m256 s = fast_sigmoid_avx2(_mm256_mul_ps(two, x));
    return _mm256_sub_ps(_mm256_mul_ps(two, s), one);
}

static inline __m256 fast_silu_avx2(__m256 x) {
    return _mm256_mul_ps(x, fast_sigmoid_avx2(x));
}


/* ── Mamba Selective Scan (fast-exp 최적화) ──────────── */

void mamba_scan_avx2(
    const float* delta,   /* [seq_len, d_inner] */
    const float* B,       /* [seq_len, d_state] */
    const float* C,       /* [seq_len, d_state] */
    const float* x,       /* [seq_len, d_inner] */
    const float* A,       /* [d_inner, d_state] (log-space, negative) */
    const float* D_skip,  /* [d_inner] */
    float* y,             /* [seq_len, d_inner] */
    float* state,         /* [d_inner, d_state] */
    int seq_len, int d_inner, int d_state
) {
    for (int t = 0; t < seq_len; t++) {
        const float* dt = delta + t * d_inner;
        const float* B_t = B + t * d_state;
        const float* C_t = C + t * d_state;
        const float* x_t = x + t * d_inner;
        float* y_t = y + t * d_inner;

        for (int i = 0; i < d_inner; i++) {
            float dt_i = dt[i];
            float x_i = x_t[i];
            float y_i = 0.0f;

            int j = 0;
            __m256 v_dt_x = _mm256_set1_ps(dt_i * x_i);
            __m256 v_dt = _mm256_set1_ps(dt_i);
            __m256 v_y = _mm256_setzero_ps();

            for (; j + 8 <= d_state; j += 8) {
                float* s = state + i * d_state + j;
                __m256 v_s = _mm256_loadu_ps(s);
                __m256 v_A = _mm256_loadu_ps(A + i * d_state + j);
                __m256 v_B = _mm256_loadu_ps(B_t + j);
                __m256 v_C = _mm256_loadu_ps(C_t + j);

                /* dA = exp(A * dt) — fast_exp 사용 */
                __m256 v_Adt = _mm256_mul_ps(v_A, v_dt);
                __m256 v_dA = fast_exp_avx2(v_Adt);
                __m256 v_dB = _mm256_mul_ps(v_B, v_dt_x);

                v_s = _mm256_fmadd_ps(v_dA, v_s, v_dB);
                _mm256_storeu_ps(s, v_s);
                v_y = _mm256_fmadd_ps(v_C, v_s, v_y);
            }

            /* horizontal sum */
            __m128 hi = _mm256_extractf128_ps(v_y, 1);
            __m128 lo = _mm256_castps256_ps128(v_y);
            __m128 sum4 = _mm_add_ps(lo, hi);
            sum4 = _mm_hadd_ps(sum4, sum4);
            sum4 = _mm_hadd_ps(sum4, sum4);
            y_i += _mm_cvtss_f32(sum4);

            /* scalar tail */
            for (; j < d_state; j++) {
                float* s = state + i * d_state + j;
                float dA = expf(A[i * d_state + j] * dt_i);
                *s = dA * (*s) + dt_i * B_t[j] * x_i;
                y_i += C_t[j] * (*s);
            }
            y_t[i] = y_i + D_skip[i] * x_i;
        }
    }
}


/* ── RetNet Retention Scan (최적화: gamma broadcast 밖으로) ── */

void retention_scan_avx2(
    const float* q, const float* k, const float* v,
    const float* gammas, float* output, float* state,
    int seq_len, int n_heads, int headdim
) {
    int d_model = n_heads * headdim;

    #pragma omp parallel for schedule(static) if(n_heads >= 4)
    for (int h = 0; h < n_heads; h++) {
        __m256 v_gamma = _mm256_set1_ps(gammas[h]);
        float* S = state + h * headdim * headdim;

        for (int t = 0; t < seq_len; t++) {
            int base = t * d_model + h * headdim;
            const float* q_t = q + base;
            const float* k_t = k + base;
            const float* v_t = v + base;
            float* o_t = output + base;

            /* S = gamma * S + outer(k, v) */
            for (int i = 0; i < headdim; i++) {
                __m256 v_ki = _mm256_set1_ps(k_t[i]);
                int j = 0;
                for (; j + 8 <= headdim; j += 8) {
                    float* s = S + i * headdim + j;
                    __m256 v_s = _mm256_loadu_ps(s);
                    __m256 v_vj = _mm256_loadu_ps(v_t + j);
                    v_s = _mm256_fmadd_ps(v_gamma, v_s, _mm256_mul_ps(v_ki, v_vj));
                    _mm256_storeu_ps(s, v_s);
                }
                for (; j < headdim; j++) {
                    S[i * headdim + j] = gammas[h] * S[i * headdim + j] + k_t[i] * v_t[j];
                }
            }

            /* o = S @ q */
            for (int i = 0; i < headdim; i++) {
                int j = 0;
                __m256 v_sum = _mm256_setzero_ps();
                for (; j + 8 <= headdim; j += 8) {
                    __m256 v_s = _mm256_loadu_ps(S + i * headdim + j);
                    __m256 v_q = _mm256_loadu_ps(q_t + j);
                    v_sum = _mm256_fmadd_ps(v_s, v_q, v_sum);
                }
                __m128 hi128 = _mm256_extractf128_ps(v_sum, 1);
                __m128 lo128 = _mm256_castps256_ps128(v_sum);
                __m128 s4 = _mm_add_ps(lo128, hi128);
                s4 = _mm_hadd_ps(s4, s4);
                s4 = _mm_hadd_ps(s4, s4);
                float sum = _mm_cvtss_f32(s4);
                for (; j < headdim; j++) sum += S[i * headdim + j] * q_t[j];
                o_t[i] = sum;
            }
        }
    }
}


/* ── sLSTM Scan (vectorized activations) ──────────── */

void slstm_scan_avx2(
    const float* i_gate, const float* f_gate,
    const float* z_gate, const float* o_gate,
    float* output, float* state_c, float* state_n,
    int seq_len, int d_model
) {
    __m256 v_one = _mm256_set1_ps(1.0f);
    __m256 v_clamp_lo = _mm256_set1_ps(-10.0f);
    __m256 v_clamp_hi = _mm256_set1_ps(10.0f);

    for (int t = 0; t < seq_len; t++) {
        int base = t * d_model;
        int d = 0;

        for (; d + 8 <= d_model; d += 8) {
            __m256 v_f = fast_sigmoid_avx2(_mm256_loadu_ps(f_gate + base + d));
            __m256 v_i_raw = _mm256_loadu_ps(i_gate + base + d);
            v_i_raw = _mm256_max_ps(v_i_raw, v_clamp_lo);
            v_i_raw = _mm256_min_ps(v_i_raw, v_clamp_hi);
            __m256 v_i = fast_exp_avx2(v_i_raw);
            __m256 v_z = fast_tanh_avx2(_mm256_loadu_ps(z_gate + base + d));
            __m256 v_o = fast_sigmoid_avx2(_mm256_loadu_ps(o_gate + base + d));

            __m256 v_c = _mm256_loadu_ps(state_c + d);
            __m256 v_n = _mm256_loadu_ps(state_n + d);

            /* c = f*c + i*z */
            v_c = _mm256_fmadd_ps(v_f, v_c, _mm256_mul_ps(v_i, v_z));
            /* n = f*n + i */
            v_n = _mm256_fmadd_ps(v_f, v_n, v_i);

            _mm256_storeu_ps(state_c + d, v_c);
            _mm256_storeu_ps(state_n + d, v_n);

            /* h = o * c / max(|n|, 1) */
            /* |n| */
            __m256 v_sign = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
            __m256 v_abs_n = _mm256_and_ps(v_n, v_sign);
            __m256 v_denom = _mm256_max_ps(v_abs_n, v_one);
            __m256 v_h = _mm256_mul_ps(v_o, _mm256_div_ps(v_c, v_denom));
            _mm256_storeu_ps(output + base + d, v_h);
        }

        /* scalar tail */
        for (; d < d_model; d++) {
            float f_k = 1.0f / (1.0f + expf(-f_gate[base + d]));
            float i_clamp = i_gate[base + d];
            if (i_clamp < -10.0f) i_clamp = -10.0f;
            if (i_clamp > 10.0f) i_clamp = 10.0f;
            float i_k = expf(i_clamp);
            float z_k = tanhf(z_gate[base + d]);
            float o_k = 1.0f / (1.0f + expf(-o_gate[base + d]));
            float c_new = f_k * state_c[d] + i_k * z_k;
            float n_new = f_k * state_n[d] + i_k;
            float abs_n = n_new < 0 ? -n_new : n_new;
            output[base + d] = o_k * c_new / (abs_n > 1.0f ? abs_n : 1.0f);
            state_c[d] = c_new;
            state_n[d] = n_new;
        }
    }
}


/* ── mLSTM Scan (matrix memory, O(d²) per step) ──── */

void mlstm_scan_avx2(
    const float* q,       /* [seq_len, n_heads * headdim] — query (post-projected) */
    const float* k,       /* [seq_len, n_heads * headdim] — key */
    const float* v,       /* [seq_len, n_heads * headdim] — value */
    const float* i_gate,  /* [seq_len, n_heads * headdim] — input gate (pre-exp) */
    const float* f_gate,  /* [seq_len, n_heads * headdim] — forget gate (pre-sigmoid) */
    float* output,        /* [seq_len, n_heads * headdim] */
    float* state_C,       /* [n_heads, headdim, headdim] — matrix cell */
    float* state_n,       /* [n_heads, headdim] — normalizer vector */
    int seq_len, int n_heads, int headdim
) {
    int d_model = n_heads * headdim;

    #pragma omp parallel for schedule(static) if(n_heads >= 4)
    for (int h = 0; h < n_heads; h++) {
        float* C = state_C + h * headdim * headdim;
        float* n = state_n + h * headdim;

        for (int t = 0; t < seq_len; t++) {
            int base = t * d_model + h * headdim;
            const float* q_t = q + base;
            const float* k_t = k + base;
            const float* v_t = v + base;

            /* per-head scalar gates (max over head dim for stability) */
            float max_if = -1e30f;
            for (int d = 0; d < headdim; d++) {
                float f_val = f_gate[base + d];
                float i_val = i_gate[base + d];
                if (f_val > max_if) max_if = f_val;
                if (i_val > max_if) max_if = i_val;
            }

            /* 간소화: per-head 단일 f, i 사용 (mean) */
            float f_sum = 0.0f, i_sum = 0.0f;
            for (int d = 0; d < headdim; d++) {
                f_sum += f_gate[base + d];
                i_sum += i_gate[base + d];
            }
            float f_scalar = 1.0f / (1.0f + expf(-f_sum / headdim));
            float i_clamp = i_sum / headdim;
            if (i_clamp < -10.0f) i_clamp = -10.0f;
            if (i_clamp > 10.0f) i_clamp = 10.0f;
            float i_scalar = expf(i_clamp);

            /* C = f * C + i * outer(k, v) */
            __m256 v_f = _mm256_set1_ps(f_scalar);
            for (int ki = 0; ki < headdim; ki++) {
                __m256 v_ki_val = _mm256_set1_ps(i_scalar * k_t[ki]);
                int j = 0;
                for (; j + 8 <= headdim; j += 8) {
                    float* c = C + ki * headdim + j;
                    __m256 v_c = _mm256_loadu_ps(c);
                    __m256 v_vj = _mm256_loadu_ps(v_t + j);
                    v_c = _mm256_fmadd_ps(v_f, v_c, _mm256_mul_ps(v_ki_val, v_vj));
                    _mm256_storeu_ps(c, v_c);
                }
                for (; j < headdim; j++) {
                    C[ki * headdim + j] = f_scalar * C[ki * headdim + j] + i_scalar * k_t[ki] * v_t[j];
                }
            }

            /* n = f * n + i * k */
            __m256 v_i_sc = _mm256_set1_ps(i_scalar);
            {
                int j = 0;
                for (; j + 8 <= headdim; j += 8) {
                    __m256 v_n = _mm256_loadu_ps(n + j);
                    __m256 v_k = _mm256_loadu_ps(k_t + j);
                    v_n = _mm256_fmadd_ps(v_f, v_n, _mm256_mul_ps(v_i_sc, v_k));
                    _mm256_storeu_ps(n + j, v_n);
                }
                for (; j < headdim; j++) {
                    n[j] = f_scalar * n[j] + i_scalar * k_t[j];
                }
            }

            /* h = C @ q */
            float* o_t = output + base;
            for (int i = 0; i < headdim; i++) {
                int j = 0;
                __m256 v_sum = _mm256_setzero_ps();
                for (; j + 8 <= headdim; j += 8) {
                    __m256 v_c = _mm256_loadu_ps(C + i * headdim + j);
                    __m256 v_q = _mm256_loadu_ps(q_t + j);
                    v_sum = _mm256_fmadd_ps(v_c, v_q, v_sum);
                }
                __m128 hi128 = _mm256_extractf128_ps(v_sum, 1);
                __m128 lo128 = _mm256_castps256_ps128(v_sum);
                __m128 s4 = _mm_add_ps(lo128, hi128);
                s4 = _mm_hadd_ps(s4, s4);
                s4 = _mm_hadd_ps(s4, s4);
                float val = _mm_cvtss_f32(s4);
                for (; j < headdim; j++) val += C[i * headdim + j] * q_t[j];

                /* normalize: h / max(|n @ q|, 1) */
                float nq = 0.0f;
                for (int jj = 0; jj < headdim; jj++) nq += n[jj] * q_t[jj];
                float abs_nq = nq < 0 ? -nq : nq;
                o_t[i] = val / (abs_nq > 1.0f ? abs_nq : 1.0f);
            }
        }
    }
}


/* ── Mamba-2 SSD Scan (스칼라 decay, head 병렬화) ──── */
/*
 * Mamba-1 vs Mamba-2 핵심 차이:
 *   Mamba-1: exp(A[d,n]*dt) 매 state 원소마다 → fast_exp 필수
 *   Mamba-2: decay[h] 스칼라 broadcast → FMA만으로 충분
 *
 * State: (nheads, d_state, headdim) matrix
 * 벡터화: headdim=64 → 8 AVX2 iterations per state dim (fully FMA)
 * 병렬화: OpenMP parallel over heads
 */

void mamba2_scan_avx2(
    const float* x,       /* [seq_len, nheads * headdim] — 입력 (conv+silu 후) */
    const float* B,       /* [seq_len, ngroups * d_state] — input projection */
    const float* C,       /* [seq_len, ngroups * d_state] — output projection */
    const float* decay,   /* [seq_len * nheads] — per-timestep decay: decay[t*nh+h] */
    const float* D_skip,  /* [nheads] — skip connection */
    const float* dt,      /* [seq_len * nheads] — per-timestep dt for x scaling */
    float* y,             /* [seq_len, nheads * headdim] — output */
    float* state,         /* [nheads, d_state, headdim] */
    int seq_len, int nheads, int headdim, int d_state, int ngroups
) {
    int d_inner = nheads * headdim;
    int heads_per_group = nheads / ngroups;

    #pragma omp parallel for schedule(static) if(nheads >= 4)
    for (int h = 0; h < nheads; h++) {
        int g = h / heads_per_group;
        float d_skip_h = D_skip[h];
        float* S = state + h * d_state * headdim;

        for (int t = 0; t < seq_len; t++) {
            float a = decay[t * nheads + h];
            float dt_val = dt[t * nheads + h];
            __m256 v_a = _mm256_set1_ps(a);
            __m256 v_dt = _mm256_set1_ps(dt_val);
            const float* x_t = x + t * d_inner + h * headdim;
            const float* B_t = B + t * ngroups * d_state + g * d_state;
            const float* C_t = C + t * ngroups * d_state + g * d_state;
            float* y_t = y + t * d_inner + h * headdim;

            /* State update: S[n,d] = a * S[n,d] + dt * B[n] * x[d] */
            for (int n = 0; n < d_state; n++) {
                float b_n = B_t[n] * dt_val;  /* dt scaling on B*x */
                __m256 v_b = _mm256_set1_ps(b_n);
                float* s = S + n * headdim;
                int d = 0;
                for (; d + 8 <= headdim; d += 8) {
                    __m256 v_s = _mm256_loadu_ps(s + d);
                    __m256 v_x = _mm256_loadu_ps(x_t + d);
                    v_s = _mm256_fmadd_ps(v_a, v_s, _mm256_mul_ps(v_b, v_x));
                    _mm256_storeu_ps(s + d, v_s);
                }
                for (; d < headdim; d++) {
                    s[d] = a * s[d] + b_n * x_t[d];
                }
            }

            /* Output: y[d] = Σ_n C[n] * S[n,d] + D * x[d] */
            int d = 0;
            for (; d + 8 <= headdim; d += 8) {
                __m256 v_y = _mm256_setzero_ps();
                for (int n = 0; n < d_state; n++) {
                    __m256 v_c = _mm256_set1_ps(C_t[n]);
                    __m256 v_s = _mm256_loadu_ps(S + n * headdim + d);
                    v_y = _mm256_fmadd_ps(v_c, v_s, v_y);
                }
                /* Skip connection: + D * x (original, not dt-scaled) */
                __m256 v_d = _mm256_set1_ps(d_skip_h);
                __m256 v_x = _mm256_loadu_ps(x_t + d);
                v_y = _mm256_fmadd_ps(v_d, v_x, v_y);
                _mm256_storeu_ps(y_t + d, v_y);
            }
            /* scalar tail */
            for (; d < headdim; d++) {
                float val = 0.0f;
                for (int n = 0; n < d_state; n++) {
                    val += C_t[n] * S[n * headdim + d];
                }
                y_t[d] = val + d_skip_h * x_t[d];
            }
        }
    }
}


/* ── Causal Depthwise 1D Conv (Mamba용, 왼쪽 패딩만) ── */

void causal_conv1d_avx2(
    const float* input,   /* [seq_len, channels] */
    const float* weight,  /* [channels, kernel_size] */
    const float* bias,    /* [channels] or NULL */
    float* output,        /* [seq_len, channels] */
    int seq_len, int channels, int kernel_size
) {
    #pragma omp parallel for schedule(static) if(seq_len >= 64)
    for (int t = 0; t < seq_len; t++) {
        int d = 0;
        for (; d + 8 <= channels; d += 8) {
            __m256 v_sum = bias ? _mm256_loadu_ps(bias + d) : _mm256_setzero_ps();
            for (int ki = 0; ki < kernel_size; ki++) {
                int src_t = t - ki;  /* causal: t, t-1, t-2, ... */
                if (src_t < 0) continue;
                float w_arr[8];
                for (int c = 0; c < 8; c++)
                    w_arr[c] = weight[(d + c) * kernel_size + ki];
                __m256 v_w = _mm256_loadu_ps(w_arr);
                __m256 v_x = _mm256_loadu_ps(input + src_t * channels + d);
                v_sum = _mm256_fmadd_ps(v_w, v_x, v_sum);
            }
            _mm256_storeu_ps(output + t * channels + d, v_sum);
        }
        for (; d < channels; d++) {
            float sum = bias ? bias[d] : 0.0f;
            for (int ki = 0; ki < kernel_size; ki++) {
                int src_t = t - ki;
                if (src_t < 0) continue;
                sum += weight[d * kernel_size + ki] * input[src_t * channels + d];
            }
            output[t * channels + d] = sum;
        }
    }
}


/* ── Depthwise 1D Dilated Conv (전치 weight 레이아웃) ── */

void depthwise_conv1d_avx2(
    const float* input,   /* [seq_len, d_model] */
    const float* weight,  /* [d_model, kernel_size] */
    const float* bias,    /* [d_model] or NULL */
    float* output,        /* [seq_len, d_model] */
    int seq_len, int d_model, int kernel_size, int dilation
) {
    #pragma omp parallel for schedule(static) if(seq_len >= 64)
    for (int t = 0; t < seq_len; t++) {
        int d = 0;
        for (; d + 8 <= d_model; d += 8) {
            __m256 v_sum = bias ? _mm256_loadu_ps(bias + d) : _mm256_setzero_ps();
            for (int ki = 0; ki < kernel_size; ki++) {
                int src_t = t + (ki - kernel_size / 2) * dilation;
                if (src_t < 0 || src_t >= seq_len) continue;
                float w_arr[8];
                for (int c = 0; c < 8; c++)
                    w_arr[c] = weight[(d + c) * kernel_size + ki];
                __m256 v_w = _mm256_loadu_ps(w_arr);
                __m256 v_x = _mm256_loadu_ps(input + src_t * d_model + d);
                v_sum = _mm256_fmadd_ps(v_w, v_x, v_sum);
            }
            _mm256_storeu_ps(output + t * d_model + d, v_sum);
        }
        for (; d < d_model; d++) {
            float sum = bias ? bias[d] : 0.0f;
            for (int ki = 0; ki < kernel_size; ki++) {
                int src_t = t + (ki - kernel_size / 2) * dilation;
                if (src_t < 0 || src_t >= seq_len) continue;
                sum += weight[d * kernel_size + ki] * input[src_t * d_model + d];
            }
            output[t * d_model + d] = sum;
        }
    }
}
