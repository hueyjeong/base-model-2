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
#include <stdio.h>
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

    static int _ssd_prof_count = 0;
    int do_prof = (_ssd_prof_count < 2);
    double _tp[5];
    if (do_prof) _tp[0] = omp_get_wtime();

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

    if (do_prof) _tp[1] = omp_get_wtime();

    /* 2. chunk_state: per-chunk accumulated state (OMP 병렬화, scalar 누적 — bit-exact)
     *    state[c, h, p, n] = Σ_l exp(dA[c,h,L-1] - dA[c,h,l]) * dt[l,h] * B[l,g,n] * x[l,h,p]
     *    Shape: [nchunks, nheads, headdim, d_state] */
    float* chunk_states = (float*)calloc(nchunks * nheads * headdim * d_state, sizeof(float));

    #pragma omp parallel for schedule(static) if(nheads >= 4)
    for (int ch = 0; ch < nchunks * nheads; ch++) {
        int c = ch / nheads;
        int h = ch % nheads;
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

    if (do_prof) _tp[2] = omp_get_wtime();

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

    if (do_prof) _tp[3] = omp_get_wtime();

    /* 4. chunk_scan: compute output per timestep
     *
     * 최적화: exp(dA_l - dA_s)는 p(headdim)에 독립 → p 루프 밖에서 1회만 계산
     * exp() 호출 64x 감소 (headdim=64): 57M → 0.9M
     * 수치적으로 동일 (대수적 재배치, 동일 double 누적 순서)
     */
    #pragma omp parallel for schedule(static) if(nheads >= 4)
    for (int h = 0; h < nheads; h++) {
        int g = h / heads_per_group;

        /* 스레드-로컬 버퍼 (스택 할당 — malloc 회피) */
        double intra_buf[256];  /* max headdim */
        double score_buf[256];  /* max chunk_size */
        float CB[256 * 256];    /* max chunk_size² — 스택 (256KB, 스레드별) */

        for (int c = 0; c < nchunks; c++) {
            /* Precompute CB scores: CB[l,s] = C[c*L+l] · B[c*L+s] (AVX2 벡터화) */
            memset(CB, 0, chunk_size * chunk_size * sizeof(float));

            for (int l = 0; l < chunk_size; l++) {
                int t_l = c * chunk_size + l;
                if (t_l >= seq_len) break;
                for (int s = 0; s <= l; s++) {
                    int t_s = c * chunk_size + s;
                    if (t_s >= seq_len) break;
                    /* scalar dot — accumulation 순서 보존 (bit-exact) */
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

                /* Phase 1: score_buf[s] = CB[l,s] * exp(dA_l - dA_s) * dt[s,h]
                 *          p(headdim)에 독립 → 1회만 계산 (64x exp() 절감) */
                for (int s = 0; s <= l; s++) {
                    int t_s = c * chunk_size + s;
                    float dA_s = dA_cumsum[(c * nheads + h) * chunk_size + s];
                    double decay_ls = exp((double)(dA_l - dA_s));
                    double dt_s = (double)dt[t_s * nheads + h];
                    score_buf[s] = (double)CB[l * chunk_size + s] * decay_ls * dt_s;
                }

                /* Phase 2: intra[p] = Σ_s score[s] * x[s,h,p]
                 * scalar 누적 — 원본과 동일한 FMA contraction 보장 (bit-exact) */
                for (int p = 0; p < headdim; p++) {
                    double intra = 0.0;
                    for (int s = 0; s <= l; s++) {
                        int t_s = c * chunk_size + s;
                        intra += score_buf[s] * (double)x[t_s * d_inner + h * headdim + p];
                    }
                    intra_buf[p] = intra;
                }

                /* Phase 3: inter-chunk + skip → output (scalar — bit-exact 보장) */
                double state_decay = exp((double)dA_l);
                int ps_base_h = ((c * nheads + h) * headdim) * d_state;

                for (int p = 0; p < headdim; p++) {
                    double inter = 0.0;
                    int ps_base = ps_base_h + p * d_state;
                    for (int n = 0; n < d_state; n++) {
                        inter += (double)C[t * ngroups * d_state + g * d_state + n]
                               * (double)prev_states[ps_base + n] * state_decay;
                    }

                    float skip = D[h] * x[t * d_inner + h * headdim + p];
                    y[t * d_inner + h * headdim + p] = (float)(intra_buf[p] + inter) + skip;
                }
            }

            /* CB는 스택 할당 — free 불필요 */
        }
    }

    if (do_prof) {
        _tp[4] = omp_get_wtime();
        fprintf(stderr, "      [ssd_fwd %d] cumsum=%.1fms chunk_state=%.1fms state_pass=%.1fms scan=%.1fms total=%.1fms\n",
            _ssd_prof_count,
            (_tp[1]-_tp[0])*1000, (_tp[2]-_tp[1])*1000,
            (_tp[3]-_tp[2])*1000, (_tp[4]-_tp[3])*1000, (_tp[4]-_tp[0])*1000);
        _ssd_prof_count++;
    }

    free(dA_cumsum);
    free(chunk_states);
    free(prev_states);
}

/* ── Bitmask LUT 기반 ternary matmul ─────────────────────────
 * 곱셈 없이 sign-XOR + zero-AND + ADD로 ternary matmul 수행.
 * 가중치 메모리: 0.25 byte/weight (vs i8: 1 byte, f32: 4 bytes)
 * 스레드당 가중치가 L1 캐시에 적합 → 메모리 대역폭 병목 해소.
 */

/* 256-entry LUT: byte → 8×32-bit mask (16KB 합계, L1 상주) */
static __m256 _sign_flip_lut[256] __attribute__((aligned(32)));
static __m256 _nonzero_lut[256] __attribute__((aligned(32)));
static int _bitmask_lut_init = 0;

void init_bitmask_luts(void) {
    if (_bitmask_lut_init) return;
    for (int b = 0; b < 256; b++) {
        /* byte의 각 비트 (MSB-first: bit7→idx0, bit0→idx7) */
        uint32_t sf[8] __attribute__((aligned(32)));
        uint32_t nz[8] __attribute__((aligned(32)));
        for (int bit = 0; bit < 8; bit++) {
            int set = (b >> (7 - bit)) & 1;
            sf[bit] = set ? 0x80000000u : 0x00000000u;
            nz[bit] = set ? 0xFFFFFFFFu : 0x00000000u;
        }
        _sign_flip_lut[b] = _mm256_load_ps((const float*)sf);
        _nonzero_lut[b] = _mm256_load_ps((const float*)nz);
    }
    _bitmask_lut_init = 1;
}

/* packed 2-bit에서 sign_bits와 nonzero_bits를 추출 (모델 로드 시 1회)
 *
 * 2-bit 인코딩 (MSB-first, 4값/byte):
 *   0b00=0, 0b01=+1, 0b11=-1
 *   → lo bit = nonzero, hi bit = sign
 *
 * 출력 bitmask: MSB-first (bit7 = 첫 가중치)
 *   bitmask_stride = (cols + 7) / 8
 */
void extract_bitmasks(
    const uint8_t* packed,      /* [rows × packed_stride] */
    uint8_t* sign_bits,         /* [rows × bitmask_stride] output */
    uint8_t* nonzero_bits,      /* [rows × bitmask_stride] output */
    int rows, int cols, int packed_stride
) {
    int bitmask_stride = (cols + 7) / 8;
    memset(sign_bits, 0, (size_t)rows * bitmask_stride);
    memset(nonzero_bits, 0, (size_t)rows * bitmask_stride);

    for (int r = 0; r < rows; r++) {
        const uint8_t* p_row = packed + (int64_t)r * packed_stride;
        uint8_t* s_row = sign_bits + (int64_t)r * bitmask_stride;
        uint8_t* nz_row = nonzero_bits + (int64_t)r * bitmask_stride;

        for (int c = 0; c < cols; c++) {
            int byte_idx = c / 4;
            int bit_pos = (3 - (c % 4)) * 2;  /* MSB-first: c%4=0→6, 1→4, 2→2, 3→0 */
            uint8_t code = (p_row[byte_idx] >> bit_pos) & 0x03;

            /* lo bit (bit0) = nonzero, hi bit (bit1) = sign */
            int nz = code & 1;
            int sign = (code >> 1) & 1;

            /* bitmask도 MSB-first: c번째 가중치 → bit (7 - c%8) */
            int bm_byte = c / 8;
            int bm_bit = 7 - (c % 8);

            if (nz)   nz_row[bm_byte] |= (1 << bm_bit);
            if (sign)  s_row[bm_byte] |= (1 << bm_bit);
        }
    }
}

/* Bitmask 기반 ternary sgemm (AVX2)
 * y[n,m] = gamma * Σ (x_i with sign flip where w=-1, zeroed where w=0)
 *
 * 내부 루프:
 *   vx = load 8 f32 activations
 *   sf = sign_flip_lut[sign_byte]  → 0x80000000 where w=-1
 *   nz = nonzero_lut[nz_byte]     → 0xFFFFFFFF where w≠0
 *   vsum += (vx XOR sf) AND nz
 */
void ternary_bitmask_sgemm_avx2(
    const uint8_t* sign_bits,     /* [m × bitmask_stride] */
    const uint8_t* nonzero_bits,  /* [m × bitmask_stride] */
    const float* x,               /* [n × k] */
    float* y,                     /* [n × m] */
    float gamma,
    int m, int n, int k
) {
    int bitmask_stride = (k + 7) / 8;

    #pragma omp parallel for schedule(static) if(m >= 64)
    for (int j = 0; j < m; j++) {
        const uint8_t* s_row = sign_bits + (int64_t)j * bitmask_stride;
        const uint8_t* nz_row = nonzero_bits + (int64_t)j * bitmask_stride;

        for (int t = 0; t < n; t++) {
            const float* x_t = x + (int64_t)t * k;
            __m256 vsum = _mm256_setzero_ps();
            int i = 0;

            for (; i + 8 <= k; i += 8) {
                __m256 vx = _mm256_loadu_ps(x_t + i);
                __m256 sf = _sign_flip_lut[s_row[i >> 3]];
                __m256 nz = _nonzero_lut[nz_row[i >> 3]];
                __m256 val = _mm256_xor_ps(vx, sf);
                val = _mm256_and_ps(val, nz);
                vsum = _mm256_add_ps(vsum, val);
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
                int bm_byte = i / 8;
                int bm_bit = 7 - (i % 8);
                int nz = (nz_row[bm_byte] >> bm_bit) & 1;
                int sign = (s_row[bm_byte] >> bm_bit) & 1;
                if (nz) {
                    sum += sign ? -x_t[i] : x_t[i];
                }
            }

            y[(int64_t)t * m + j] = gamma * sum;
        }
    }
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
    #pragma omp parallel for schedule(static) if(m >= 64)
    for (int j = 0; j < m; j++) {
        const int8_t* w_row = w + j * k;
        for (int t = 0; t < n; t++) {
            const float* x_t = x + t * k;
            __m256 vsum = _mm256_setzero_ps();
            int i = 0;
            /* 8 i8 → 8 i32 → 8 f32, then FMA with x */
            for (; i + 8 <= k; i += 8) {
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
