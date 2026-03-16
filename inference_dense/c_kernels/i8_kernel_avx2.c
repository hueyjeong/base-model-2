/**
 * AVX2-only int8 matmul 커널 — BitEditor CPU 추론 엔진용
 *
 * AVX-VNNI(vpdpbusd) 없이 AVX2만으로 u8 × i8 → i32 dot product 수행.
 *
 * ternary {-1,0,+1} 가중치: _mm256_maddubs_epi16 사용 가능 (i16 오버플로 없음)
 * 일반 i8 가중치: i16 widening 방식 (_mm256_cvtepu8_epi16 + _mm256_madd_epi16)
 *
 * 활성화는 u8 = (i8_quantized + 128)로 오프셋.
 * 보정: dot(x_u8, w_i8) = dot(x_i8, w_i8) + 128 * sum(w_row)
 *       → corrected = raw_dot - 128 * row_sum
 */

#include <immintrin.h>
#include <stdint.h>
#include <math.h>

/* ── AVX2 i8 dot product (i16 widening, 오버플로 안전) ── */

static inline int32_t avx2_dot_i8(const uint8_t* a, const int8_t* b, int n) {
    __m256i acc0 = _mm256_setzero_si256();
    __m256i acc1 = _mm256_setzero_si256();
    int i;

    /* 32 bytes/iter: 16 elements × 2 반복 (i16 widening) */
    for (i = 0; i + 32 <= n; i += 32) {
        /* 첫 16 elements */
        __m128i a128_0 = _mm_loadu_si128((const __m128i*)(a + i));
        __m128i b128_0 = _mm_loadu_si128((const __m128i*)(b + i));
        __m256i a16_0 = _mm256_cvtepu8_epi16(a128_0);   /* u8 → u16 zero-extend */
        __m256i b16_0 = _mm256_cvtepi8_epi16(b128_0);   /* i8 → i16 sign-extend */
        acc0 = _mm256_add_epi32(acc0, _mm256_madd_epi16(a16_0, b16_0));

        /* 다음 16 elements */
        __m128i a128_1 = _mm_loadu_si128((const __m128i*)(a + i + 16));
        __m128i b128_1 = _mm_loadu_si128((const __m128i*)(b + i + 16));
        __m256i a16_1 = _mm256_cvtepu8_epi16(a128_1);
        __m256i b16_1 = _mm256_cvtepi8_epi16(b128_1);
        acc1 = _mm256_add_epi32(acc1, _mm256_madd_epi16(a16_1, b16_1));
    }
    /* 나머지 16-byte 청크 */
    for (; i + 16 <= n; i += 16) {
        __m128i a128 = _mm_loadu_si128((const __m128i*)(a + i));
        __m128i b128 = _mm_loadu_si128((const __m128i*)(b + i));
        __m256i a16 = _mm256_cvtepu8_epi16(a128);
        __m256i b16 = _mm256_cvtepi8_epi16(b128);
        acc0 = _mm256_add_epi32(acc0, _mm256_madd_epi16(a16, b16));
    }
    /* acc0 + acc1 합산 */
    acc0 = _mm256_add_epi32(acc0, acc1);
    /* horizontal sum of 8 × i32 */
    __m128i hi = _mm256_extracti128_si256(acc0, 1);
    __m128i lo = _mm256_castsi256_si128(acc0);
    __m128i s = _mm_add_epi32(lo, hi);
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0x4E));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0xB1));
    int32_t result = _mm_cvtsi128_si32(s);
    /* scalar tail */
    for (; i < n; i++) {
        result += (int32_t)a[i] * (int32_t)b[i];
    }
    return result;
}

/* ── AVX2 ternary dot product (maddubs 사용, ternary 안전) ── */

static inline int32_t avx2_dot_ternary(const uint8_t* a, const int8_t* b, int n) {
    __m256i acc0 = _mm256_setzero_si256();
    __m256i acc1 = _mm256_setzero_si256();
    __m256i ones = _mm256_set1_epi16(1);
    int i;

    /* 64 bytes/iter, 2x 언롤
     * maddubs: u8*i8 → i16 pairwise sum (ternary이므로 최대 ±510, i16 안전)
     * madd: i16 pairwise → i32 */
    for (i = 0; i + 64 <= n; i += 64) {
        __m256i va0 = _mm256_loadu_si256((const __m256i*)(a + i));
        __m256i vb0 = _mm256_loadu_si256((const __m256i*)(b + i));
        __m256i va1 = _mm256_loadu_si256((const __m256i*)(a + i + 32));
        __m256i vb1 = _mm256_loadu_si256((const __m256i*)(b + i + 32));
        __m256i prod0 = _mm256_maddubs_epi16(va0, vb0);
        __m256i prod1 = _mm256_maddubs_epi16(va1, vb1);
        acc0 = _mm256_add_epi32(acc0, _mm256_madd_epi16(prod0, ones));
        acc1 = _mm256_add_epi32(acc1, _mm256_madd_epi16(prod1, ones));
    }
    /* 나머지 32-byte 청크 */
    for (; i + 32 <= n; i += 32) {
        __m256i va = _mm256_loadu_si256((const __m256i*)(a + i));
        __m256i vb = _mm256_loadu_si256((const __m256i*)(b + i));
        __m256i prod = _mm256_maddubs_epi16(va, vb);
        acc0 = _mm256_add_epi32(acc0, _mm256_madd_epi16(prod, ones));
    }
    /* acc0 + acc1 합산 */
    acc0 = _mm256_add_epi32(acc0, acc1);
    /* horizontal sum of 8 × i32 */
    __m128i hi = _mm256_extracti128_si256(acc0, 1);
    __m128i lo = _mm256_castsi256_si128(acc0);
    __m128i s = _mm_add_epi32(lo, hi);
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0x4E));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0xB1));
    int32_t result = _mm_cvtsi128_si32(s);
    /* scalar tail */
    for (; i < n; i++) {
        result += (int32_t)a[i] * (int32_t)b[i];
    }
    return result;
}

/* ── sgemv: y[m] = (A_i8[m,n] · x_u8[n] - 128*row_sum[m]) * row_scale[m] * x_scale ── */

void i8_sgemv(
    const int8_t* weights,        /* [m × n] row-major */
    const uint8_t* x_u8,          /* [n] = clamp(round(x/eta*127), -128, 127) + 128 */
    float* y,                      /* [m] output */
    int m, int n,
    const int32_t* row_sums,      /* [m] Σ_j weights[i,j] — 사전 계산 */
    const float* row_scales,      /* [m] per-row dequant scale (NULL → use w_scale) */
    float x_scale,                 /* activation dequant: eta / 127 */
    float w_scale                  /* global weight scale (BitLinear gamma; Linear이면 0) */
) {
    #pragma omp parallel for schedule(static)
    for (int row = 0; row < m; row++) {
        int32_t dot = avx2_dot_i8(x_u8, (const int8_t*)(weights + (int64_t)row * n), n);
        int32_t corrected = dot - 128 * row_sums[row];
        float scale = (row_scales != 0) ? row_scales[row] * x_scale
                                        : w_scale * x_scale;
        y[row] = (float)corrected * scale;
    }
}

/* ── 2-bit ternary → i8 언팩 (byte LUT 방식, AVX2 전용) ──── */

static int32_t _byte_lut[256]; /* 4xi8 packed as int32_t */
static int _byte_lut_init = 0;

static const int8_t _2bit_val[4] = {0, 1, 0, -1};

static void init_byte_lut(void) {
    if (_byte_lut_init) return;
    for (int b = 0; b < 256; b++) {
        int8_t vals[4];
        vals[0] = _2bit_val[(b >> 6) & 3];
        vals[1] = _2bit_val[(b >> 4) & 3];
        vals[2] = _2bit_val[(b >> 2) & 3];
        vals[3] = _2bit_val[ b       & 3];
        __builtin_memcpy(&_byte_lut[b], vals, 4);
    }
    _byte_lut_init = 1;
}

static inline void unpack_2bit_row(
    const uint8_t* packed, int8_t* out, int cols
) {
    init_byte_lut();
    int full_bytes = cols / 4;
    int b = 0;
    /* 8 bytes → 32 i8 values */
    for (; b + 8 <= full_bytes; b += 8) {
        int8_t* dst = out + b * 4;
        __builtin_memcpy(dst,      &_byte_lut[packed[b]],     4);
        __builtin_memcpy(dst + 4,  &_byte_lut[packed[b + 1]], 4);
        __builtin_memcpy(dst + 8,  &_byte_lut[packed[b + 2]], 4);
        __builtin_memcpy(dst + 12, &_byte_lut[packed[b + 3]], 4);
        __builtin_memcpy(dst + 16, &_byte_lut[packed[b + 4]], 4);
        __builtin_memcpy(dst + 20, &_byte_lut[packed[b + 5]], 4);
        __builtin_memcpy(dst + 24, &_byte_lut[packed[b + 6]], 4);
        __builtin_memcpy(dst + 28, &_byte_lut[packed[b + 7]], 4);
    }
    /* 나머지 */
    for (; b < full_bytes; b++) {
        __builtin_memcpy(out + b * 4, &_byte_lut[packed[b]], 4);
    }
    /* tail (cols가 4의 배수가 아닌 경우) */
    int c = full_bytes * 4;
    if (c < cols) {
        int8_t tmp[4];
        __builtin_memcpy(tmp, &_byte_lut[packed[full_bytes]], 4);
        for (int i = 0; c < cols; i++, c++) out[c] = tmp[i];
    }
    /* 32바이트 정렬 패딩 */
    int aligned = (cols + 31) & ~31;
    for (int i = cols; i < aligned; i++) out[i] = 0;
}

/* ── 2-bit packed → i8 배치 언팩 (모델 로드 시 1회 호출) ──── */

void unpack_2bit_rows(
    const uint8_t* packed,   /* [rows × packed_stride] packed 2-bit */
    int8_t* out,             /* [rows × cols] i8 output (row-major) */
    int rows, int cols, int packed_stride
) {
    init_byte_lut();
    for (int row = 0; row < rows; row++) {
        unpack_2bit_row(
            packed + (int64_t)row * packed_stride,
            out + (int64_t)row * cols,
            cols
        );
    }
}

/* ── ternary_sgemv: packed 2-bit weight × u8 activation → f32 output ── */

void ternary_sgemv(
    const uint8_t* packed_weights,  /* [m × packed_stride] packed 2-bit */
    const uint8_t* x_u8,           /* [n] quantized activation */
    float* y,                       /* [m] output */
    int m, int n, int packed_stride,
    const int32_t* row_sums,       /* [m] Σ_j w[i,j] — 사전 계산 */
    float gamma,                    /* BitLinear gamma */
    float x_scale                   /* activation dequant scale */
) {
    float combined_scale = gamma * x_scale;
    for (int row = 0; row < m; row++) {
        /* thread-local 언팩 버퍼 */
        static __thread int8_t unpack_buf[8192];
        unpack_2bit_row(
            packed_weights + (int64_t)row * packed_stride,
            unpack_buf, n
        );
        int32_t dot = avx2_dot_ternary(x_u8, unpack_buf, n);
        int32_t corrected = dot - 128 * row_sums[row];
        y[row] = (float)corrected * combined_scale;
    }
}

/* ── f32 → u8 양자화 (AVX2 벡터화, eta 반환) ──────── */

float quantize_f32_to_u8(const float* x, uint8_t* out, int n) {
    /* Pass 1: max abs via AVX2 */
    __m256 vmax = _mm256_setzero_ps();
    __m256 sign_mask = _mm256_set1_ps(-0.0f);
    int i;
    for (i = 0; i + 8 <= n; i += 8) {
        __m256 v = _mm256_loadu_ps(x + i);
        vmax = _mm256_max_ps(vmax, _mm256_andnot_ps(sign_mask, v));
    }
    /* horizontal max reduce */
    __m128 h = _mm256_extractf128_ps(vmax, 1);
    __m128 l = _mm256_castps256_ps128(vmax);
    __m128 m128 = _mm_max_ps(l, h);
    m128 = _mm_max_ps(m128, _mm_shuffle_ps(m128, m128, 0x4E));
    m128 = _mm_max_ps(m128, _mm_shuffle_ps(m128, m128, 0xB1));
    float eta;
    _mm_store_ss(&eta, m128);
    for (; i < n; i++) {
        float a = fabsf(x[i]);
        if (a > eta) eta = a;
    }
    if (eta < 1e-5f) eta = 1e-5f;

    /* Pass 2: 양자화 — AVX2 벡터화 */
    __m256 v_inv_scale = _mm256_set1_ps(127.0f / eta);
    __m256 v_min = _mm256_set1_ps(-128.0f);
    __m256 v_max = _mm256_set1_ps(127.0f);
    __m256 v_offset = _mm256_set1_ps(128.0f);

    for (i = 0; i + 8 <= n; i += 8) {
        /* scale */
        __m256 v = _mm256_mul_ps(_mm256_loadu_ps(x + i), v_inv_scale);
        /* round to nearest */
        v = _mm256_round_ps(v, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        /* clamp [-128, 127] */
        v = _mm256_max_ps(v, v_min);
        v = _mm256_min_ps(v, v_max);
        /* +128 → [0, 255] */
        v = _mm256_add_ps(v, v_offset);
        /* float → int32 → pack to u8 */
        __m256i vi = _mm256_cvtps_epi32(v);
        /* pack i32 → i16 (with saturation) */
        vi = _mm256_packs_epi32(vi, vi);
        /* pack i16 → u8 (with unsigned saturation) */
        vi = _mm256_packus_epi16(vi, vi);
        /* 256-bit pack 결과는 lane 꼬임 — 하위 4바이트씩 추출 */
        uint32_t lo4 = (uint32_t)_mm256_extract_epi32(vi, 0);
        uint32_t hi4 = (uint32_t)_mm256_extract_epi32(vi, 4);
        __builtin_memcpy(out + i, &lo4, 4);
        __builtin_memcpy(out + i + 4, &hi4, 4);
    }
    /* scalar tail */
    float inv_scale = 127.0f / eta;
    for (; i < n; i++) {
        float v = x[i] * inv_scale;
        int iv = (int)roundf(v);
        if (iv > 127) iv = 127;
        if (iv < -128) iv = -128;
        out[i] = (uint8_t)(iv + 128);
    }
    return eta / 127.0f;  /* x_scale */
}

/* ── 배치 f32 → u8 양자화 (seq_len 토큰을 한 번에) ──── */

void batch_quantize_f32_to_u8(
    const float* x,       /* [seq_len × d] row-major */
    uint8_t* out,         /* [seq_len × d] row-major */
    float* scales,        /* [seq_len] per-token x_scale 출력 */
    int seq_len, int d
) {
    #pragma omp parallel for schedule(static)
    for (int t = 0; t < seq_len; t++) {
        scales[t] = quantize_f32_to_u8(x + (int64_t)t * d, out + (int64_t)t * d, d);
    }
}

/* ── i8 sgemm: X_u8[n,k] × W_i8[m,k]^T → Y[n,m] ──── */

void i8_sgemm(
    const int8_t* W,          /* [m, k] row-major */
    const uint8_t* X_u8,      /* [n, k] row-major (n=seq_len) */
    float* Y,                 /* [n, m] row-major */
    int m, int n, int k,
    const int32_t* row_sums,  /* [m] Σ W[j,:] */
    const float* row_scales,  /* [m] per-row scale (NULL → w_scale) */
    const float* x_scales,    /* [n] per-token scale */
    float w_scale             /* global scale (BitLinear gamma; 0 for Linear) */
) {
    #pragma omp parallel for schedule(dynamic, 4)
    for (int j = 0; j < m; j++) {
        const int8_t* w_row = W + (int64_t)j * k;
        int32_t rsum = row_sums[j];
        float w_s = (row_scales != 0) ? row_scales[j] : w_scale;

        for (int t = 0; t < n; t++) {
            int32_t dot = avx2_dot_i8(X_u8 + (int64_t)t * k, w_row, k);
            int32_t corrected = dot - 128 * rsum;
            Y[(int64_t)t * m + j] = (float)corrected * w_s * x_scales[t];
        }
    }
}
void wkv6_scan_avx2(
    const float* r,       /* [seq_len × d_model] */
    const float* k,       /* [seq_len × d_model] */
    const float* v,       /* [seq_len × d_model] */
    const float* w,       /* [seq_len × d_model] — raw decay (before -softplus) */
    const float* u_param, /* [n_heads × headdim] — in-context bonus */
    float* output,        /* [seq_len × d_model] */
    float* state,         /* [n_heads × 32 × 32] — 호출자가 0으로 초기화 */
    int seq_len, int n_heads, int headdim, int d_model
) {
    /* headdim=32 전용 최적화 (4 × __m256) */
    const int HD = 32;
    /* assert(headdim == 32) */

    for (int t = 0; t < seq_len; t++) {
        const float* r_t = r + (int64_t)t * d_model;
        const float* k_t = k + (int64_t)t * d_model;
        const float* v_t = v + (int64_t)t * d_model;
        const float* w_t = w + (int64_t)t * d_model;
        float* out_t = output + (int64_t)t * d_model;

        for (int h = 0; h < n_heads; h++) {
            int h_off = h * HD;
            float* S = state + h * HD * HD;  /* state[h]: 32×32 */

            /* v[t,h,:] → 4 AVX2 레지스터 */
            __m256 v0 = _mm256_loadu_ps(v_t + h_off);
            __m256 v1 = _mm256_loadu_ps(v_t + h_off + 8);
            __m256 v2 = _mm256_loadu_ps(v_t + h_off + 16);
            __m256 v3 = _mm256_loadu_ps(v_t + h_off + 24);

            /* output 누적기 초기화 */
            __m256 out0 = _mm256_setzero_ps();
            __m256 out1 = _mm256_setzero_ps();
            __m256 out2 = _mm256_setzero_ps();
            __m256 out3 = _mm256_setzero_ps();

            /* 행별 순회: i = 0..31 */
            for (int i = 0; i < HD; i++) {
                float* si = S + i * HD;  /* state[h][i][:] */

                /* 스칼라 값 로드 */
                float u_val = u_param[h_off + i];
                float k_val = k_t[h_off + i];
                float r_val = r_t[h_off + i];
                float w_raw = w_t[h_off + i];
                float decay = 1.0f / (1.0f + expf(w_raw));

                /* uk = u[i] * k[i] */
                __m256 vuk = _mm256_set1_ps(u_val * k_val);
                /* r_i = r[i] */
                __m256 vr = _mm256_set1_ps(r_val);
                /* decay_i */
                __m256 vdecay = _mm256_set1_ps(decay);
                /* k_i */
                __m256 vk = _mm256_set1_ps(k_val);

                /* state[i][:] 로드 (4 × __m256) */
                __m256 s0 = _mm256_loadu_ps(si);
                __m256 s1 = _mm256_loadu_ps(si + 8);
                __m256 s2 = _mm256_loadu_ps(si + 16);
                __m256 s3 = _mm256_loadu_ps(si + 24);

                /* kv_bonus[i][:] = state[i][:] + uk * v[:] */
                __m256 b0 = _mm256_fmadd_ps(vuk, v0, s0);
                __m256 b1 = _mm256_fmadd_ps(vuk, v1, s1);
                __m256 b2 = _mm256_fmadd_ps(vuk, v2, s2);
                __m256 b3 = _mm256_fmadd_ps(vuk, v3, s3);

                /* output[:] += kv_bonus[i][:] * r[i] */
                out0 = _mm256_fmadd_ps(b0, vr, out0);
                out1 = _mm256_fmadd_ps(b1, vr, out1);
                out2 = _mm256_fmadd_ps(b2, vr, out2);
                out3 = _mm256_fmadd_ps(b3, vr, out3);

                /* state[i][:] = decay * state[i][:] + k[i] * v[:] */
                s0 = _mm256_fmadd_ps(vdecay, s0, _mm256_mul_ps(vk, v0));
                s1 = _mm256_fmadd_ps(vdecay, s1, _mm256_mul_ps(vk, v1));
                s2 = _mm256_fmadd_ps(vdecay, s2, _mm256_mul_ps(vk, v2));
                s3 = _mm256_fmadd_ps(vdecay, s3, _mm256_mul_ps(vk, v3));

                /* state 저장 */
                _mm256_storeu_ps(si, s0);
                _mm256_storeu_ps(si + 8, s1);
                _mm256_storeu_ps(si + 16, s2);
                _mm256_storeu_ps(si + 24, s3);
            }

            /* output 저장 */
            _mm256_storeu_ps(out_t + h_off, out0);
            _mm256_storeu_ps(out_t + h_off + 8, out1);
            _mm256_storeu_ps(out_t + h_off + 16, out2);
            _mm256_storeu_ps(out_t + h_off + 24, out3);
        }
    }
}
