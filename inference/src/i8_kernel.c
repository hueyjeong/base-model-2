/**
 * AVX-VNNI int8 matmul 커널 — BitMamba CPU 추론 엔진용
 *
 * vpdpbusd: u8 × i8 → i32 누적, 32개 동시 처리
 * BitLinear ternary {-1,0,+1} 및 Linear int8 양자화 weight 모두 지원
 *
 * 활성화는 u8 = (i8_quantized + 128)로 오프셋.
 * 보정: dot(x_u8, w_i8) = dot(x_i8, w_i8) + 128 * sum(w_row)
 *       → corrected = raw_dot - 128 * row_sum
 */

#include <immintrin.h>
#include <stdint.h>
#include <math.h>

/* ── AVX-VNNI i8 dot product ────────────────────────── */

static inline int32_t vnni_dot(const uint8_t* a, const int8_t* b, int n) {
    __m256i acc = _mm256_setzero_si256();
    int i;
    for (i = 0; i + 32 <= n; i += 32) {
        __m256i va = _mm256_loadu_si256((const __m256i*)(a + i));
        __m256i vb = _mm256_loadu_si256((const __m256i*)(b + i));
        acc = _mm256_dpbusd_avx_epi32(acc, va, vb);
    }
    /* horizontal sum of 8 × i32 */
    __m128i hi = _mm256_extracti128_si256(acc, 1);
    __m128i lo = _mm256_castsi256_si128(acc);
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
    #pragma omp parallel for schedule(static) if(m >= 64)
    for (int row = 0; row < m; row++) {
        int32_t dot = vnni_dot(x_u8, weights + (int64_t)row * n, n);
        int32_t corrected = dot - 128 * row_sums[row];
        float scale = (row_scales != 0) ? row_scales[row] * x_scale
                                        : w_scale * x_scale;
        y[row] = (float)corrected * scale;
    }
}

/* ── f32 → u8 양자화 (eta 반환) ───────────────────── */

float quantize_f32_to_u8(const float* x, uint8_t* out, int n) {
    /* max abs via AVX */
    __m256 vmax = _mm256_setzero_ps();
    __m256 sign_mask = _mm256_set1_ps(-0.0f);
    int i;
    for (i = 0; i + 8 <= n; i += 8) {
        __m256 v = _mm256_loadu_ps(x + i);
        vmax = _mm256_max_ps(vmax, _mm256_andnot_ps(sign_mask, v));
    }
    /* reduce */
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

    float inv_scale = 127.0f / eta;
    for (i = 0; i < n; i++) {
        float v = x[i] * inv_scale;
        int iv = (int)(v + (v >= 0 ? 0.5f : -0.5f));
        if (iv > 127) iv = 127;
        if (iv < -128) iv = -128;
        out[i] = (uint8_t)(iv + 128);
    }
    return eta / 127.0f;  /* x_scale */
}
