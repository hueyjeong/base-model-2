//! BitEditor 모델 — CPU 추론 엔진
//!
//! RWKV6 양방향 인코더 + MoE FFN + Shared Linear Self-Attention 기반
//! 편집 태그 예측을 통한 한국어 문법 교정

use anyhow::{bail, Context, Result};
use std::collections::{HashMap, HashSet};

use crate::bmmq::{self, TensorData};
use crate::config::ModelConfig;

// ── OpenBLAS FFI ─────────────────────────────────────

#[allow(non_camel_case_types)]
type c_int = i32;

const CBLAS_ROW_MAJOR: c_int = 101;
#[allow(dead_code)]
const CBLAS_NO_TRANS: c_int = 111;

extern "C" {
    fn cblas_sgemv(
        order: c_int,
        trans: c_int,
        m: c_int,
        n: c_int,
        alpha: f32,
        a: *const f32,
        lda: c_int,
        x: *const f32,
        incx: c_int,
        beta: f32,
        y: *mut f32,
        incy: c_int,
    );
}

// ── AVX-VNNI int8 커널 FFI ──────────────────────────

extern "C" {
    /// i8 weight × u8 activation → f32 output (AVX-VNNI vpdpbusd)
    fn i8_sgemv(
        weights: *const i8,
        x_u8: *const u8,
        y: *mut f32,
        m: c_int,
        n: c_int,
        row_sums: *const i32,
        row_scales: *const f32,  // NULL이면 w_scale 사용
        x_scale: f32,
        w_scale: f32,
    );

    /// 2-bit packed ternary weight × u8 activation → f32 output
    fn ternary_sgemv(
        packed_weights: *const u8,
        x_u8: *const u8,
        y: *mut f32,
        m: c_int,
        n: c_int,
        packed_stride: c_int,
        row_sums: *const i32,
        gamma: f32,
        x_scale: f32,
    );

    /// f32 → u8 양자화 (absmax 기반, x_scale 반환)
    fn quantize_f32_to_u8(
        x: *const f32,
        out: *mut u8,
        n: c_int,
    ) -> f32;

    /// 2-bit packed → i8 배치 언팩 (모델 로드 시 1회 호출)
    fn unpack_2bit_rows(
        packed: *const u8,
        out: *mut i8,
        rows: c_int,
        cols: c_int,
        packed_stride: c_int,
    );

    /// 배치 f32 → u8 양자화 (seq_len 토큰을 한 번에)
    fn batch_quantize_f32_to_u8(
        x: *const f32,
        out: *mut u8,
        scales: *mut f32,
        seq_len: c_int,
        d: c_int,
    );

    /// i8 sgemm: X_u8[n,k] × W_i8[m,k]^T → Y[n,m]
    fn i8_sgemm(
        w: *const i8,
        x_u8: *const u8,
        y: *mut f32,
        m: c_int,
        n: c_int,
        k: c_int,
        row_sums: *const i32,
        row_scales: *const f32,
        x_scales: *const f32,
        w_scale: f32,
    );

    /// WKV-6 순차 스캔 (AVX2 FMA, headdim=32 전용)
    fn wkv6_scan_avx2(
        r: *const f32,
        k: *const f32,
        v: *const f32,
        w: *const f32,
        u_param: *const f32,
        output: *mut f32,
        state: *mut f32,
        seq_len: c_int,
        n_heads: c_int,
        headdim: c_int,
        d_model: c_int,
    );
}

// ── BMMQ 헬퍼 ────────────────────────────────────────

/// BMMQ TensorData에서 f32 Vec 추출 (소유권 이전)
fn bmmq_take_f32(tensors: &mut HashMap<String, TensorData>, key: &str) -> Result<Vec<f32>> {
    match tensors.remove(key).context(format!("텐서 없음: {}", key))? {
        TensorData::F32 { data, .. } => Ok(data),
        _ => bail!("f32 타입이어야 함: {}", key),
    }
}

/// BMMQ TensorData에서 i8 데이터 추출 (I8Quantized)
fn bmmq_take_i8(tensors: &mut HashMap<String, TensorData>, key: &str)
    -> Result<(Vec<i8>, Vec<f32>, Vec<i32>, usize, usize)>
{
    match tensors.remove(key).context(format!("텐서 없음: {}", key))? {
        TensorData::I8Quantized { data, row_scales, row_sums, rows, cols } => {
            Ok((data, row_scales, row_sums, rows, cols))
        }
        TensorData::F32 { data, shape } => {
            // f32 → per-row i8 양자화 (소형 텐서 폴백)
            let (rows, cols) = (shape[0], shape[1]);
            let mut w_i8 = vec![0i8; rows * cols];
            let mut row_scales = vec![0.0f32; rows];
            let mut row_sums = vec![0i32; rows];
            for row in 0..rows {
                let base = row * cols;
                let mut max_abs = 0.0f32;
                for col in 0..cols { max_abs = max_abs.max(data[base + col].abs()); }
                if max_abs < 1e-10 { max_abs = 1e-10; }
                row_scales[row] = max_abs / 127.0;
                let inv_scale = 127.0 / max_abs;
                let mut rsum = 0i32;
                for col in 0..cols {
                    let v = (data[base + col] * inv_scale).round().clamp(-128.0, 127.0) as i8;
                    w_i8[base + col] = v;
                    rsum += v as i32;
                }
                row_sums[row] = rsum;
            }
            Ok((w_i8, row_scales, row_sums, rows, cols))
        }
        _ => bail!("I8Quantized 또는 F32 타입이어야 함: {}", key),
    }
}

// ── 활성화 함수 ──────────────────────────────────────

#[inline(always)]
fn silu_scalar(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

#[inline(always)]
#[allow(dead_code)]
fn softplus_scalar(x: f32) -> f32 {
    if x > 20.0 { x } else { (1.0 + x.exp()).ln() }
}

#[inline(always)]
fn gelu1p_scalar(x: f32) -> f32 {
    let x3 = x * x * x;
    let inner = (x + 0.044715 * x3) * 0.7978845608;
    x * 0.5 * (1.0 + inner.tanh()) + 1.0
}

#[inline(always)]
fn sigmoid_scalar(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

// ── RMSNorm ──────────────────────────────────────────

pub struct RMSNorm {
    weight_vec: Vec<f32>,
    eps: f32,
}

impl RMSNorm {
    pub fn load_bmmq(tensors: &mut HashMap<String, TensorData>, prefix: &str, eps: f64) -> Result<Self> {
        let key = format!("{}.weight", prefix);
        let weight_vec = bmmq_take_f32(tensors, &key)?;
        Ok(Self { weight_vec, eps: eps as f32 })
    }

    #[inline]
    fn forward_vec(&self, x: &[f32], out: &mut [f32]) {
        let n = x.len();
        let mut sq_sum = 0.0f32;
        for &v in x { sq_sum += v * v; }
        let rms = (sq_sum / n as f32 + self.eps).sqrt().recip();
        for i in 0..n {
            out[i] = x[i] * rms * self.weight_vec[i];
        }
    }
}

// ── RMSNorm (elementwise_affine=False, BitLinear 내부용) ─

#[inline]
fn rms_norm_no_affine_vec(x: &[f32], out: &mut [f32], eps: f32) {
    let n = x.len();
    let mut sq_sum = 0.0f32;
    for &v in x { sq_sum += v * v; }
    let rms_inv = (sq_sum / n as f32 + eps).sqrt().recip();
    for i in 0..n {
        out[i] = x[i] * rms_inv;
    }
}

// ── BitLinear (i8 ternary + AVX-VNNI) ───────────────

pub struct BitLinear {
    gamma: f32,
    out_dim: usize,
    in_dim: usize,
    w_i8: Vec<i8>,       // 사전 언팩된 ternary 가중치 (로드 시 캐시)
    row_sums: Vec<i32>,
}

impl BitLinear {
    pub fn load_bmmq(tensors: &mut HashMap<String, TensorData>, prefix: &str) -> Result<Self> {
        let key = format!("{}.weight", prefix);
        match tensors.remove(&key).context(format!("BitLinear weight 없음: {}", key))? {
            TensorData::Packed2Bit { data, gamma, row_sums, rows, cols, packed_stride } => {
                // 로드 시 2-bit → i8 사전 언팩 (추론 중 반복 언팩 제거)
                let mut w_i8 = vec![0i8; rows * cols];
                unsafe {
                    unpack_2bit_rows(
                        data.as_ptr(),
                        w_i8.as_mut_ptr(),
                        rows as c_int,
                        cols as c_int,
                        packed_stride as c_int,
                    );
                }
                Ok(Self {
                    gamma,
                    out_dim: rows,
                    in_dim: cols,
                    w_i8,
                    row_sums,
                })
            }
            _ => bail!("BitLinear은 Packed2Bit 타입이어야 함: {}", key),
        }
    }

    /// i8 matmul (사전 언팩된 가중치 사용 — 할당 없음)
    fn forward_vec(&self, x: &[f32], out: &mut [f32], x_norm: &mut [f32], x_u8: &mut [u8]) {
        debug_assert_eq!(x.len(), self.in_dim);
        let n = self.in_dim;

        // 1. RMSNorm (BitLinear 내부 — 평균 빼기 없이 RMS만)
        rms_norm_no_affine_vec(x, &mut x_norm[..n], 1e-5);

        // 2. f32 → u8 양자화
        let x_scale = unsafe {
            quantize_f32_to_u8(x_norm.as_ptr(), x_u8.as_mut_ptr(), n as c_int)
        };

        // 3. i8 matmul
        self.matmul_preq(x_u8, x_scale, out);
    }

    /// 이미 양자화된 입력으로 matmul만 수행 (RMSNorm+quantize 공유용)
    #[inline]
    fn matmul_preq(&self, x_u8: &[u8], x_scale: f32, out: &mut [f32]) {
        unsafe {
            i8_sgemv(
                self.w_i8.as_ptr(),
                x_u8.as_ptr(),
                out.as_mut_ptr(),
                self.out_dim as c_int,
                self.in_dim as c_int,
                self.row_sums.as_ptr(),
                std::ptr::null(),
                x_scale,
                self.gamma,
            );
        }
    }
}

impl BitLinear {
    /// 배치 matmul: RMSNorm → quantize → sgemm (seq_len 토큰을 한 번에)
    /// x: (seq_len * in_dim), out: (seq_len * out_dim)
    fn forward_batch(&self, x: &[f32], seq_len: usize, out: &mut [f32],
                      x_norm_buf: &mut Vec<f32>, x_u8_buf: &mut Vec<u8>,
                      x_scales_buf: &mut Vec<f32>) {
        let k = self.in_dim;
        let m = self.out_dim;

        // 1. 토큰별 RMSNorm
        x_norm_buf.resize(seq_len * k, 0.0);
        for t in 0..seq_len {
            rms_norm_no_affine_vec(
                &x[t * k..(t + 1) * k],
                &mut x_norm_buf[t * k..(t + 1) * k],
                1e-5,
            );
        }

        // 2. 배치 양자화
        x_u8_buf.resize(seq_len * k, 0);
        x_scales_buf.resize(seq_len, 0.0);
        unsafe {
            batch_quantize_f32_to_u8(
                x_norm_buf.as_ptr(),
                x_u8_buf.as_mut_ptr(),
                x_scales_buf.as_mut_ptr(),
                seq_len as c_int,
                k as c_int,
            );
        }

        // 3. sgemm: X_u8 (seq_len, k) × W_i8^T (m, k) → Y (seq_len, m)
        unsafe {
            i8_sgemm(
                self.w_i8.as_ptr(),
                x_u8_buf.as_ptr(),
                out.as_mut_ptr(),
                m as c_int,
                seq_len as c_int,
                k as c_int,
                self.row_sums.as_ptr(),
                std::ptr::null(),
                x_scales_buf.as_ptr(),
                self.gamma,
            );
        }
    }
}

// ── Linear (i8 양자화 + AVX-VNNI) ───────────────────

pub struct Linear {
    w_i8: Vec<i8>,
    row_scales: Vec<f32>,
    row_sums: Vec<i32>,
    out_dim: usize,
    in_dim: usize,
}

impl Linear {
    pub fn load_bmmq(tensors: &mut HashMap<String, TensorData>, prefix: &str) -> Result<Self> {
        let key = format!("{}.weight", prefix);
        match tensors.remove(&key).context(format!("Linear weight 없음: {}", key))? {
            TensorData::I8Quantized { data, row_scales, row_sums, rows, cols } => {
                Ok(Self {
                    w_i8: data,
                    row_scales,
                    row_sums,
                    out_dim: rows,
                    in_dim: cols,
                })
            }
            _ => bail!("Linear은 I8Quantized 타입이어야 함: {}", key),
        }
    }

    /// i8 matmul via AVX-VNNI (할당 없음)
    fn forward_vec(&self, x: &[f32], out: &mut [f32], x_u8: &mut [u8]) {
        let n = self.in_dim;
        let x_scale = unsafe {
            quantize_f32_to_u8(x.as_ptr(), x_u8.as_mut_ptr(), n as c_int)
        };
        self.matmul_preq(x_u8, x_scale, out);
    }

    /// 이미 양자화된 입력으로 matmul만 수행 (양자화 공유용)
    #[inline]
    fn matmul_preq(&self, x_u8: &[u8], x_scale: f32, out: &mut [f32]) {
        unsafe {
            i8_sgemv(
                self.w_i8.as_ptr(),
                x_u8.as_ptr(),
                out.as_mut_ptr(),
                self.out_dim as c_int,
                self.in_dim as c_int,
                self.row_sums.as_ptr(),
                self.row_scales.as_ptr(),
                x_scale,
                0.0,
            );
        }
    }
}

impl Linear {
    /// 배치 matmul: quantize → sgemm (seq_len 토큰을 한 번에)
    fn forward_batch(&self, x: &[f32], seq_len: usize, out: &mut [f32],
                      x_u8_buf: &mut Vec<u8>, x_scales_buf: &mut Vec<f32>) {
        let k = self.in_dim;
        let m = self.out_dim;

        x_u8_buf.resize(seq_len * k, 0);
        x_scales_buf.resize(seq_len, 0.0);
        unsafe {
            batch_quantize_f32_to_u8(
                x.as_ptr(),
                x_u8_buf.as_mut_ptr(),
                x_scales_buf.as_mut_ptr(),
                seq_len as c_int,
                k as c_int,
            );
            i8_sgemm(
                self.w_i8.as_ptr(),
                x_u8_buf.as_ptr(),
                out.as_mut_ptr(),
                m as c_int,
                seq_len as c_int,
                k as c_int,
                self.row_sums.as_ptr(),
                self.row_scales.as_ptr(),
                x_scales_buf.as_ptr(),
                0.0,
            );
        }
    }
}

// ── BitNetFFN (Sigmoid-Gated) ────────────────────────

pub struct BitNetFFN {
    gate_proj: BitLinear,
    up_proj: BitLinear,
    down_proj: BitLinear,
    d_ff: usize,
}

impl BitNetFFN {
    pub fn load_bmmq(tensors: &mut HashMap<String, TensorData>, prefix: &str) -> Result<Self> {
        let gate_proj = BitLinear::load_bmmq(tensors, &format!("{}.gate_proj", prefix))?;
        let d_ff = gate_proj.out_dim;
        Ok(Self {
            gate_proj,
            up_proj: BitLinear::load_bmmq(tensors, &format!("{}.up_proj", prefix))?,
            down_proj: BitLinear::load_bmmq(tensors, &format!("{}.down_proj", prefix))?,
            d_ff,
        })
    }

    fn forward_vec(&self, x: &[f32], buf_ff: &mut Vec<f32>, buf_ff2: &mut Vec<f32>,
                    x_norm_buf: &mut [f32], x_u8: &mut [u8], out: &mut [f32]) {
        let n = x.len();
        buf_ff.resize(self.d_ff, 0.0);
        buf_ff2.resize(self.d_ff, 0.0);

        // gate/up은 같은 입력 → RMSNorm+quantize 1회 공유
        rms_norm_no_affine_vec(x, &mut x_norm_buf[..n], 1e-5);
        let x_scale = unsafe {
            quantize_f32_to_u8(x_norm_buf.as_ptr(), x_u8.as_mut_ptr(), n as c_int)
        };
        self.gate_proj.matmul_preq(&x_u8[..n], x_scale, buf_ff);
        self.up_proj.matmul_preq(&x_u8[..n], x_scale, buf_ff2);

        // sigmoid(gate) * up → reuse buf_ff
        for i in 0..self.d_ff {
            buf_ff[i] = sigmoid_scalar(buf_ff[i]) * buf_ff2[i];
        }

        self.down_proj.forward_vec(buf_ff, out, x_norm_buf, x_u8);
    }
}

// ── RWKV6TimeMix ─────────────────────────────────────

struct RWKV6TimeMix {
    r_proj: BitLinear,
    k_proj: BitLinear,
    v_proj: BitLinear,
    o_proj: BitLinear,
    g_proj: Linear,       // gate
    u: Vec<f32>,          // (n_heads * headdim,) in-context bonus
    w_base: Vec<f32>,     // (n_heads * headdim,)
    w_lora_down: Linear,  // (d_model -> w_lora_rank)
    w_lora_up: Linear,    // (w_lora_rank -> d_model)
    output_norm_weight: Vec<f32>,  // LayerNorm per-head weight
    output_norm_bias: Vec<f32>,    // LayerNorm per-head bias
    n_heads: usize,
    headdim: usize,
    d_model: usize,
    w_lora_rank: usize,   // w_lora_down의 실제 출력 차원
}

impl RWKV6TimeMix {
    fn load_bmmq(
        tensors: &mut HashMap<String, TensorData>,
        prefix: &str,
        n_heads: usize,
        headdim: usize,
        _lora_rank: usize,
    ) -> Result<Self> {
        let d_model = n_heads * headdim;
        let w_lora_down = Linear::load_bmmq(tensors, &format!("{}.w_lora_down", prefix))?;
        let w_lora_rank = w_lora_down.out_dim;  // 실제 차원 사용 (config 값 대신)
        Ok(Self {
            r_proj: BitLinear::load_bmmq(tensors, &format!("{}.r_proj", prefix))?,
            k_proj: BitLinear::load_bmmq(tensors, &format!("{}.k_proj", prefix))?,
            v_proj: BitLinear::load_bmmq(tensors, &format!("{}.v_proj", prefix))?,
            o_proj: BitLinear::load_bmmq(tensors, &format!("{}.o_proj", prefix))?,
            g_proj: Linear::load_bmmq(tensors, &format!("{}.g_proj", prefix))?,
            u: bmmq_take_f32(tensors, &format!("{}.u", prefix))?,
            w_base: bmmq_take_f32(tensors, &format!("{}.w_base", prefix))?,
            w_lora_down,
            w_lora_up: Linear::load_bmmq(tensors, &format!("{}.w_lora_up", prefix))?,
            output_norm_weight: bmmq_take_f32(tensors, &format!("{}.output_norm.weight", prefix))?,
            output_norm_bias: bmmq_take_f32(tensors, &format!("{}.output_norm.bias", prefix))?,
            n_heads,
            headdim,
            d_model,
            w_lora_rank,
        })
    }

    /// 전체 시퀀스 배치 처리 (seq_len, d_model) → (seq_len, d_model)
    fn forward_batch(
        &self,
        x: &[f32],        // (seq_len * d_model)
        seq_len: usize,
        bufs: &mut RWKVBufs,
    ) {
        let d = self.d_model;
        let nh = self.n_heads;
        let hd = self.headdim;

        // R, K, V, G 프로젝션: 토큰별 sgemv (384×384에서 L2 캐시 활용 최적)
        bufs.r.resize(seq_len * d, 0.0);
        bufs.k.resize(seq_len * d, 0.0);
        bufs.v.resize(seq_len * d, 0.0);
        bufs.g.resize(seq_len * d, 0.0);
        bufs.w.resize(seq_len * d, 0.0);

        for t in 0..seq_len {
            let x_t = &x[t * d..(t + 1) * d];

            // r/k/v는 동일 입력에 대해 RMSNorm+quantize 1회만 수행 → matmul 3회 공유
            rms_norm_no_affine_vec(x_t, &mut bufs.x_norm[..d], 1e-5);
            let x_scale = unsafe {
                quantize_f32_to_u8(bufs.x_norm.as_ptr(), bufs.x_u8.as_mut_ptr(), d as c_int)
            };
            self.r_proj.matmul_preq(&bufs.x_u8[..d], x_scale, &mut bufs.r[t * d..(t + 1) * d]);
            self.k_proj.matmul_preq(&bufs.x_u8[..d], x_scale, &mut bufs.k[t * d..(t + 1) * d]);
            self.v_proj.matmul_preq(&bufs.x_u8[..d], x_scale, &mut bufs.v[t * d..(t + 1) * d]);

            // g_proj는 Linear (RMSNorm 없이 직접 quantize) — x_t로 별도 양자화
            self.g_proj.forward_vec(x_t, &mut bufs.g[t * d..(t + 1) * d],
                                    &mut bufs.x_u8);
        }

        // Data-dependent decay: 토큰별 (w_lora 입력 의존)
        for t in 0..seq_len {
            let x_t = &x[t * d..(t + 1) * d];
            bufs.lora_down.resize(self.w_lora_rank, 0.0);
            self.w_lora_down.forward_vec(x_t, &mut bufs.lora_down, &mut bufs.x_u8);
            for v in bufs.lora_down.iter_mut() { *v = v.tanh(); }
            bufs.lora_up.resize(d, 0.0);
            self.w_lora_up.forward_vec(&bufs.lora_down, &mut bufs.lora_up, &mut bufs.x_u8);
            for i in 0..d {
                bufs.w[t * d + i] = self.w_base[i] + bufs.lora_up[i];
            }
        }

        // WKV sequential scan (AVX2 FMA 벡터화)
        bufs.state.resize(nh * hd * hd, 0.0);
        bufs.state.fill(0.0);
        bufs.output.resize(seq_len * d, 0.0);

        unsafe {
            wkv6_scan_avx2(
                bufs.r.as_ptr(),
                bufs.k.as_ptr(),
                bufs.v.as_ptr(),
                bufs.w.as_ptr(),
                self.u.as_ptr(),
                bufs.output.as_mut_ptr(),
                bufs.state.as_mut_ptr(),
                seq_len as c_int,
                nh as c_int,
                hd as c_int,
                d as c_int,
            );
        }

        // Per-head LayerNorm + gate + output projection
        bufs.normed_head.resize(d, 0.0);
        bufs.gated.resize(d, 0.0);
        bufs.final_out.resize(seq_len * d, 0.0);

        for t in 0..seq_len {
            let out_t = &bufs.output[t * d..];
            let g_t = &bufs.g[t * d..];

            // Per-head LayerNorm (with weight and bias)
            for h in 0..nh {
                let off = h * hd;
                // Compute mean and var for this head
                let mut mean = 0.0f32;
                for i in 0..hd { mean += out_t[off + i]; }
                mean /= hd as f32;

                let mut var = 0.0f32;
                for i in 0..hd {
                    let d_val = out_t[off + i] - mean;
                    var += d_val * d_val;
                }
                let inv_std = (var / hd as f32 + 1e-5f32).sqrt().recip();

                for i in 0..hd {
                    let idx = off + i;
                    let normed = (out_t[idx] - mean) * inv_std;
                    // output_norm은 (headdim,) 크기로 head 간 공유
                    bufs.normed_head[idx] = normed * self.output_norm_weight[i]
                        + self.output_norm_bias[i];
                }
            }

            // Gate: sigmoid(g) * normed (Python: g = torch.sigmoid(g_proj(x)))
            for i in 0..d {
                bufs.gated[i] = sigmoid_scalar(g_t[i]) * bufs.normed_head[i];
            }

            // Output projection
            self.o_proj.forward_vec(&bufs.gated, &mut bufs.final_out[t * d..(t + 1) * d],
                                    &mut bufs.x_norm, &mut bufs.x_u8);
        }
    }
}

/// RWKV 연산용 재활용 버퍼
struct RWKVBufs {
    r: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    g: Vec<f32>,
    w: Vec<f32>,
    state: Vec<f32>,
    kv_bonus: Vec<f32>,
    output: Vec<f32>,
    normed_head: Vec<f32>,
    gated: Vec<f32>,
    final_out: Vec<f32>,
    lora_down: Vec<f32>,
    lora_up: Vec<f32>,
    x_norm: Vec<f32>,
    x_u8: Vec<u8>,
    // 배치 프로젝션용 버퍼
    batch_norm: Vec<f32>,
    batch_u8: Vec<u8>,
    batch_scales: Vec<f32>,
}

impl RWKVBufs {
    fn new(max_in_dim: usize) -> Self {
        Self {
            r: Vec::new(),
            k: Vec::new(),
            v: Vec::new(),
            g: Vec::new(),
            w: Vec::new(),
            state: Vec::new(),
            kv_bonus: Vec::new(),
            output: Vec::new(),
            normed_head: Vec::new(),
            gated: Vec::new(),
            final_out: Vec::new(),
            lora_down: Vec::new(),
            lora_up: Vec::new(),
            x_norm: vec![0.0; max_in_dim],
            x_u8: vec![0u8; max_in_dim],
            batch_norm: Vec::new(),
            batch_u8: Vec::new(),
            batch_scales: Vec::new(),
        }
    }
}

// ── BiRWKV ───────────────────────────────────────────

struct BiRWKV {
    forward_rwkv: RWKV6TimeMix,
    backward_rwkv: RWKV6TimeMix,
}

impl BiRWKV {
    fn load_bmmq(
        tensors: &mut HashMap<String, TensorData>,
        prefix: &str,
        n_heads: usize,
        headdim: usize,
        lora_rank: usize,
    ) -> Result<Self> {
        Ok(Self {
            forward_rwkv: RWKV6TimeMix::load_bmmq(
                tensors, &format!("{}.forward_rwkv", prefix), n_heads, headdim, lora_rank,
            )?,
            backward_rwkv: RWKV6TimeMix::load_bmmq(
                tensors, &format!("{}.backward_rwkv", prefix), n_heads, headdim, lora_rank,
            )?,
        })
    }

    /// 양방향 RWKV: 순방향 + 역순 입력의 역방향 결과를 합산
    fn forward_batch(
        &self,
        x: &[f32],       // (seq_len * d_model)
        seq_len: usize,
        d_model: usize,
        fwd_bufs: &mut RWKVBufs,
        bwd_bufs: &mut RWKVBufs,
        x_rev: &mut Vec<f32>,
    ) {
        // 순방향
        self.forward_rwkv.forward_batch(x, seq_len, fwd_bufs);

        // 입력 반전
        x_rev.resize(seq_len * d_model, 0.0);
        for t in 0..seq_len {
            let src = (seq_len - 1 - t) * d_model;
            let dst = t * d_model;
            x_rev[dst..dst + d_model].copy_from_slice(&x[src..src + d_model]);
        }

        // 역방향
        self.backward_rwkv.forward_batch(x_rev, seq_len, bwd_bufs);

        // 역방향 결과를 다시 뒤집어서 순방향 결과에 합산
        for t in 0..seq_len {
            let fwd_off = t * d_model;
            let bwd_off = (seq_len - 1 - t) * d_model;
            for i in 0..d_model {
                fwd_bufs.final_out[fwd_off + i] += bwd_bufs.final_out[bwd_off + i];
            }
        }
    }

    /// 프로파일링 버전: 내부 구간별 시간 측정 (static 누적)
    fn forward_batch_profiled(
        &self,
        x: &[f32],
        seq_len: usize,
        d_model: usize,
        fwd_bufs: &mut RWKVBufs,
        bwd_bufs: &mut RWKVBufs,
        x_rev: &mut Vec<f32>,
        timings: &mut [u128; 6], // [fwd_proj, fwd_wkv, fwd_post, bwd_proj, bwd_wkv, bwd_post]
    ) {
        use std::time::Instant;

        // 순방향 프로젝션
        let t0 = Instant::now();
        let d = self.forward_rwkv.d_model;
        let nh = self.forward_rwkv.n_heads;
        let hd = self.forward_rwkv.headdim;

        fwd_bufs.r.resize(seq_len * d, 0.0);
        fwd_bufs.k.resize(seq_len * d, 0.0);
        fwd_bufs.v.resize(seq_len * d, 0.0);
        fwd_bufs.g.resize(seq_len * d, 0.0);
        fwd_bufs.w.resize(seq_len * d, 0.0);

        for t in 0..seq_len {
            let x_t = &x[t * d..(t + 1) * d];
            self.forward_rwkv.r_proj.forward_vec(x_t, &mut fwd_bufs.r[t*d..(t+1)*d], &mut fwd_bufs.x_norm, &mut fwd_bufs.x_u8);
            self.forward_rwkv.k_proj.forward_vec(x_t, &mut fwd_bufs.k[t*d..(t+1)*d], &mut fwd_bufs.x_norm, &mut fwd_bufs.x_u8);
            self.forward_rwkv.v_proj.forward_vec(x_t, &mut fwd_bufs.v[t*d..(t+1)*d], &mut fwd_bufs.x_norm, &mut fwd_bufs.x_u8);
            self.forward_rwkv.g_proj.forward_vec(x_t, &mut fwd_bufs.g[t*d..(t+1)*d], &mut fwd_bufs.x_u8);
            // w_lora
            fwd_bufs.lora_down.resize(self.forward_rwkv.w_lora_rank, 0.0);
            self.forward_rwkv.w_lora_down.forward_vec(x_t, &mut fwd_bufs.lora_down, &mut fwd_bufs.x_u8);
            for v in fwd_bufs.lora_down.iter_mut() { *v = v.tanh(); }
            fwd_bufs.lora_up.resize(d, 0.0);
            self.forward_rwkv.w_lora_up.forward_vec(&fwd_bufs.lora_down, &mut fwd_bufs.lora_up, &mut fwd_bufs.x_u8);
            for i in 0..d { fwd_bufs.w[t*d+i] = self.forward_rwkv.w_base[i] + fwd_bufs.lora_up[i]; }
        }
        timings[0] += t0.elapsed().as_micros();

        // 순방향 WKV scan (AVX2 FMA)
        let t0 = Instant::now();
        fwd_bufs.state.resize(nh * hd * hd, 0.0);
        fwd_bufs.state.fill(0.0);
        fwd_bufs.output.resize(seq_len * d, 0.0);
        unsafe {
            wkv6_scan_avx2(
                fwd_bufs.r.as_ptr(), fwd_bufs.k.as_ptr(),
                fwd_bufs.v.as_ptr(), fwd_bufs.w.as_ptr(),
                self.forward_rwkv.u.as_ptr(),
                fwd_bufs.output.as_mut_ptr(), fwd_bufs.state.as_mut_ptr(),
                seq_len as c_int, nh as c_int, hd as c_int, d as c_int,
            );
        }
        timings[1] += t0.elapsed().as_micros();

        // 순방향 후처리 (LayerNorm + gate + o_proj)
        let t0 = Instant::now();
        fwd_bufs.normed_head.resize(d, 0.0);
        fwd_bufs.gated.resize(d, 0.0);
        fwd_bufs.final_out.resize(seq_len * d, 0.0);
        for t in 0..seq_len {
            let out_t = &fwd_bufs.output[t*d..];
            let g_t = &fwd_bufs.g[t*d..];
            for h in 0..nh {
                let off = h * hd;
                let mut mean = 0.0f32;
                for i in 0..hd { mean += out_t[off+i]; }
                mean /= hd as f32;
                let mut var = 0.0f32;
                for i in 0..hd { let dv = out_t[off+i] - mean; var += dv*dv; }
                let inv_std = (var / hd as f32 + 1e-5f32).sqrt().recip();
                for i in 0..hd {
                    let idx = off + i;
                    let normed = (out_t[idx] - mean) * inv_std;
                    fwd_bufs.normed_head[idx] = normed * self.forward_rwkv.output_norm_weight[i]
                        + self.forward_rwkv.output_norm_bias[i];
                }
            }
            for i in 0..d { fwd_bufs.gated[i] = sigmoid_scalar(g_t[i]) * fwd_bufs.normed_head[i]; }
            self.forward_rwkv.o_proj.forward_vec(&fwd_bufs.gated, &mut fwd_bufs.final_out[t*d..(t+1)*d],
                                                  &mut fwd_bufs.x_norm, &mut fwd_bufs.x_u8);
        }
        timings[2] += t0.elapsed().as_micros();

        // 역방향: 입력 반전
        x_rev.resize(seq_len * d_model, 0.0);
        for t in 0..seq_len {
            let src = (seq_len - 1 - t) * d_model;
            let dst = t * d_model;
            x_rev[dst..dst + d_model].copy_from_slice(&x[src..src + d_model]);
        }

        // 역방향 프로젝션
        let t0 = Instant::now();
        bwd_bufs.r.resize(seq_len * d, 0.0);
        bwd_bufs.k.resize(seq_len * d, 0.0);
        bwd_bufs.v.resize(seq_len * d, 0.0);
        bwd_bufs.g.resize(seq_len * d, 0.0);
        bwd_bufs.w.resize(seq_len * d, 0.0);
        for t in 0..seq_len {
            let x_t = &x_rev[t * d..(t + 1) * d];
            // r/k/v 양자화 공유
            rms_norm_no_affine_vec(x_t, &mut bwd_bufs.x_norm[..d], 1e-5);
            let x_scale = unsafe {
                quantize_f32_to_u8(bwd_bufs.x_norm.as_ptr(), bwd_bufs.x_u8.as_mut_ptr(), d as c_int)
            };
            self.backward_rwkv.r_proj.matmul_preq(&bwd_bufs.x_u8[..d], x_scale, &mut bwd_bufs.r[t*d..(t+1)*d]);
            self.backward_rwkv.k_proj.matmul_preq(&bwd_bufs.x_u8[..d], x_scale, &mut bwd_bufs.k[t*d..(t+1)*d]);
            self.backward_rwkv.v_proj.matmul_preq(&bwd_bufs.x_u8[..d], x_scale, &mut bwd_bufs.v[t*d..(t+1)*d]);
            self.backward_rwkv.g_proj.forward_vec(x_t, &mut bwd_bufs.g[t*d..(t+1)*d], &mut bwd_bufs.x_u8);
            bwd_bufs.lora_down.resize(self.backward_rwkv.w_lora_rank, 0.0);
            self.backward_rwkv.w_lora_down.forward_vec(x_t, &mut bwd_bufs.lora_down, &mut bwd_bufs.x_u8);
            for v in bwd_bufs.lora_down.iter_mut() { *v = v.tanh(); }
            bwd_bufs.lora_up.resize(d, 0.0);
            self.backward_rwkv.w_lora_up.forward_vec(&bwd_bufs.lora_down, &mut bwd_bufs.lora_up, &mut bwd_bufs.x_u8);
            for i in 0..d { bwd_bufs.w[t*d+i] = self.backward_rwkv.w_base[i] + bwd_bufs.lora_up[i]; }
        }
        timings[3] += t0.elapsed().as_micros();

        // 역방향 WKV scan (AVX2 FMA)
        let t0 = Instant::now();
        bwd_bufs.state.resize(nh * hd * hd, 0.0);
        bwd_bufs.state.fill(0.0);
        bwd_bufs.output.resize(seq_len * d, 0.0);
        unsafe {
            wkv6_scan_avx2(
                bwd_bufs.r.as_ptr(), bwd_bufs.k.as_ptr(),
                bwd_bufs.v.as_ptr(), bwd_bufs.w.as_ptr(),
                self.backward_rwkv.u.as_ptr(),
                bwd_bufs.output.as_mut_ptr(), bwd_bufs.state.as_mut_ptr(),
                seq_len as c_int, nh as c_int, hd as c_int, d as c_int,
            );
        }
        timings[4] += t0.elapsed().as_micros();

        // 역방향 후처리 + 합산
        let t0 = Instant::now();
        bwd_bufs.normed_head.resize(d, 0.0);
        bwd_bufs.gated.resize(d, 0.0);
        bwd_bufs.final_out.resize(seq_len * d, 0.0);
        for t in 0..seq_len {
            let out_t = &bwd_bufs.output[t*d..];
            let g_t = &bwd_bufs.g[t*d..];
            for h in 0..nh {
                let off = h * hd;
                let mut mean = 0.0f32;
                for i in 0..hd { mean += out_t[off+i]; }
                mean /= hd as f32;
                let mut var = 0.0f32;
                for i in 0..hd { let dv = out_t[off+i] - mean; var += dv*dv; }
                let inv_std = (var / hd as f32 + 1e-5f32).sqrt().recip();
                for i in 0..hd {
                    let idx = off + i;
                    let normed = (out_t[idx] - mean) * inv_std;
                    bwd_bufs.normed_head[idx] = normed * self.backward_rwkv.output_norm_weight[i]
                        + self.backward_rwkv.output_norm_bias[i];
                }
            }
            for i in 0..d { bwd_bufs.gated[i] = sigmoid_scalar(g_t[i]) * bwd_bufs.normed_head[i]; }
            self.backward_rwkv.o_proj.forward_vec(&bwd_bufs.gated, &mut bwd_bufs.final_out[t*d..(t+1)*d],
                                                   &mut bwd_bufs.x_norm, &mut bwd_bufs.x_u8);
        }
        // 역방향 결과를 뒤집어서 순방향에 합산
        for t in 0..seq_len {
            let fwd_off = t * d_model;
            let bwd_off = (seq_len - 1 - t) * d_model;
            for i in 0..d_model { fwd_bufs.final_out[fwd_off + i] += bwd_bufs.final_out[bwd_off + i]; }
        }
        timings[5] += t0.elapsed().as_micros();
    }
}

// ── MoEBitNetFFN ─────────────────────────────────────

struct MoEBitNetFFN {
    router: Linear,
    experts: Vec<BitNetFFN>,
    n_experts: usize,
    top_k: usize,
}

impl MoEBitNetFFN {
    fn load_bmmq(
        tensors: &mut HashMap<String, TensorData>,
        prefix: &str,
        n_experts: usize,
        top_k: usize,
    ) -> Result<Self> {
        let router = Linear::load_bmmq(tensors, &format!("{}.router", prefix))?;
        let mut experts = Vec::with_capacity(n_experts);
        for i in 0..n_experts {
            experts.push(BitNetFFN::load_bmmq(tensors, &format!("{}.experts.{}", prefix, i))?);
        }
        Ok(Self { router, experts, n_experts, top_k })
    }

    /// 배치 MoE forward: 토큰별 라우터 softmax → top_k 전문가 디스패치
    fn forward_batch(
        &self,
        x: &[f32],         // (seq_len * d_model)
        seq_len: usize,
        d_model: usize,
        bufs: &mut MoEBufs,
    ) {
        bufs.output.resize(seq_len * d_model, 0.0);
        bufs.output.fill(0.0);

        for t in 0..seq_len {
            let x_t = &x[t * d_model..(t + 1) * d_model];

            // 라우터 logits
            bufs.router_logits.resize(self.n_experts, 0.0);
            self.router.forward_vec(x_t, &mut bufs.router_logits, &mut bufs.x_u8);

            // Softmax
            let max_logit = bufs.router_logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum_exp = 0.0f32;
            bufs.router_probs.resize(self.n_experts, 0.0);
            for i in 0..self.n_experts {
                bufs.router_probs[i] = (bufs.router_logits[i] - max_logit).exp();
                sum_exp += bufs.router_probs[i];
            }
            let inv_sum = 1.0 / sum_exp;
            for v in bufs.router_probs.iter_mut() { *v *= inv_sum; }

            // Top-k 선택
            bufs.top_indices.clear();
            let mut probs_copy = bufs.router_probs.clone();
            for _ in 0..self.top_k.min(self.n_experts) {
                let (best_idx, _) = probs_copy.iter().enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .unwrap();
                bufs.top_indices.push(best_idx);
                probs_copy[best_idx] = f32::NEG_INFINITY;
            }

            // Top-k 가중치 재정규화
            let mut top_sum = 0.0f32;
            for &idx in &bufs.top_indices {
                top_sum += bufs.router_probs[idx];
            }
            let inv_top = 1.0 / top_sum;

            // 전문가 실행 & 가중 합산
            bufs.expert_out.resize(d_model, 0.0);
            let out_t = &mut bufs.output[t * d_model..(t + 1) * d_model];

            // gate/up 입력 양자화 1회 → 별도 버퍼에 저장 (down_proj가 덮어쓰지 않도록)
            let n = d_model;
            rms_norm_no_affine_vec(x_t, &mut bufs.x_norm[..n], 1e-5);
            bufs.x_scale_shared = unsafe {
                quantize_f32_to_u8(bufs.x_norm.as_ptr(), bufs.x_u8_shared.as_mut_ptr(), n as c_int)
            };

            for &idx in &bufs.top_indices {
                let weight = bufs.router_probs[idx] * inv_top;
                let expert = &self.experts[idx];

                // gate/up: 공유 버퍼(x_u8_shared)에서 읽기 — 재계산 없음
                bufs.ff1.resize(expert.d_ff, 0.0);
                bufs.ff2.resize(expert.d_ff, 0.0);
                expert.gate_proj.matmul_preq(&bufs.x_u8_shared[..n], bufs.x_scale_shared, &mut bufs.ff1);
                expert.up_proj.matmul_preq(&bufs.x_u8_shared[..n], bufs.x_scale_shared, &mut bufs.ff2);

                for i in 0..expert.d_ff {
                    bufs.ff1[i] = sigmoid_scalar(bufs.ff1[i]) * bufs.ff2[i];
                }

                // down_proj는 x_u8(일반 버퍼)를 사용 — x_u8_shared는 보존됨
                expert.down_proj.forward_vec(&bufs.ff1, &mut bufs.expert_out,
                                              &mut bufs.x_norm, &mut bufs.x_u8);

                for i in 0..d_model {
                    out_t[i] += weight * bufs.expert_out[i];
                }
            }
        }
    }
}

/// MoE 연산용 재활용 버퍼
struct MoEBufs {
    output: Vec<f32>,
    router_logits: Vec<f32>,
    router_probs: Vec<f32>,
    top_indices: Vec<usize>,
    expert_out: Vec<f32>,
    ff1: Vec<f32>,
    ff2: Vec<f32>,
    x_norm: Vec<f32>,
    x_u8: Vec<u8>,
    // top-k > 1일 때 공유 양자화 저장용
    x_u8_shared: Vec<u8>,
    x_scale_shared: f32,
}

impl MoEBufs {
    fn new(max_in_dim: usize) -> Self {
        Self {
            output: Vec::new(),
            router_logits: Vec::new(),
            router_probs: Vec::new(),
            top_indices: Vec::new(),
            expert_out: Vec::new(),
            ff1: Vec::new(),
            ff2: Vec::new(),
            x_norm: vec![0.0; max_in_dim],
            x_u8: vec![0u8; max_in_dim],
            x_u8_shared: vec![0u8; max_in_dim],
            x_scale_shared: 0.0,
        }
    }
}

// ── LoRA ─────────────────────────────────────────────

struct LoRA {
    down: Linear,
    up: Linear,
}

impl LoRA {
    fn load_bmmq(tensors: &mut HashMap<String, TensorData>, prefix: &str) -> Result<Self> {
        Ok(Self {
            down: Linear::load_bmmq(tensors, &format!("{}.down", prefix))?,
            up: Linear::load_bmmq(tensors, &format!("{}.up", prefix))?,
        })
    }

    /// LoRA forward: up(down(x))
    fn forward_vec(&self, x: &[f32], out: &mut [f32], lora_buf: &mut Vec<f32>, x_u8: &mut [u8]) {
        lora_buf.resize(self.down.out_dim, 0.0);
        self.down.forward_vec(x, lora_buf, x_u8);
        self.up.forward_vec(lora_buf, out, x_u8);
    }
}

// ── SharedLinearSelfAttention ────────────────────────

struct SharedLinearSelfAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    lora_q: Vec<LoRA>,
    lora_k: Vec<LoRA>,
    lora_v: Vec<LoRA>,
    lora_o: Vec<LoRA>,
    n_heads: usize,
    d_head: usize,
    d_model: usize,
}

impl SharedLinearSelfAttention {
    fn load_bmmq(
        tensors: &mut HashMap<String, TensorData>,
        prefix: &str,
        n_heads: usize,
        d_head: usize,
        n_layers: usize,  // LoRA 레이어 수 (삽입 포인트 수)
    ) -> Result<Self> {
        let d_model = n_heads * d_head;
        let q_proj = Linear::load_bmmq(tensors, &format!("{}.q_proj", prefix))?;
        let k_proj = Linear::load_bmmq(tensors, &format!("{}.k_proj", prefix))?;
        let v_proj = Linear::load_bmmq(tensors, &format!("{}.v_proj", prefix))?;
        let o_proj = Linear::load_bmmq(tensors, &format!("{}.o_proj", prefix))?;

        let mut lora_q = Vec::with_capacity(n_layers);
        let mut lora_k = Vec::with_capacity(n_layers);
        let mut lora_v = Vec::with_capacity(n_layers);
        let mut lora_o = Vec::with_capacity(n_layers);

        for i in 0..n_layers {
            lora_q.push(LoRA::load_bmmq(tensors, &format!("{}.lora_q.{}", prefix, i))?);
            lora_k.push(LoRA::load_bmmq(tensors, &format!("{}.lora_k.{}", prefix, i))?);
            lora_v.push(LoRA::load_bmmq(tensors, &format!("{}.lora_v.{}", prefix, i))?);
            lora_o.push(LoRA::load_bmmq(tensors, &format!("{}.lora_o.{}", prefix, i))?);
        }

        Ok(Self {
            q_proj, k_proj, v_proj, o_proj,
            lora_q, lora_k, lora_v, lora_o,
            n_heads, d_head, d_model,
        })
    }

    /// O(N) linear self-attention
    /// lora_idx: 몇 번째 삽입 포인트인지 (LoRA 인덱스)
    fn forward_batch(
        &self,
        x: &[f32],        // (seq_len * d_model)
        seq_len: usize,
        lora_idx: usize,
        bufs: &mut AttnBufs,
    ) {
        let d = self.d_model;
        let nh = self.n_heads;
        let dh = self.d_head;

        // 토큰별 Q, K, V 프로젝션 + LoRA
        bufs.q.resize(seq_len * d, 0.0);
        bufs.k.resize(seq_len * d, 0.0);
        bufs.v.resize(seq_len * d, 0.0);
        bufs.output.resize(seq_len * d, 0.0);

        for t in 0..seq_len {
            let x_t = &x[t * d..(t + 1) * d];

            // q/k/v는 같은 입력 → quantize 1회 공유
            let x_scale = unsafe {
                quantize_f32_to_u8(x_t.as_ptr(), bufs.x_u8.as_mut_ptr(), d as c_int)
            };
            self.q_proj.matmul_preq(&bufs.x_u8[..d], x_scale, &mut bufs.q[t * d..(t + 1) * d]);
            self.k_proj.matmul_preq(&bufs.x_u8[..d], x_scale, &mut bufs.k[t * d..(t + 1) * d]);
            self.v_proj.matmul_preq(&bufs.x_u8[..d], x_scale, &mut bufs.v[t * d..(t + 1) * d]);

            // LoRA 보정
            if lora_idx < self.lora_q.len() {
                bufs.lora_out.resize(d, 0.0);

                self.lora_q[lora_idx].forward_vec(x_t, &mut bufs.lora_out, &mut bufs.lora_buf, &mut bufs.x_u8);
                for i in 0..d { bufs.q[t * d + i] += bufs.lora_out[i]; }

                self.lora_k[lora_idx].forward_vec(x_t, &mut bufs.lora_out, &mut bufs.lora_buf, &mut bufs.x_u8);
                for i in 0..d { bufs.k[t * d + i] += bufs.lora_out[i]; }

                self.lora_v[lora_idx].forward_vec(x_t, &mut bufs.lora_out, &mut bufs.lora_buf, &mut bufs.x_u8);
                for i in 0..d { bufs.v[t * d + i] += bufs.lora_out[i]; }
            }
        }

        // gelu1p feature map on Q and K
        for v in bufs.q.iter_mut() { *v = gelu1p_scalar(*v); }
        for v in bufs.k.iter_mut() { *v = gelu1p_scalar(*v); }

        // O(N) linear self-attention per head:
        // context[h] = K[h]^T @ V[h]  →  (dh, dh)
        // z[h] = sum(K[h], dim=0)      →  (dh,)
        // out[t,h] = (Q[t,h] @ context[h]) / (Q[t,h] . z[h] + eps)
        bufs.context.resize(nh * dh * dh, 0.0);
        bufs.context.fill(0.0);
        bufs.z.resize(nh * dh, 0.0);
        bufs.z.fill(0.0);

        // 누적: K^T @ V 와 sum(K)
        for t in 0..seq_len {
            for h in 0..nh {
                let h_off = h * dh;
                let ctx_off = h * dh * dh;
                let z_off = h * dh;

                for ki in 0..dh {
                    let k_val = bufs.k[t * d + h_off + ki];
                    bufs.z[z_off + ki] += k_val;
                    for vi in 0..dh {
                        bufs.context[ctx_off + ki * dh + vi] +=
                            k_val * bufs.v[t * d + h_off + vi];
                    }
                }
            }
        }

        // 토큰별 출력 계산
        for t in 0..seq_len {
            for h in 0..nh {
                let h_off = h * dh;
                let ctx_off = h * dh * dh;
                let z_off = h * dh;
                let out_off = t * d + h_off;

                // Q[t,h] @ context[h] via cblas_sgemv
                unsafe {
                    cblas_sgemv(
                        CBLAS_ROW_MAJOR,
                        112, // CblasTrans
                        dh as c_int,
                        dh as c_int,
                        1.0,
                        bufs.context[ctx_off..].as_ptr(),
                        dh as c_int,
                        bufs.q[t * d + h_off..].as_ptr(),
                        1,
                        0.0,
                        bufs.output[out_off..].as_mut_ptr(),
                        1,
                    );
                }

                // normalizer
                let mut den = 0.0f32;
                for i in 0..dh {
                    den += bufs.q[t * d + h_off + i] * bufs.z[z_off + i];
                }
                den = (den + 1e-5f32).recip();
                for i in 0..dh {
                    bufs.output[out_off + i] *= den;
                }
            }
        }

        // O projection + LoRA
        bufs.final_out.resize(seq_len * d, 0.0);
        for t in 0..seq_len {
            let o_t = &bufs.output[t * d..(t + 1) * d];
            self.o_proj.forward_vec(o_t, &mut bufs.final_out[t * d..(t + 1) * d], &mut bufs.x_u8);

            if lora_idx < self.lora_o.len() {
                bufs.lora_out.resize(d, 0.0);
                self.lora_o[lora_idx].forward_vec(o_t, &mut bufs.lora_out, &mut bufs.lora_buf, &mut bufs.x_u8);
                for i in 0..d {
                    bufs.final_out[t * d + i] += bufs.lora_out[i];
                }
            }
        }
    }
}

/// Attention 연산용 재활용 버퍼
struct AttnBufs {
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    output: Vec<f32>,
    final_out: Vec<f32>,
    context: Vec<f32>,
    z: Vec<f32>,
    lora_out: Vec<f32>,
    lora_buf: Vec<f32>,
    x_u8: Vec<u8>,
}

impl AttnBufs {
    fn new(max_in_dim: usize) -> Self {
        Self {
            q: Vec::new(),
            k: Vec::new(),
            v: Vec::new(),
            output: Vec::new(),
            final_out: Vec::new(),
            context: Vec::new(),
            z: Vec::new(),
            lora_out: Vec::new(),
            lora_buf: Vec::new(),
            x_u8: vec![0u8; max_in_dim],
        }
    }
}

// ── BitEditorLayer ───────────────────────────────────

struct BitEditorLayer {
    bi_rwkv: BiRWKV,
    norm1: RMSNorm,
    moe_ffn: MoEBitNetFFN,
    norm2: RMSNorm,
}

impl BitEditorLayer {
    fn load_bmmq(
        tensors: &mut HashMap<String, TensorData>,
        prefix: &str,
        cfg: &ModelConfig,
    ) -> Result<Self> {
        Ok(Self {
            bi_rwkv: BiRWKV::load_bmmq(
                tensors, &format!("{}.bi_rwkv", prefix),
                cfg.n_heads, cfg.headdim, cfg.lora_rank,
            )?,
            norm1: RMSNorm::load_bmmq(tensors, &format!("{}.norm1", prefix), cfg.rms_norm_eps)?,
            moe_ffn: MoEBitNetFFN::load_bmmq(
                tensors, &format!("{}.moe_ffn", prefix),
                cfg.n_experts, cfg.top_k,
            )?,
            norm2: RMSNorm::load_bmmq(tensors, &format!("{}.norm2", prefix), cfg.rms_norm_eps)?,
        })
    }

    /// 레이어 forward: pre-norm → BiRWKV → (+residual) → pre-norm → MoE FFN → (+residual)
    fn forward_batch(
        &self,
        x: &mut Vec<f32>,        // (seq_len * d_model), 입력 겸 출력
        seq_len: usize,
        d_model: usize,
        fwd_bufs: &mut RWKVBufs,
        bwd_bufs: &mut RWKVBufs,
        moe_bufs: &mut MoEBufs,
        x_rev: &mut Vec<f32>,
        normed: &mut Vec<f32>,
    ) {
        // pre-norm → BiRWKV → residual
        normed.resize(seq_len * d_model, 0.0);
        for t in 0..seq_len {
            let off = t * d_model;
            self.norm1.forward_vec(&x[off..off + d_model], &mut normed[off..off + d_model]);
        }
        self.bi_rwkv.forward_batch(normed, seq_len, d_model, fwd_bufs, bwd_bufs, x_rev);
        // x += birwkv_output
        for i in 0..(seq_len * d_model) {
            x[i] += fwd_bufs.final_out[i];
        }

        // pre-norm → MoE FFN → residual
        for t in 0..seq_len {
            let off = t * d_model;
            self.norm2.forward_vec(&x[off..off + d_model], &mut normed[off..off + d_model]);
        }
        self.moe_ffn.forward_batch(normed, seq_len, d_model, moe_bufs);
        for i in 0..(seq_len * d_model) {
            x[i] += moe_bufs.output[i];
        }
    }
}

// ── Edit Tag 상수 및 적용 ─────────────────────────────

const TAG_KEEP: u32 = 0;
const TAG_DELETE: u32 = 1;
// REPLACE_x = 2 .. vocab_size + 1
// INSERT_x  = vocab_size + 2 .. 2 * vocab_size + 1

/// 편집 태그를 원본 시퀀스에 적용하여 교정된 시퀀스 생성
fn apply_edit_tags(src_ids: &[u32], tags: &[u32], vocab_size: u32) -> Vec<u32> {
    let mut result = Vec::with_capacity(src_ids.len());
    for (i, &src_id) in src_ids.iter().enumerate() {
        let tag = if i < tags.len() { tags[i] } else { TAG_KEEP };

        if tag == TAG_KEEP {
            result.push(src_id);
        } else if tag == TAG_DELETE {
            // 삭제: 아무것도 추가하지 않음
        } else if tag >= 2 && tag < vocab_size + 2 {
            // REPLACE: 기존 토큰을 대체
            result.push(tag - 2);
        } else if tag >= vocab_size + 2 && tag < 2 * vocab_size + 2 {
            // INSERT: 기존 토큰 유지 + 새 토큰 삽입
            result.push(src_id);
            result.push(tag - vocab_size - 2);
        } else {
            // 알 수 없는 태그 → KEEP 처리
            result.push(src_id);
        }
    }
    result
}

// ── BitEditor ────────────────────────────────────────

pub struct BitEditor {
    embedding: Vec<f32>,       // (vocab_size * d_model) flat
    layers: Vec<BitEditorLayer>,
    shared_attn: SharedLinearSelfAttention,
    attn_norms: Vec<RMSNorm>,
    final_norm: RMSNorm,
    tag_head_w: Vec<i8>,       // i8 quantized (n_tags * d_model)
    tag_head_scales: Vec<f32>, // per-row scales
    tag_head_sums: Vec<i32>,   // per-row sums
    tag_head_bias: Vec<f32>,   // (n_tags,)
    cfg: ModelConfig,
    attn_insertion_set: HashSet<usize>,
}

impl BitEditor {
    pub fn load_bmmq(model_path: &str, cfg: &ModelConfig) -> Result<Self> {
        eprintln!("BitEditor BMMQ 모델 로드 중: {}", model_path);

        let mut tensors = bmmq::load_bmmq(model_path)
            .context("BMMQ 파일 로드 실패")?;

        eprintln!("  텐서 수: {}", tensors.len());

        // 임베딩 (f32)
        let embedding = bmmq_take_f32(&mut tensors, "embedding.weight")?;

        // 레이어
        let mut layers = Vec::with_capacity(cfg.n_rwkv_layers);
        for i in 0..cfg.n_rwkv_layers {
            let prefix = format!("layers.{}", i);
            layers.push(BitEditorLayer::load_bmmq(&mut tensors, &prefix, cfg)?);
            eprintln!("  레이어 {} 로드", i);
        }

        // Shared attention
        let attn_insertion_set: HashSet<usize> = cfg.attn_insertion_points.iter().cloned().collect();
        let n_attn_insertions = cfg.attn_insertion_points.len();
        let shared_attn = SharedLinearSelfAttention::load_bmmq(
            &mut tensors,
            "shared_attn",
            cfg.n_attn_heads,
            cfg.attn_d_head(),
            n_attn_insertions,
        )?;

        // Attention norms
        let mut attn_norms = Vec::with_capacity(n_attn_insertions);
        for i in 0..n_attn_insertions {
            attn_norms.push(RMSNorm::load_bmmq(
                &mut tensors,
                &format!("attn_norms.{}", i),
                cfg.rms_norm_eps,
            )?);
        }

        // Final norm
        let final_norm = RMSNorm::load_bmmq(&mut tensors, "final_norm", cfg.rms_norm_eps)?;

        // Tag head (Linear with bias → i8 양자화)
        let tag_key = "tag_head.weight";
        let (tag_head_w, tag_head_scales, tag_head_sums, _rows, _cols) =
            bmmq_take_i8(&mut tensors, tag_key)?;
        let tag_head_bias = bmmq_take_f32(&mut tensors, "tag_head.bias")?;

        eprintln!("BitEditor 모델 로드 완료 (레이어 {}개, 전문가 {}×{}, 어텐션 삽입 {}개)",
                  cfg.n_rwkv_layers, cfg.n_experts, cfg.top_k, n_attn_insertions);

        Ok(Self {
            embedding,
            layers,
            shared_attn,
            attn_norms,
            final_norm,
            tag_head_w,
            tag_head_scales,
            tag_head_sums,
            tag_head_bias,
            cfg: cfg.clone(),
            attn_insertion_set,
        })
    }

    /// 메인 추론 진입점: 원문 토큰 ID → 교정된 토큰 ID
    pub fn correct(&self, src_ids: &[u32]) -> Vec<u32> {
        let d = self.cfg.d_model;
        let scale = self.cfg.embed_scale();
        let vocab_size = self.cfg.vocab_size;
        let n_tags = self.cfg.n_tags;
        let max_in_dim = d.max(self.cfg.d_ff).max(n_tags);

        // 작업 버퍼 할당
        let mut fwd_bufs = RWKVBufs::new(max_in_dim);
        let mut bwd_bufs = RWKVBufs::new(max_in_dim);
        let mut moe_bufs = MoEBufs::new(max_in_dim);
        let mut attn_bufs = AttnBufs::new(max_in_dim);
        let mut x_rev = Vec::new();
        let mut normed = Vec::new();

        let mut current_ids = src_ids.to_vec();

        for _iter in 0..self.cfg.n_iterations {
            let cur_len = current_ids.len();

            // 1. 임베딩
            let mut x = vec![0.0f32; cur_len * d];
            for (t, &id) in current_ids.iter().enumerate() {
                let emb_off = id as usize * d;
                for i in 0..d {
                    x[t * d + i] = self.embedding[emb_off + i] * scale;
                }
            }

            // 2. 레이어 처리 (BiRWKV + MoE FFN + shared attention 삽입)
            let mut attn_insert_idx = 0usize;
            for (layer_idx, layer) in self.layers.iter().enumerate() {
                layer.forward_batch(
                    &mut x, cur_len, d,
                    &mut fwd_bufs, &mut bwd_bufs, &mut moe_bufs,
                    &mut x_rev, &mut normed,
                );

                // Shared attention 삽입 (해당 레이어 이후)
                // Python: x = attn_norm(x + shared_attn(x)) — post-norm
                if self.attn_insertion_set.contains(&layer_idx) {
                    // shared self-attention (정규화 없이 x 직접 입력)
                    self.shared_attn.forward_batch(
                        &x, cur_len, attn_insert_idx, &mut attn_bufs,
                    );

                    // x = x + attn_out
                    for i in 0..cur_len * d {
                        x[i] += attn_bufs.final_out[i];
                    }

                    // post-norm: x = RMSNorm(x)
                    for t in 0..cur_len {
                        let off = t * d;
                        // in-place 불가 → normed를 임시 버퍼로 사용
                        self.attn_norms[attn_insert_idx].forward_vec(
                            &x[off..off + d], &mut normed[off..off + d],
                        );
                    }
                    x[..cur_len * d].copy_from_slice(&normed[..cur_len * d]);

                    attn_insert_idx += 1;
                }
            }

            // 3. Final norm
            normed.resize(cur_len * d, 0.0);
            {
                let mut norm_in = vec![0.0f32; d];
                for t in 0..cur_len {
                    let off = t * d;
                    norm_in.copy_from_slice(&x[off..off + d]);
                    self.final_norm.forward_vec(&norm_in, &mut normed[off..off + d]);
                }
            }

            // 4. Tag head: 토큰별 tag logits 계산
            let mut x_u8 = vec![0u8; d];
            let mut tag_logits = vec![0.0f32; n_tags];
            let mut tags = Vec::with_capacity(cur_len);

            for t in 0..cur_len {
                let h_t = &normed[t * d..(t + 1) * d];

                // i8 matmul for tag head
                let x_scale = unsafe {
                    quantize_f32_to_u8(h_t.as_ptr(), x_u8.as_mut_ptr(), d as c_int)
                };
                unsafe {
                    i8_sgemv(
                        self.tag_head_w.as_ptr(),
                        x_u8.as_ptr(),
                        tag_logits.as_mut_ptr(),
                        n_tags as c_int,
                        d as c_int,
                        self.tag_head_sums.as_ptr(),
                        self.tag_head_scales.as_ptr(),
                        x_scale,
                        0.0,
                    );
                }
                // bias 가산
                for i in 0..n_tags {
                    tag_logits[i] += self.tag_head_bias[i];
                }

                // argmax
                let best_tag = tag_logits[..n_tags].iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .map(|(idx, _)| idx as u32)
                    .unwrap_or(TAG_KEEP);

                tags.push(best_tag);
            }

            // 5. 편집 태그 적용
            current_ids = apply_edit_tags(&current_ids, &tags, vocab_size as u32);

            // 모든 태그가 KEEP이면 조기 종료
            if tags.iter().all(|&t| t == TAG_KEEP) {
                break;
            }
        }

        current_ids
    }

    /// 프로파일링 버전: 구간별 시간 측정
    pub fn correct_profiled(&self, src_ids: &[u32]) -> Vec<u32> {
        use std::time::Instant;

        let d = self.cfg.d_model;
        let scale = self.cfg.embed_scale();
        let vocab_size = self.cfg.vocab_size;
        let n_tags = self.cfg.n_tags;
        let max_in_dim = d.max(self.cfg.d_ff).max(n_tags);

        let mut fwd_bufs = RWKVBufs::new(max_in_dim);
        let mut bwd_bufs = RWKVBufs::new(max_in_dim);
        let mut moe_bufs = MoEBufs::new(max_in_dim);
        let mut attn_bufs = AttnBufs::new(max_in_dim);
        let mut x_rev = Vec::new();
        let mut normed = Vec::new();

        let mut current_ids = src_ids.to_vec();
        let cur_len = current_ids.len();

        // 타이머
        let mut t_embed = 0u128;
        let mut t_norm = 0u128;
        let mut t_birwkv = 0u128;
        let mut t_moe = 0u128;
        let mut t_shared_attn = 0u128;
        let mut t_final = 0u128;
        let mut rwkv_detail = [0u128; 6]; // [fwd_proj, fwd_wkv, fwd_post, bwd_proj, bwd_wkv, bwd_post]

        // 임베딩
        let t0 = Instant::now();
        let mut x = vec![0.0f32; cur_len * d];
        for (t, &id) in current_ids.iter().enumerate() {
            let emb_off = id as usize * d;
            for i in 0..d { x[t * d + i] = self.embedding[emb_off + i] * scale; }
        }
        t_embed = t0.elapsed().as_micros();

        // 레이어
        let mut attn_insert_idx = 0usize;
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            // pre-norm
            let t0 = Instant::now();
            normed.resize(cur_len * d, 0.0);
            for t in 0..cur_len {
                let off = t * d;
                layer.norm1.forward_vec(&x[off..off + d], &mut normed[off..off + d]);
            }
            t_norm += t0.elapsed().as_micros();

            // BiRWKV (세분화 프로파일링)
            let t0 = Instant::now();
            layer.bi_rwkv.forward_batch_profiled(&normed, cur_len, d, &mut fwd_bufs, &mut bwd_bufs, &mut x_rev, &mut rwkv_detail);
            for i in 0..(cur_len * d) { x[i] += fwd_bufs.final_out[i]; }
            t_birwkv += t0.elapsed().as_micros();

            // pre-norm → MoE
            let t0 = Instant::now();
            for t in 0..cur_len {
                let off = t * d;
                layer.norm2.forward_vec(&x[off..off + d], &mut normed[off..off + d]);
            }
            t_norm += t0.elapsed().as_micros();

            let t0 = Instant::now();
            layer.moe_ffn.forward_batch(&normed, cur_len, d, &mut moe_bufs);
            for i in 0..(cur_len * d) { x[i] += moe_bufs.output[i]; }
            t_moe += t0.elapsed().as_micros();

            // Shared attention
            if self.attn_insertion_set.contains(&layer_idx) {
                let t0 = Instant::now();
                self.shared_attn.forward_batch(&x, cur_len, attn_insert_idx, &mut attn_bufs);
                for i in 0..cur_len * d { x[i] += attn_bufs.final_out[i]; }
                for t in 0..cur_len {
                    let off = t * d;
                    self.attn_norms[attn_insert_idx].forward_vec(
                        &x[off..off + d], &mut normed[off..off + d]);
                }
                x[..cur_len * d].copy_from_slice(&normed[..cur_len * d]);
                t_shared_attn += t0.elapsed().as_micros();
                attn_insert_idx += 1;
            }
        }

        // Final norm + tag head
        let t0 = Instant::now();
        normed.resize(cur_len * d, 0.0);
        {
            let mut norm_in = vec![0.0f32; d];
            for t in 0..cur_len {
                let off = t * d;
                norm_in.copy_from_slice(&x[off..off + d]);
                self.final_norm.forward_vec(&norm_in, &mut normed[off..off + d]);
            }
        }
        let mut x_u8 = vec![0u8; d];
        let mut tag_logits = vec![0.0f32; n_tags];
        let mut tags = Vec::with_capacity(cur_len);
        for t in 0..cur_len {
            let h_t = &normed[t * d..(t + 1) * d];
            let x_scale = unsafe { quantize_f32_to_u8(h_t.as_ptr(), x_u8.as_mut_ptr(), d as c_int) };
            unsafe {
                i8_sgemv(self.tag_head_w.as_ptr(), x_u8.as_ptr(), tag_logits.as_mut_ptr(),
                         n_tags as c_int, d as c_int, self.tag_head_sums.as_ptr(),
                         self.tag_head_scales.as_ptr(), x_scale, 0.0);
            }
            for i in 0..n_tags { tag_logits[i] += self.tag_head_bias[i]; }
            let best_tag = tag_logits[..n_tags].iter().enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(idx, _)| idx as u32).unwrap_or(TAG_KEEP);
            tags.push(best_tag);
        }
        t_final = t0.elapsed().as_micros();

        current_ids = apply_edit_tags(&current_ids, &tags, vocab_size as u32);

        let total = t_embed + t_norm + t_birwkv + t_moe + t_shared_attn + t_final;
        eprintln!("\n──── 프로파일 ({} 토큰) ────", cur_len);
        eprintln!("  임베딩:          {:>6}µs ({:.1}%)", t_embed, t_embed as f64 / total as f64 * 100.0);
        eprintln!("  RMSNorm:         {:>6}µs ({:.1}%)", t_norm, t_norm as f64 / total as f64 * 100.0);
        eprintln!("  BiRWKV:          {:>6}µs ({:.1}%)", t_birwkv, t_birwkv as f64 / total as f64 * 100.0);
        eprintln!("    ├─ fwd proj:   {:>6}µs ({:.1}%)", rwkv_detail[0], rwkv_detail[0] as f64 / total as f64 * 100.0);
        eprintln!("    ├─ fwd wkv:    {:>6}µs ({:.1}%)", rwkv_detail[1], rwkv_detail[1] as f64 / total as f64 * 100.0);
        eprintln!("    ├─ fwd post:   {:>6}µs ({:.1}%)", rwkv_detail[2], rwkv_detail[2] as f64 / total as f64 * 100.0);
        eprintln!("    ├─ bwd proj:   {:>6}µs ({:.1}%)", rwkv_detail[3], rwkv_detail[3] as f64 / total as f64 * 100.0);
        eprintln!("    ├─ bwd wkv:    {:>6}µs ({:.1}%)", rwkv_detail[4], rwkv_detail[4] as f64 / total as f64 * 100.0);
        eprintln!("    └─ bwd post:   {:>6}µs ({:.1}%)", rwkv_detail[5], rwkv_detail[5] as f64 / total as f64 * 100.0);
        eprintln!("  MoE FFN:         {:>6}µs ({:.1}%)", t_moe, t_moe as f64 / total as f64 * 100.0);
        eprintln!("  SharedAttention:  {:>6}µs ({:.1}%)", t_shared_attn, t_shared_attn as f64 / total as f64 * 100.0);
        eprintln!("  Final+TagHead:   {:>6}µs ({:.1}%)", t_final, t_final as f64 / total as f64 * 100.0);
        eprintln!("  합계:            {:>6}µs", total);

        current_ids
    }
}
