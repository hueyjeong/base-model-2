//! 공통 빌딩 블록 — RMSNorm, BitLinear, Linear, BitNetFFN
//!
//! 기존 BitEditor Rust 추론 엔진에서 이식.
//! AVX-VNNI/AVX2 C 커널을 통한 i8 matmul 가속.

use anyhow::{bail, Context, Result};
use std::collections::HashMap;

use crate::bmmq::TensorData;

// ── C 커널 FFI ──────────────────────────────────────

#[allow(non_camel_case_types)]
type c_int = i32;

extern "C" {
    pub fn i8_sgemv(
        weights: *const i8, x_u8: *const u8, y: *mut f32,
        m: c_int, n: c_int,
        row_sums: *const i32, row_scales: *const f32,
        x_scale: f32, w_scale: f32,
    );
    pub fn i8_sgemm(
        w: *const i8, x_u8: *const u8, y: *mut f32,
        m: c_int, n: c_int, k: c_int,
        row_sums: *const i32, row_scales: *const f32,
        x_scales: *const f32, w_scale: f32,
    );
    pub fn quantize_f32_to_u8(x: *const f32, out: *mut u8, n: c_int) -> f32;
    pub fn batch_quantize_f32_to_u8(
        x: *const f32, out: *mut u8, scales: *mut f32,
        seq_len: c_int, d: c_int,
    );
    pub fn unpack_2bit_rows(
        packed: *const u8, out: *mut i8,
        rows: c_int, cols: c_int, packed_stride: c_int,
    );

    // Mixing 커널
    pub fn wkv6_scan_avx2(
        r: *const f32, k: *const f32, v: *const f32, w: *const f32,
        u_param: *const f32, output: *mut f32, state: *mut f32,
        seq_len: c_int, n_heads: c_int, headdim: c_int, d_model: c_int,
    );
    pub fn mamba_scan_avx2(
        delta: *const f32, B: *const f32, C: *const f32, x: *const f32,
        A: *const f32, D_skip: *const f32, y: *mut f32, state: *mut f32,
        seq_len: c_int, d_inner: c_int, d_state: c_int,
    );
    pub fn retention_scan_avx2(
        q: *const f32, k: *const f32, v: *const f32, gammas: *const f32,
        output: *mut f32, state: *mut f32,
        seq_len: c_int, n_heads: c_int, headdim: c_int,
    );
    pub fn slstm_scan_avx2(
        i_gate: *const f32, f_gate: *const f32, z_gate: *const f32, o_gate: *const f32,
        output: *mut f32, state_c: *mut f32, state_n: *mut f32,
        seq_len: c_int, d_model: c_int,
    );
    pub fn depthwise_conv1d_avx2(
        input: *const f32, weight: *const f32, bias: *const f32,
        output: *mut f32,
        seq_len: c_int, d_model: c_int, kernel_size: c_int, dilation: c_int,
    );
    pub fn mlstm_scan_avx2(
        q: *const f32, k: *const f32, v: *const f32,
        i_gate: *const f32, f_gate: *const f32,
        output: *mut f32,
        state_C: *mut f32, state_n: *mut f32,
        seq_len: c_int, n_heads: c_int, headdim: c_int,
    );
    pub fn causal_conv1d_avx2(
        input: *const f32, weight: *const f32, bias: *const f32,
        output: *mut f32,
        seq_len: c_int, channels: c_int, kernel_size: c_int,
    );
    pub fn mamba2_scan_avx2(
        x: *const f32, B: *const f32, C: *const f32,
        decay: *const f32, D_skip: *const f32, dt: *const f32,
        y: *mut f32, state: *mut f32,
        seq_len: c_int, nheads: c_int, headdim: c_int,
        d_state: c_int, ngroups: c_int,
    );

    /// FP32 batch sgemm (AVX2 + FMA): y[n,m] = w[m,k] @ x[n,k]^T
    pub fn f32_sgemm_avx2(
        w: *const f32, x: *const f32, y: *mut f32,
        m: c_int, n: c_int, k: c_int,
    );

    /// Ternary matmul (AVX2): y[n,m] = gamma * (w_i8[m,k] @ x[n,k]^T)
    pub fn ternary_f32_sgemm_avx2(
        w: *const i8, x: *const f32, y: *mut f32,
        gamma: f32, m: c_int, n: c_int, k: c_int,
    );

    /// Chunk-parallel SSD forward — mamba_ssm CUDA 커널과 수치 호환
    pub fn mamba2_ssd_fwd(
        x: *const f32, B: *const f32, C: *const f32,
        dt: *const f32, A: *const f32, D: *const f32,
        y: *mut f32,
        chunk_size: c_int,
        seq_len: c_int, nheads: c_int, headdim: c_int,
        d_state: c_int, ngroups: c_int,
    );
}

// ── 활성화 함수 ──────────────────────────────────────

#[inline(always)]
pub fn relu_scalar(x: f32) -> f32 {
    if x > 0.0 { x } else { 0.0 }
}

#[inline(always)]
pub fn silu_scalar(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

#[inline(always)]
pub fn sigmoid_scalar(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[inline(always)]
pub fn softplus_scalar(x: f32) -> f32 {
    if x > 20.0 { x } else { (1.0 + x.exp()).ln() }
}

// ── BMMQ 헬퍼 ────────────────────────────────────────

pub fn bmmq_take_f32(tensors: &mut HashMap<String, TensorData>, key: &str) -> Result<Vec<f32>> {
    match tensors.remove(key).context(format!("텐서 없음: {}", key))? {
        TensorData::F32 { data, .. } => Ok(data),
        _ => bail!("f32 타입이어야 함: {}", key),
    }
}

pub fn bmmq_take_i8(tensors: &mut HashMap<String, TensorData>, key: &str)
    -> Result<(Vec<i8>, Vec<f32>, Vec<i32>, usize, usize)>
{
    match tensors.remove(key).context(format!("텐서 없음: {}", key))? {
        TensorData::I8Quantized { data, row_scales, row_sums, rows, cols } => {
            Ok((data, row_scales, row_sums, rows, cols))
        }
        TensorData::F32 { data, shape } => {
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

// ── RMSNorm ──────────────────────────────────────────

pub struct RMSNorm {
    pub weight: Vec<f32>,
    pub eps: f32,
}

impl RMSNorm {
    pub fn load_bmmq(tensors: &mut HashMap<String, TensorData>, prefix: &str, eps: f64) -> Result<Self> {
        let key = format!("{}.weight", prefix);
        let weight = bmmq_take_f32(tensors, &key)?;
        Ok(Self { weight, eps: eps as f32 })
    }

    /// RMSNorm: x[i] * rsqrt(mean(x²) + eps) * weight[i]
    #[inline]
    pub fn forward(&self, x: &[f32], out: &mut [f32]) {
        let n = x.len();
        let mut sq_sum = 0.0f32;
        for &v in x { sq_sum += v * v; }
        let rms = (sq_sum / n as f32 + self.eps).sqrt().recip();
        for i in 0..n {
            out[i] = x[i] * rms * self.weight[i];
        }
    }
}

/// RMSNorm without affine (벤치마크용)
#[inline]
pub fn rms_norm_no_affine(x: &[f32], out: &mut [f32], eps: f32) {
    let n = x.len();
    let mut sq_sum = 0.0f32;
    for &v in x { sq_sum += v * v; }
    let rms_inv = (sq_sum / n as f32 + eps).sqrt().recip();
    for i in 0..n {
        out[i] = x[i] * rms_inv;
    }
}

/// LayerNorm without affine (BitLinear 내부용 — Python nn.LayerNorm과 동일)
/// (x - mean(x)) / sqrt(var(x) + eps)
#[inline]
pub fn layer_norm_no_affine(x: &[f32], out: &mut [f32], eps: f32) {
    let n = x.len();
    let mut sum = 0.0f32;
    for &v in x { sum += v; }
    let mean = sum / n as f32;

    let mut var_sum = 0.0f32;
    for &v in x {
        let d = v - mean;
        var_sum += d * d;
    }
    let inv_std = (var_sum / n as f32 + eps).sqrt().recip();

    for i in 0..n {
        out[i] = (x[i] - mean) * inv_std;
    }
}

// ── BitLinear (1.58-bit ternary) ─────────────────────

pub struct BitLinear {
    pub gamma: f32,
    pub out_dim: usize,
    pub in_dim: usize,
    pub w_i8: Vec<i8>,
    pub row_sums: Vec<i32>,
}

impl BitLinear {
    pub fn load_bmmq(tensors: &mut HashMap<String, TensorData>, prefix: &str) -> Result<Self> {
        let key = format!("{}.weight", prefix);
        match tensors.remove(&key).context(format!("BitLinear weight 없음: {}", key))? {
            TensorData::Packed2Bit { data, gamma, row_sums, rows, cols, packed_stride } => {
                // C 커널이 32-byte 정렬 패딩을 쓰므로 여유 확보
                let aligned_cols = (cols + 31) & !31;
                let mut w_i8 = vec![0i8; rows * aligned_cols];
                unsafe {
                    unpack_2bit_rows(
                        data.as_ptr(), w_i8.as_mut_ptr(),
                        rows as c_int, cols as c_int, packed_stride as c_int,
                    );
                }
                // 패딩된 부분 잘라내기 (matmul은 cols 기준)
                if aligned_cols != cols {
                    let mut compact = vec![0i8; rows * cols];
                    for r in 0..rows {
                        compact[r*cols..(r+1)*cols]
                            .copy_from_slice(&w_i8[r*aligned_cols..r*aligned_cols+cols]);
                    }
                    w_i8 = compact;
                }
                Ok(Self { gamma, out_dim: rows, in_dim: cols, w_i8, row_sums })
            }
            _ => bail!("BitLinear은 Packed2Bit 타입이어야 함: {}", key),
        }
    }

    /// 단일 벡터 forward: LayerNorm → quantize → i8_sgemv
    pub fn forward_vec(&self, x: &[f32], out: &mut [f32], norm_buf: &mut [f32], u8_buf: &mut [u8]) {
        let n = self.in_dim;
        layer_norm_no_affine(x, &mut norm_buf[..n], 1e-5);
        let x_scale = unsafe {
            quantize_f32_to_u8(norm_buf.as_ptr(), u8_buf.as_mut_ptr(), n as c_int)
        };
        self.matmul_preq(&u8_buf[..n], x_scale, out);
    }

    /// 이미 양자화된 입력으로 matmul만
    #[inline]
    pub fn matmul_preq(&self, x_u8: &[u8], x_scale: f32, out: &mut [f32]) {
        unsafe {
            i8_sgemv(
                self.w_i8.as_ptr(), x_u8.as_ptr(), out.as_mut_ptr(),
                self.out_dim as c_int, self.in_dim as c_int,
                self.row_sums.as_ptr(), std::ptr::null(),
                x_scale, self.gamma,
            );
        }
    }

    /// 배치 forward: LayerNorm → quantize → i8_sgemm
    pub fn forward_batch(&self, x: &[f32], seq_len: usize, out: &mut [f32], bufs: &mut BatchBufs) {
        let k = self.in_dim;
        let m = self.out_dim;

        bufs.norm.resize(seq_len * k, 0.0);
        for t in 0..seq_len {
            layer_norm_no_affine(&x[t*k..(t+1)*k], &mut bufs.norm[t*k..(t+1)*k], 1e-5);
        }

        bufs.u8_buf.resize(seq_len * k, 0);
        bufs.scales.resize(seq_len, 0.0);
        unsafe {
            batch_quantize_f32_to_u8(
                bufs.norm.as_ptr(), bufs.u8_buf.as_mut_ptr(), bufs.scales.as_mut_ptr(),
                seq_len as c_int, k as c_int,
            );
            i8_sgemm(
                self.w_i8.as_ptr(), bufs.u8_buf.as_ptr(), out.as_mut_ptr(),
                m as c_int, seq_len as c_int, k as c_int,
                self.row_sums.as_ptr(), std::ptr::null(),
                bufs.scales.as_ptr(), self.gamma,
            );
        }
    }
}

// ── LinearF32 (FP32 matmul — mamba_ssm.Mamba2 projection용) ────

pub struct LinearF32 {
    pub weight: Vec<f32>,  // [out_dim × in_dim] row-major
    pub out_dim: usize,
    pub in_dim: usize,
}

impl LinearF32 {
    pub fn load_bmmq(tensors: &mut HashMap<String, TensorData>, prefix: &str) -> Result<Self> {
        let key = format!("{}.weight", prefix);
        match tensors.remove(&key).context(format!("LinearF32 weight 없음: {}", key))? {
            TensorData::F32 { data, shape } => {
                let (out_dim, in_dim) = (shape[0], shape[1]);
                Ok(Self { weight: data, out_dim, in_dim })
            }
            _ => bail!("LinearF32는 F32 타입이어야 함: {}", key),
        }
    }

    /// FP32 배치 matmul (AVX2 C 커널): out[t,j] = Σ_k weight[j,k] * x[t,k]
    pub fn forward_batch(&self, x: &[f32], seq_len: usize, out: &mut [f32]) {
        unsafe {
            f32_sgemm_avx2(
                self.weight.as_ptr(), x.as_ptr(), out.as_mut_ptr(),
                self.out_dim as i32, seq_len as i32, self.in_dim as i32,
            );
        }
    }

    /// FP32 배치 matmul (순수 Rust — 폴백용)
    #[allow(dead_code)]
    pub fn forward_batch_rust(&self, x: &[f32], seq_len: usize, out: &mut [f32]) {
        let m = self.out_dim;
        let k = self.in_dim;
        for t in 0..seq_len {
            let x_t = &x[t * k..(t + 1) * k];
            let o_t = &mut out[t * m..(t + 1) * m];
            for j in 0..m {
                let w_row = &self.weight[j * k..(j + 1) * k];
                let mut sum = 0.0f32;
                for i in 0..k {
                    sum += w_row[i] * x_t[i];
                }
                o_t[j] = sum;
            }
        }
    }
}

// ── TernaryLinear (1.58-bit ternary weight + FP32 activation) ──

pub struct TernaryLinear {
    pub w_i8: Vec<i8>,   // [out_dim × in_dim] ternary {-1, 0, +1}
    pub gamma: f32,      // 단일 scale factor (per-tensor absmean)
    pub out_dim: usize,
    pub in_dim: usize,
}

impl TernaryLinear {
    pub fn load_bmmq(tensors: &mut HashMap<String, TensorData>, prefix: &str) -> Result<Self> {
        let key = format!("{}.weight", prefix);
        match tensors.remove(&key).context(format!("TernaryLinear weight 없음: {}", key))? {
            TensorData::Packed2Bit { data, gamma, row_sums: _, rows, cols, packed_stride } => {
                let aligned_cols = (cols + 31) & !31;
                let mut w_i8 = vec![0i8; rows * aligned_cols];
                unsafe {
                    unpack_2bit_rows(
                        data.as_ptr(), w_i8.as_mut_ptr(),
                        rows as c_int, cols as c_int, packed_stride as c_int,
                    );
                }
                if aligned_cols != cols {
                    let mut compact = vec![0i8; rows * cols];
                    for r in 0..rows {
                        compact[r*cols..(r+1)*cols]
                            .copy_from_slice(&w_i8[r*aligned_cols..r*aligned_cols+cols]);
                    }
                    w_i8 = compact;
                }
                Ok(Self { w_i8, gamma, out_dim: rows, in_dim: cols })
            }
            _ => bail!("TernaryLinear은 Packed2Bit 타입이어야 함: {}", key),
        }
    }

    /// Ternary 배치 matmul: out[t,j] = gamma * Σ_k w_i8[j,k] * x[t,k]
    pub fn forward_batch(&self, x: &[f32], seq_len: usize, out: &mut [f32]) {
        unsafe {
            ternary_f32_sgemm_avx2(
                self.w_i8.as_ptr(), x.as_ptr(), out.as_mut_ptr(),
                self.gamma, self.out_dim as i32, seq_len as i32, self.in_dim as i32,
            );
        }
    }
}

// ── Linear (per-row i8 quantized) ────────────────────

pub struct Linear {
    pub w_i8: Vec<i8>,
    pub row_scales: Vec<f32>,
    pub row_sums: Vec<i32>,
    pub out_dim: usize,
    pub in_dim: usize,
}

impl Linear {
    pub fn load_bmmq(tensors: &mut HashMap<String, TensorData>, prefix: &str) -> Result<Self> {
        let key = format!("{}.weight", prefix);
        let (data, row_scales, row_sums, rows, cols) = bmmq_take_i8(tensors, &key)?;
        Ok(Self { w_i8: data, row_scales, row_sums, out_dim: rows, in_dim: cols })
    }

    pub fn forward_batch(&self, x: &[f32], seq_len: usize, out: &mut [f32], bufs: &mut BatchBufs) {
        let k = self.in_dim;
        let m = self.out_dim;
        bufs.u8_buf.resize(seq_len * k, 0);
        bufs.scales.resize(seq_len, 0.0);
        unsafe {
            batch_quantize_f32_to_u8(
                x.as_ptr(), bufs.u8_buf.as_mut_ptr(), bufs.scales.as_mut_ptr(),
                seq_len as c_int, k as c_int,
            );
            i8_sgemm(
                self.w_i8.as_ptr(), bufs.u8_buf.as_ptr(), out.as_mut_ptr(),
                m as c_int, seq_len as c_int, k as c_int,
                self.row_sums.as_ptr(), self.row_scales.as_ptr(),
                bufs.scales.as_ptr(), 0.0,
            );
        }
    }
}

// ── BitNetFFN (ReLU gating) ──────────────────────────

pub struct BitNetFFN {
    pub gate_proj: BitLinear,
    pub up_proj: BitLinear,
    pub down_proj: BitLinear,
    pub d_ff: usize,
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

    /// 배치 forward: gate/up 공유 양자화 → relu(gate)*up → down_proj
    pub fn forward_batch(
        &self, x: &[f32], seq_len: usize, d_model: usize,
        out: &mut [f32], bufs: &mut BatchBufs, ff_buf: &mut Vec<f32>, ff_buf2: &mut Vec<f32>,
    ) {
        let d_ff = self.d_ff;
        ff_buf.resize(seq_len * d_ff, 0.0);
        ff_buf2.resize(seq_len * d_ff, 0.0);

        // gate와 up은 같은 입력 → 양자화 1회 공유
        let k = d_model;
        bufs.norm.resize(seq_len * k, 0.0);
        for t in 0..seq_len {
            layer_norm_no_affine(&x[t*k..(t+1)*k], &mut bufs.norm[t*k..(t+1)*k], 1e-5);
        }
        bufs.u8_buf.resize(seq_len * k, 0);
        bufs.scales.resize(seq_len, 0.0);
        unsafe {
            batch_quantize_f32_to_u8(
                bufs.norm.as_ptr(), bufs.u8_buf.as_mut_ptr(), bufs.scales.as_mut_ptr(),
                seq_len as c_int, k as c_int,
            );
        }
        // gate
        unsafe {
            i8_sgemm(
                self.gate_proj.w_i8.as_ptr(), bufs.u8_buf.as_ptr(), ff_buf.as_mut_ptr(),
                d_ff as c_int, seq_len as c_int, k as c_int,
                self.gate_proj.row_sums.as_ptr(), std::ptr::null(),
                bufs.scales.as_ptr(), self.gate_proj.gamma,
            );
        }
        // up
        unsafe {
            i8_sgemm(
                self.up_proj.w_i8.as_ptr(), bufs.u8_buf.as_ptr(), ff_buf2.as_mut_ptr(),
                d_ff as c_int, seq_len as c_int, k as c_int,
                self.up_proj.row_sums.as_ptr(), std::ptr::null(),
                bufs.scales.as_ptr(), self.up_proj.gamma,
            );
        }
        // relu(gate) * up
        for i in 0..seq_len * d_ff {
            ff_buf[i] = relu_scalar(ff_buf[i]) * ff_buf2[i];
        }
        // down_proj
        self.down_proj.forward_batch(ff_buf, seq_len, out, bufs);
    }
}

// ── 배치 버퍼 ─────────────────────────────────────────

pub struct BatchBufs {
    pub norm: Vec<f32>,
    pub u8_buf: Vec<u8>,
    pub scales: Vec<f32>,
}

impl BatchBufs {
    pub fn new(max_dim: usize) -> Self {
        Self {
            norm: vec![0.0; max_dim],
            u8_buf: vec![0u8; max_dim],
            scales: vec![0.0; max_dim],
        }
    }
}
