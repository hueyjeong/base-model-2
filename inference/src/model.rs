//! BitMamba Seq2Seq 모델 — CPU 추론 엔진 (v2 최적화)
//!
//! v2 최적화: i8 ternary BitLinear (mul→add/sub), rayon 병렬 matmul,
//!           cross-attn 순수 scalar, 버퍼 재활용, 정밀 타이밍

use anyhow::{bail, Context, Result};
use candle_core::{DType, Device, IndexOp, Tensor, D};
use std::collections::HashMap;

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
}

// ── BMMQ 헬퍼 ────────────────────────────────────────

/// BMMQ TensorData에서 f32 Vec 추출 (소유권 이전)
fn bmmq_take_f32(tensors: &mut HashMap<String, TensorData>, key: &str) -> Result<Vec<f32>> {
    match tensors.remove(key).context(format!("텐서 없음: {}", key))? {
        TensorData::F32 { data, .. } => Ok(data),
        _ => bail!("f32 타입이어야 함: {}", key),
    }
}


// ── 활성화 함수 ──────────────────────────────────────

fn silu(x: &Tensor) -> Result<Tensor> {
    let neg = x.neg()?;
    let sig = (neg.exp()? + 1.0)?.recip()?;
    Ok((x * &sig)?)
}

#[inline(always)]
fn silu_scalar(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

fn sigmoid(x: &Tensor) -> Result<Tensor> {
    let neg = x.neg()?;
    Ok((neg.exp()? + 1.0)?.recip()?)
}

#[inline(always)]
fn softplus_scalar(x: f32) -> f32 {
    if x > 20.0 { x } else { (1.0 + x.exp()).ln() }
}

fn gelu(x: &Tensor) -> Result<Tensor> {
    let x3 = (x * x)?.broadcast_mul(x)?;
    let inner = ((x + &(x3 * 0.044715)?)? * 0.7978845608)?;
    let tanh_val = inner.tanh()?;
    let one_plus_tanh = (tanh_val + 1.0)?;
    let scaled = x.broadcast_mul(&one_plus_tanh)?;
    Ok((scaled * 0.5)?)
}

fn gelu1p(x: &Tensor) -> Result<Tensor> {
    Ok((gelu(x)? + 1.0)?)
}

#[inline(always)]
fn gelu1p_scalar(x: f32) -> f32 {
    let x3 = x * x * x;
    let inner = (x + 0.044715 * x3) * 0.7978845608;
    x * 0.5 * (1.0 + inner.tanh()) + 1.0
}

// ── RMSNorm ──────────────────────────────────────────

pub struct RMSNorm {
    weight: Option<Tensor>,
    weight_vec: Vec<f32>,
    eps: f32,
}

impl RMSNorm {
    pub fn load(tensors: &HashMap<String, Tensor>, prefix: &str, eps: f64) -> Result<Self> {
        let weight = tensors.get(&format!("{}.weight", prefix))
            .context(format!("RMSNorm weight 없음: {}", prefix))?.clone();
        let weight_vec: Vec<f32> = weight.flatten_all()?.to_vec1()?;
        Ok(Self { weight: Some(weight), weight_vec, eps: eps as f32 })
    }

    pub fn load_bmmq(tensors: &mut HashMap<String, TensorData>, prefix: &str, eps: f64) -> Result<Self> {
        let key = format!("{}.weight", prefix);
        let weight_vec = bmmq_take_f32(tensors, &key)?;
        Ok(Self { weight: None, weight_vec, eps: eps as f32 })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let w = self.weight.as_ref().expect("RMSNorm Tensor dropped");
        let var = x.sqr()?.mean_keepdim(D::Minus1)?;
        let rms = (var + self.eps as f64)?.sqrt()?.recip()?;
        Ok((x.broadcast_mul(&rms))?.broadcast_mul(w)?)
    }

    fn drop_tensor(&mut self) { self.weight = None; }

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

// ── LayerNorm (elementwise_affine=False) ─────────────

fn layer_norm_no_affine(x: &Tensor, eps: f64) -> Result<Tensor> {
    let mean = x.mean_keepdim(D::Minus1)?;
    let centered = x.broadcast_sub(&mean)?;
    let var = centered.sqr()?.mean_keepdim(D::Minus1)?;
    let inv_std = (var + eps)?.sqrt()?.recip()?;
    Ok(centered.broadcast_mul(&inv_std)?)
}

#[inline]
fn layer_norm_no_affine_vec(x: &[f32], out: &mut [f32], eps: f32) {
    let n = x.len();
    let inv_n = 1.0 / n as f32;
    let mean: f32 = x.iter().sum::<f32>() * inv_n;
    let mut var = 0.0f32;
    for i in 0..n {
        let d = x[i] - mean;
        out[i] = d;
        var += d * d;
    }
    let inv_std = (var * inv_n + eps).sqrt().recip();
    for v in out.iter_mut().take(n) { *v *= inv_std; }
}

// ── BitLinear (i8 ternary + AVX-VNNI) ───────────────

pub struct BitLinear {
    w_quant: Option<Tensor>,  // 인코더 Tensor path용 (디코딩 후 해제)
    gamma: f32,
    out_dim: usize,
    in_dim: usize,
    // 2-bit packed 모드 (BMMQ): w_packed 사용, w_i8은 빈 Vec
    // i8 모드 (safetensors): w_i8 사용, w_packed은 빈 Vec
    w_i8: Vec<i8>,             // i8 ternary {-1,0,+1} (safetensors 경로)
    w_packed: Vec<u8>,         // 2-bit packed ternary (BMMQ 경로)
    packed_stride: usize,      // (in_dim + 3) / 4
    row_sums: Vec<i32>,        // Σ_j w[i,j]
}

impl BitLinear {
    pub fn load(tensors: &HashMap<String, Tensor>, prefix: &str) -> Result<Self> {
        let weight = tensors.get(&format!("{}.weight", prefix))
            .context(format!("BitLinear weight 없음: {}", prefix))?.clone();

        let out_dim = weight.dim(0)?;
        let in_dim = weight.dim(1)?;

        // 사전 양자화
        let gamma_t = weight.abs()?.mean_all()?.clamp(1e-5, f64::MAX)?;
        let gamma: f32 = gamma_t.to_scalar()?;
        let w_scaled = weight.broadcast_div(&gamma_t)?;
        let w_quant = w_scaled.clamp(-1.0, 1.0)?.round()?;

        // i8 ternary + row_sums 사전계산
        let w_f32: Vec<f32> = w_quant.flatten_all()?.to_vec1()?;
        let mut w_i8 = vec![0i8; out_dim * in_dim];
        let mut row_sums = vec![0i32; out_dim];
        for row in 0..out_dim {
            let mut rsum = 0i32;
            for col in 0..in_dim {
                let v = w_f32[row * in_dim + col] as i8;
                w_i8[row * in_dim + col] = v;
                rsum += v as i32;
            }
            row_sums[row] = rsum;
        }

        Ok(Self {
            w_quant: Some(w_quant), gamma, out_dim, in_dim,
            w_i8, w_packed: Vec::new(), packed_stride: 0, row_sums,
        })
    }

    pub fn load_bmmq(tensors: &mut HashMap<String, TensorData>, prefix: &str) -> Result<Self> {
        let key = format!("{}.weight", prefix);
        match tensors.remove(&key).context(format!("BitLinear weight 없음: {}", key))? {
            TensorData::Packed2Bit { data, gamma, row_sums, rows, cols, packed_stride } => {
                Ok(Self {
                    w_quant: None,
                    gamma,
                    out_dim: rows,
                    in_dim: cols,
                    w_i8: Vec::new(),
                    w_packed: data,
                    packed_stride,
                    row_sums,
                })
            }
            _ => bail!("BitLinear은 Packed2Bit 타입이어야 함: {}", key),
        }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let w_quant = self.w_quant.as_ref().expect("Tensor already dropped");
        let x_norm = layer_norm_no_affine(x, 1e-5)?;
        let eta = x_norm.abs()?.max_keepdim(D::Minus1)?.clamp(1e-5, f64::MAX)?;
        let x_quant = x_norm.broadcast_div(&eta)?
            .broadcast_mul(&Tensor::new(127.0f32, x.device())?)?
            .clamp(-128.0, 127.0)?.round()?;
        let x_scale = (eta / 127.0)?;

        let wt = w_quant.t()?;
        let dims = x_quant.dims();
        let out = if dims.len() == 3 {
            let (b, l, d) = x_quant.dims3()?;
            let flat = x_quant.reshape((b * l, d))?;
            let r = flat.matmul(&wt)?;
            r.reshape((b, l, self.out_dim))?
        } else {
            x_quant.matmul(&wt)?
        };

        let gamma_t = Tensor::new(self.gamma, x.device())?;
        let scale = x_scale.broadcast_mul(&gamma_t)?;
        Ok(out.broadcast_mul(&scale)?)
    }

    /// 인코딩 후 Tensor 해제 (메모리 절감)
    fn drop_tensor(&mut self) { self.w_quant = None; }

    /// AVX-VNNI matmul (할당 없음 — 외부 버퍼)
    /// packed 모드: ternary_sgemv (2-bit → 언팩 → VNNI)
    /// i8 모드: i8_sgemv
    fn forward_vec(&self, x: &[f32], out: &mut [f32], x_norm: &mut [f32], x_u8: &mut [u8]) {
        debug_assert_eq!(x.len(), self.in_dim);
        let n = self.in_dim;

        // 1. LayerNorm
        layer_norm_no_affine_vec(x, &mut x_norm[..n], 1e-5);

        // 2. f32 → u8 양자화
        let x_scale = unsafe {
            quantize_f32_to_u8(x_norm.as_ptr(), x_u8.as_mut_ptr(), n as c_int)
        };

        // 3. matmul
        if !self.w_packed.is_empty() {
            // 2-bit packed → ternary_sgemv (on-the-fly 언팩)
            unsafe {
                ternary_sgemv(
                    self.w_packed.as_ptr(),
                    x_u8.as_ptr(),
                    out.as_mut_ptr(),
                    self.out_dim as c_int,
                    n as c_int,
                    self.packed_stride as c_int,
                    self.row_sums.as_ptr(),
                    self.gamma,
                    x_scale,
                );
            }
        } else {
            // i8 → i8_sgemv (safetensors 경로)
            unsafe {
                i8_sgemv(
                    self.w_i8.as_ptr(),
                    x_u8.as_ptr(),
                    out.as_mut_ptr(),
                    self.out_dim as c_int,
                    n as c_int,
                    self.row_sums.as_ptr(),
                    std::ptr::null(),
                    x_scale,
                    self.gamma,
                );
            }
        }
    }
}

// ── Linear (i8 양자화 + AVX-VNNI) ───────────────────

pub struct Linear {
    weight: Option<Tensor>,   // 인코더 Tensor path용 (디코딩 후 해제)
    w_i8: Vec<i8>,            // per-row int8 양자화 weight
    row_scales: Vec<f32>,     // per-row dequant scale: max(abs(row))/127
    row_sums: Vec<i32>,       // Σ_j w_i8[i,j] — u8 오프셋 보정용
    out_dim: usize,
    in_dim: usize,
}

impl Linear {
    pub fn load(tensors: &HashMap<String, Tensor>, prefix: &str) -> Result<Self> {
        let weight = tensors.get(&format!("{}.weight", prefix))
            .context(format!("Linear weight 없음: {}", prefix))?.clone();
        let out_dim = weight.dim(0)?;
        let in_dim = weight.dim(1)?;
        let w_f32: Vec<f32> = weight.flatten_all()?.to_vec1()?;

        // per-row int8 양자화
        let mut w_i8 = vec![0i8; out_dim * in_dim];
        let mut row_scales = vec![0.0f32; out_dim];
        let mut row_sums = vec![0i32; out_dim];

        for row in 0..out_dim {
            let base = row * in_dim;
            let mut max_abs = 0.0f32;
            for col in 0..in_dim {
                let a = w_f32[base + col].abs();
                if a > max_abs { max_abs = a; }
            }
            if max_abs < 1e-10 { max_abs = 1e-10; }
            let scale = max_abs / 127.0;
            let inv_scale = 127.0 / max_abs;
            row_scales[row] = scale;

            let mut rsum = 0i32;
            for col in 0..in_dim {
                let v = (w_f32[base + col] * inv_scale).round().clamp(-128.0, 127.0) as i8;
                w_i8[base + col] = v;
                rsum += v as i32;
            }
            row_sums[row] = rsum;
        }

        Ok(Self { weight: Some(weight), w_i8, row_scales, row_sums, out_dim, in_dim })
    }

    pub fn load_bmmq(tensors: &mut HashMap<String, TensorData>, prefix: &str) -> Result<Self> {
        let key = format!("{}.weight", prefix);
        match tensors.remove(&key).context(format!("Linear weight 없음: {}", key))? {
            TensorData::I8Quantized { data, row_scales, row_sums, rows, cols } => {
                Ok(Self {
                    weight: None,
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

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let weight = self.weight.as_ref().expect("Tensor already dropped");
        let wt = weight.t()?;
        let dims = x.dims();
        if dims.len() == 3 {
            let (b, l, d) = x.dims3()?;
            let flat = x.reshape((b * l, d))?;
            let result = flat.matmul(&wt)?;
            Ok(result.reshape((b, l, self.out_dim))?)
        } else {
            Ok(x.matmul(&wt)?)
        }
    }

    /// 인코딩 후 Tensor 해제 (메모리 절감)
    fn drop_tensor(&mut self) { self.weight = None; }

    /// i8 matmul via AVX-VNNI (할당 없음)
    fn forward_vec(&self, x: &[f32], out: &mut [f32], x_u8: &mut [u8]) {
        let n = self.in_dim;
        let x_scale = unsafe {
            quantize_f32_to_u8(x.as_ptr(), x_u8.as_mut_ptr(), n as c_int)
        };
        unsafe {
            i8_sgemv(
                self.w_i8.as_ptr(),
                x_u8.as_ptr(),
                out.as_mut_ptr(),
                self.out_dim as c_int,
                n as c_int,
                self.row_sums.as_ptr(),
                self.row_scales.as_ptr(),  // per-row scales
                x_scale,
                0.0,  // w_scale 미사용
            );
        }
    }
}

// ── LinearWithBias (Copy Gate용 — i8 양자화) ─────────

pub struct LinearWithBias {
    w_i8: Vec<i8>,
    row_scales: Vec<f32>,
    row_sums: Vec<i32>,
    bias_data: Vec<f32>,
    out_dim: usize,
    in_dim: usize,
}

impl LinearWithBias {
    pub fn load(tensors: &HashMap<String, Tensor>, prefix: &str) -> Result<Self> {
        let weight = tensors.get(&format!("{}.weight", prefix))
            .context(format!("Linear weight 없음: {}", prefix))?.clone();
        let bias = tensors.get(&format!("{}.bias", prefix))
            .context(format!("Linear bias 없음: {}", prefix))?.clone();
        let out_dim = weight.dim(0)?;
        let in_dim = weight.dim(1)?;
        let w_f32: Vec<f32> = weight.flatten_all()?.to_vec1()?;
        let bias_data: Vec<f32> = bias.flatten_all()?.to_vec1()?;

        let mut w_i8 = vec![0i8; out_dim * in_dim];
        let mut row_scales = vec![0.0f32; out_dim];
        let mut row_sums = vec![0i32; out_dim];
        for row in 0..out_dim {
            let base = row * in_dim;
            let mut max_abs = 0.0f32;
            for col in 0..in_dim { max_abs = max_abs.max(w_f32[base + col].abs()); }
            if max_abs < 1e-10 { max_abs = 1e-10; }
            row_scales[row] = max_abs / 127.0;
            let inv_scale = 127.0 / max_abs;
            let mut rsum = 0i32;
            for col in 0..in_dim {
                let v = (w_f32[base + col] * inv_scale).round().clamp(-128.0, 127.0) as i8;
                w_i8[base + col] = v;
                rsum += v as i32;
            }
            row_sums[row] = rsum;
        }

        Ok(Self { w_i8, row_scales, row_sums, bias_data, out_dim, in_dim })
    }

    pub fn load_bmmq(tensors: &mut HashMap<String, TensorData>, prefix: &str) -> Result<Self> {
        let w_key = format!("{}.weight", prefix);
        let td = tensors.remove(&w_key).context(format!("LinearWithBias weight 없음: {}", w_key))?;

        let (w_i8, row_scales, row_sums, out_dim, in_dim) = match td {
            TensorData::I8Quantized { data, row_scales, row_sums, rows, cols } => {
                (data, row_scales, row_sums, rows, cols)
            }
            TensorData::F32 { data, shape } => {
                // f32 → per-row i8 양자화 (copy_gate 등 소형 텐서)
                let (out_dim, in_dim) = (shape[0], shape[1]);
                let mut w_i8 = vec![0i8; out_dim * in_dim];
                let mut row_scales = vec![0.0f32; out_dim];
                let mut row_sums = vec![0i32; out_dim];
                for row in 0..out_dim {
                    let base = row * in_dim;
                    let mut max_abs = 0.0f32;
                    for col in 0..in_dim { max_abs = max_abs.max(data[base + col].abs()); }
                    if max_abs < 1e-10 { max_abs = 1e-10; }
                    row_scales[row] = max_abs / 127.0;
                    let inv_scale = 127.0 / max_abs;
                    let mut rsum = 0i32;
                    for col in 0..in_dim {
                        let v = (data[base + col] * inv_scale).round().clamp(-128.0, 127.0) as i8;
                        w_i8[base + col] = v;
                        rsum += v as i32;
                    }
                    row_sums[row] = rsum;
                }
                (w_i8, row_scales, row_sums, out_dim, in_dim)
            }
            _ => bail!("LinearWithBias weight은 I8 또는 F32 타입이어야 함: {}", w_key),
        };

        let bias_key = format!("{}.bias", prefix);
        let bias_data = bmmq_take_f32(tensors, &bias_key)?;

        Ok(Self { w_i8, row_scales, row_sums, bias_data, out_dim, in_dim })
    }

    fn forward_vec(&self, x: &[f32], out: &mut [f32], x_u8: &mut [u8]) {
        let n = self.in_dim;
        let x_scale = unsafe {
            quantize_f32_to_u8(x.as_ptr(), x_u8.as_mut_ptr(), n as c_int)
        };
        unsafe {
            i8_sgemv(
                self.w_i8.as_ptr(),
                x_u8.as_ptr(),
                out.as_mut_ptr(),
                self.out_dim as c_int,
                n as c_int,
                self.row_sums.as_ptr(),
                self.row_scales.as_ptr(),
                x_scale,
                0.0,
            );
        }
        for o in 0..self.out_dim {
            out[o] += self.bias_data[o];
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
    pub fn load(tensors: &HashMap<String, Tensor>, prefix: &str) -> Result<Self> {
        let gate_proj = BitLinear::load(tensors, &format!("{}.gate_proj", prefix))?;
        let d_ff = gate_proj.out_dim;
        Ok(Self {
            gate_proj,
            up_proj: BitLinear::load(tensors, &format!("{}.up_proj", prefix))?,
            down_proj: BitLinear::load(tensors, &format!("{}.down_proj", prefix))?,
            d_ff,
        })
    }

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

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = sigmoid(&self.gate_proj.forward(x)?)?;
        let up = self.up_proj.forward(x)?;
        let x = (gate * up)?;
        self.down_proj.forward(&x)
    }

    fn forward_vec(&self, x: &[f32], buf_ff: &mut Vec<f32>, buf_ff2: &mut Vec<f32>,
                    x_norm_buf: &mut [f32], x_u8: &mut [u8], out: &mut [f32]) {
        buf_ff.resize(self.d_ff, 0.0);
        buf_ff2.resize(self.d_ff, 0.0);
        self.gate_proj.forward_vec(x, buf_ff, x_norm_buf, x_u8);
        self.up_proj.forward_vec(x, buf_ff2, x_norm_buf, x_u8);

        // sigmoid(gate) * up → reuse buf_ff
        for i in 0..self.d_ff {
            let sig = 1.0 / (1.0 + (-buf_ff[i]).exp());
            buf_ff[i] = sig * buf_ff2[i];
        }

        self.down_proj.forward_vec(buf_ff, out, x_norm_buf, x_u8);
    }

    fn drop_tensors(&mut self) {
        self.gate_proj.drop_tensor();
        self.up_proj.drop_tensor();
        self.down_proj.drop_tensor();
    }
}

// ── Mamba2 State ─────────────────────────────────────

pub struct Mamba2State {
    pub ssm_state: Vec<f32>,
    pub conv_buf: Vec<f32>,
    pub conv_buf_len: usize,
}

// ── Mamba2Block ──────────────────────────────────────

pub struct Mamba2Block {
    in_proj: Linear,
    conv1d_weight: Option<Tensor>,  // 인코더 Tensor path용
    conv1d_bias: Option<Tensor>,
    norm_weight: Option<Tensor>,
    out_proj: Linear,
    // scalar용
    conv_weight_data: Vec<f32>,
    conv_bias_data: Vec<f32>,
    norm_weight_data: Vec<f32>,
    dt_bias_data: Vec<f32>,
    a_data: Vec<f32>,
    d_data: Vec<f32>,
    // 파라미터
    d_inner: usize,
    d_state: usize,
    d_conv: usize,
    nheads: usize,
    headdim: usize,
    ngroups: usize,
    conv_channels: usize,
    proj_dim: usize,
}

impl Mamba2Block {
    pub fn load(
        tensors: &HashMap<String, Tensor>,
        prefix: &str,
        cfg: &ModelConfig,
    ) -> Result<Self> {
        let p = format!("{}.mamba2", prefix);
        let conv1d_weight = tensors.get(&format!("{}.conv1d.weight", p))
            .context("conv1d weight 없음")?.clone();
        let conv1d_bias = tensors.get(&format!("{}.conv1d.bias", p))
            .context("conv1d bias 없음")?.clone();
        let norm_weight = tensors.get(&format!("{}.norm.weight", p))
            .context("norm weight 없음")?.clone();
        let dt_bias_raw = tensors.get(&format!("{}.dt_bias", p))
            .context("dt_bias 없음")?.clone();
        let a_log = tensors.get(&format!("{}.A_log", p))
            .context("A_log 없음")?.clone();
        let d_param = tensors.get(&format!("{}.D", p))
            .context("D 없음")?.clone();

        let conv_channels = cfg.d_inner + 2 * cfg.ngroups * cfg.d_state;
        let nheads = cfg.nheads();
        let proj_dim = 2 * cfg.d_inner + 2 * cfg.ngroups * cfg.d_state + nheads;

        let conv_weight_data: Vec<f32> = conv1d_weight.flatten_all()?.to_vec1()?;
        let conv_bias_data: Vec<f32> = conv1d_bias.flatten_all()?.to_vec1()?;
        let norm_weight_data: Vec<f32> = norm_weight.flatten_all()?.to_vec1()?;
        let dt_bias_data: Vec<f32> = dt_bias_raw.flatten_all()?.to_vec1()?;
        let a_data: Vec<f32> = a_log.exp()?.neg()?.to_vec1()?;
        let d_data: Vec<f32> = d_param.flatten_all()?.to_vec1()?;

        Ok(Self {
            in_proj: Linear::load(tensors, &format!("{}.in_proj", p))?,
            conv1d_weight: Some(conv1d_weight),
            conv1d_bias: Some(conv1d_bias),
            norm_weight: Some(norm_weight),
            out_proj: Linear::load(tensors, &format!("{}.out_proj", p))?,
            conv_weight_data,
            conv_bias_data,
            norm_weight_data,
            dt_bias_data,
            a_data,
            d_data,
            d_inner: cfg.d_inner,
            d_state: cfg.d_state,
            d_conv: cfg.d_conv,
            nheads,
            headdim: cfg.headdim,
            ngroups: cfg.ngroups,
            conv_channels,
            proj_dim,
        })
    }

    pub fn load_bmmq(
        tensors: &mut HashMap<String, TensorData>,
        prefix: &str,
        cfg: &ModelConfig,
    ) -> Result<Self> {
        let p = format!("{}.mamba2", prefix);

        let conv_weight_data = bmmq_take_f32(tensors, &format!("{}.conv1d.weight", p))?;
        let conv_bias_data = bmmq_take_f32(tensors, &format!("{}.conv1d.bias", p))?;
        let norm_weight_data = bmmq_take_f32(tensors, &format!("{}.norm.weight", p))?;
        let dt_bias_data = bmmq_take_f32(tensors, &format!("{}.dt_bias", p))?;

        let a_log_data = bmmq_take_f32(tensors, &format!("{}.A_log", p))?;
        let a_data: Vec<f32> = a_log_data.iter().map(|&v| -(v.exp())).collect();

        let d_data = bmmq_take_f32(tensors, &format!("{}.D", p))?;

        let conv_channels = cfg.d_inner + 2 * cfg.ngroups * cfg.d_state;
        let nheads = cfg.nheads();
        let proj_dim = 2 * cfg.d_inner + 2 * cfg.ngroups * cfg.d_state + nheads;

        Ok(Self {
            in_proj: Linear::load_bmmq(tensors, &format!("{}.in_proj", p))?,
            conv1d_weight: None,
            conv1d_bias: None,
            norm_weight: None,
            out_proj: Linear::load_bmmq(tensors, &format!("{}.out_proj", p))?,
            conv_weight_data,
            conv_bias_data,
            norm_weight_data,
            dt_bias_data,
            a_data,
            d_data,
            d_inner: cfg.d_inner,
            d_state: cfg.d_state,
            d_conv: cfg.d_conv,
            nheads,
            headdim: cfg.headdim,
            ngroups: cfg.ngroups,
            conv_channels,
            proj_dim,
        })
    }

    pub fn new_state(&self) -> Mamba2State {
        Mamba2State {
            ssm_state: vec![0.0f32; self.nheads * self.d_state * self.headdim],
            conv_buf: vec![0.0f32; (self.d_conv - 1) * self.conv_channels],
            conv_buf_len: 0,
        }
    }

    fn causal_conv1d(&self, x: &Tensor) -> Result<Tensor> {
        let conv1d_weight = self.conv1d_weight.as_ref().expect("Tensor dropped");
        let conv1d_bias = self.conv1d_bias.as_ref().expect("Tensor dropped");
        let (_, seq_len, channels) = x.dims3()?;
        let dev = x.device();
        let pad = Tensor::zeros((1, self.d_conv - 1, channels), DType::F32, dev)?;
        let x_padded = Tensor::cat(&[&pad, x], 1)?;

        let mut out = conv1d_bias.unsqueeze(0)?.unsqueeze(0)?
            .broadcast_as((1, seq_len, channels))?.contiguous()?;

        for k in 0..self.d_conv {
            let w_k = conv1d_weight.i((.., 0, k))?;
            let x_k = x_padded.narrow(1, k, seq_len)?;
            out = (out + x_k.broadcast_mul(&w_k)?)?;
        }
        Ok(out)
    }

    fn sequential_scan(
        &self,
        x: &Tensor,
        dt: &Tensor,
        b_ssm: &Tensor,
        c_ssm: &Tensor,
    ) -> Result<Tensor> {
        let seq_len = x.dim(1)?;
        let dev = x.device();

        let x_data: Vec<f32> = x.flatten_all()?.to_vec1()?;
        let dt_data: Vec<f32> = dt.flatten_all()?.to_vec1()?;
        let b_data: Vec<f32> = b_ssm.flatten_all()?.to_vec1()?;
        let c_data: Vec<f32> = c_ssm.flatten_all()?.to_vec1()?;

        let mut state = vec![0.0f32; self.nheads * self.d_state * self.headdim];
        let mut y_data = vec![0.0f32; seq_len * self.nheads * self.headdim];

        let nh_hd = self.nheads * self.headdim;
        let ng_ds = self.ngroups * self.d_state;

        for t in 0..seq_len {
            let t_xoff = t * nh_hd;
            let t_boff = t * ng_ds;
            for h in 0..self.nheads {
                let g = h * self.ngroups / self.nheads;
                let dt_val = dt_data[t * self.nheads + h];
                let da = (self.a_data[h] * dt_val).exp();
                let h_hd = h * self.headdim;
                let h_ds_hd = h * self.d_state * self.headdim;

                for s in 0..self.d_state {
                    let b_val = b_data[t_boff + g * self.d_state + s];
                    let c_val = c_data[t_boff + g * self.d_state + s];
                    let db = dt_val * b_val;
                    let s_hd = s * self.headdim;

                    for d in 0..self.headdim {
                        let state_idx = h_ds_hd + s_hd + d;
                        let x_val = x_data[t_xoff + h_hd + d];

                        state[state_idx] = da * state[state_idx] + db * x_val;
                        y_data[t_xoff + h_hd + d] += c_val * state[state_idx];
                    }
                }

                let d_h = self.d_data[h];
                for d in 0..self.headdim {
                    y_data[t_xoff + h_hd + d] += d_h * x_data[t_xoff + h_hd + d];
                }
            }
        }

        Ok(Tensor::from_vec(y_data, (1, seq_len, self.nheads, self.headdim), dev)?)
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (_, seq_len, _) = x.dims3()?;

        let proj = self.in_proj.forward(x)?;

        let z_dim = self.d_inner;
        let xbc_dim = self.conv_channels;

        let z = proj.narrow(D::Minus1, 0, z_dim)?;
        let xbc = proj.narrow(D::Minus1, z_dim, xbc_dim)?;
        let dt_raw = proj.narrow(D::Minus1, z_dim + xbc_dim, self.nheads)?;

        let xbc = silu(&self.causal_conv1d(&xbc)?)?;

        let x_ssm = xbc.narrow(D::Minus1, 0, self.d_inner)?;
        let b_ssm = xbc.narrow(D::Minus1, self.d_inner, self.ngroups * self.d_state)?;
        let c_ssm = xbc.narrow(D::Minus1, self.d_inner + self.ngroups * self.d_state, self.ngroups * self.d_state)?;

        let x_ssm = x_ssm.reshape((1, seq_len, self.nheads, self.headdim))?;
        let b_ssm = b_ssm.reshape((1, seq_len, self.ngroups, self.d_state))?;
        let c_ssm = c_ssm.reshape((1, seq_len, self.ngroups, self.d_state))?;

        let dt = {
            let raw_data: Vec<f32> = dt_raw.flatten_all()?.to_vec1()?;
            let mut dt_data = vec![0.0f32; seq_len * self.nheads];
            for t in 0..seq_len {
                for h in 0..self.nheads {
                    let idx = t * self.nheads + h;
                    dt_data[idx] = softplus_scalar(raw_data[idx] + self.dt_bias_data[h]);
                }
            }
            Tensor::from_vec(dt_data, (1, seq_len, self.nheads), x.device())?
        };

        let y = self.sequential_scan(&x_ssm, &dt, &b_ssm, &c_ssm)?;
        let y = y.reshape((1, seq_len, self.d_inner))?;
        let y_norm = rmsnorm_with_weight(&y, self.norm_weight.as_ref().expect("Tensor dropped"), 1e-5)?;
        let z_gate = silu(&z.reshape((1, seq_len, self.d_inner))?)?;
        let y_gated = (y_norm * z_gate)?;

        self.out_proj.forward(&y_gated)
    }

    /// Incremental 1-token step (ssm_y: 재활용 버퍼)
    pub fn step(&self, x: &[f32], state: &mut Mamba2State, proj_buf: &mut Vec<f32>, xbc_buf: &mut Vec<f32>, ssm_y: &mut [f32], x_u8: &mut [u8]) {
        // 1. in_proj
        proj_buf.resize(self.proj_dim, 0.0);
        self.in_proj.forward_vec(x, proj_buf, x_u8);

        let z_dim = self.d_inner;
        let xbc_dim = self.conv_channels;

        // 2. Conv1d step (먼저 conv → 버퍼 업데이트)
        let buf_max = self.d_conv - 1;
        xbc_buf.resize(self.conv_channels, 0.0);

        // xbc_raw 범위: proj_buf[z_dim..z_dim+xbc_dim]
        let xbc_raw_start = z_dim;
        for c in 0..self.conv_channels {
            let mut sum = self.conv_bias_data[c];
            for k in 0..self.d_conv {
                let w = self.conv_weight_data[c * self.d_conv + k];
                let time_back = self.d_conv - 1 - k;
                let val = if time_back == 0 {
                    proj_buf[xbc_raw_start + c]
                } else if state.conv_buf_len >= time_back {
                    let buf_row = state.conv_buf_len - time_back;
                    state.conv_buf[buf_row * self.conv_channels + c]
                } else {
                    0.0
                };
                sum += w * val;
            }
            xbc_buf[c] = silu_scalar(sum);
        }

        // 버퍼 업데이트
        if state.conv_buf_len >= buf_max {
            let cc = self.conv_channels;
            state.conv_buf.copy_within(cc.., 0);
            let last = (buf_max - 1) * cc;
            state.conv_buf[last..last + cc].copy_from_slice(&proj_buf[xbc_raw_start..xbc_raw_start + cc]);
        } else {
            let off = state.conv_buf_len * self.conv_channels;
            let cc = self.conv_channels;
            state.conv_buf[off..off + cc].copy_from_slice(&proj_buf[xbc_raw_start..xbc_raw_start + cc]);
            state.conv_buf_len += 1;
        }

        // 3. SSM step
        let ng_ds = self.ngroups * self.d_state;
        let dt_start = z_dim + xbc_dim;

        // y 버퍼 초기화 (외부에서 받은 재활용 버퍼)
        for v in ssm_y[..self.d_inner].iter_mut() { *v = 0.0; }

        for h in 0..self.nheads {
            let g = h * self.ngroups / self.nheads;
            let dt_val = softplus_scalar(proj_buf[dt_start + h] + self.dt_bias_data[h]);
            let da = (self.a_data[h] * dt_val).exp();
            let h_hd = h * self.headdim;
            let h_ds_hd = h * self.d_state * self.headdim;
            let g_ds = g * self.d_state;

            // Phase 1: state 업데이트 — state[h,s,:] = da * state[h,s,:] + (dt*b[s]) * x[:]
            for s in 0..self.d_state {
                let db = dt_val * xbc_buf[self.d_inner + g_ds + s];
                let s_off = h_ds_hd + s * self.headdim;
                for d in 0..self.headdim {
                    state.ssm_state[s_off + d] = da * state.ssm_state[s_off + d] + db * xbc_buf[h_hd + d];
                }
            }

            // Phase 2: output — y[h,:] += state[h,:,:]^T @ c via sgemv
            // state[h] is (d_state, headdim) row-major, c is (d_state,)
            // y[d] = sum_s c[s] * state[h,s,d] = state^T @ c
            let c_off = self.d_inner + ng_ds + g_ds;
            unsafe {
                cblas_sgemv(
                    CBLAS_ROW_MAJOR,
                    112, // CblasTrans
                    self.d_state as c_int,
                    self.headdim as c_int,
                    1.0,
                    state.ssm_state[h_ds_hd..].as_ptr(),
                    self.headdim as c_int,
                    xbc_buf[c_off..].as_ptr(),
                    1,
                    1.0, // beta=1.0: 누적
                    ssm_y[h_hd..].as_mut_ptr(),
                    1,
                );
            }

            // D skip connection
            let d_h = self.d_data[h];
            for d in 0..self.headdim {
                ssm_y[h_hd + d] += d_h * xbc_buf[h_hd + d];
            }
        }

        // 4. RMSNorm + silu(z) gate → out_proj
        // y_norm → reuse xbc_buf (충분한 크기)
        xbc_buf.resize(self.d_inner, 0.0);
        let mut sq_sum = 0.0f32;
        for i in 0..self.d_inner { sq_sum += ssm_y[i] * ssm_y[i]; }
        let rms_inv = (sq_sum / self.d_inner as f32 + 1e-5).sqrt().recip();
        for i in 0..self.d_inner {
            xbc_buf[i] = ssm_y[i] * rms_inv * self.norm_weight_data[i] * silu_scalar(proj_buf[i]); // proj_buf[0..z_dim] = z
        }

        // 5. out_proj → proj_buf 재활용 (d_model 크기로 resize됨)
        proj_buf.resize(self.out_proj.out_dim, 0.0);
        self.out_proj.forward_vec(xbc_buf, proj_buf, x_u8);
    }

    fn drop_tensors(&mut self) {
        self.in_proj.drop_tensor();
        self.out_proj.drop_tensor();
        self.conv1d_weight = None;
        self.conv1d_bias = None;
        self.norm_weight = None;
    }
}

fn rmsnorm_with_weight(x: &Tensor, weight: &Tensor, eps: f64) -> Result<Tensor> {
    let var = x.sqr()?.mean_keepdim(D::Minus1)?;
    let rms = (var + eps)?.sqrt()?.recip()?;
    Ok(x.broadcast_mul(&rms)?.broadcast_mul(weight)?)
}

// ── Cross-Attention KV 캐시 (순수 scalar) ───────────

pub struct CrossAttnCache {
    /// K^T @ V: (n_heads, d_head, d_head) flat
    pub kv_data: Vec<f32>,
    /// sum(K, dim=seq): (n_heads, d_head) flat
    pub z_data: Vec<f32>,
    pub n_heads: usize,
    pub d_head: usize,
}

// ── LinearCrossAttention ─────────────────────────────

pub struct LinearCrossAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    n_heads: usize,
    d_head: usize,
    eps: f32,
}

impl LinearCrossAttention {
    pub fn load(
        tensors: &HashMap<String, Tensor>,
        prefix: &str,
        n_heads: usize,
        d_head: usize,
    ) -> Result<Self> {
        Ok(Self {
            q_proj: Linear::load(tensors, &format!("{}.q_proj", prefix))?,
            k_proj: Linear::load(tensors, &format!("{}.k_proj", prefix))?,
            v_proj: Linear::load(tensors, &format!("{}.v_proj", prefix))?,
            o_proj: Linear::load(tensors, &format!("{}.o_proj", prefix))?,
            n_heads,
            d_head,
            eps: 1e-5,
        })
    }

    pub fn load_bmmq(
        tensors: &mut HashMap<String, TensorData>,
        prefix: &str,
        n_heads: usize,
        d_head: usize,
    ) -> Result<Self> {
        Ok(Self {
            q_proj: Linear::load_bmmq(tensors, &format!("{}.q_proj", prefix))?,
            k_proj: Linear::load_bmmq(tensors, &format!("{}.k_proj", prefix))?,
            v_proj: Linear::load_bmmq(tensors, &format!("{}.v_proj", prefix))?,
            o_proj: Linear::load_bmmq(tensors, &format!("{}.o_proj", prefix))?,
            n_heads,
            d_head,
            eps: 1e-5,
        })
    }

    /// Encoder KV 캐시 생성 (순수 scalar)
    pub fn cache_encoder(&self, encoder_out: &Tensor) -> Result<CrossAttnCache> {
        let (_, src_len, _) = encoder_out.dims3()?;

        let k = self.k_proj.forward(encoder_out)?;
        let v = self.v_proj.forward(encoder_out)?;

        // (1, src_len, n_heads*d_head) → reshape+transpose → (n_heads, src_len, d_head)
        let k = k.reshape((1, src_len, self.n_heads, self.d_head))?.transpose(1, 2)?;
        let v = v.reshape((1, src_len, self.n_heads, self.d_head))?.transpose(1, 2)?;

        // gelu1p feature map
        let k = gelu1p(&k)?;

        // K^T @ V: (1, nh, d_head, d_head)
        let kt = k.transpose(2, 3)?;
        let kv = kt.matmul(&v)?;
        let z = k.sum(2)?; // (1, nh, d_head)

        let kv_data: Vec<f32> = kv.flatten_all()?.to_vec1()?;
        let z_data: Vec<f32> = z.flatten_all()?.to_vec1()?;

        Ok(CrossAttnCache { kv_data, z_data, n_heads: self.n_heads, d_head: self.d_head })
    }

    /// 캐시된 KV로 1-token forward (할당 없음 — 외부 버퍼 재활용)
    fn forward_cached_vec(&self, x: &[f32], cache: &CrossAttnCache, q_buf: &mut Vec<f32>,
                          attn_out: &mut [f32], q_h: &mut [f32], out: &mut [f32], x_u8: &mut [u8]) {
        let nh = self.n_heads;
        let dh = self.d_head;
        let d_model = nh * dh;

        // Q projection
        q_buf.resize(d_model, 0.0);
        self.q_proj.forward_vec(x, q_buf, x_u8);

        for h in 0..nh {
            // gelu1p on Q[h]
            let q_off = h * dh;
            for d in 0..dh {
                q_h[d] = gelu1p_scalar(q_buf[q_off + d]);
            }

            // Q @ KV[h]: (1, dh) @ (dh, dh) → (1, dh) via sgemv
            // KV[h]는 (dh, dh) row-major: q_h^T @ KV = (KV^T @ q_h)^T
            // 하지만 KV가 row-major이고 q는 행벡터이므로 그냥 transpose해서 계산
            // y = KV^T @ q_h 도 sgemv(Trans)로 가능하지만, 여기선 KV를 row-major로
            // 행렬이 row-major (dh, dh)이고 q_h가 (dh,)일 때:
            // 결과[d_out] = sum_d_in q_h[d_in] * KV[d_in, d_out]
            // 이건 y = KV^T @ q_h 와 같음 → CblasTrans 사용
            let kv_off = h * dh * dh;
            let attn_off = h * dh;
            unsafe {
                cblas_sgemv(
                    CBLAS_ROW_MAJOR,
                    112, // CblasTrans
                    dh as c_int,
                    dh as c_int,
                    1.0,
                    cache.kv_data[kv_off..].as_ptr(),
                    dh as c_int,
                    q_h.as_ptr(),
                    1,
                    0.0,
                    attn_out[attn_off..].as_mut_ptr(),
                    1,
                );
            }

            // Q @ z[h]: (1, dh) . (dh) → scalar
            let z_off = h * dh;
            let mut den = 0.0f32;
            for d in 0..dh {
                den += q_h[d] * cache.z_data[z_off + d];
            }
            den = (den + self.eps).recip();

            // normalize
            for d_out in 0..dh {
                attn_out[attn_off + d_out] *= den;
            }
        }

        // O projection
        self.o_proj.forward_vec(&attn_out, out, x_u8);
    }

    fn drop_tensors(&mut self) {
        self.q_proj.drop_tensor();
        self.k_proj.drop_tensor();
        self.v_proj.drop_tensor();
        self.o_proj.drop_tensor();
    }
}

// ── EncoderLayer ─────────────────────────────────────

pub struct EncoderLayer {
    mamba: Mamba2Block,
    norm1: RMSNorm,
    ffn: BitNetFFN,
    norm2: RMSNorm,
}

impl EncoderLayer {
    pub fn load(
        tensors: &HashMap<String, Tensor>,
        prefix: &str,
        cfg: &ModelConfig,
    ) -> Result<Self> {
        Ok(Self {
            mamba: Mamba2Block::load(tensors, &format!("{}.mamba", prefix), cfg)?,
            norm1: RMSNorm::load(tensors, &format!("{}.norm1", prefix), cfg.rms_norm_eps)?,
            ffn: BitNetFFN::load(tensors, &format!("{}.ffn", prefix))?,
            norm2: RMSNorm::load(tensors, &format!("{}.norm2", prefix), cfg.rms_norm_eps)?,
        })
    }

    pub fn load_bmmq(
        tensors: &mut HashMap<String, TensorData>,
        prefix: &str,
        cfg: &ModelConfig,
    ) -> Result<Self> {
        Ok(Self {
            mamba: Mamba2Block::load_bmmq(tensors, &format!("{}.mamba", prefix), cfg)?,
            norm1: RMSNorm::load_bmmq(tensors, &format!("{}.norm1", prefix), cfg.rms_norm_eps)?,
            ffn: BitNetFFN::load_bmmq(tensors, &format!("{}.ffn", prefix))?,
            norm2: RMSNorm::load_bmmq(tensors, &format!("{}.norm2", prefix), cfg.rms_norm_eps)?,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let residual = x.clone();
        let x = self.mamba.forward(x)?;
        let x = self.norm1.forward(&(residual + &x)?)?;

        let residual = x.clone();
        let x = self.ffn.forward(&x)?;
        self.norm2.forward(&(residual + &x)?)
    }

    fn drop_tensors(&mut self) {
        self.mamba.drop_tensors();
        self.norm1.drop_tensor();
        self.ffn.drop_tensors();
        self.norm2.drop_tensor();
    }
}

// ── Decoder State ────────────────────────────────────

pub struct DecoderState {
    pub mamba_states: Vec<Mamba2State>,
    pub cross_attn_caches: Vec<CrossAttnCache>,
}

// ── 디코더 작업 버퍼 (재할당 방지) ──────────────────

struct DecoderBufs {
    proj: Vec<f32>,
    xbc: Vec<f32>,
    q: Vec<f32>,
    normed: Vec<f32>,
    residual: Vec<f32>,
    out: Vec<f32>,
    ff1: Vec<f32>,
    ff2: Vec<f32>,
    // SSM + cross-attn 버퍼 (per-step 할당 제거)
    ssm_y: Vec<f32>,       // d_inner
    attn_out: Vec<f32>,    // d_model
    q_h: Vec<f32>,         // d_head (64)
    x_norm_buf: Vec<f32>,  // max(in_dim) for BitLinear layernorm
    x_u8: Vec<u8>,         // int8 양자화된 활성화 버퍼 (AVX-VNNI용)
    // per-token 할당 제거용 사전할당 버퍼
    logits: Vec<f32>,      // vocab_size
    embed: Vec<f32>,       // d_model (embed_one 결과)
}

impl DecoderBufs {
    fn new(d_model: usize, proj_dim: usize, conv_ch: usize, d_ff: usize,
           d_inner: usize, d_head: usize, max_in_dim: usize, vocab_size: usize) -> Self {
        Self {
            proj: vec![0.0; proj_dim],
            xbc: vec![0.0; conv_ch],
            q: vec![0.0; d_model],
            normed: vec![0.0; d_model],
            residual: vec![0.0; d_model],
            out: vec![0.0; d_model],
            ff1: vec![0.0; d_ff],
            ff2: vec![0.0; d_ff],
            ssm_y: vec![0.0; d_inner],
            attn_out: vec![0.0; d_model],
            q_h: vec![0.0; d_head],
            x_norm_buf: vec![0.0; max_in_dim],
            x_u8: vec![0u8; max_in_dim],
            logits: vec![0.0; vocab_size],
            embed: vec![0.0; d_model],
        }
    }
}

// ── DecoderLayer ─────────────────────────────────────

pub struct DecoderLayer {
    mamba: Mamba2Block,
    norm1: RMSNorm,
    cross_attn: LinearCrossAttention,
    norm_cross: RMSNorm,
    ffn: BitNetFFN,
    norm2: RMSNorm,
    d_model: usize,
}

impl DecoderLayer {
    pub fn load(
        tensors: &HashMap<String, Tensor>,
        prefix: &str,
        cfg: &ModelConfig,
    ) -> Result<Self> {
        let n_heads = cfg.d_model / 64;
        let d_head = 64;
        Ok(Self {
            mamba: Mamba2Block::load(tensors, &format!("{}.mamba", prefix), cfg)?,
            norm1: RMSNorm::load(tensors, &format!("{}.norm1", prefix), cfg.rms_norm_eps)?,
            cross_attn: LinearCrossAttention::load(
                tensors,
                &format!("{}.cross_attn", prefix),
                n_heads,
                d_head,
            )?,
            norm_cross: RMSNorm::load(tensors, &format!("{}.norm_cross", prefix), cfg.rms_norm_eps)?,
            ffn: BitNetFFN::load(tensors, &format!("{}.ffn", prefix))?,
            norm2: RMSNorm::load(tensors, &format!("{}.norm2", prefix), cfg.rms_norm_eps)?,
            d_model: cfg.d_model,
        })
    }

    pub fn load_bmmq(
        tensors: &mut HashMap<String, TensorData>,
        prefix: &str,
        cfg: &ModelConfig,
    ) -> Result<Self> {
        let n_heads = cfg.d_model / 64;
        let d_head = 64;
        Ok(Self {
            mamba: Mamba2Block::load_bmmq(tensors, &format!("{}.mamba", prefix), cfg)?,
            norm1: RMSNorm::load_bmmq(tensors, &format!("{}.norm1", prefix), cfg.rms_norm_eps)?,
            cross_attn: LinearCrossAttention::load_bmmq(
                tensors,
                &format!("{}.cross_attn", prefix),
                n_heads,
                d_head,
            )?,
            norm_cross: RMSNorm::load_bmmq(tensors, &format!("{}.norm_cross", prefix), cfg.rms_norm_eps)?,
            ffn: BitNetFFN::load_bmmq(tensors, &format!("{}.ffn", prefix))?,
            norm2: RMSNorm::load_bmmq(tensors, &format!("{}.norm2", prefix), cfg.rms_norm_eps)?,
            d_model: cfg.d_model,
        })
    }

    /// Incremental 1-token step (버퍼 재활용)
    fn step(
        &self,
        x: &[f32],
        mamba_state: &mut Mamba2State,
        cross_cache: &CrossAttnCache,
        bufs: &mut DecoderBufs,
    ) {
        let d = self.d_model;

        // 1. Mamba step → proj_buf에 결과 저장
        self.mamba.step(x, mamba_state, &mut bufs.proj, &mut bufs.xbc, &mut bufs.ssm_y, &mut bufs.x_u8);
        // mamba 결과가 bufs.proj에 있음 (out_proj 출력)
        for i in 0..d { bufs.residual[i] = x[i] + bufs.proj[i]; }
        self.norm1.forward_vec(&bufs.residual, &mut bufs.normed);

        // 2. Cross-attention (cached scalar)
        self.cross_attn.forward_cached_vec(&bufs.normed, cross_cache, &mut bufs.q,
                                           &mut bufs.attn_out, &mut bufs.q_h, &mut bufs.out, &mut bufs.x_u8);
        for i in 0..d { bufs.residual[i] = bufs.normed[i] + bufs.out[i]; }
        self.norm_cross.forward_vec(&bufs.residual, &mut bufs.normed);

        // 3. FFN
        self.ffn.forward_vec(&bufs.normed, &mut bufs.ff1, &mut bufs.ff2, &mut bufs.x_norm_buf, &mut bufs.x_u8, &mut bufs.out);
        for i in 0..d { bufs.residual[i] = bufs.normed[i] + bufs.out[i]; }
        self.norm2.forward_vec(&bufs.residual, &mut bufs.normed);
        // normed가 이 레이어의 최종 출력
    }

    fn drop_tensors(&mut self) {
        self.mamba.drop_tensors();
        self.norm1.drop_tensor();
        self.cross_attn.drop_tensors();
        self.norm_cross.drop_tensor();
        self.ffn.drop_tensors();
        self.norm2.drop_tensor();
    }
}

// ── BitMambaSeq2Seq ──────────────────────────────────

pub struct BitMambaSeq2Seq {
    encoder_embedding: Option<Tensor>,  // 인코딩 후 해제
    embed_data: Vec<f32>,
    encoder_layers: Vec<EncoderLayer>,
    decoder_layers: Vec<DecoderLayer>,
    final_norm: RMSNorm,
    // lm_head도 i8 양자화
    lm_head_i8: Vec<i8>,
    lm_head_row_scales: Vec<f32>,
    lm_head_row_sums: Vec<i32>,
    copy_gate: Option<LinearWithBias>,
    pub cfg: ModelConfig,
}

impl BitMambaSeq2Seq {
    pub fn load(model_path: &str, cfg: &ModelConfig) -> Result<Self> {
        let device = Device::Cpu;
        eprintln!("모델 로드 중: {}", model_path);

        let tensors = candle_core::safetensors::load(model_path, &device)
            .context("safetensors 로드 실패")?;

        eprintln!("  텐서 수: {}", tensors.len());

        let encoder_embedding = tensors.get("encoder_embedding.weight")
            .context("encoder_embedding 없음")?.clone();
        let embed_data: Vec<f32> = encoder_embedding.flatten_all()?.to_vec1()?;

        let lm_head_weight = if cfg.tie_lm_head {
            encoder_embedding.clone()
        } else {
            tensors.get("lm_head.weight")
                .context("lm_head weight 없음")?.clone()
        };
        // lm_head i8 양자화 (vocab_size × d_model)
        let lm_head_f32: Vec<f32> = lm_head_weight.flatten_all()?.to_vec1()?;
        let lm_vocab = cfg.vocab_size;
        let lm_dim = cfg.d_model;
        let mut lm_head_i8 = vec![0i8; lm_vocab * lm_dim];
        let mut lm_head_row_scales = vec![0.0f32; lm_vocab];
        let mut lm_head_row_sums = vec![0i32; lm_vocab];
        for row in 0..lm_vocab {
            let base = row * lm_dim;
            let mut max_abs = 0.0f32;
            for col in 0..lm_dim { max_abs = max_abs.max(lm_head_f32[base + col].abs()); }
            if max_abs < 1e-10 { max_abs = 1e-10; }
            lm_head_row_scales[row] = max_abs / 127.0;
            let inv_scale = 127.0 / max_abs;
            let mut rsum = 0i32;
            for col in 0..lm_dim {
                let v = (lm_head_f32[base + col] * inv_scale).round().clamp(-128.0, 127.0) as i8;
                lm_head_i8[base + col] = v;
                rsum += v as i32;
            }
            lm_head_row_sums[row] = rsum;
        }

        let mut encoder_layers = Vec::new();
        for i in 0..cfg.n_encoder_layers {
            let prefix = format!("encoder.layers.{}", i);
            encoder_layers.push(EncoderLayer::load(&tensors, &prefix, cfg)?);
            eprintln!("  인코더 레이어 {} 로드", i);
        }

        let mut decoder_layers = Vec::new();
        for i in 0..cfg.n_decoder_layers {
            let prefix = format!("decoder.layers.{}", i);
            decoder_layers.push(DecoderLayer::load(&tensors, &prefix, cfg)?);
            eprintln!("  디코더 레이어 {} 로드", i);
        }

        let final_norm = RMSNorm::load(&tensors, "final_norm", cfg.rms_norm_eps)?;

        let copy_gate = if cfg.use_copy_gate {
            Some(LinearWithBias::load(&tensors, "copy_gate")?)
        } else {
            None
        };

        // i8 BitLinear 메모리 절약량 계산
        let bitlinear_count = (cfg.n_encoder_layers + cfg.n_decoder_layers) * 3; // gate+up+down per layer
        eprintln!("모델 로드 완료 (BitLinear i8 ternary ×{}, AVX-VNNI int8 커널)", bitlinear_count);

        Ok(Self {
            encoder_embedding: Some(encoder_embedding),
            embed_data,
            encoder_layers,
            decoder_layers,
            final_norm,
            lm_head_i8,
            lm_head_row_scales,
            lm_head_row_sums,
            copy_gate,
            cfg: cfg.clone(),
        })
    }

    pub fn load_bmmq(model_path: &str, cfg: &ModelConfig) -> Result<Self> {
        eprintln!("BMMQ 모델 로드 중: {}", model_path);

        let mut tensors = bmmq::load_bmmq(model_path)
            .context("BMMQ 파일 로드 실패")?;

        eprintln!("  텐서 수: {}", tensors.len());

        // 임베딩 (f32)
        let embed_data = bmmq_take_f32(&mut tensors, "encoder_embedding.weight")?;

        // lm_head: tie_lm_head이면 encoder_embedding과 동일
        let lm_head_f32 = if cfg.tie_lm_head {
            embed_data.clone()
        } else {
            bmmq_take_f32(&mut tensors, "lm_head.weight")?
        };
        // lm_head i8 양자화
        let lm_vocab = cfg.vocab_size;
        let lm_dim = cfg.d_model;
        let mut lm_head_i8 = vec![0i8; lm_vocab * lm_dim];
        let mut lm_head_row_scales = vec![0.0f32; lm_vocab];
        let mut lm_head_row_sums = vec![0i32; lm_vocab];
        for row in 0..lm_vocab {
            let base = row * lm_dim;
            let mut max_abs = 0.0f32;
            for col in 0..lm_dim { max_abs = max_abs.max(lm_head_f32[base + col].abs()); }
            if max_abs < 1e-10 { max_abs = 1e-10; }
            lm_head_row_scales[row] = max_abs / 127.0;
            let inv_scale = 127.0 / max_abs;
            let mut rsum = 0i32;
            for col in 0..lm_dim {
                let v = (lm_head_f32[base + col] * inv_scale).round().clamp(-128.0, 127.0) as i8;
                lm_head_i8[base + col] = v;
                rsum += v as i32;
            }
            lm_head_row_sums[row] = rsum;
        }

        let mut encoder_layers = Vec::new();
        for i in 0..cfg.n_encoder_layers {
            let prefix = format!("encoder.layers.{}", i);
            encoder_layers.push(EncoderLayer::load_bmmq(&mut tensors, &prefix, cfg)?);
            eprintln!("  인코더 레이어 {} 로드", i);
        }

        let mut decoder_layers = Vec::new();
        for i in 0..cfg.n_decoder_layers {
            let prefix = format!("decoder.layers.{}", i);
            decoder_layers.push(DecoderLayer::load_bmmq(&mut tensors, &prefix, cfg)?);
            eprintln!("  디코더 레이어 {} 로드", i);
        }

        let final_norm = RMSNorm::load_bmmq(&mut tensors, "final_norm", cfg.rms_norm_eps)?;

        let copy_gate = if cfg.use_copy_gate {
            Some(LinearWithBias::load_bmmq(&mut tensors, "copy_gate")?)
        } else {
            None
        };

        let bitlinear_count = (cfg.n_encoder_layers + cfg.n_decoder_layers) * 3;
        eprintln!("BMMQ 모델 로드 완료 (BitLinear 2-bit ×{}, AVX-VNNI int8 커널)", bitlinear_count);

        Ok(Self {
            encoder_embedding: None,  // Vec path 사용 — Tensor 불필요
            embed_data,
            encoder_layers,
            decoder_layers,
            final_norm,
            lm_head_i8,
            lm_head_row_scales,
            lm_head_row_sums,
            copy_gate,
            cfg: cfg.clone(),
        })
    }

    /// 인코더 Vec path: 토큰별 순차 처리 (Tensor 불필요)
    pub fn encode_vec(&self, src_ids: &[u32]) -> Vec<f32> {
        let d = self.cfg.d_model;
        let seq_len = src_ids.len();
        let scale = self.cfg.embed_scale();

        // 인코더 출력: (seq_len, d_model) flat
        let mut x = vec![0.0f32; seq_len * d];
        for (t, &id) in src_ids.iter().enumerate() {
            let offset = id as usize * d;
            for i in 0..d {
                x[t * d + i] = self.embed_data[offset + i] * scale;
            }
        }

        // 인코더 버퍼
        let proj_dim = self.encoder_layers[0].mamba.proj_dim;
        let conv_ch = self.encoder_layers[0].mamba.conv_channels;
        let d_ff = self.encoder_layers[0].ffn.d_ff;
        let d_inner = self.cfg.d_inner;
        let max_in_dim = d.max(d_inner).max(self.cfg.d_ff);
        let mut proj_buf = vec![0.0f32; proj_dim];
        let mut xbc_buf = vec![0.0f32; conv_ch];
        let mut ssm_y = vec![0.0f32; d_inner];
        let mut x_u8 = vec![0u8; max_in_dim];
        let mut x_norm_buf = vec![0.0f32; max_in_dim];
        let mut ff1 = vec![0.0f32; d_ff];
        let mut ff2 = vec![0.0f32; d_ff];
        let mut residual = vec![0.0f32; d];
        let mut normed = vec![0.0f32; d];
        let mut ffn_out = vec![0.0f32; d];
        let mut x_next = vec![0.0f32; seq_len * d];

        for layer in &self.encoder_layers {
            let mut mamba_state = layer.mamba.new_state();

            for t in 0..seq_len {
                let x_t = &x[t * d..(t + 1) * d];

                // Mamba step
                layer.mamba.step(x_t, &mut mamba_state, &mut proj_buf, &mut xbc_buf, &mut ssm_y, &mut x_u8);
                // proj_buf는 mamba 결과 (d_model)
                for i in 0..d { residual[i] = x_t[i] + proj_buf[i]; }
                layer.norm1.forward_vec(&residual, &mut normed);

                // FFN
                layer.ffn.forward_vec(&normed, &mut ff1, &mut ff2, &mut x_norm_buf, &mut x_u8, &mut ffn_out);
                for i in 0..d { residual[i] = normed[i] + ffn_out[i]; }
                layer.norm2.forward_vec(&residual, &mut normed);

                // 결과를 x_next에 저장
                x_next[t * d..(t + 1) * d].copy_from_slice(&normed[..d]);
            }

            // 레이어 출력을 다음 레이어 입력으로
            std::mem::swap(&mut x, &mut x_next);
        }

        x
    }

    /// Vec path 인코더 출력으로 디코더 state 초기화
    pub fn init_decoder_state_vec(&self, enc_out: &[f32], seq_len: usize) -> DecoderState {
        let d = self.cfg.d_model;
        let max_in_dim = d.max(self.cfg.d_inner).max(self.cfg.d_ff);
        let mut x_u8 = vec![0u8; max_in_dim];

        let mut mamba_states = Vec::new();
        let mut cross_attn_caches = Vec::new();

        for layer in &self.decoder_layers {
            mamba_states.push(layer.mamba.new_state());

            let ca = &layer.cross_attn;
            let nh = ca.n_heads;
            let dh = ca.d_head;

            // KV 캐시: 토큰별 k_proj, v_proj → 누적
            let mut kv_data = vec![0.0f32; nh * dh * dh];
            let mut z_data = vec![0.0f32; nh * dh];
            let mut k_buf = vec![0.0f32; d];
            let mut v_buf = vec![0.0f32; d];

            for t in 0..seq_len {
                let x_t = &enc_out[t * d..(t + 1) * d];

                // K, V projection
                ca.k_proj.forward_vec(x_t, &mut k_buf, &mut x_u8);
                ca.v_proj.forward_vec(x_t, &mut v_buf, &mut x_u8);

                // gelu1p on K
                for i in 0..d { k_buf[i] = gelu1p_scalar(k_buf[i]); }

                // 헤드별 outer product 누적: KV[h] += k_h^T @ v_h
                for h in 0..nh {
                    let k_off = h * dh;
                    let v_off = h * dh;
                    let kv_off = h * dh * dh;
                    let z_off = h * dh;

                    for ki in 0..dh {
                        let k_val = k_buf[k_off + ki];
                        z_data[z_off + ki] += k_val;
                        for vi in 0..dh {
                            kv_data[kv_off + ki * dh + vi] += k_val * v_buf[v_off + vi];
                        }
                    }
                }
            }

            cross_attn_caches.push(CrossAttnCache {
                kv_data, z_data, n_heads: nh, d_head: dh,
            });
        }

        DecoderState { mamba_states, cross_attn_caches }
    }

    fn embed(&self, ids: &Tensor) -> Result<Tensor> {
        let enc_emb = self.encoder_embedding.as_ref().expect("encoder_embedding already dropped");
        let (batch, seq_len) = ids.dims2()?;
        let flat_ids = ids.flatten_all()?;
        let emb = enc_emb.index_select(&flat_ids, 0)?;
        let emb = emb.reshape((batch, seq_len, self.cfg.d_model))?;
        let scale = Tensor::new(self.cfg.embed_scale(), ids.device())?;
        Ok(emb.broadcast_mul(&scale)?)
    }

    #[inline]
    fn embed_one_into(&self, id: u32, out: &mut [f32]) {
        let d = self.cfg.d_model;
        let scale = self.cfg.embed_scale();
        let offset = id as usize * d;
        for i in 0..d {
            out[i] = self.embed_data[offset + i] * scale;
        }
    }

    pub fn encode(&self, src_ids: &[u32]) -> Result<Tensor> {
        let dev = Device::Cpu;
        let src = Tensor::new(src_ids, &dev)?.unsqueeze(0)?;
        let mut x = self.embed(&src)?;
        for layer in &self.encoder_layers {
            x = layer.forward(&x)?;
        }
        Ok(x)
    }

    pub fn init_decoder_state(&self, encoder_out: &Tensor) -> Result<DecoderState> {
        let mut mamba_states = Vec::new();
        let mut cross_attn_caches = Vec::new();

        for layer in &self.decoder_layers {
            mamba_states.push(layer.mamba.new_state());
            cross_attn_caches.push(layer.cross_attn.cache_encoder(encoder_out)?);
        }

        Ok(DecoderState { mamba_states, cross_attn_caches })
    }

    /// 인코딩+KV캐시 후 불필요한 데이터 해제 — 메모리 대폭 절감
    pub fn drop_tensors(&mut self) {
        self.encoder_embedding = None;
        // 인코더 레이어 전체 해제 (인코딩 완료 후 불필요)
        self.encoder_layers.clear();
        self.encoder_layers.shrink_to_fit();
        for layer in &mut self.decoder_layers {
            layer.drop_tensors();
        }
    }

    /// Incremental 1-token decode step → logits (bufs.logits에 결과 저장, 할당 없음)
    fn decode_step(
        &self,
        token_id: u32,
        state: &mut DecoderState,
        copy_logits: &[f32],
        bufs: &mut DecoderBufs,
    ) {
        let d = self.cfg.d_model;

        // embed를 bufs에서 분리하여 borrow 충돌 방지
        let mut x = std::mem::take(&mut bufs.embed);
        self.embed_one_into(token_id, &mut x);

        for (i, layer) in self.decoder_layers.iter().enumerate() {
            layer.step(
                &x,
                &mut state.mamba_states[i],
                &state.cross_attn_caches[i],
                bufs,
            );
            x[..d].copy_from_slice(&bufs.normed[..d]);
        }

        // final_norm
        self.final_norm.forward_vec(&x, &mut bufs.normed);
        bufs.embed = x;  // 반환

        // LM Head via AVX-VNNI i8
        let x_scale = unsafe {
            quantize_f32_to_u8(bufs.normed.as_ptr(), bufs.x_u8.as_mut_ptr(), d as c_int)
        };
        unsafe {
            i8_sgemv(
                self.lm_head_i8.as_ptr(),
                bufs.x_u8.as_ptr(),
                bufs.logits.as_mut_ptr(),
                self.cfg.vocab_size as c_int,
                d as c_int,
                self.lm_head_row_sums.as_ptr(),
                self.lm_head_row_scales.as_ptr(),
                x_scale,
                0.0,
            );
        }

        // Copy Gate (사전계산된 copy_logits 사용)
        if self.copy_gate.is_some() {
            let gate_proj = self.copy_gate.as_ref().unwrap();
            gate_proj.forward_vec(&bufs.normed, &mut bufs.out, &mut bufs.x_u8);
            let raw_sig = 1.0 / (1.0 + (-bufs.out[0]).exp());
            let gate = 0.5 + 0.5 * raw_sig;
            let one_minus_gate = 1.0 - gate;

            for v in 0..self.cfg.vocab_size {
                bufs.logits[v] = gate * bufs.logits[v] + one_minus_gate * copy_logits[v];
            }
        }
    }

    /// Auto-regressive 생성 (state를 외부에서 받음 — 타이밍 분리용)
    pub fn generate_with_state(
        &self,
        src_ids: &[u32],
        max_len: usize,
        bos_id: u32,
        eos_id: u32,
        pad_id: u32,
        dec_state: &mut DecoderState,
    ) -> Result<Vec<u32>> {
        let d_model = self.cfg.d_model;
        let vocab_size = self.cfg.vocab_size;
        let proj_dim = self.decoder_layers[0].mamba.proj_dim;
        let conv_ch = self.decoder_layers[0].mamba.conv_channels;
        let d_ff = self.decoder_layers[0].ffn.d_ff;

        let d_inner = self.cfg.d_inner;
        let d_head = 64usize;
        let max_in_dim = d_model.max(d_inner).max(self.cfg.d_ff);
        let mut bufs = DecoderBufs::new(d_model, proj_dim, conv_ch, d_ff,
                                         d_inner, d_head, max_in_dim, vocab_size);

        // Copy gate: 소스 분포 사전계산 (전체 디코딩에 걸쳐 불변)
        let copy_logits = if self.copy_gate.is_some() {
            let mut p_copy = vec![0.0f32; vocab_size];
            let mut count = 0.0f32;
            for &id in src_ids {
                if id != self.cfg.pad_id {
                    p_copy[id as usize] += 1.0;
                    count += 1.0;
                }
            }
            if count > 0.0 {
                let inv_count = 1.0 / count;
                for v in p_copy.iter_mut() { *v *= inv_count; }
            }
            // log 변환 (매 토큰마다 반복하지 않음)
            for v in p_copy.iter_mut() {
                *v = (*v).max(1e-5).ln();
            }
            p_copy
        } else {
            Vec::new()
        };

        let mut generated = vec![bos_id];
        let mut current_token = bos_id;

        for step in 0..max_len {
            self.decode_step(current_token, dec_state, &copy_logits, &mut bufs);

            let next_token = bufs.logits[..vocab_size].iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(idx, _)| idx as u32)
                .unwrap();

            if next_token == eos_id || next_token == pad_id {
                break;
            }

            generated.push(next_token);
            current_token = next_token;

            if (step + 1) % 10 == 0 {
                eprint!(".");
            }
        }
        eprintln!();

        Ok(generated)
    }
}
