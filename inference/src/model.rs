//! BitMamba Seq2Seq 모델 — CPU 추론 엔진 (v2 최적화)
//!
//! v2 최적화: i8 ternary BitLinear (mul→add/sub), rayon 병렬 matmul,
//!           cross-attn 순수 scalar, 버퍼 재활용, 정밀 타이밍

use anyhow::{Context, Result};
use candle_core::{DType, Device, IndexOp, Tensor, D};
use rayon::prelude::*;
use std::collections::HashMap;

use crate::config::ModelConfig;

// ── OpenBLAS FFI ─────────────────────────────────────

#[allow(non_camel_case_types)]
type c_int = i32;

const CBLAS_ROW_MAJOR: c_int = 101;
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

/// y = A @ x  (A: m×n row-major, x: n, y: m)
#[inline]
fn sgemv(a: &[f32], x: &[f32], y: &mut [f32], m: usize, n: usize) {
    unsafe {
        cblas_sgemv(
            CBLAS_ROW_MAJOR,
            CBLAS_NO_TRANS,
            m as c_int,
            n as c_int,
            1.0,
            a.as_ptr(),
            n as c_int,
            x.as_ptr(),
            1,
            0.0,
            y.as_mut_ptr(),
            1,
        );
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
    weight: Tensor,
    weight_vec: Vec<f32>,
    eps: f32,
}

impl RMSNorm {
    pub fn load(tensors: &HashMap<String, Tensor>, prefix: &str, eps: f64) -> Result<Self> {
        let weight = tensors.get(&format!("{}.weight", prefix))
            .context(format!("RMSNorm weight 없음: {}", prefix))?.clone();
        let weight_vec: Vec<f32> = weight.flatten_all()?.to_vec1()?;
        Ok(Self { weight, weight_vec, eps: eps as f32 })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let var = x.sqr()?.mean_keepdim(D::Minus1)?;
        let rms = (var + self.eps as f64)?.sqrt()?.recip()?;
        Ok((x.broadcast_mul(&rms))?.broadcast_mul(&self.weight)?)
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

// ── BitLinear (i8 ternary + rayon) ───────────────────

pub struct BitLinear {
    w_quant: Tensor,          // 사전 양자화된 ternary weight (인코더 Tensor path용)
    gamma: f32,
    out_dim: usize,
    in_dim: usize,
    w_ternary_f32: Vec<f32>,  // f32 {-1,0,+1} (sgemv용 — BLAS > auto-vec)
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

        // f32 ternary (BLAS sgemv용 — hand-tuned AVX2가 auto-vec i8보다 빠름)
        let w_ternary_f32: Vec<f32> = w_quant.flatten_all()?.to_vec1()?;

        Ok(Self { w_quant, gamma, out_dim, in_dim, w_ternary_f32 })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x_norm = layer_norm_no_affine(x, 1e-5)?;
        let eta = x_norm.abs()?.max_keepdim(D::Minus1)?.clamp(1e-5, f64::MAX)?;
        let x_quant = x_norm.broadcast_div(&eta)?
            .broadcast_mul(&Tensor::new(127.0f32, x.device())?)?
            .clamp(-128.0, 127.0)?.round()?;
        let x_scale = (eta / 127.0)?;

        let wt = self.w_quant.t()?;
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

    /// ternary matmul via sgemv (할당 없음 — 외부 버퍼)
    fn forward_vec(&self, x: &[f32], out: &mut [f32], x_norm: &mut [f32]) {
        debug_assert_eq!(x.len(), self.in_dim);
        let n = self.in_dim;

        // 1. LayerNorm
        layer_norm_no_affine_vec(x, &mut x_norm[..n], 1e-5);

        // 2. 활성화 양자화
        let mut eta = 0.0f32;
        for i in 0..n { eta = eta.max(x_norm[i].abs()); }
        eta = eta.max(1e-5);
        let scale_in = 127.0 / eta;
        for i in 0..n {
            x_norm[i] = (x_norm[i] * scale_in).clamp(-128.0, 127.0).round();
        }
        let combined_scale = (eta / 127.0) * self.gamma;

        // 3. Ternary matmul via BLAS sgemv (hand-tuned AVX2 > auto-vec i8)
        sgemv(&self.w_ternary_f32, &x_norm[..n], out, self.out_dim, n);
        for o in 0..self.out_dim {
            out[o] *= combined_scale;
        }
    }
}

// ── Linear (rayon 병렬) ──────────────────────────────

pub struct Linear {
    weight: Tensor,
    weight_data: Vec<f32>,
    out_dim: usize,
    in_dim: usize,
}

impl Linear {
    pub fn load(tensors: &HashMap<String, Tensor>, prefix: &str) -> Result<Self> {
        let weight = tensors.get(&format!("{}.weight", prefix))
            .context(format!("Linear weight 없음: {}", prefix))?.clone();
        let out_dim = weight.dim(0)?;
        let in_dim = weight.dim(1)?;
        let weight_data: Vec<f32> = weight.flatten_all()?.to_vec1()?;
        Ok(Self { weight, weight_data, out_dim, in_dim })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let wt = self.weight.t()?;
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

    fn forward_vec(&self, x: &[f32], out: &mut [f32]) {
        sgemv(&self.weight_data, x, out, self.out_dim, self.in_dim);
    }
}

// ── LinearWithBias (Copy Gate용) ─────────────────────

pub struct LinearWithBias {
    weight_data: Vec<f32>,
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
        let weight_data: Vec<f32> = weight.flatten_all()?.to_vec1()?;
        let bias_data: Vec<f32> = bias.flatten_all()?.to_vec1()?;
        Ok(Self { weight_data, bias_data, out_dim, in_dim })
    }

    fn forward_vec(&self, x: &[f32], out: &mut [f32]) {
        sgemv(&self.weight_data, x, out, self.out_dim, self.in_dim);
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

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = sigmoid(&self.gate_proj.forward(x)?)?;
        let up = self.up_proj.forward(x)?;
        let x = (gate * up)?;
        self.down_proj.forward(&x)
    }

    fn forward_vec(&self, x: &[f32], buf_ff: &mut Vec<f32>, buf_ff2: &mut Vec<f32>,
                    x_norm_buf: &mut [f32], out: &mut [f32]) {
        buf_ff.resize(self.d_ff, 0.0);
        buf_ff2.resize(self.d_ff, 0.0);
        self.gate_proj.forward_vec(x, buf_ff, x_norm_buf);
        self.up_proj.forward_vec(x, buf_ff2, x_norm_buf);

        // sigmoid(gate) * up → reuse buf_ff
        for i in 0..self.d_ff {
            let sig = 1.0 / (1.0 + (-buf_ff[i]).exp());
            buf_ff[i] = sig * buf_ff2[i];
        }

        self.down_proj.forward_vec(buf_ff, out, x_norm_buf);
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
    conv1d_weight: Tensor,
    conv1d_bias: Tensor,
    norm_weight: Tensor,
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
            conv1d_weight,
            conv1d_bias,
            norm_weight,
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

    pub fn new_state(&self) -> Mamba2State {
        Mamba2State {
            ssm_state: vec![0.0f32; self.nheads * self.d_state * self.headdim],
            conv_buf: vec![0.0f32; (self.d_conv - 1) * self.conv_channels],
            conv_buf_len: 0,
        }
    }

    fn causal_conv1d(&self, x: &Tensor) -> Result<Tensor> {
        let (_, seq_len, channels) = x.dims3()?;
        let dev = x.device();
        let pad = Tensor::zeros((1, self.d_conv - 1, channels), DType::F32, dev)?;
        let x_padded = Tensor::cat(&[&pad, x], 1)?;

        let mut out = self.conv1d_bias.unsqueeze(0)?.unsqueeze(0)?
            .broadcast_as((1, seq_len, channels))?.contiguous()?;

        for k in 0..self.d_conv {
            let w_k = self.conv1d_weight.i((.., 0, k))?;
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
        let y_norm = rmsnorm_with_weight(&y, &self.norm_weight, 1e-5)?;
        let z_gate = silu(&z.reshape((1, seq_len, self.d_inner))?)?;
        let y_gated = (y_norm * z_gate)?;

        self.out_proj.forward(&y_gated)
    }

    /// Incremental 1-token step (ssm_y: 재활용 버퍼)
    pub fn step(&self, x: &[f32], state: &mut Mamba2State, proj_buf: &mut Vec<f32>, xbc_buf: &mut Vec<f32>, ssm_y: &mut [f32]) {
        // 1. in_proj
        proj_buf.resize(self.proj_dim, 0.0);
        self.in_proj.forward_vec(x, proj_buf);

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
        self.out_proj.forward_vec(xbc_buf, proj_buf);
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
                          attn_out: &mut [f32], q_h: &mut [f32], out: &mut [f32]) {
        let nh = self.n_heads;
        let dh = self.d_head;
        let d_model = nh * dh;

        // Q projection
        q_buf.resize(d_model, 0.0);
        self.q_proj.forward_vec(x, q_buf);

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
        self.o_proj.forward_vec(&attn_out, out);
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

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let residual = x.clone();
        let x = self.mamba.forward(x)?;
        let x = self.norm1.forward(&(residual + &x)?)?;

        let residual = x.clone();
        let x = self.ffn.forward(&x)?;
        self.norm2.forward(&(residual + &x)?)
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
}

impl DecoderBufs {
    fn new(d_model: usize, proj_dim: usize, conv_ch: usize, d_ff: usize,
           d_inner: usize, d_head: usize, max_in_dim: usize) -> Self {
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
        self.mamba.step(x, mamba_state, &mut bufs.proj, &mut bufs.xbc, &mut bufs.ssm_y);
        // mamba 결과가 bufs.proj에 있음 (out_proj 출력)
        for i in 0..d { bufs.residual[i] = x[i] + bufs.proj[i]; }
        self.norm1.forward_vec(&bufs.residual, &mut bufs.normed);

        // 2. Cross-attention (cached scalar)
        self.cross_attn.forward_cached_vec(&bufs.normed, cross_cache, &mut bufs.q,
                                           &mut bufs.attn_out, &mut bufs.q_h, &mut bufs.out);
        for i in 0..d { bufs.residual[i] = bufs.normed[i] + bufs.out[i]; }
        self.norm_cross.forward_vec(&bufs.residual, &mut bufs.normed);

        // 3. FFN
        self.ffn.forward_vec(&bufs.normed, &mut bufs.ff1, &mut bufs.ff2, &mut bufs.x_norm_buf, &mut bufs.out);
        for i in 0..d { bufs.residual[i] = bufs.normed[i] + bufs.out[i]; }
        self.norm2.forward_vec(&bufs.residual, &mut bufs.normed);
        // normed가 이 레이어의 최종 출력
    }
}

// ── BitMambaSeq2Seq ──────────────────────────────────

pub struct BitMambaSeq2Seq {
    encoder_embedding: Tensor,
    embed_data: Vec<f32>,
    encoder_layers: Vec<EncoderLayer>,
    decoder_layers: Vec<DecoderLayer>,
    final_norm: RMSNorm,
    lm_head_data: Vec<f32>,
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
        let lm_head_data: Vec<f32> = lm_head_weight.flatten_all()?.to_vec1()?;

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
        eprintln!("모델 로드 완료 (BitLinear i8 ternary ×{}, rayon 병렬화)", bitlinear_count);

        Ok(Self {
            encoder_embedding,
            embed_data,
            encoder_layers,
            decoder_layers,
            final_norm,
            lm_head_data,
            copy_gate,
            cfg: cfg.clone(),
        })
    }

    fn embed(&self, ids: &Tensor) -> Result<Tensor> {
        let (batch, seq_len) = ids.dims2()?;
        let flat_ids = ids.flatten_all()?;
        let emb = self.encoder_embedding.index_select(&flat_ids, 0)?;
        let emb = emb.reshape((batch, seq_len, self.cfg.d_model))?;
        let scale = Tensor::new(self.cfg.embed_scale(), ids.device())?;
        Ok(emb.broadcast_mul(&scale)?)
    }

    #[inline]
    fn embed_one(&self, id: u32) -> Vec<f32> {
        let d = self.cfg.d_model;
        let scale = self.cfg.embed_scale();
        let offset = id as usize * d;
        let mut out = vec![0.0f32; d];
        for i in 0..d {
            out[i] = self.embed_data[offset + i] * scale;
        }
        out
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

    /// Incremental 1-token decode step → logits
    fn decode_step(
        &self,
        token_id: u32,
        state: &mut DecoderState,
        src_ids: &[u32],
        bufs: &mut DecoderBufs,
    ) -> Vec<f32> {
        let d = self.cfg.d_model;

        let mut x = self.embed_one(token_id);

        for (i, layer) in self.decoder_layers.iter().enumerate() {
            layer.step(
                &x,
                &mut state.mamba_states[i],
                &state.cross_attn_caches[i],
                bufs,
            );
            // normed가 결과 — x에 복사
            x.copy_from_slice(&bufs.normed[..d]);
        }

        // final_norm
        self.final_norm.forward_vec(&x, &mut bufs.normed);

        // LM Head via sgemv
        let mut logits = vec![0.0f32; self.cfg.vocab_size];
        sgemv(&self.lm_head_data, &bufs.normed[..d], &mut logits, self.cfg.vocab_size, d);

        // Copy Gate
        if let Some(ref gate_proj) = self.copy_gate {
            let mut gate_out = [0.0f32; 1];
            gate_proj.forward_vec(&bufs.normed, &mut bufs.out);
            let raw_sig = 1.0 / (1.0 + (-bufs.out[0]).exp());
            let gate = 0.5 + 0.5 * raw_sig;
            let one_minus_gate = 1.0 - gate;

            let mut p_copy = vec![0.0f32; self.cfg.vocab_size];
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

            for v in 0..self.cfg.vocab_size {
                let copy_logit = (p_copy[v].max(1e-5)).ln();
                logits[v] = gate * logits[v] + one_minus_gate * copy_logit;
            }
        }

        logits
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
        // 첫 디코더 레이어에서 proj_dim과 conv_channels, d_ff 가져오기
        let proj_dim = self.decoder_layers[0].mamba.proj_dim;
        let conv_ch = self.decoder_layers[0].mamba.conv_channels;
        let d_ff = self.decoder_layers[0].ffn.d_ff;

        let d_inner = self.cfg.d_inner;
        let d_head = 64usize;
        let max_in_dim = d_model.max(d_inner).max(self.cfg.d_ff);
        let mut bufs = DecoderBufs::new(d_model, proj_dim, conv_ch, d_ff,
                                         d_inner, d_head, max_in_dim);
        let mut generated = vec![bos_id];
        let mut current_token = bos_id;

        for step in 0..max_len {
            let logits = self.decode_step(current_token, dec_state, src_ids, &mut bufs);

            let next_token = logits.iter()
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
