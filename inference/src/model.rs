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

// ── LayerNorm (elementwise_affine=False) ─────────────

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
    gamma: f32,
    out_dim: usize,
    in_dim: usize,
    w_i8: Vec<i8>,
    w_packed: Vec<u8>,
    packed_stride: usize,
    row_sums: Vec<i32>,
}

impl BitLinear {
    pub fn load_bmmq(tensors: &mut HashMap<String, TensorData>, prefix: &str) -> Result<Self> {
        let key = format!("{}.weight", prefix);
        match tensors.remove(&key).context(format!("BitLinear weight 없음: {}", key))? {
            TensorData::Packed2Bit { data, gamma, row_sums, rows, cols, packed_stride } => {
                Ok(Self {
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

    /// AVX-VNNI matmul (할당 없음 — 외부 버퍼)
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
        buf_ff.resize(self.d_ff, 0.0);
        buf_ff2.resize(self.d_ff, 0.0);
        self.gate_proj.forward_vec(x, buf_ff, x_norm_buf, x_u8);
        self.up_proj.forward_vec(x, buf_ff2, x_norm_buf, x_u8);

        // sigmoid(gate) * up → reuse buf_ff
        for i in 0..self.d_ff {
            let sig = sigmoid_scalar(buf_ff[i]);
            buf_ff[i] = sig * buf_ff2[i];
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
    w_base: Vec<f32>,     // (n_heads * headdim,)
    w_lora_down: Linear,  // (d_model -> lora_rank)
    w_lora_up: Linear,    // (lora_rank -> d_model)
    output_norm_weight: Vec<f32>,  // LayerNorm per-head weight
    output_norm_bias: Vec<f32>,    // LayerNorm per-head bias
    n_heads: usize,
    headdim: usize,
    d_model: usize,
    lora_rank: usize,
}

impl RWKV6TimeMix {
    fn load_bmmq(
        tensors: &mut HashMap<String, TensorData>,
        prefix: &str,
        n_heads: usize,
        headdim: usize,
        lora_rank: usize,
    ) -> Result<Self> {
        let d_model = n_heads * headdim;
        Ok(Self {
            r_proj: BitLinear::load_bmmq(tensors, &format!("{}.r_proj", prefix))?,
            k_proj: BitLinear::load_bmmq(tensors, &format!("{}.k_proj", prefix))?,
            v_proj: BitLinear::load_bmmq(tensors, &format!("{}.v_proj", prefix))?,
            o_proj: BitLinear::load_bmmq(tensors, &format!("{}.o_proj", prefix))?,
            g_proj: Linear::load_bmmq(tensors, &format!("{}.g_proj", prefix))?,
            w_base: bmmq_take_f32(tensors, &format!("{}.w_base", prefix))?,
            w_lora_down: Linear::load_bmmq(tensors, &format!("{}.w_lora_down", prefix))?,
            w_lora_up: Linear::load_bmmq(tensors, &format!("{}.w_lora_up", prefix))?,
            output_norm_weight: bmmq_take_f32(tensors, &format!("{}.output_norm.weight", prefix))?,
            output_norm_bias: bmmq_take_f32(tensors, &format!("{}.output_norm.bias", prefix))?,
            n_heads,
            headdim,
            d_model,
            lora_rank,
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

        // R, K, V, G 프로젝션: 토큰별 순회
        bufs.r.resize(seq_len * d, 0.0);
        bufs.k.resize(seq_len * d, 0.0);
        bufs.v.resize(seq_len * d, 0.0);
        bufs.g.resize(seq_len * d, 0.0);
        bufs.w.resize(seq_len * d, 0.0);

        for t in 0..seq_len {
            let x_t = &x[t * d..(t + 1) * d];
            self.r_proj.forward_vec(x_t, &mut bufs.r[t * d..(t + 1) * d],
                                    &mut bufs.x_norm, &mut bufs.x_u8);
            self.k_proj.forward_vec(x_t, &mut bufs.k[t * d..(t + 1) * d],
                                    &mut bufs.x_norm, &mut bufs.x_u8);
            self.v_proj.forward_vec(x_t, &mut bufs.v[t * d..(t + 1) * d],
                                    &mut bufs.x_norm, &mut bufs.x_u8);
            self.g_proj.forward_vec(x_t, &mut bufs.g[t * d..(t + 1) * d],
                                    &mut bufs.x_u8);

            // Data-dependent decay: w = w_base + w_lora_up(tanh(w_lora_down(x)))
            bufs.lora_down.resize(self.lora_rank, 0.0);
            self.w_lora_down.forward_vec(x_t, &mut bufs.lora_down, &mut bufs.x_u8);
            for v in bufs.lora_down.iter_mut() { *v = v.tanh(); }
            bufs.lora_up.resize(d, 0.0);
            self.w_lora_up.forward_vec(&bufs.lora_down, &mut bufs.lora_up, &mut bufs.x_u8);
            for i in 0..d {
                bufs.w[t * d + i] = self.w_base[i] + bufs.lora_up[i];
            }
        }

        // WKV sequential scan
        // state: (n_heads, headdim, headdim) — state[h][i][j]
        bufs.state.resize(nh * hd * hd, 0.0);
        bufs.state.fill(0.0);
        bufs.output.resize(seq_len * d, 0.0);

        for t in 0..seq_len {
            let k_t = &bufs.k[t * d..];
            let v_t = &bufs.v[t * d..];
            let r_t = &bufs.r[t * d..];

            for h in 0..nh {
                let h_off = h * hd;
                let s_off = h * hd * hd;

                // state[h] = diag(exp(w_t[h])) @ state[h] + k[t,h] ⊗ v[t,h]
                // Python: w = -softplus(w_raw) → exp(w) = 1/(1+exp(w_raw))
                for i in 0..hd {
                    let w_raw = bufs.w[t * d + h_off + i];
                    let decay = 1.0 / (1.0 + w_raw.exp());
                    let k_val = k_t[h_off + i];
                    let si = s_off + i * hd;
                    for j in 0..hd {
                        bufs.state[si + j] = decay * bufs.state[si + j]
                            + k_val * v_t[h_off + j];
                    }
                }

                // out[t,h] = state[h] @ r[t,h]  via cblas_sgemv
                // state[h] is (headdim, headdim) row-major
                // out[d] = sum_i state[h][i][d] * r[h][i]  →  state^T @ r
                unsafe {
                    cblas_sgemv(
                        CBLAS_ROW_MAJOR,
                        112, // CblasTrans
                        hd as c_int,
                        hd as c_int,
                        1.0,
                        bufs.state[s_off..].as_ptr(),
                        hd as c_int,
                        r_t[h_off..].as_ptr(),
                        1,
                        0.0,
                        bufs.output[t * d + h_off..].as_mut_ptr(),
                        1,
                    );
                }
            }
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
                    bufs.normed_head[idx] = normed * self.output_norm_weight[idx]
                        + self.output_norm_bias[idx];
                }
            }

            // Gate: silu(g) * normed
            for i in 0..d {
                bufs.gated[i] = silu_scalar(g_t[i]) * bufs.normed_head[i];
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
    output: Vec<f32>,
    normed_head: Vec<f32>,
    gated: Vec<f32>,
    final_out: Vec<f32>,
    lora_down: Vec<f32>,
    lora_up: Vec<f32>,
    x_norm: Vec<f32>,
    x_u8: Vec<u8>,
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
            output: Vec::new(),
            normed_head: Vec::new(),
            gated: Vec::new(),
            final_out: Vec::new(),
            lora_down: Vec::new(),
            lora_up: Vec::new(),
            x_norm: vec![0.0; max_in_dim],
            x_u8: vec![0u8; max_in_dim],
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
        // fwd_bufs.final_out에 결과 누적
        for t in 0..seq_len {
            let fwd_off = t * d_model;
            let bwd_off = (seq_len - 1 - t) * d_model;
            for i in 0..d_model {
                fwd_bufs.final_out[fwd_off + i] += bwd_bufs.final_out[bwd_off + i];
            }
        }
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

            for &idx in &bufs.top_indices {
                let weight = bufs.router_probs[idx] * inv_top;
                self.experts[idx].forward_vec(
                    x_t, &mut bufs.ff1, &mut bufs.ff2,
                    &mut bufs.x_norm, &mut bufs.x_u8, &mut bufs.expert_out,
                );
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

            // Base projection
            self.q_proj.forward_vec(x_t, &mut bufs.q[t * d..(t + 1) * d], &mut bufs.x_u8);
            self.k_proj.forward_vec(x_t, &mut bufs.k[t * d..(t + 1) * d], &mut bufs.x_u8);
            self.v_proj.forward_vec(x_t, &mut bufs.v[t * d..(t + 1) * d], &mut bufs.x_u8);

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

                // Q[t,h] @ context[h]  via cblas_sgemv
                // context is (dh, dh) row-major
                // want: out[d] = sum_i Q[i] * context[i][d]  →  context^T @ Q
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

                // normalizer: Q[t,h] . z[h]
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
                if self.attn_insertion_set.contains(&layer_idx) {
                    // attention norm
                    normed.resize(cur_len * d, 0.0);
                    let mut norm_in = vec![0.0f32; d];
                    for t in 0..cur_len {
                        let off = t * d;
                        norm_in.copy_from_slice(&x[off..off + d]);
                        self.attn_norms[attn_insert_idx].forward_vec(
                            &norm_in, &mut normed[off..off + d],
                        );
                    }

                    // shared self-attention
                    self.shared_attn.forward_batch(
                        &normed, cur_len, attn_insert_idx, &mut attn_bufs,
                    );

                    // residual
                    for i in 0..cur_len * d {
                        x[i] += attn_bufs.final_out[i];
                    }

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
}
