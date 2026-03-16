//! BiMamba mixing layer — 양방향 Mamba-1 selective scan
//!
//! mamba_scan_avx2 C 커널 사용.
//! State: d_inner × d_state = 512×16 = 32KB (L1/L2 경계)

use std::collections::HashMap;
use anyhow::Result;

use crate::bmmq::TensorData;
use crate::common::*;
use super::MixingLayer;

/// 단방향 Mamba-1 블록
pub struct MambaBlock {
    in_proj: BitLinear,   // d_model → 2*d_inner
    conv1d_weight: Vec<f32>,  // (d_inner, d_conv)
    conv1d_bias: Vec<f32>,    // (d_inner,)
    x_proj: Linear,       // d_inner → dt_rank + 2*d_state
    dt_proj: Linear,      // dt_rank → d_inner
    dt_proj_bias: Vec<f32>,   // (d_inner,)
    A_log: Vec<f32>,      // (d_inner, d_state) — log-space
    D: Vec<f32>,          // (d_inner,) — skip
    out_proj: BitLinear,  // d_inner → d_model

    d_model: usize,
    d_inner: usize,
    d_state: usize,
    d_conv: usize,
    dt_rank: usize,
}

impl MambaBlock {
    pub fn load_bmmq(
        tensors: &mut HashMap<String, TensorData>,
        prefix: &str, d_model: usize, d_state: usize, d_conv: usize, expand: usize,
    ) -> Result<Self> {
        let d_inner = d_model * expand;
        let dt_rank = (d_model / 16).max(1);
        Ok(Self {
            in_proj: BitLinear::load_bmmq(tensors, &format!("{}.in_proj", prefix))?,
            conv1d_weight: bmmq_take_f32(tensors, &format!("{}.conv1d.weight", prefix))?,
            conv1d_bias: bmmq_take_f32(tensors, &format!("{}.conv1d.bias", prefix))?,
            x_proj: Linear::load_bmmq(tensors, &format!("{}.x_proj", prefix))?,
            dt_proj: Linear::load_bmmq(tensors, &format!("{}.dt_proj", prefix))?,
            dt_proj_bias: bmmq_take_f32(tensors, &format!("{}.dt_proj.bias", prefix))
                .unwrap_or_else(|_| vec![0.0; d_inner]),
            A_log: bmmq_take_f32(tensors, &format!("{}.A_log", prefix))?,
            D: bmmq_take_f32(tensors, &format!("{}.D", prefix))?,
            out_proj: BitLinear::load_bmmq(tensors, &format!("{}.out_proj", prefix))?,
            d_model, d_inner, d_state, d_conv, dt_rank,
        })
    }
}

impl MixingLayer for MambaBlock {
    fn forward_batch(
        &self, x: &[f32], seq_len: usize, d_model: usize,
        out: &mut [f32], bufs: &mut BatchBufs,
    ) {
        let di = self.d_inner;
        let ds = self.d_state;

        // in_proj: (seq_len, d_model) → (seq_len, 2*d_inner)
        let mut xz = vec![0.0f32; seq_len * 2 * di];
        self.in_proj.forward_batch(x, seq_len, &mut xz, bufs);

        // Split x_branch, z
        let mut x_branch = vec![0.0f32; seq_len * di];
        let mut z = vec![0.0f32; seq_len * di];
        for t in 0..seq_len {
            for d in 0..di {
                x_branch[t * di + d] = xz[t * 2 * di + d];
                z[t * di + d] = xz[t * 2 * di + di + d];
            }
        }

        // Conv1d (depthwise, kernel=d_conv, causal padding)
        let mut x_conv = vec![0.0f32; seq_len * di];
        for d in 0..di {
            for t in 0..seq_len {
                let mut sum = self.conv1d_bias[d];
                for ki in 0..self.d_conv {
                    let src_t = t as i32 - ki as i32;
                    if src_t >= 0 {
                        sum += self.conv1d_weight[d * self.d_conv + ki]
                            * x_branch[src_t as usize * di + d];
                    }
                }
                x_conv[t * di + d] = silu_scalar(sum);
            }
        }

        // SSM parameters: x_proj → dt, B, C
        let proj_dim = self.dt_rank + 2 * ds;
        let mut x_ssm = vec![0.0f32; seq_len * proj_dim];
        self.x_proj.forward_batch(&x_conv, seq_len, &mut x_ssm, bufs);

        // dt_proj: dt_rank → d_inner + softplus
        let mut delta = vec![0.0f32; seq_len * di];
        let mut dt_raw = vec![0.0f32; seq_len * self.dt_rank];
        for t in 0..seq_len {
            for d in 0..self.dt_rank {
                dt_raw[t * self.dt_rank + d] = x_ssm[t * proj_dim + d];
            }
        }
        let mut dt_proj_out = vec![0.0f32; seq_len * di];
        self.dt_proj.forward_batch(&dt_raw, seq_len, &mut dt_proj_out, bufs);
        for i in 0..seq_len * di {
            delta[i] = softplus_scalar(dt_proj_out[i] + self.dt_proj_bias[i % di]);
        }

        // B, C 추출
        let mut B = vec![0.0f32; seq_len * ds];
        let mut C = vec![0.0f32; seq_len * ds];
        for t in 0..seq_len {
            for d in 0..ds {
                B[t * ds + d] = x_ssm[t * proj_dim + self.dt_rank + d];
                C[t * ds + d] = x_ssm[t * proj_dim + self.dt_rank + ds + d];
            }
        }

        // A = -exp(A_log)
        let mut A = vec![0.0f32; di * ds];
        for i in 0..di * ds {
            A[i] = -self.A_log[i].exp();
        }

        // Selective scan (C 커널)
        let mut y = vec![0.0f32; seq_len * di];
        let mut state = vec![0.0f32; di * ds];
        unsafe {
            mamba_scan_avx2(
                delta.as_ptr(), B.as_ptr(), C.as_ptr(), x_conv.as_ptr(),
                A.as_ptr(), self.D.as_ptr(), y.as_mut_ptr(), state.as_mut_ptr(),
                seq_len as i32, di as i32, ds as i32,
            );
        }

        // Gate: y * silu(z)
        for i in 0..seq_len * di {
            y[i] *= silu_scalar(z[i]);
        }

        // out_proj: (seq_len, d_inner) → (seq_len, d_model)
        self.out_proj.forward_batch(&y, seq_len, out, bufs);
    }
}

/// 양방향 Mamba
pub struct BiMamba {
    pub fwd: MambaBlock,
    pub bwd: MambaBlock,
}

impl BiMamba {
    pub fn load_bmmq(
        tensors: &mut HashMap<String, TensorData>,
        prefix: &str, d_model: usize, d_state: usize, d_conv: usize, expand: usize,
    ) -> Result<Self> {
        Ok(Self {
            fwd: MambaBlock::load_bmmq(tensors, &format!("{}.fwd", prefix), d_model, d_state, d_conv, expand)?,
            bwd: MambaBlock::load_bmmq(tensors, &format!("{}.bwd", prefix), d_model, d_state, d_conv, expand)?,
        })
    }
}
