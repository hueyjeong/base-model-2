//! BiMamba2 mixing layer — 양방향 Mamba-2 SSD scan
//!
//! mamba2_scan_avx2 C 커널 사용.
//! Mamba-1 대비: exp() 제거, 스칼라 decay, head 병렬화 (OpenMP)
//!
//! State: nheads × d_state × headdim
//!   d=640, expand=2, headdim=64, d_state=16: 20×16×64 = 80KB (L2)
//!   d=640, expand=2, headdim=64, d_state=64: 20×64×64 = 320KB (L2/L3 경계)

use std::collections::HashMap;
use anyhow::Result;

use crate::bmmq::TensorData;
use crate::common::*;
use super::MixingLayer;

/// 단방향 Mamba-2 SSD 블록
pub struct Mamba2Block {
    in_proj: LinearF32,       // d_model → d_in_proj (일반 nn.Linear — mamba_ssm.Mamba2)
    conv1d_weight: Vec<f32>,  // (d_conv_in, d_conv) — depthwise
    conv1d_bias: Vec<f32>,    // (d_conv_in,)
    a_log: Vec<f32>,          // (nheads,) — log-space
    d_skip: Vec<f32>,         // (nheads,)
    dt_bias: Vec<f32>,        // (nheads,)
    norm_weight: Vec<f32>,    // (d_inner,) — RMSNorm
    out_proj: LinearF32,      // d_inner → d_model (일반 nn.Linear)

    d_model: usize,
    d_inner: usize,
    d_state: usize,
    d_conv: usize,
    nheads: usize,
    headdim: usize,
    ngroups: usize,
}

impl Mamba2Block {
    pub fn load_bmmq(
        tensors: &mut HashMap<String, TensorData>,
        prefix: &str,
        d_model: usize, d_state: usize, d_conv: usize,
        expand: usize, headdim: usize, ngroups: usize,
    ) -> Result<Self> {
        let d_inner = d_model * expand;
        let nheads = d_inner / headdim;

        Ok(Self {
            in_proj: LinearF32::load_bmmq(tensors, &format!("{}.mamba2.in_proj", prefix))?,
            conv1d_weight: bmmq_take_f32(tensors, &format!("{}.mamba2.conv1d.weight", prefix))?,
            conv1d_bias: bmmq_take_f32(tensors, &format!("{}.mamba2.conv1d.bias", prefix))?,
            a_log: bmmq_take_f32(tensors, &format!("{}.mamba2.A_log", prefix))?,
            d_skip: bmmq_take_f32(tensors, &format!("{}.mamba2.D", prefix))?,
            dt_bias: bmmq_take_f32(tensors, &format!("{}.mamba2.dt_bias", prefix))?,
            norm_weight: bmmq_take_f32(tensors, &format!("{}.mamba2.norm.weight", prefix))?,
            out_proj: LinearF32::load_bmmq(tensors, &format!("{}.mamba2.out_proj", prefix))?,
            d_model, d_inner, d_state, d_conv, nheads, headdim, ngroups,
        })
    }
}

impl MixingLayer for Mamba2Block {
    fn forward_batch(
        &self, x: &[f32], seq_len: usize, d_model: usize,
        out: &mut [f32], bufs: &mut BatchBufs,
    ) {
        let di = self.d_inner;
        let ds = self.d_state;
        let nh = self.nheads;
        let hd = self.headdim;
        let ng = self.ngroups;
        let d_conv_in = di + 2 * ng * ds;
        let d_in_proj = 2 * di + 2 * ng * ds + nh;

        // in_proj: d_model → d_in_proj (FP32 matmul)
        let mut proj = vec![0.0f32; seq_len * d_in_proj];
        self.in_proj.forward_batch(x, seq_len, &mut proj);

        // DEBUG (첫 호출만)
        static DEBUG_ONCE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        let debug = !DEBUG_ONCE.swap(true, std::sync::atomic::Ordering::Relaxed);

        // Split: mamba_ssm.Mamba2 순서 — z(di) + xBC(di+2*ng*ds) + dt_raw(nh)
        // xBC = x(di) + B(ng*ds) + C(ng*ds)
        let mut z = vec![0.0f32; seq_len * di];
        let mut x_branch = vec![0.0f32; seq_len * di];
        let mut b_raw = vec![0.0f32; seq_len * ng * ds];
        let mut c_raw = vec![0.0f32; seq_len * ng * ds];
        let mut dt_raw = vec![0.0f32; seq_len * nh];

        for t in 0..seq_len {
            let src = t * d_in_proj;
            let mut off = 0;
            // z 먼저 (mamba_ssm 순서)
            for d in 0..di {
                z[t * di + d] = proj[src + off + d];
            }
            off += di;
            // xBC: x, B, C
            for d in 0..di {
                x_branch[t * di + d] = proj[src + off + d];
            }
            off += di;
            for d in 0..(ng * ds) {
                b_raw[t * ng * ds + d] = proj[src + off + d];
            }
            off += ng * ds;
            for d in 0..(ng * ds) {
                c_raw[t * ng * ds + d] = proj[src + off + d];
            }
            off += ng * ds;
            // dt
            for d in 0..nh {
                dt_raw[t * nh + d] = proj[src + off + d];
            }
        }

        // conv1d on [x_branch, B, C] concatenation
        let mut xbc = vec![0.0f32; seq_len * d_conv_in];
        for t in 0..seq_len {
            for d in 0..di {
                xbc[t * d_conv_in + d] = x_branch[t * di + d];
            }
            for d in 0..(ng * ds) {
                xbc[t * d_conv_in + di + d] = b_raw[t * ng * ds + d];
            }
            for d in 0..(ng * ds) {
                xbc[t * d_conv_in + di + ng * ds + d] = c_raw[t * ng * ds + d];
            }
        }

        // Depthwise causal conv1d (PyTorch cross-correlation 규칙: w[ki] * x[t-K+1+ki])
        let dc = self.d_conv;
        let mut xbc_conv = vec![0.0f32; seq_len * d_conv_in];
        for ch in 0..d_conv_in {
            for t in 0..seq_len {
                let mut sum = self.conv1d_bias[ch];
                for ki in 0..dc {
                    let src_t = t as i32 - (dc as i32 - 1) + ki as i32;
                    if src_t >= 0 && (src_t as usize) < seq_len {
                        sum += self.conv1d_weight[ch * dc + ki]
                            * xbc[src_t as usize * d_conv_in + ch];
                    }
                }
                xbc_conv[t * d_conv_in + ch] = sum;
            }
        }

        if debug {
            eprintln!("  conv1d[0:5] = [{:.6},{:.6},{:.6},{:.6},{:.6}]",
                xbc_conv[0], xbc_conv[1], xbc_conv[2], xbc_conv[3], xbc_conv[4]);
        }

        // SiLU on ALL xBC channels, then split (matching mamba_ssm combined kernel)
        let mut x_conv = vec![0.0f32; seq_len * di];
        let mut b_conv = vec![0.0f32; seq_len * ng * ds];
        let mut c_conv = vec![0.0f32; seq_len * ng * ds];
        for t in 0..seq_len {
            for d in 0..di {
                x_conv[t * di + d] = silu_scalar(xbc_conv[t * d_conv_in + d]);
            }
            for d in 0..(ng * ds) {
                b_conv[t * ng * ds + d] = silu_scalar(xbc_conv[t * d_conv_in + di + d]);
            }
            for d in 0..(ng * ds) {
                c_conv[t * ng * ds + d] = silu_scalar(xbc_conv[t * d_conv_in + di + ng * ds + d]);
            }
        }

        // Per-timestep dt 계산 (softplus(raw + bias))
        let mut dt_arr = vec![0.0f32; seq_len * nh];
        for t in 0..seq_len {
            for h in 0..nh {
                dt_arr[t * nh + h] = softplus_scalar(dt_raw[t * nh + h] + self.dt_bias[h]);
            }
        }

        // A = -exp(A_log) per head
        let a_neg: Vec<f32> = self.a_log.iter().map(|v| -v.exp()).collect();

        // Chunk-parallel SSD forward (mamba_ssm CUDA 호환)
        let mut y_scan = vec![0.0f32; seq_len * di];
        unsafe {
            mamba2_ssd_fwd(
                x_conv.as_ptr(), b_conv.as_ptr(), c_conv.as_ptr(),
                dt_arr.as_ptr(), a_neg.as_ptr(), self.d_skip.as_ptr(),
                y_scan.as_mut_ptr(),
                256,  // chunk_size
                seq_len as i32, nh as i32, hd as i32, ds as i32, ng as i32,
            );
        }

        // DEBUG: bwd scan token 10 h=0 p=0
        static DEBUG_SSD: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let ssd_call = DEBUG_SSD.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if ssd_call == 1 {  // bwd만
            let t = seq_len - 1;
            eprintln!("  bwd_scan[{},h0p0]={:.8}", t, y_scan[t*di]);
            eprintln!("  bwd_scan[{},h0p1]={:.8}", t, y_scan[t*di+1]);
            eprintln!("  bwd_scan[{},h0p2]={:.8}", t, y_scan[t*di+2]);
        }

        // Gate FIRST, then RMSNorm (norm_before_gate=False)
        // = RMSNorm(y * silu(z)) * weight
        for i in 0..seq_len * di {
            y_scan[i] *= silu_scalar(z[i]);
        }
        for t in 0..seq_len {
            let base = t * di;
            let mut sq = 0.0f32;
            for d in 0..di { sq += y_scan[base + d] * y_scan[base + d]; }
            let rms = (sq / di as f32 + 1e-5).sqrt().recip();
            for d in 0..di {
                y_scan[base + d] = y_scan[base + d] * rms * self.norm_weight[d];
            }
        }

        // out_proj: d_inner → d_model (FP32 matmul)
        self.out_proj.forward_batch(&y_scan, seq_len, out);
    }
}

/// 양방향 Mamba-2
pub struct BiMamba2 {
    pub fwd: Mamba2Block,
    pub bwd: Mamba2Block,
}

impl BiMamba2 {
    pub fn load_bmmq(
        tensors: &mut HashMap<String, TensorData>,
        prefix: &str,
        d_model: usize, d_state: usize, d_conv: usize,
        expand: usize, headdim: usize, ngroups: usize,
    ) -> Result<Self> {
        Ok(Self {
            fwd: Mamba2Block::load_bmmq(tensors, &format!("{}.fwd", prefix),
                d_model, d_state, d_conv, expand, headdim, ngroups)?,
            bwd: Mamba2Block::load_bmmq(tensors, &format!("{}.bwd", prefix),
                d_model, d_state, d_conv, expand, headdim, ngroups)?,
        })
    }
}
