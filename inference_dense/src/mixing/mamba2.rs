//! BiMamba2 mixing layer — 양방향 Mamba-2 SSD scan
//!
//! mamba2_scan_avx2 C 커널 사용.
//! Mamba-1 대비: exp() 제거, 스칼라 decay, head 병렬화 (OpenMP)
//!
//! State: nheads × d_state × headdim
//!   d=640, expand=2, headdim=64, d_state=64: 20×64×64 = 320KB (L2/L3 경계)

use std::collections::HashMap;
use anyhow::Result;

use crate::bmmq::TensorData;
use crate::common::*;
use super::MixingLayer;

/// 단방향 Mamba-2 SSD 블록 — BitLinear projection (Ternary/F32 자동 감지)
///
/// 저랭크 모드 (in_proj_rank > 0):
///   in_proj_down (d_model → rank, F32) → in_proj_up (rank → d_in_proj, Ternary)
/// 기존 모드 (in_proj_rank = 0):
///   in_proj_up (d_model → d_in_proj) 단일 projection
pub struct Mamba2Block {
    in_proj_down: Option<Projection>, // 저랭크일 때만: d_model → rank
    in_proj_up: Projection,           // rank → d_in_proj (또는 full: d_model → d_in_proj)
    conv1d_weight: Vec<f32>,          // (d_conv_in, d_conv) — depthwise
    conv1d_bias: Vec<f32>,            // (d_conv_in,)
    a_log: Vec<f32>,                  // (nheads,) — log-space
    d_skip: Vec<f32>,                 // (nheads,)
    dt_bias: Vec<f32>,                // (nheads,)
    norm_weight: Vec<f32>,            // (d_inner,) — RMSNorm
    out_proj: Projection,             // d_inner → d_model (BitLinear → Ternary)

    d_model: usize,
    d_inner: usize,
    d_state: usize,
    d_conv: usize,
    nheads: usize,
    headdim: usize,
    ngroups: usize,

    // pre-allocated 버퍼 (a_neg는 상수)
    a_neg: Vec<f32>,
}

impl Mamba2Block {
    pub fn load_bmmq(
        tensors: &mut HashMap<String, TensorData>,
        prefix: &str,
        d_model: usize, d_state: usize, d_conv: usize,
        expand: usize, headdim: usize, ngroups: usize,
        in_proj_rank: usize,
    ) -> Result<Self> {
        let d_inner = d_model * expand;
        let nheads = d_inner / headdim;

        // BitLinear QAT 모델: in_proj_rank > 0이면 저랭크 + BitLinear (LayerNorm 포함)
        // 기존 모델: in_proj_rank = 0이면 단일 Ternary (LayerNorm 없음)
        let use_bitlinear = in_proj_rank > 0;
        let load_proj = if use_bitlinear {
            Projection::load_bmmq_bitlinear
        } else {
            Projection::load_bmmq
        };

        let (in_proj_down, in_proj_up) = if in_proj_rank > 0 {
            let down = Projection::load_bmmq(tensors, &format!("{}.in_proj_down", prefix))?;
            let up = load_proj(tensors, &format!("{}.in_proj_up", prefix))?;
            (Some(down), up)
        } else {
            (None, Projection::load_bmmq(tensors, &format!("{}.in_proj", prefix))?)
        };

        let a_log = bmmq_take_f32(tensors, &format!("{}.A_log", prefix))?;
        let a_neg: Vec<f32> = a_log.iter().map(|v| -v.exp()).collect();

        Ok(Self {
            in_proj_down,
            in_proj_up,
            conv1d_weight: bmmq_take_f32(tensors, &format!("{}.conv1d.weight", prefix))?,
            conv1d_bias: bmmq_take_f32(tensors, &format!("{}.conv1d.bias", prefix))?,
            a_log,
            d_skip: bmmq_take_f32(tensors, &format!("{}.D", prefix))?,
            dt_bias: bmmq_take_f32(tensors, &format!("{}.dt_bias", prefix))?,
            norm_weight: bmmq_take_f32(tensors, &format!("{}.norm.weight", prefix))?,
            out_proj: load_proj(tensors, &format!("{}.out_proj", prefix))?,
            d_model, d_inner, d_state, d_conv, nheads, headdim, ngroups,
            a_neg,
        })
    }
}

impl MixingLayer for Mamba2Block {
    fn forward_batch(
        &self, x: &[f32], seq_len: usize, _d_model: usize,
        out: &mut [f32], _bufs: &mut BatchBufs,
    ) {
        let di = self.d_inner;
        let ds = self.d_state;
        let nh = self.nheads;
        let hd = self.headdim;
        let ng = self.ngroups;
        let d_conv_in = di + 2 * ng * ds;
        let d_in_proj = self.in_proj_up.out_dim();

        let _t0 = std::time::Instant::now();

        // in_proj → proj 버퍼 (저랭크: down→mid→up, 기존: 단일 matmul)
        let mut proj = vec![0.0f32; seq_len * d_in_proj];
        if let Some(ref down) = self.in_proj_down {
            let rank = down.out_dim();
            let mut mid = vec![0.0f32; seq_len * rank];
            down.forward_batch(x, seq_len, &mut mid);
            self.in_proj_up.forward_batch(&mid, seq_len, &mut proj);
        } else {
            self.in_proj_up.forward_batch(x, seq_len, &mut proj);
        }

        let _t_proj = _t0.elapsed();

        // proj 레이아웃: [z(di) | xBC(d_conv_in) | dt(nh)] per token
        // xBC 부분을 직접 conv1d 입력으로 사용 — 별도 split+concat 불필요
        // 단, conv1d는 [seq_len × d_conv_in] 연속 배열 필요 → proj에서 추출
        let mut xbc = vec![0.0f32; seq_len * d_conv_in];
        for t in 0..seq_len {
            let src = &proj[t * d_in_proj + di..t * d_in_proj + di + d_conv_in];
            xbc[t * d_conv_in..(t + 1) * d_conv_in].copy_from_slice(src);
        }

        let _t1 = std::time::Instant::now();

        // Depthwise causal conv1d
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

        let _t_conv = _t1.elapsed();

        // SiLU on ALL xBC → split into x_conv, b_conv, c_conv
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

        // dt = softplus(raw + bias)
        let mut dt_arr = vec![0.0f32; seq_len * nh];
        for t in 0..seq_len {
            for h in 0..nh {
                let raw = proj[t * d_in_proj + di + d_conv_in + h];
                dt_arr[t * nh + h] = softplus_scalar(raw + self.dt_bias[h]);
            }
        }

        let _t2 = std::time::Instant::now();

        // Chunk-parallel SSD forward
        let mut y_scan = vec![0.0f32; seq_len * di];
        unsafe {
            mamba2_ssd_fwd(
                x_conv.as_ptr(), b_conv.as_ptr(), c_conv.as_ptr(),
                dt_arr.as_ptr(), self.a_neg.as_ptr(), self.d_skip.as_ptr(),
                y_scan.as_mut_ptr(),
                256,
                seq_len as i32, nh as i32, hd as i32, ds as i32, ng as i32,
            );
        }

        let _t_scan = _t2.elapsed();

        // Gate FIRST, then RMSNorm (norm_before_gate=False)
        // z는 proj의 첫 di 원소에서 직접 참조
        for t in 0..seq_len {
            for d in 0..di {
                y_scan[t * di + d] *= silu_scalar(proj[t * d_in_proj + d]);
            }
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

        let _t3 = std::time::Instant::now();

        self.out_proj.forward_batch(&y_scan, seq_len, out);

        let _t4 = std::time::Instant::now();

        static PROF_COUNT: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        let cnt = PROF_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if cnt < 2 {
            eprintln!("    [mamba2 L0-{}] in_proj={:.1}ms conv1d={:.1}ms ssd_scan={:.1}ms gate+norm+out={:.1}ms total={:.1}ms",
                if cnt == 0 { "fwd" } else { "bwd" },
                _t_proj.as_secs_f64() * 1000.0,
                _t_conv.as_secs_f64() * 1000.0,
                _t_scan.as_secs_f64() * 1000.0,
                (_t4 - _t3).as_secs_f64() * 1000.0 + (_t2 - _t1).as_secs_f64() * 1000.0 - _t_conv.as_secs_f64() * 1000.0,
                (_t4 - _t0).as_secs_f64() * 1000.0,
            );
        }
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
        in_proj_rank: usize,
    ) -> Result<Self> {
        Ok(Self {
            fwd: Mamba2Block::load_bmmq(tensors, &format!("{}.fwd", prefix),
                d_model, d_state, d_conv, expand, headdim, ngroups, in_proj_rank)?,
            bwd: Mamba2Block::load_bmmq(tensors, &format!("{}.bwd", prefix),
                d_model, d_state, d_conv, expand, headdim, ngroups, in_proj_rank)?,
        })
    }
}
