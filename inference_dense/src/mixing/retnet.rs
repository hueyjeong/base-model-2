//! BiRetention mixing layer — 양방향 multi-scale retention
//!
//! retention_scan_avx2 C 커널 사용.
//! State: n_heads × headdim × headdim = 8×32×32 = 8KB (L1 적중)

use std::collections::HashMap;
use anyhow::Result;

use crate::bmmq::TensorData;
use crate::common::*;
use super::MixingLayer;

/// 단방향 Retention scan
pub struct RetentionScan {
    q_proj: BitLinear,
    k_proj: BitLinear,
    v_proj: BitLinear,
    o_proj: BitLinear,
    g_proj: Linear,  // gating (SiLU)
    gammas: Vec<f32>,
    n_heads: usize,
    headdim: usize,
    d_model: usize,
}

impl RetentionScan {
    pub fn load_bmmq(
        tensors: &mut HashMap<String, TensorData>,
        prefix: &str, n_heads: usize, headdim: usize,
        gamma_min: f32, gamma_max: f32,
    ) -> Result<Self> {
        let d_model = n_heads * headdim;
        let mut gammas = vec![0.0f32; n_heads];
        for i in 0..n_heads {
            gammas[i] = gamma_min + (gamma_max - gamma_min) * i as f32 / (n_heads - 1).max(1) as f32;
        }
        Ok(Self {
            q_proj: BitLinear::load_bmmq(tensors, &format!("{}.q_proj", prefix))?,
            k_proj: BitLinear::load_bmmq(tensors, &format!("{}.k_proj", prefix))?,
            v_proj: BitLinear::load_bmmq(tensors, &format!("{}.v_proj", prefix))?,
            o_proj: BitLinear::load_bmmq(tensors, &format!("{}.o_proj", prefix))?,
            g_proj: Linear::load_bmmq(tensors, &format!("{}.g_proj", prefix))?,
            gammas, n_heads, headdim, d_model,
        })
    }
}

impl MixingLayer for RetentionScan {
    fn forward_batch(
        &self, x: &[f32], seq_len: usize, d_model: usize,
        out: &mut [f32], bufs: &mut BatchBufs,
    ) {
        let total = seq_len * d_model;

        let mut q = vec![0.0f32; total];
        let mut k = vec![0.0f32; total];
        let mut v = vec![0.0f32; total];
        let mut g = vec![0.0f32; total];
        let mut scan_out = vec![0.0f32; total];

        // 프로젝션
        self.q_proj.forward_batch(x, seq_len, &mut q, bufs);
        self.k_proj.forward_batch(x, seq_len, &mut k, bufs);
        self.v_proj.forward_batch(x, seq_len, &mut v, bufs);
        self.g_proj.forward_batch(x, seq_len, &mut g, bufs);

        // Retention scan (C 커널)
        let mut state = vec![0.0f32; self.n_heads * self.headdim * self.headdim];
        unsafe {
            retention_scan_avx2(
                q.as_ptr(), k.as_ptr(), v.as_ptr(), self.gammas.as_ptr(),
                scan_out.as_mut_ptr(), state.as_mut_ptr(),
                seq_len as i32, self.n_heads as i32, self.headdim as i32,
            );
        }

        // Gating: silu(g) * scan_out
        for i in 0..total {
            scan_out[i] *= silu_scalar(g[i]);
        }

        // Output projection
        self.o_proj.forward_batch(&scan_out, seq_len, out, bufs);
    }
}

/// 양방향 Retention
pub struct BiRetention {
    pub fwd: RetentionScan,
    pub bwd: RetentionScan,
}

impl BiRetention {
    pub fn load_bmmq(
        tensors: &mut HashMap<String, TensorData>,
        prefix: &str, n_heads: usize, headdim: usize,
        gamma_min: f32, gamma_max: f32,
    ) -> Result<Self> {
        Ok(Self {
            fwd: RetentionScan::load_bmmq(tensors, &format!("{}.fwd", prefix), n_heads, headdim, gamma_min, gamma_max)?,
            bwd: RetentionScan::load_bmmq(tensors, &format!("{}.bwd", prefix), n_heads, headdim, gamma_min, gamma_max)?,
        })
    }
}
