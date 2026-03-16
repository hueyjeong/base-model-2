//! BiRWKV mixing layer — RWKV-6 양방향 스캔
//!
//! 기존 wkv6_scan_avx2 C 커널 재사용.
//! State: n_heads × headdim × headdim = 8×32×32 = 8KB (L1 적중)

use std::collections::HashMap;
use anyhow::Result;

use crate::bmmq::TensorData;
use crate::common::*;
use super::MixingLayer;

/// 단방향 RWKV-6 TimeMix
pub struct RWKV6TimeMix {
    // 프로젝션 (BitLinear)
    r_proj: BitLinear,
    k_proj: BitLinear,
    v_proj: BitLinear,
    o_proj: BitLinear,
    g_proj: Linear,  // gating (non-ternary)

    // LoRA decay
    w_base: Vec<f32>,         // (d_model,) — base decay
    w_lora_down: Linear,      // d_model → lora_rank
    w_lora_up: Linear,        // lora_rank → d_model

    // in-context bonus
    u_param: Vec<f32>,        // (d_model,)

    n_heads: usize,
    headdim: usize,
    d_model: usize,
}

impl RWKV6TimeMix {
    pub fn load_bmmq(
        tensors: &mut HashMap<String, TensorData>,
        prefix: &str, n_heads: usize, headdim: usize,
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
            u_param: bmmq_take_f32(tensors, &format!("{}.u", prefix))?,
            n_heads, headdim, d_model,
        })
    }
}

impl MixingLayer for RWKV6TimeMix {
    fn forward_batch(
        &self, x: &[f32], seq_len: usize, d_model: usize,
        out: &mut [f32], bufs: &mut BatchBufs,
    ) {
        let total = seq_len * d_model;

        // 프로젝션 버퍼
        let mut r = vec![0.0f32; total];
        let mut k = vec![0.0f32; total];
        let mut v = vec![0.0f32; total];
        let mut g = vec![0.0f32; total];
        let mut w = vec![0.0f32; total];
        let mut o_buf = vec![0.0f32; total];

        // 프로젝션
        self.r_proj.forward_batch(x, seq_len, &mut r, bufs);
        self.k_proj.forward_batch(x, seq_len, &mut k, bufs);
        self.v_proj.forward_batch(x, seq_len, &mut v, bufs);
        self.g_proj.forward_batch(x, seq_len, &mut g, bufs);

        // Data-dependent decay: w = w_base + w_lora_up(tanh(w_lora_down(x)))
        let lora_rank = self.w_lora_down.out_dim;
        let mut lora_mid = vec![0.0f32; seq_len * lora_rank];
        self.w_lora_down.forward_batch(x, seq_len, &mut lora_mid, bufs);
        // tanh
        for v in lora_mid.iter_mut() { *v = v.tanh(); }
        let mut lora_out = vec![0.0f32; total];
        self.w_lora_up.forward_batch(&lora_mid, seq_len, &mut lora_out, bufs);
        // w = -softplus(w_base + lora_out)
        for t in 0..seq_len {
            for d in 0..d_model {
                let idx = t * d_model + d;
                w[idx] = -softplus_scalar(self.w_base[d] + lora_out[idx]);
            }
        }

        // WKV6 scan (C 커널)
        let mut state = vec![0.0f32; self.n_heads * self.headdim * self.headdim];
        unsafe {
            wkv6_scan_avx2(
                r.as_ptr(), k.as_ptr(), v.as_ptr(), w.as_ptr(),
                self.u_param.as_ptr(), o_buf.as_mut_ptr(), state.as_mut_ptr(),
                seq_len as i32, self.n_heads as i32, self.headdim as i32, d_model as i32,
            );
        }

        // Gating: out = silu(g) * o_buf
        for i in 0..total {
            out[i] = silu_scalar(g[i]) * o_buf[i];
        }

        // Output projection
        let mut final_out = vec![0.0f32; total];
        self.o_proj.forward_batch(out, seq_len, &mut final_out, bufs);
        out.copy_from_slice(&final_out);
    }
}

/// 양방향 RWKV
pub struct BiRWKV {
    pub fwd: RWKV6TimeMix,
    pub bwd: RWKV6TimeMix,
}

impl BiRWKV {
    pub fn load_bmmq(
        tensors: &mut HashMap<String, TensorData>,
        prefix: &str, n_heads: usize, headdim: usize,
    ) -> Result<Self> {
        Ok(Self {
            fwd: RWKV6TimeMix::load_bmmq(tensors, &format!("{}.forward_rwkv", prefix), n_heads, headdim)?,
            bwd: RWKV6TimeMix::load_bmmq(tensors, &format!("{}.backward_rwkv", prefix), n_heads, headdim)?,
        })
    }
}
