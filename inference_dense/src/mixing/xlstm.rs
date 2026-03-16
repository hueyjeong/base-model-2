//! BiSLSTM mixing layer — 양방향 sLSTM
//!
//! slstm_scan_avx2 C 커널 사용.
//! State per head: c(scalar) + n(scalar) = 수 바이트 (레지스터 적중)

use std::collections::HashMap;
use anyhow::Result;

use crate::bmmq::TensorData;
use crate::common::*;
use super::MixingLayer;

/// 단방향 sLSTM
pub struct SLSTMScan {
    i_proj: BitLinear,
    f_proj: BitLinear,
    z_proj: BitLinear,
    o_proj: BitLinear,
    d_model: usize,
}

impl SLSTMScan {
    pub fn load_bmmq(
        tensors: &mut HashMap<String, TensorData>,
        prefix: &str, d_model: usize,
    ) -> Result<Self> {
        Ok(Self {
            i_proj: BitLinear::load_bmmq(tensors, &format!("{}.i_proj", prefix))?,
            f_proj: BitLinear::load_bmmq(tensors, &format!("{}.f_proj", prefix))?,
            z_proj: BitLinear::load_bmmq(tensors, &format!("{}.z_proj", prefix))?,
            o_proj: BitLinear::load_bmmq(tensors, &format!("{}.o_proj", prefix))?,
            d_model,
        })
    }
}

impl MixingLayer for SLSTMScan {
    fn forward_batch(
        &self, x: &[f32], seq_len: usize, d_model: usize,
        out: &mut [f32], bufs: &mut BatchBufs,
    ) {
        let total = seq_len * d_model;

        let mut i_gate = vec![0.0f32; total];
        let mut f_gate = vec![0.0f32; total];
        let mut z_gate = vec![0.0f32; total];
        let mut o_gate = vec![0.0f32; total];

        // Gate projections
        self.i_proj.forward_batch(x, seq_len, &mut i_gate, bufs);
        self.f_proj.forward_batch(x, seq_len, &mut f_gate, bufs);
        self.z_proj.forward_batch(x, seq_len, &mut z_gate, bufs);
        self.o_proj.forward_batch(x, seq_len, &mut o_gate, bufs);

        // sLSTM scan (C 커널)
        let mut state_c = vec![0.0f32; d_model];
        let mut state_n = vec![0.0f32; d_model];
        unsafe {
            slstm_scan_avx2(
                i_gate.as_ptr(), f_gate.as_ptr(), z_gate.as_ptr(), o_gate.as_ptr(),
                out.as_mut_ptr(), state_c.as_mut_ptr(), state_n.as_mut_ptr(),
                seq_len as i32, d_model as i32,
            );
        }
    }
}

/// 양방향 sLSTM
pub struct BiSLSTM {
    pub fwd: SLSTMScan,
    pub bwd: SLSTMScan,
}

impl BiSLSTM {
    pub fn load_bmmq(
        tensors: &mut HashMap<String, TensorData>,
        prefix: &str, d_model: usize,
    ) -> Result<Self> {
        Ok(Self {
            fwd: SLSTMScan::load_bmmq(tensors, &format!("{}.fwd", prefix), d_model)?,
            bwd: SLSTMScan::load_bmmq(tensors, &format!("{}.bwd", prefix), d_model)?,
        })
    }
}
