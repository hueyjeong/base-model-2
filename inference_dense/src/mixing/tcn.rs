//! TCN mixing layer — Non-causal dilated depthwise conv + BitLinear pointwise
//!
//! depthwise_conv1d_avx2 C 커널 사용.
//! 본질적으로 양방향 (non-causal symmetric padding).

use std::collections::HashMap;
use anyhow::Result;

use crate::bmmq::TensorData;
use crate::common::*;
use super::MixingLayer;

/// 단일 dilation의 depthwise conv 가중치
struct DepthwiseConv {
    weight: Vec<f32>,  // (d_model, kernel_size) — row-major
    kernel_size: usize,
    dilation: usize,
    d_model: usize,
}

impl DepthwiseConv {
    fn forward(&self, x: &[f32], seq_len: usize, out: &mut [f32]) {
        unsafe {
            depthwise_conv1d_avx2(
                x.as_ptr(),
                self.weight.as_ptr(),
                std::ptr::null(),  // no bias
                out.as_mut_ptr(),
                seq_len as i32, self.d_model as i32,
                self.kernel_size as i32, self.dilation as i32,
            );
        }
    }
}

pub struct TCNMixing {
    convs: Vec<DepthwiseConv>,
    proj: BitLinear,
    d_model: usize,
}

impl TCNMixing {
    pub fn load_bmmq(
        tensors: &mut HashMap<String, TensorData>,
        prefix: &str, d_model: usize, kernel_size: usize, n_dilations: usize,
    ) -> Result<Self> {
        let mut convs = Vec::new();
        for i in 0..n_dilations {
            let key = format!("{}.convs.{}.weight", prefix, i);
            // Conv1d weight: (d_model, 1, kernel_size) in PyTorch → (d_model, kernel_size)
            let weight = bmmq_take_f32(tensors, &key)?;
            convs.push(DepthwiseConv {
                weight,
                kernel_size,
                dilation: 1 << i,
                d_model,
            });
        }
        let proj = BitLinear::load_bmmq(tensors, &format!("{}.proj", prefix))?;
        Ok(Self { convs, proj, d_model })
    }
}

impl MixingLayer for TCNMixing {
    fn forward_batch(
        &self, x: &[f32], seq_len: usize, d_model: usize,
        out: &mut [f32], bufs: &mut BatchBufs,
    ) {
        let total = seq_len * d_model;
        let mut acc = vec![0.0f32; total];
        let mut conv_out = vec![0.0f32; total];

        // 다중 dilation 합산
        for conv in &self.convs {
            conv.forward(x, seq_len, &mut conv_out);
            for i in 0..total {
                acc[i] += conv_out[i];
            }
        }

        // ReLU
        for v in acc.iter_mut() {
            *v = relu_scalar(*v);
        }

        // Pointwise BitLinear projection
        self.proj.forward_batch(&acc, seq_len, out, bufs);
    }

    fn forward_bidirectional(
        &self, x: &[f32], seq_len: usize, d_model: usize,
        out: &mut [f32], bufs: &mut BatchBufs,
        _rev_buf: &mut Vec<f32>, _fwd_buf: &mut Vec<f32>, _bwd_buf: &mut Vec<f32>,
    ) {
        // TCN은 non-causal → 본질적으로 양방향
        self.forward_batch(x, seq_len, d_model, out, bufs);
    }
}
