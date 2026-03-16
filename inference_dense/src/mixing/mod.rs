//! Mixing Layer trait 및 레지스트리

pub mod rwkv;
pub mod fnet;
pub mod tcn;
pub mod retnet;
pub mod mamba;
pub mod xlstm;

use crate::common::BatchBufs;

/// 교체 가능한 토큰 믹싱 레이어 trait
pub trait MixingLayer {
    /// 배치 forward: x (seq_len * d_model) → out (seq_len * d_model)
    fn forward_batch(
        &self, x: &[f32], seq_len: usize, d_model: usize,
        out: &mut [f32], bufs: &mut BatchBufs,
    );

    /// 양방향 forward: fwd + bwd scan → element-wise addition
    /// 기본 구현: 순차 모델용 (FNet, TCN은 override)
    fn forward_bidirectional(
        &self, x: &[f32], seq_len: usize, d_model: usize,
        out: &mut [f32], bufs: &mut BatchBufs,
        rev_buf: &mut Vec<f32>, fwd_buf: &mut Vec<f32>, bwd_buf: &mut Vec<f32>,
    ) {
        let total = seq_len * d_model;
        fwd_buf.resize(total, 0.0);
        bwd_buf.resize(total, 0.0);
        rev_buf.resize(total, 0.0);

        // Forward scan
        self.forward_batch(x, seq_len, d_model, fwd_buf, bufs);

        // Reverse input
        for t in 0..seq_len {
            let src = (seq_len - 1 - t) * d_model;
            let dst = t * d_model;
            rev_buf[dst..dst + d_model].copy_from_slice(&x[src..src + d_model]);
        }

        // Backward scan (same function, reversed input)
        self.forward_batch(rev_buf, seq_len, d_model, bwd_buf, bufs);

        // Reverse backward output + add to forward
        for t in 0..seq_len {
            let fwd_base = t * d_model;
            let bwd_base = (seq_len - 1 - t) * d_model;
            for d in 0..d_model {
                out[fwd_base + d] = fwd_buf[fwd_base + d] + bwd_buf[bwd_base + d];
            }
        }
    }
}
