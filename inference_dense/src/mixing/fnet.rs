//! FNet mixing layer — FFT 기반 토큰 믹싱
//!
//! 파라미터 없음, rustfft crate 사용.
//! 2D FFT (seq × feature) → 실수부 추출.

use rustfft::{FftPlanner, num_complex::Complex};

use crate::common::BatchBufs;
use super::MixingLayer;

pub struct FNetMixing {
    d_model: usize,
}

impl FNetMixing {
    pub fn new(d_model: usize) -> Self {
        Self { d_model }
    }
}

impl MixingLayer for FNetMixing {
    fn forward_batch(
        &self, x: &[f32], seq_len: usize, d_model: usize,
        out: &mut [f32], _bufs: &mut BatchBufs,
    ) {
        // 2D FFT: seq 차원 + feature 차원, 실수부 추출
        // Step 1: 각 feature 채널별 seq 차원 FFT
        let mut planner = FftPlanner::<f32>::new();
        let fft_seq = planner.plan_fft_forward(seq_len);
        let fft_feat = planner.plan_fft_forward(d_model);

        // intermediate: complex buffer (seq_len × d_model)
        let mut buf: Vec<Complex<f32>> = vec![Complex::new(0.0, 0.0); seq_len * d_model];

        // x → complex
        for i in 0..seq_len * d_model {
            buf[i] = Complex::new(x[i], 0.0);
        }

        // FFT along seq dimension (per feature channel)
        let mut col_buf = vec![Complex::new(0.0, 0.0); seq_len];
        for d in 0..d_model {
            for t in 0..seq_len {
                col_buf[t] = buf[t * d_model + d];
            }
            fft_seq.process(&mut col_buf);
            for t in 0..seq_len {
                buf[t * d_model + d] = col_buf[t];
            }
        }

        // FFT along feature dimension (per time step)
        let mut row_buf = vec![Complex::new(0.0, 0.0); d_model];
        for t in 0..seq_len {
            let base = t * d_model;
            row_buf[..d_model].copy_from_slice(&buf[base..base + d_model]);
            fft_feat.process(&mut row_buf);
            buf[base..base + d_model].copy_from_slice(&row_buf[..d_model]);
        }

        // 실수부 추출
        for i in 0..seq_len * d_model {
            out[i] = buf[i].re;
        }
    }

    fn forward_bidirectional(
        &self, x: &[f32], seq_len: usize, d_model: usize,
        out: &mut [f32], bufs: &mut BatchBufs,
        _rev_buf: &mut Vec<f32>, _fwd_buf: &mut Vec<f32>, _bwd_buf: &mut Vec<f32>,
    ) {
        // FNet은 본질적으로 양방향 — 단방향 forward가 곧 양방향
        self.forward_batch(x, seq_len, d_model, out, bufs);
    }
}
