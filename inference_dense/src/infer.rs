//! DenseEditor 실제 추론 — BMMQ 모델 로드 → 토큰 입력 → 태그 예측
//!
//! stdin에서 JSON Lines (`{"ids": [2, 45, ...]}`) 읽고
//! stdout으로 JSON Lines (`{"tags": [0, 0, 305, ...]}`) 출력.

use std::collections::HashMap;
use std::io::{self, BufRead, Write};

use anyhow::{ Context, Result};
use serde::{Deserialize, Serialize};

use crate::bmmq::{self, TensorData};
use crate::common::*;
use crate::config::DenseEditorConfig;
use crate::mixing::MixingLayer;
use crate::mixing::mamba2::{BiMamba2, Mamba2Block};

// ── I/O 포맷 ─────────────────────────────────────────

#[derive(Deserialize)]
struct InputLine {
    ids: Vec<u32>,
}

#[derive(Serialize)]
struct OutputLine {
    tags: Vec<u32>,
}

// ── Fused BitNetFFN (FP32 matmul — INT8 양자화 오차 제거) ─────────────

struct FusedBitNetFFN {
    gate_up_proj: LinearF32,  // d_model → 2*d_ff (FP32 weight — ternary 포함)
    down_proj: LinearF32,     // d_ff → d_model
    d_ff: usize,
}

impl FusedBitNetFFN {
    fn load_bmmq(
        tensors: &mut HashMap<String, TensorData>, prefix: &str,
    ) -> Result<Self> {
        let gate_up = LinearF32::load_bmmq(tensors, &format!("{}.gate_up_proj", prefix))?;
        let d_ff = gate_up.out_dim / 2;
        Ok(Self {
            gate_up_proj: gate_up,
            down_proj: LinearF32::load_bmmq(tensors, &format!("{}.down_proj", prefix))?,
            d_ff,
        })
    }

    fn forward_batch(
        &self, x: &[f32], seq_len: usize,
        out: &mut [f32], _bufs: &mut BatchBufs,
        gu_buf: &mut Vec<f32>, mid_buf: &mut Vec<f32>,
    ) {
        let d_in = self.gate_up_proj.in_dim;
        let dff2 = self.gate_up_proj.out_dim;
        let dff = self.d_ff;

        // BitLinear의 LayerNorm(no affine) 적용
        let mut normed = vec![0.0f32; seq_len * d_in];
        for t in 0..seq_len {
            layer_norm_no_affine(&x[t*d_in..(t+1)*d_in], &mut normed[t*d_in..(t+1)*d_in], 1e-5);
        }

        // FP32 matmul (gate_up)
        gu_buf.resize(seq_len * dff2, 0.0);
        self.gate_up_proj.forward_batch(&normed, seq_len, gu_buf);

        // relu(gate) * up
        mid_buf.resize(seq_len * dff, 0.0);
        for t in 0..seq_len {
            for i in 0..dff {
                let gate = gu_buf[t * dff2 + i];
                let up = gu_buf[t * dff2 + dff + i];
                mid_buf[t * dff + i] = relu_scalar(gate) * up;
            }
        }

        // down_proj도 LayerNorm + FP32 matmul
        let d_mid = self.down_proj.in_dim;
        let mut normed_mid = vec![0.0f32; seq_len * d_mid];
        for t in 0..seq_len {
            layer_norm_no_affine(&mid_buf[t*d_mid..(t+1)*d_mid], &mut normed_mid[t*d_mid..(t+1)*d_mid], 1e-5);
        }
        self.down_proj.forward_batch(&normed_mid, seq_len, out);
    }
}

// ── DenseEditor 레이어 ───────────────────────────────

struct DenseEditorLayer {
    norm1: RMSNorm,
    mixing: BiMamba2,
    norm2: RMSNorm,
    ffn: FusedBitNetFFN,
}

// ── 모델 전체 ────────────────────────────────────────

struct DenseEditorModel {
    embedding: Vec<f32>,  // (vocab_size, d_model) — row-major
    embed_scale: f32,
    layers: Vec<DenseEditorLayer>,
    final_norm: RMSNorm,
    tag_head: LinearF32,
    d_model: usize,
    n_tags: usize,
}

impl DenseEditorModel {
    fn load(config_path: &str, model_path: &str) -> Result<Self> {
        let config_str = std::fs::read_to_string(config_path)
            .context("config.json 읽기 실패")?;
        let config: DenseEditorConfig = serde_json::from_str(&config_str)
            .context("config.json 파싱 실패")?;

        let d = config.d_model;
        let nl = config.n_layers;
        let ds = config.mamba2_d_state;
        let hd = config.mamba2_headdim;
        let ng = config.mamba2_ngroups;
        let expand = config.mamba_expand;
        let d_conv = config.mamba_d_conv;
        let n_tags = config.n_tags;
        let eps = config.rms_norm_eps;

        eprintln!("BMMQ 로드: {} (d={}, n_layers={})", model_path, d, nl);
        let mut tensors = bmmq::load_bmmq(model_path)?;
        eprintln!("  텐서 {}개 로드됨", tensors.len());

        // Embedding
        let embedding = bmmq_take_f32(&mut tensors, "embedding.weight")?;
        let embed_scale = (d as f32).sqrt();

        // Layers
        let mut layers = Vec::with_capacity(nl);
        for i in 0..nl {
            let prefix = format!("layers.{}", i);
            let norm1 = RMSNorm::load_bmmq(&mut tensors, &format!("{}.norm1", prefix), eps as f64)?;
            let mixing = BiMamba2::load_bmmq(
                &mut tensors, &format!("{}.mixing", prefix),
                d, ds, d_conv, expand, hd, ng,
            )?;
            let norm2 = RMSNorm::load_bmmq(&mut tensors, &format!("{}.norm2", prefix), eps as f64)?;
            let ffn = FusedBitNetFFN::load_bmmq(&mut tensors, &format!("{}.ffn", prefix))?;
            layers.push(DenseEditorLayer { norm1, mixing, norm2, ffn });
        }

        let final_norm = RMSNorm::load_bmmq(&mut tensors, "final_norm", eps as f64)?;
        let tag_head = LinearF32::load_bmmq(&mut tensors, "tag_head")?;

        // 남은 텐서 보고 (tag_head.norm.weight 등은 BitLinear에 포함되지 않음 — 무시)
        if !tensors.is_empty() {
            eprintln!("  미사용 텐서 {}개 (norm 등): {:?}",
                tensors.len(),
                tensors.keys().take(5).collect::<Vec<_>>());
        }

        eprintln!("  모델 로드 완료 (embed_scale={:.1})", embed_scale);

        Ok(Self { embedding, embed_scale, layers, final_norm, tag_head, d_model: d, n_tags })
    }

    /// 단일 시퀀스 추론: input_ids → predicted tags
    fn infer(&self, input_ids: &[u32]) -> Vec<u32> {
        let sl_orig = input_ids.len();
        let d = self.d_model;
        let vocab_size = self.embedding.len() / d;
        let chunk_size = 256;

        // 패딩 없이 원래 길이 사용 (C SSD 커널이 seq_len으로 패딩 처리)
        let sl = sl_orig;

        // Embedding lookup + scale (패딩 위치는 0)
        let mut x = vec![0.0f32; sl * d];
        for t in 0..sl_orig {
            let tok = input_ids[t] as usize;
            if tok < vocab_size {
                let src = tok * d;
                for i in 0..d {
                    x[t * d + i] = self.embedding[src + i] * self.embed_scale;
                }
            }
        }

        let mut bufs = BatchBufs::new(sl * d * 4);
        let mut nm = vec![0.0f32; sl * d];
        let mut mo = vec![0.0f32; sl * d];
        let mut fo = vec![0.0f32; sl * d];
        let mut gu_buf = Vec::new();
        let mut mid_buf = Vec::new();

        // 양방향 BiMamba2용 버퍼
        let mut x_rev = vec![0.0f32; sl * d];
        let mut fwd_out = vec![0.0f32; sl * d];
        let mut bwd_out = vec![0.0f32; sl * d];

        // Layer loop
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            // Mixing sub-layer: RMSNorm → BiMamba2 → residual
            for t in 0..sl {
                layer.norm1.forward(&x[t*d..(t+1)*d], &mut nm[t*d..(t+1)*d]);
            }

            if layer_idx == 0 {
                let mean: f32 = x.iter().sum::<f32>() / x.len() as f32;
                let std: f32 = (x.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / x.len() as f32).sqrt();
                eprintln!("  emb: mean={:.6} std={:.6} first5=[{:.6},{:.6},{:.6},{:.6},{:.6}]",
                    mean, std, x[0], x[1], x[2], x[3], x[4]);
                let mean_n: f32 = nm.iter().sum::<f32>() / nm.len() as f32;
                let std_n: f32 = (nm.iter().map(|v| (v - mean_n).powi(2)).sum::<f32>() / nm.len() as f32).sqrt();
                eprintln!("  norm1: mean={:.6} std={:.6} first5=[{:.6},{:.6},{:.6},{:.6},{:.6}]",
                    mean_n, std_n, nm[0], nm[1], nm[2], nm[3], nm[4]);
            }

            // BiMamba2: forward + backward(reversed) + add
            layer.mixing.fwd.forward_batch(&nm, sl, d, &mut fwd_out, &mut bufs);

            // Reverse input for backward direction
            for t in 0..sl {
                x_rev[t*d..(t+1)*d].copy_from_slice(&nm[(sl-1-t)*d..(sl-t)*d]);
            }
            layer.mixing.bwd.forward_batch(&x_rev, sl, d, &mut bwd_out, &mut bufs);

            // Reverse backward output and add with forward
            for t in 0..sl {
                for i in 0..d {
                    mo[t*d+i] = fwd_out[t*d+i] + bwd_out[(sl-1-t)*d+i];
                }
            }

            if layer_idx == 0 {
                // bwd의 scan 출력 (norm/gate/outproj 전) — Mamba2Block.forward_batch는
                // 이미 norm+gate+outproj를 포함하므로, 직접 비교 불가.
                // 대신 최종 bwd_out(=after outproj) 비교
                for t in [0usize, 5, sl-1] {
                    eprintln!("  bwd_final[{},:3]=[{:.6},{:.6},{:.6}]",
                        t, bwd_out[t*d], bwd_out[t*d+1], bwd_out[t*d+2]);
                }
            }

            // Residual add
            for i in 0..sl*d { x[i] += mo[i]; }

            // FFN sub-layer: RMSNorm → FFN → residual
            for t in 0..sl {
                layer.norm2.forward(&x[t*d..(t+1)*d], &mut nm[t*d..(t+1)*d]);
            }
            layer.ffn.forward_batch(&nm, sl, &mut fo, &mut bufs, &mut gu_buf, &mut mid_buf);
            for i in 0..sl*d { x[i] += fo[i]; }
        }

        // Final norm
        for t in 0..sl {
            self.final_norm.forward(&x[t*d..(t+1)*d], &mut nm[t*d..(t+1)*d]);
        }

        // DEBUG: final hidden state
        eprintln!("  final_x[0,:5]=[{:.4},{:.4},{:.4},{:.4},{:.4}]", nm[0], nm[1], nm[2], nm[3], nm[4]);

        // Tag head: LayerNorm + FP32 matmul (d_model → n_tags)
        let mut nm_tag = vec![0.0f32; sl * d];
        for t in 0..sl {
            layer_norm_no_affine(&nm[t*d..(t+1)*d], &mut nm_tag[t*d..(t+1)*d], 1e-5);
        }
        let mut logits = vec![0.0f32; sl * self.n_tags];
        self.tag_head.forward_batch(&nm_tag, sl, &mut logits);

        // Argmax per token (원래 길이만)
        let mut tags = vec![0u32; sl_orig];
        for t in 0..sl_orig {
            let base = t * self.n_tags;
            let mut max_val = logits[base];
            let mut max_idx = 0u32;
            for i in 1..self.n_tags {
                if logits[base + i] > max_val {
                    max_val = logits[base + i];
                    max_idx = i as u32;
                }
            }
            tags[t] = max_idx;
        }

        tags
    }
}

// ── 엔트리포인트 ─────────────────────────────────────

pub fn run_infer(config_path: &str, model_path: &str) -> Result<()> {
    let model = DenseEditorModel::load(config_path, model_path)?;

    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut out = io::BufWriter::new(stdout.lock());

    for line in stdin.lock().lines() {
        let line = line.context("stdin 읽기 실패")?;
        let line = line.trim();
        if line.is_empty() { continue; }

        let input: InputLine = serde_json::from_str(line)
            .context("JSON 파싱 실패")?;

        let tags = model.infer(&input.ids);

        let output = OutputLine { tags };
        serde_json::to_writer(&mut out, &output)?;
        out.write_all(b"\n")?;
        out.flush()?;
    }

    Ok(())
}
