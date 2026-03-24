//! DenseEditor 실제 추론 — BMMQ 모델 로드 → 토큰 입력 → 태그 예측
//!
//! stdin에서 JSON Lines (`{"ids": [2, 45, ...]}`) 읽고
//! stdout으로 JSON Lines (`{"tags": [0, 0, 305, ...]}`) 출력.
//!
//! --temperature T: Gumbel noise 기반 stochastic sampling (consensus 실험용)

use std::collections::HashMap;
use std::io::{self, BufRead, Write};

use anyhow::{ Context, Result};
use serde::{Deserialize, Serialize};

use crate::bmmq::{self, TensorData};
use crate::common::*;
use crate::config::DenseEditorConfig;
use crate::mixing::MixingLayer;
use crate::mixing::mamba2::{BiMamba2, Mamba2Block};

// ── Xoshiro256++ PRNG (빠르고 고품질) ─────────────────

struct Rng {
    s: [u64; 4],
}

impl Rng {
    fn new(seed: u64) -> Self {
        // SplitMix64로 초기 상태 생성
        let mut z = seed;
        let mut s = [0u64; 4];
        for i in 0..4 {
            z = z.wrapping_add(0x9e3779b97f4a7c15);
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
            s[i] = z ^ (z >> 31);
        }
        Self { s }
    }

    #[inline]
    fn next_u64(&mut self) -> u64 {
        let result = (self.s[0].wrapping_add(self.s[3]))
            .rotate_left(23)
            .wrapping_add(self.s[0]);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        result
    }

    /// (0, 1) 범위 f64 — Gumbel noise용
    #[inline]
    fn next_f64(&mut self) -> f64 {
        let u = self.next_u64();
        // (0, 1) open interval: 정밀도 52비트
        ((u >> 11) as f64 + 0.5) * (1.0 / (1u64 << 53) as f64)
    }

    /// Gumbel(0,1) noise: -ln(-ln(U)), U ~ Uniform(0,1)
    #[inline]
    fn gumbel(&mut self) -> f32 {
        let u = self.next_f64();
        (-(-u.ln()).ln()) as f32
    }
}

// ── I/O 포맷 ─────────────────────────────────────────

#[derive(Deserialize)]
struct InputLine {
    ids: Vec<u32>,
}

#[derive(Serialize)]
struct OutputLine {
    tags: Vec<u32>,
}

// ── Fused BitNetFFN (F32 또는 Ternary matmul) ──────────

struct FusedBitNetFFN {
    gate_up_proj: Projection,  // d_model → 2*d_ff
    down_proj: Projection,     // d_ff → d_model
    d_ff: usize,
}

impl FusedBitNetFFN {
    fn load_bmmq(
        tensors: &mut HashMap<String, TensorData>, prefix: &str,
    ) -> Result<Self> {
        let gate_up = Projection::load_bmmq(tensors, &format!("{}.gate_up_proj", prefix))?;
        let d_ff = gate_up.out_dim() / 2;
        Ok(Self {
            gate_up_proj: gate_up,
            down_proj: Projection::load_bmmq(tensors, &format!("{}.down_proj", prefix))?,
            d_ff,
        })
    }

    fn forward_batch(
        &self, x: &[f32], seq_len: usize,
        out: &mut [f32], _bufs: &mut BatchBufs,
        gu_buf: &mut Vec<f32>, mid_buf: &mut Vec<f32>,
    ) {
        let d_in = self.gate_up_proj.in_dim();
        let dff2 = self.gate_up_proj.out_dim();
        let dff = self.d_ff;

        // BitLinear의 LayerNorm(no affine) — gu_buf를 normed 버퍼로 재사용
        gu_buf.resize(seq_len * d_in, 0.0);
        for t in 0..seq_len {
            layer_norm_no_affine(&x[t*d_in..(t+1)*d_in], &mut gu_buf[t*d_in..(t+1)*d_in], 1e-5);
        }

        // matmul (gate_up) — normed → mid_buf로 출력
        mid_buf.resize(seq_len * dff2, 0.0);
        self.gate_up_proj.forward_batch(gu_buf, seq_len, mid_buf);

        // relu(gate) * up → gu_buf 재사용 (dff 크기)
        gu_buf.resize(seq_len * dff, 0.0);
        for t in 0..seq_len {
            for i in 0..dff {
                let gate = mid_buf[t * dff2 + i];
                let up = mid_buf[t * dff2 + dff + i];
                gu_buf[t * dff + i] = relu_scalar(gate) * up;
            }
        }

        // down_proj: in-place LayerNorm on gu_buf, then matmul
        let d_mid = self.down_proj.in_dim();
        for t in 0..seq_len {
            let sl = &mut gu_buf[t*d_mid..(t+1)*d_mid];
            let n = sl.len();
            let mut sum = 0.0f32;
            for &v in sl.iter() { sum += v; }
            let mean = sum / n as f32;
            let mut var_sum = 0.0f32;
            for &v in sl.iter() { let d = v - mean; var_sum += d * d; }
            let inv_std = (var_sum / n as f32 + 1e-5).sqrt().recip();
            for v in sl.iter_mut() { *v = (*v - mean) * inv_std; }
        }
        self.down_proj.forward_batch(gu_buf, seq_len, out);
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
    tag_head: Projection,
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
        let in_proj_rank = if config.bitlinear_mamba {
            config.mamba2_in_proj_rank
        } else {
            0
        };

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
                d, ds, d_conv, expand, hd, ng, in_proj_rank,
            )?;
            let norm2 = RMSNorm::load_bmmq(&mut tensors, &format!("{}.norm2", prefix), eps as f64)?;
            let ffn = FusedBitNetFFN::load_bmmq(&mut tensors, &format!("{}.ffn", prefix))?;
            layers.push(DenseEditorLayer { norm1, mixing, norm2, ffn });
        }

        let final_norm = RMSNorm::load_bmmq(&mut tensors, "final_norm", eps as f64)?;
        let tag_head = Projection::load_bmmq(&mut tensors, "tag_head")?;

        // 남은 텐서 보고 (tag_head.norm.weight 등은 BitLinear에 포함되지 않음 — 무시)
        if !tensors.is_empty() {
            eprintln!("  미사용 텐서 {}개 (norm 등): {:?}",
                tensors.len(),
                tensors.keys().take(5).collect::<Vec<_>>());
        }

        eprintln!("  모델 로드 완료 (embed_scale={:.1})", embed_scale);

        Ok(Self { embedding, embed_scale, layers, final_norm, tag_head, d_model: d, n_tags })
    }

    /// Forward pass: input_ids → logits (비싼 부분)
    /// 반환: (logits, seq_len_orig)
    fn forward_logits(&self, input_ids: &[u32]) -> Vec<f32> {
        let sl_orig = input_ids.len();
        let d = self.d_model;
        let vocab_size = self.embedding.len() / d;
        let sl = sl_orig;

        // Embedding lookup + scale
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
        let mut x_rev = vec![0.0f32; sl * d];
        let mut fwd_out = vec![0.0f32; sl * d];
        let mut bwd_out = vec![0.0f32; sl * d];

        let mut total_mixing = 0.0f64;
        let mut total_ffn = 0.0f64;

        for (_layer_idx, layer) in self.layers.iter().enumerate() {
            let t_mix = std::time::Instant::now();
            for t in 0..sl {
                layer.norm1.forward(&x[t*d..(t+1)*d], &mut nm[t*d..(t+1)*d]);
            }
            layer.mixing.fwd.forward_batch(&nm, sl, d, &mut fwd_out, &mut bufs);
            for t in 0..sl {
                x_rev[t*d..(t+1)*d].copy_from_slice(&nm[(sl-1-t)*d..(sl-t)*d]);
            }
            layer.mixing.bwd.forward_batch(&x_rev, sl, d, &mut bwd_out, &mut bufs);
            for t in 0..sl {
                for i in 0..d {
                    mo[t*d+i] = fwd_out[t*d+i] + bwd_out[(sl-1-t)*d+i];
                }
            }
            for i in 0..sl*d { x[i] += mo[i]; }
            total_mixing += t_mix.elapsed().as_secs_f64();

            let t_ffn = std::time::Instant::now();
            for t in 0..sl {
                layer.norm2.forward(&x[t*d..(t+1)*d], &mut nm[t*d..(t+1)*d]);
            }
            layer.ffn.forward_batch(&nm, sl, &mut fo, &mut bufs, &mut gu_buf, &mut mid_buf);
            for i in 0..sl*d { x[i] += fo[i]; }
            total_ffn += t_ffn.elapsed().as_secs_f64();
        }

        static PROF_FWD: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        let cnt = PROF_FWD.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if cnt < 2 {
            eprintln!("  [profile] mixing={:.0}ms, ffn={:.0}ms, total={:.0}ms (seq={})",
                total_mixing * 1000.0, total_ffn * 1000.0,
                (total_mixing + total_ffn) * 1000.0, sl_orig);
        }

        // Final norm → tag head
        for t in 0..sl {
            self.final_norm.forward(&x[t*d..(t+1)*d], &mut nm[t*d..(t+1)*d]);
        }
        let mut nm_tag = vec![0.0f32; sl * d];
        for t in 0..sl {
            layer_norm_no_affine(&nm[t*d..(t+1)*d], &mut nm_tag[t*d..(t+1)*d], 1e-5);
        }
        let mut logits = vec![0.0f32; sl * self.n_tags];
        self.tag_head.forward_batch(&nm_tag, sl, &mut logits);
        logits
    }

    /// Logits → tags (거의 무료: Gumbel noise + argmax)
    fn sample_tags(&self, logits: &[f32], sl: usize, rng: Option<&mut Rng>, temperature: f32) -> Vec<u32> {
        let mut tags = vec![0u32; sl];
        let use_gumbel = temperature > 0.0 && rng.is_some();

        if use_gumbel {
            let rng = rng.unwrap();
            let inv_t = 1.0 / temperature;
            for t in 0..sl {
                let base = t * self.n_tags;
                let mut max_val = logits[base] * inv_t + rng.gumbel();
                let mut max_idx = 0u32;
                for i in 1..self.n_tags {
                    let v = logits[base + i] * inv_t + rng.gumbel();
                    if v > max_val {
                        max_val = v;
                        max_idx = i as u32;
                    }
                }
                tags[t] = max_idx;
            }
        } else {
            for t in 0..sl {
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
        }
        tags
    }

    /// N-sample majority vote: logits에서 N회 Gumbel → min_agree개 동의 시 채택
    /// forward 1회만! sampling N회는 거의 무료.
    fn majority_tags(
        &self, logits: &[f32], sl: usize,
        rngs: &mut [Rng], temperature: f32, min_agree: usize,
    ) -> Vec<u32> {
        let inv_t = 1.0 / temperature;
        let n_samples = rngs.len();
        let mut tags = vec![0u32; sl];
        let mut samples = vec![0u32; n_samples];

        for t in 0..sl {
            let base = t * self.n_tags;

            // N회 독립 Gumbel sampling
            for (s, rng) in rngs.iter_mut().enumerate() {
                let mut max_val = logits[base] * inv_t + rng.gumbel();
                let mut max_idx = 0u32;
                for i in 1..self.n_tags {
                    let v = logits[base + i] * inv_t + rng.gumbel();
                    if v > max_val { max_val = v; max_idx = i as u32; }
                }
                samples[s] = max_idx;
            }

            // Majority vote: 가장 많은 표를 받은 tag가 min_agree 이상이면 채택
            // 대부분의 토큰이 TAG_KEEP(0)이므로 빠른 경로 우선
            let first = samples[0];
            let all_same = samples[1..].iter().all(|&s| s == first);
            if all_same {
                tags[t] = first;
                continue;
            }

            // 느린 경로: 투표 카운트 (n_samples ≤ 16 가정)
            let mut best_tag = 0u32;
            let mut best_count = 0usize;
            for &s in &samples {
                let count = samples.iter().filter(|&&x| x == s).count();
                if count > best_count {
                    best_count = count;
                    best_tag = s;
                }
            }
            tags[t] = if best_count >= min_agree { best_tag } else { 0 };
        }
        tags
    }

    /// 단일 시퀀스 추론 (기존 API 호환)
    fn infer(&self, input_ids: &[u32], rng: Option<&mut Rng>, temperature: f32) -> Vec<u32> {
        let logits = self.forward_logits(input_ids);
        self.sample_tags(&logits, input_ids.len(), rng, temperature)
    }
}

// ── 엔트리포인트 ─────────────────────────────────────

/// 기본 추론: stdin → forward → sample → stdout
pub fn run_infer(config_path: &str, model_path: &str, temperature: f32, seed: u64) -> Result<()> {
    let model = DenseEditorModel::load(config_path, model_path)?;

    let mut rng = if temperature > 0.0 {
        eprintln!("  Gumbel noise 활성 (temperature={}, seed={})", temperature, seed);
        Some(Rng::new(seed))
    } else {
        None
    };

    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut out = io::BufWriter::new(stdout.lock());

    for line in stdin.lock().lines() {
        let line = line.context("stdin 읽기 실패")?;
        let line = line.trim();
        if line.is_empty() { continue; }

        let input: InputLine = serde_json::from_str(line)
            .context("JSON 파싱 실패")?;

        let tags = model.infer(&input.ids, rng.as_mut(), temperature);

        let output = OutputLine { tags };
        serde_json::to_writer(&mut out, &output)?;
        out.write_all(b"\n")?;
        out.flush()?;
    }

    Ok(())
}

/// Consensus 추론: forward 1회 → Gumbel N회 sampling → majority vote
/// n_samples=2, min_agree=2: 원래 consensus (2/2 동의)
/// n_samples=4, min_agree=3: 3/4 majority vote
pub fn run_consensus(
    config_path: &str, model_path: &str,
    temperature: f32, seed: u64,
    n_samples: usize, min_agree: usize,
) -> Result<()> {
    let model = DenseEditorModel::load(config_path, model_path)?;
    eprintln!("  Consensus 모드 (n_samples={}, min_agree={}, temperature={}, seed={})",
        n_samples, min_agree, temperature, seed);

    // N개 독립 RNG 생성
    let mut rngs: Vec<Rng> = (0..n_samples)
        .map(|i| Rng::new(seed.wrapping_add(i as u64 * 0x9E3779B97F4A7C15)))
        .collect();

    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut out = io::BufWriter::new(stdout.lock());

    let mut n_sents = 0u64;
    let mut total_fwd_ms = 0.0f64;
    let mut total_sample_ms = 0.0f64;
    let mut total_tokens = 0u64;

    for line in stdin.lock().lines() {
        let line = line.context("stdin 읽기 실패")?;
        let line = line.trim();
        if line.is_empty() { continue; }

        let input: InputLine = serde_json::from_str(line)
            .context("JSON 파싱 실패")?;

        let sl = input.ids.len();

        // Forward pass (비싼 부분)
        let t_fwd = std::time::Instant::now();
        let logits = model.forward_logits(&input.ids);
        let fwd_ms = t_fwd.elapsed().as_secs_f64() * 1000.0;

        // Majority vote sampling (거의 무료)
        let t_sample = std::time::Instant::now();
        let tags = model.majority_tags(&logits, sl, &mut rngs, temperature, min_agree);
        let sample_ms = t_sample.elapsed().as_secs_f64() * 1000.0;

        total_fwd_ms += fwd_ms;
        total_sample_ms += sample_ms;
        total_tokens += sl as u64;
        n_sents += 1;

        let output = OutputLine { tags };
        serde_json::to_writer(&mut out, &output)?;
        out.write_all(b"\n")?;
        out.flush()?;
    }

    // 타이밍 리포트
    if n_sents > 0 {
        eprintln!("\n  [consensus 타이밍] {}문장, {}토큰", n_sents, total_tokens);
        eprintln!("    forward:   {:.0}ms 합계, {:.1}ms/문장 (비싼 부분)",
            total_fwd_ms, total_fwd_ms / n_sents as f64);
        eprintln!("    sampling:  {:.1}ms 합계, {:.3}ms/문장 ({}회 Gumbel + vote)",
            total_sample_ms, total_sample_ms / n_sents as f64, n_samples);
        eprintln!("    합계:      {:.0}ms, {:.1}ms/문장",
            total_fwd_ms + total_sample_ms,
            (total_fwd_ms + total_sample_ms) / n_sents as f64);
        eprintln!("    sampling 오버헤드: {:.2}%",
            total_sample_ms / total_fwd_ms * 100.0);
    }

    Ok(())
}
