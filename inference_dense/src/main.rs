//! DenseEditor CPU 추론 엔진 — 6종 mixing layer 벤치마크
//!
//! 사용법:
//!   cargo run --release -- --config config.json --model model.bmmq --benchmark
//!   cargo run --release -- --benchmark-dummy --mixing-type all --seq-len 2048

mod bmmq;
mod config;
mod common;
mod mixing;
mod bench;

use std::time::Instant;
use clap::Parser;
use anyhow::Result;

use config::DenseEditorConfig;
use common::*;
use mixing::MixingLayer;

#[derive(Parser)]
#[command(name = "dense-editor-inference")]
struct Args {
    /// 설정 파일 경로
    #[arg(long)]
    config: Option<String>,

    /// BMMQ 모델 파일 경로
    #[arg(long)]
    model: Option<String>,

    /// 더미 가중치 벤치마크 모드 (scan 커널만)
    #[arg(long)]
    benchmark_dummy: bool,

    /// 전체 모델 벤치마크 (projection + scan + FFN)
    #[arg(long)]
    benchmark_full: bool,

    /// Mixing type (all|rwkv|fnet|tcn|retnet|mamba|mamba2|xlstm)
    #[arg(long, default_value = "all")]
    mixing_type: String,

    /// 시퀀스 길이
    #[arg(long, default_value = "512")]
    seq_len: usize,

    /// 모델 히든 차원 (headdim=32의 배수)
    #[arg(long, default_value = "256")]
    d_model: usize,

    /// Warmup 횟수
    #[arg(long, default_value = "5")]
    warmup: usize,

    /// 측정 횟수
    #[arg(long, default_value = "50")]
    n_runs: usize,
}

/// 더미 가중치로 벤치마크 — BMMQ 파일 없이 순수 연산 성능 측정
fn benchmark_dummy(mixing_type: &str, seq_len: usize, warmup: usize, n_runs: usize) {
    // 더미 데이터 (d_model=256)
    let d_model = 256;
    let input: Vec<f32> = (0..seq_len * d_model)
        .map(|i| ((i as f32 * 0.01).sin() * 0.1))
        .collect();
    let mut output = vec![0.0f32; seq_len * d_model];
    let mut bufs = BatchBufs::new(seq_len * d_model);

    // Mixing 커널만 벤치마크 (프로젝션 제외)
    match mixing_type {
        "fnet" => {
            let layer = mixing::fnet::FNetMixing::new(d_model);
            let mut rev = vec![]; let mut fwd = vec![]; let mut bwd = vec![];
            // warmup
            for _ in 0..warmup {
                layer.forward_bidirectional(&input, seq_len, d_model, &mut output, &mut bufs, &mut rev, &mut fwd, &mut bwd);
            }
            // bench
            let mut latencies = Vec::with_capacity(n_runs);
            for _ in 0..n_runs {
                let t0 = Instant::now();
                layer.forward_bidirectional(&input, seq_len, d_model, &mut output, &mut bufs, &mut rev, &mut fwd, &mut bwd);
                latencies.push(t0.elapsed().as_secs_f64() * 1000.0);
            }
            report("fnet", seq_len, d_model, &latencies);
        }
        "slstm_scan" | "xlstm" => {
            // 더미 gate 데이터
            let i_gate = vec![0.1f32; seq_len * d_model];
            let f_gate = vec![0.5f32; seq_len * d_model];
            let z_gate = vec![0.0f32; seq_len * d_model];
            let o_gate = vec![0.3f32; seq_len * d_model];
            let mut state_c = vec![0.0f32; d_model];
            let mut state_n = vec![0.0f32; d_model];

            // warmup
            for _ in 0..warmup {
                state_c.fill(0.0); state_n.fill(0.0);
                unsafe {
                    slstm_scan_avx2(
                        i_gate.as_ptr(), f_gate.as_ptr(), z_gate.as_ptr(), o_gate.as_ptr(),
                        output.as_mut_ptr(), state_c.as_mut_ptr(), state_n.as_mut_ptr(),
                        seq_len as i32, d_model as i32,
                    );
                }
            }
            let mut latencies = Vec::with_capacity(n_runs);
            for _ in 0..n_runs {
                state_c.fill(0.0); state_n.fill(0.0);
                let t0 = Instant::now();
                unsafe {
                    slstm_scan_avx2(
                        i_gate.as_ptr(), f_gate.as_ptr(), z_gate.as_ptr(), o_gate.as_ptr(),
                        output.as_mut_ptr(), state_c.as_mut_ptr(), state_n.as_mut_ptr(),
                        seq_len as i32, d_model as i32,
                    );
                }
                latencies.push(t0.elapsed().as_secs_f64() * 1000.0);
            }
            report("slstm_scan", seq_len, d_model, &latencies);
        }
        "retention_scan" | "retnet" => {
            let n_heads = 8;
            let headdim = 32;
            let q = vec![0.1f32; seq_len * d_model];
            let k = vec![0.1f32; seq_len * d_model];
            let v = vec![0.1f32; seq_len * d_model];
            let gammas: Vec<f32> = (0..n_heads).map(|i| 0.8 + 0.199 * i as f32 / 7.0).collect();
            let mut state = vec![0.0f32; n_heads * headdim * headdim];

            for _ in 0..warmup {
                state.fill(0.0);
                unsafe {
                    retention_scan_avx2(
                        q.as_ptr(), k.as_ptr(), v.as_ptr(), gammas.as_ptr(),
                        output.as_mut_ptr(), state.as_mut_ptr(),
                        seq_len as i32, n_heads as i32, headdim as i32,
                    );
                }
            }
            let mut latencies = Vec::with_capacity(n_runs);
            for _ in 0..n_runs {
                state.fill(0.0);
                let t0 = Instant::now();
                unsafe {
                    retention_scan_avx2(
                        q.as_ptr(), k.as_ptr(), v.as_ptr(), gammas.as_ptr(),
                        output.as_mut_ptr(), state.as_mut_ptr(),
                        seq_len as i32, n_heads as i32, headdim as i32,
                    );
                }
                latencies.push(t0.elapsed().as_secs_f64() * 1000.0);
            }
            report("retention_scan", seq_len, d_model, &latencies);
        }
        "mamba_scan" | "mamba" => {
            let d_inner = 512;
            let d_state = 16;
            let delta = vec![0.1f32; seq_len * d_inner];
            let B = vec![0.1f32; seq_len * d_state];
            let C = vec![0.1f32; seq_len * d_state];
            let x_inner = vec![0.1f32; seq_len * d_inner];
            let A = vec![-1.0f32; d_inner * d_state];
            let D_skip = vec![1.0f32; d_inner];
            let mut y = vec![0.0f32; seq_len * d_inner];
            let mut state = vec![0.0f32; d_inner * d_state];

            for _ in 0..warmup {
                state.fill(0.0);
                unsafe {
                    mamba_scan_avx2(
                        delta.as_ptr(), B.as_ptr(), C.as_ptr(), x_inner.as_ptr(),
                        A.as_ptr(), D_skip.as_ptr(), y.as_mut_ptr(), state.as_mut_ptr(),
                        seq_len as i32, d_inner as i32, d_state as i32,
                    );
                }
            }
            let mut latencies = Vec::with_capacity(n_runs);
            for _ in 0..n_runs {
                state.fill(0.0);
                let t0 = Instant::now();
                unsafe {
                    mamba_scan_avx2(
                        delta.as_ptr(), B.as_ptr(), C.as_ptr(), x_inner.as_ptr(),
                        A.as_ptr(), D_skip.as_ptr(), y.as_mut_ptr(), state.as_mut_ptr(),
                        seq_len as i32, d_inner as i32, d_state as i32,
                    );
                }
                latencies.push(t0.elapsed().as_secs_f64() * 1000.0);
            }
            report("mamba_scan", seq_len, d_model, &latencies);
        }
        "mamba2_scan" | "mamba2" => {
            let nheads = 8; // d_model=256, expand=2, headdim=64 → d_inner=512, nheads=8
            let headdim = 64;
            let ngroups = 1;
            let d_inner = nheads * headdim;

            // d_state=16, 64, 128 비교
            for &d_state in &[16usize, 64, 128] {
                let x_inner = vec![0.1f32; seq_len * d_inner];
                let b_ssm = vec![0.1f32; seq_len * ngroups * d_state];
                let c_ssm = vec![0.1f32; seq_len * ngroups * d_state];
                let decay: Vec<f32> = (0..nheads).map(|i| 0.9 + 0.09 * i as f32 / nheads as f32).collect();
                let d_skip = vec![1.0f32; nheads];
                let mut y = vec![0.0f32; seq_len * d_inner];
                let mut state = vec![0.0f32; nheads * d_state * headdim];

                for _ in 0..warmup {
                    state.fill(0.0);
                    unsafe {
                        mamba2_scan_avx2(
                            x_inner.as_ptr(), b_ssm.as_ptr(), c_ssm.as_ptr(),
                            decay.as_ptr(), d_skip.as_ptr(),
                            y.as_mut_ptr(), state.as_mut_ptr(),
                            seq_len as i32, nheads as i32, headdim as i32,
                            d_state as i32, ngroups as i32,
                        );
                    }
                }
                let mut latencies = Vec::with_capacity(n_runs);
                for _ in 0..n_runs {
                    state.fill(0.0);
                    let t0 = Instant::now();
                    unsafe {
                        mamba2_scan_avx2(
                            x_inner.as_ptr(), b_ssm.as_ptr(), c_ssm.as_ptr(),
                            decay.as_ptr(), d_skip.as_ptr(),
                            y.as_mut_ptr(), state.as_mut_ptr(),
                            seq_len as i32, nheads as i32, headdim as i32,
                            d_state as i32, ngroups as i32,
                        );
                    }
                    latencies.push(t0.elapsed().as_secs_f64() * 1000.0);
                }
                report(&format!("mamba2_ds{}", d_state), seq_len, d_model, &latencies);
            }
        }
        "depthwise_conv" | "tcn" => {
            let kernel_size = 7;
            let weight = vec![0.1f32; d_model * kernel_size];

            for _ in 0..warmup {
                unsafe {
                    depthwise_conv1d_avx2(
                        input.as_ptr(), weight.as_ptr(), std::ptr::null(),
                        output.as_mut_ptr(),
                        seq_len as i32, d_model as i32, kernel_size as i32, 1,
                    );
                }
            }
            let mut latencies = Vec::with_capacity(n_runs);
            for _ in 0..n_runs {
                let t0 = Instant::now();
                // 6 dilations (like TCN)
                for dil_idx in 0..6 {
                    let dilation = 1 << dil_idx;
                    unsafe {
                        depthwise_conv1d_avx2(
                            input.as_ptr(), weight.as_ptr(), std::ptr::null(),
                            output.as_mut_ptr(),
                            seq_len as i32, d_model as i32, kernel_size as i32, dilation,
                        );
                    }
                }
                latencies.push(t0.elapsed().as_secs_f64() * 1000.0);
            }
            report("depthwise_conv_6dil", seq_len, d_model, &latencies);
        }
        "wkv6" | "rwkv" => {
            let n_heads = 8;
            let headdim = 32;
            let r = vec![0.1f32; seq_len * d_model];
            let k = vec![0.1f32; seq_len * d_model];
            let v = vec![0.1f32; seq_len * d_model];
            let w = vec![-0.5f32; seq_len * d_model];
            let u = vec![0.1f32; d_model];
            let mut state = vec![0.0f32; n_heads * headdim * headdim];

            for _ in 0..warmup {
                state.fill(0.0);
                unsafe {
                    wkv6_scan_avx2(
                        r.as_ptr(), k.as_ptr(), v.as_ptr(), w.as_ptr(), u.as_ptr(),
                        output.as_mut_ptr(), state.as_mut_ptr(),
                        seq_len as i32, n_heads as i32, headdim as i32, d_model as i32,
                    );
                }
            }
            let mut latencies = Vec::with_capacity(n_runs);
            for _ in 0..n_runs {
                state.fill(0.0);
                let t0 = Instant::now();
                unsafe {
                    wkv6_scan_avx2(
                        r.as_ptr(), k.as_ptr(), v.as_ptr(), w.as_ptr(), u.as_ptr(),
                        output.as_mut_ptr(), state.as_mut_ptr(),
                        seq_len as i32, n_heads as i32, headdim as i32, d_model as i32,
                    );
                }
                latencies.push(t0.elapsed().as_secs_f64() * 1000.0);
            }
            report("wkv6_scan", seq_len, d_model, &latencies);
        }
        "all" => {
            println!("=== DenseEditor Mixing Layer 벤치마크 (더미 가중치) ===");
            println!("seq_len={}, d_model={}, warmup={}, n_runs={}\n", seq_len, d_model, warmup, n_runs);
            println!("{:<22} {:>10} {:>10} {:>10}", "Kernel", "Median(ms)", "Mean(ms)", "P99(ms)");
            println!("{}", "-".repeat(56));

            for t in &["fnet", "tcn", "rwkv", "retnet", "mamba", "mamba2", "xlstm"] {
                benchmark_dummy(t, seq_len, warmup, n_runs);
            }
            return;
        }
        _ => {
            eprintln!("알 수 없는 mixing type: {}", mixing_type);
            return;
        }
    }
}

fn report(name: &str, seq_len: usize, d_model: usize, latencies: &[f64]) {
    let mut sorted = latencies.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = sorted[sorted.len() / 2];
    let mean = sorted.iter().sum::<f64>() / sorted.len() as f64;
    let p99 = sorted[(sorted.len() as f64 * 0.99) as usize];
    println!("{:<22} {:>10.3} {:>10.3} {:>10.3}", name, median, mean, p99);
}

fn main() -> Result<()> {
    let args = Args::parse();

    if args.benchmark_full {
        bench::benchmark_all_full(args.seq_len, args.d_model, args.warmup, args.n_runs);
        return Ok(());
    }

    if args.benchmark_dummy {
        benchmark_dummy(&args.mixing_type, args.seq_len, args.warmup, args.n_runs);
        return Ok(());
    }

    println!("DenseEditor CPU 추론 엔진");
    println!("사용법: --benchmark-dummy --mixing-type all --seq-len 2048");

    Ok(())
}
