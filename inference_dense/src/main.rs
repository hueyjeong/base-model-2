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
mod infer;

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

    /// Matmul 커널 벤치마크 (f32 vs ternary_i8 vs ternary_bitmask)
    #[arg(long)]
    benchmark_matmul: bool,

    /// 추론 모드 (stdin JSON Lines → stdout JSON Lines)
    #[arg(long)]
    infer: bool,

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
                let decay: Vec<f32> = (0..seq_len*nheads).map(|i| 0.9 + 0.09 * (i % nheads) as f32 / nheads as f32).collect();
                let d_skip = vec![1.0f32; nheads];
                let dt_dummy = vec![1.0f32; seq_len * nheads];
                let mut y = vec![0.0f32; seq_len * d_inner];
                let mut state = vec![0.0f32; nheads * d_state * headdim];

                for _ in 0..warmup {
                    state.fill(0.0);
                    unsafe {
                        mamba2_scan_avx2(
                            x_inner.as_ptr(), b_ssm.as_ptr(), c_ssm.as_ptr(),
                            decay.as_ptr(), d_skip.as_ptr(), dt_dummy.as_ptr(),
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
                            decay.as_ptr(), d_skip.as_ptr(), dt_dummy.as_ptr(),
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

/// Matmul 커널 벤치마크: f32 vs ternary_i8 vs ternary_bitmask
fn benchmark_matmul(d_model: usize, seq_len: usize, warmup: usize, n_runs: usize) {
    use std::time::Instant;

    // 대표적 프로젝션 크기들
    let sizes: Vec<(usize, usize, &str)> = vec![
        (2708, d_model, "in_proj"),
        (d_model, 1280, "out_proj"),
        (3414, d_model, "gate_up_proj"),
        (d_model, 1707, "down_proj"),
        (608, d_model, "tag_head"),
    ];

    println!("=== Matmul 커널 벤치마크 (d_model={}, seq_len={}) ===\n", d_model, seq_len);
    println!("{:<16} {:>8} {:>12} {:>12} {:>12} {:>10}",
        "Projection", "(m,k)", "f32(ms)", "ternary_i8", "bitmask(ms)", "max_diff");
    println!("{}", "-".repeat(76));

    // LUT 초기화
    unsafe { init_bitmask_luts(); }

    for (m, k, name) in &sizes {
        let m = *m;
        let k = *k;
        let n = seq_len;

        // 랜덤 ternary weights {-1, 0, +1}
        let mut rng_state: u32 = 42;
        let mut w_i8 = vec![0i8; m * k];
        for w in w_i8.iter_mut() {
            // 간단한 LCG 랜덤
            rng_state = rng_state.wrapping_mul(1664525).wrapping_add(1013904223);
            let r = (rng_state >> 16) % 3;
            *w = match r { 0 => -1, 1 => 0, _ => 1 };
        }
        let gamma = 0.03f32; // 대표 gamma 값

        // f32 weights (ternary 값 그대로 f32로)
        let w_f32: Vec<f32> = w_i8.iter().map(|&v| v as f32 * gamma).collect();

        // bitmasks 생성
        let bitmask_stride = (k + 7) / 8;
        let mut sign_bits = vec![0u8; m * bitmask_stride];
        let mut nonzero_bits = vec![0u8; m * bitmask_stride];
        for j in 0..m {
            for i in 0..k {
                let v = w_i8[j * k + i];
                let bm_byte = i / 8;
                let bm_bit = 7 - (i % 8);
                if v != 0 {
                    nonzero_bits[j * bitmask_stride + bm_byte] |= 1 << bm_bit;
                }
                if v == -1 {
                    sign_bits[j * bitmask_stride + bm_byte] |= 1 << bm_bit;
                }
            }
        }

        // 랜덤 activations
        let x: Vec<f32> = (0..n * k)
            .map(|i| ((i as f32 * 0.0137).sin() * 0.5))
            .collect();

        let mut y_f32 = vec![0.0f32; n * m];
        let mut y_i8 = vec![0.0f32; n * m];
        let mut y_bm = vec![0.0f32; n * m];

        // ── 정확성 테스트 ──
        unsafe {
            f32_sgemm_avx2(
                w_f32.as_ptr(), x.as_ptr(), y_f32.as_mut_ptr(),
                m as i32, n as i32, k as i32,
            );
            ternary_f32_sgemm_avx2(
                w_i8.as_ptr(), x.as_ptr(), y_i8.as_mut_ptr(),
                gamma, m as i32, n as i32, k as i32,
            );
            ternary_bitmask_sgemm_avx2(
                sign_bits.as_ptr(), nonzero_bits.as_ptr(),
                x.as_ptr(), y_bm.as_mut_ptr(),
                gamma, m as i32, n as i32, k as i32,
            );
        }

        // max diff (bitmask vs i8)
        let max_diff: f32 = y_i8.iter().zip(y_bm.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);

        // max diff (bitmask vs f32)
        let max_diff_f32: f32 = y_f32.iter().zip(y_bm.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);

        // ── 벤치마크 ──
        // f32
        for _ in 0..warmup {
            unsafe {
                f32_sgemm_avx2(
                    w_f32.as_ptr(), x.as_ptr(), y_f32.as_mut_ptr(),
                    m as i32, n as i32, k as i32,
                );
            }
        }
        let mut lat_f32 = Vec::with_capacity(n_runs);
        for _ in 0..n_runs {
            let t0 = Instant::now();
            unsafe {
                f32_sgemm_avx2(
                    w_f32.as_ptr(), x.as_ptr(), y_f32.as_mut_ptr(),
                    m as i32, n as i32, k as i32,
                );
            }
            lat_f32.push(t0.elapsed().as_secs_f64() * 1000.0);
        }

        // ternary i8
        for _ in 0..warmup {
            unsafe {
                ternary_f32_sgemm_avx2(
                    w_i8.as_ptr(), x.as_ptr(), y_i8.as_mut_ptr(),
                    gamma, m as i32, n as i32, k as i32,
                );
            }
        }
        let mut lat_i8 = Vec::with_capacity(n_runs);
        for _ in 0..n_runs {
            let t0 = Instant::now();
            unsafe {
                ternary_f32_sgemm_avx2(
                    w_i8.as_ptr(), x.as_ptr(), y_i8.as_mut_ptr(),
                    gamma, m as i32, n as i32, k as i32,
                );
            }
            lat_i8.push(t0.elapsed().as_secs_f64() * 1000.0);
        }

        // ternary bitmask
        for _ in 0..warmup {
            unsafe {
                ternary_bitmask_sgemm_avx2(
                    sign_bits.as_ptr(), nonzero_bits.as_ptr(),
                    x.as_ptr(), y_bm.as_mut_ptr(),
                    gamma, m as i32, n as i32, k as i32,
                );
            }
        }
        let mut lat_bm = Vec::with_capacity(n_runs);
        for _ in 0..n_runs {
            let t0 = Instant::now();
            unsafe {
                ternary_bitmask_sgemm_avx2(
                    sign_bits.as_ptr(), nonzero_bits.as_ptr(),
                    x.as_ptr(), y_bm.as_mut_ptr(),
                    gamma, m as i32, n as i32, k as i32,
                );
            }
            lat_bm.push(t0.elapsed().as_secs_f64() * 1000.0);
        }

        // 중간값 계산
        lat_f32.sort_by(|a, b| a.partial_cmp(b).unwrap());
        lat_i8.sort_by(|a, b| a.partial_cmp(b).unwrap());
        lat_bm.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let med_f32 = lat_f32[lat_f32.len() / 2];
        let med_i8 = lat_i8[lat_i8.len() / 2];
        let med_bm = lat_bm[lat_bm.len() / 2];

        println!("{:<16} ({:>4},{:>4}) {:>10.3} {:>10.3} {:>10.3}   {:.2e}",
            name, m, k, med_f32, med_i8, med_bm, max_diff);
    }

    println!("\n(max_diff = bitmask vs ternary_i8 커널 차이, 0이면 bit-exact)");
}

fn report(name: &str, _seq_len: usize, _d_model: usize, latencies: &[f64]) {
    let mut sorted = latencies.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = sorted[sorted.len() / 2];
    let mean = sorted.iter().sum::<f64>() / sorted.len() as f64;
    let p99 = sorted[(sorted.len() as f64 * 0.99) as usize];
    println!("{:<22} {:>10.3} {:>10.3} {:>10.3}", name, median, mean, p99);
}

fn main() -> Result<()> {
    let args = Args::parse();

    if args.infer {
        let config = args.config.as_deref()
            .expect("--infer 모드에는 --config 필요");
        let model = args.model.as_deref()
            .expect("--infer 모드에는 --model 필요");
        return infer::run_infer(config, model);
    }

    if args.benchmark_matmul {
        benchmark_matmul(args.d_model, args.seq_len, args.warmup, args.n_runs);
        return Ok(());
    }

    if args.benchmark_full {
        bench::benchmark_all_full(args.seq_len, args.d_model, args.warmup, args.n_runs);
        return Ok(());
    }

    if args.benchmark_dummy {
        benchmark_dummy(&args.mixing_type, args.seq_len, args.warmup, args.n_runs);
        return Ok(());
    }

    println!("DenseEditor CPU 추론 엔진");
    println!("사용법:");
    println!("  --infer --config config.json --model model.bmmq  (추론)");
    println!("  --benchmark-dummy --mixing-type all --seq-len 2048  (벤치마크)");

    Ok(())
}
