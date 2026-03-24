//! DenseEditor wgpu GPU 추론 엔진
//!
//! 크로스플랫폼 GPU 추론: Vulkan / Metal / DX12

mod gpu;
mod buffers;
mod model;
mod dispatch;

use anyhow::Result;
use clap::Parser;

use inference_common::config::DenseEditorConfig;
use crate::gpu::GpuContext;
use crate::model::DenseEditorGpu;

#[derive(Parser, Debug)]
#[command(name = "dense-editor-wgpu")]
#[command(about = "DenseEditor wgpu GPU inference engine")]
struct Args {
    /// Config JSON 경로
    #[arg(long)]
    config: String,

    /// BMMQ 모델 경로
    #[arg(long)]
    model: String,

    /// 추론 모드 (stdin JSON Lines)
    #[arg(long)]
    infer: bool,

    /// 벤치마크 모드
    #[arg(long)]
    benchmark: bool,

    /// 벤치마크 시퀀스 길이
    #[arg(long, default_value = "200")]
    seq_len: usize,

    /// 벤치마크 반복 횟수
    #[arg(long, default_value = "100")]
    n_runs: usize,

    /// 워밍업 횟수
    #[arg(long, default_value = "10")]
    warmup: usize,

    /// GPU 백엔드 (auto, vulkan, metal, dx12)
    #[arg(long, default_value = "auto")]
    backend: String,
}

fn parse_backend(s: &str) -> Option<wgpu::Backend> {
    match s.to_lowercase().as_str() {
        "vulkan" | "vk" => Some(wgpu::Backend::Vulkan),
        "metal" | "mtl" => Some(wgpu::Backend::Metal),
        "dx12" | "d3d12" => Some(wgpu::Backend::Dx12),
        _ => None,
    }
}

fn main() -> Result<()> {
    env_logger::init();
    let args = Args::parse();

    // GPU 초기화
    let backend = parse_backend(&args.backend);
    let gpu = GpuContext::new(backend)?;
    eprintln!("GPU: {} ({:?})", gpu.adapter_name, gpu.backend);

    // 모델 로드
    let config = DenseEditorConfig::load(&args.config)?;
    eprintln!("모델: d={}, {}L, tags={}", config.d_model, config.n_layers, config.n_tags);

    let mut model = DenseEditorGpu::load(&gpu, config, &args.model)?;
    eprintln!("모델 로드 완료 (GPU 버퍼 업로드)");

    if args.benchmark {
        run_benchmark(&gpu, &mut model, args.seq_len, args.warmup, args.n_runs)?;
    } else if args.infer {
        run_infer(&gpu, &mut model)?;
    } else {
        eprintln!("--infer 또는 --benchmark 지정 필요");
    }

    Ok(())
}

fn run_benchmark(gpu: &GpuContext, model: &mut DenseEditorGpu, seq_len: usize,
                 warmup: usize, n_runs: usize) -> Result<()> {
    // 더미 입력
    let input_ids: Vec<u32> = (0..seq_len).map(|i| (i % 303) as u32).collect();

    // 워밍업
    for _ in 0..warmup {
        let _ = model.forward(gpu, &input_ids)?;
    }

    // 측정
    let start = std::time::Instant::now();
    for _ in 0..n_runs {
        let _ = model.forward(gpu, &input_ids)?;
    }
    let elapsed = start.elapsed();
    let avg_ms = elapsed.as_secs_f64() * 1000.0 / n_runs as f64;

    eprintln!("벤치마크: seq={}, {}회 평균 {:.2}ms", seq_len, n_runs, avg_ms);
    Ok(())
}

fn run_infer(gpu: &GpuContext, model: &mut DenseEditorGpu) -> Result<()> {
    use std::io::BufRead;
    let stdin = std::io::stdin();
    for line in stdin.lock().lines() {
        let line = line?;
        if line.trim().is_empty() { continue; }

        #[derive(serde::Deserialize)]
        struct Input { ids: Vec<u32> }

        let input: Input = serde_json::from_str(&line)?;
        let tags = model.forward(gpu, &input.ids)?;

        #[derive(serde::Serialize)]
        struct Output { tags: Vec<u32> }

        let out = Output { tags };
        println!("{}", serde_json::to_string(&out)?);
    }
    Ok(())
}
