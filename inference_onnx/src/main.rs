//! Diamond ELECTRA ONNX Runtime 추론 벤치마크

use clap::Parser;
use ort::value::Tensor;
use std::time::Instant;

#[derive(Parser)]
#[command(name = "inference-onnx")]
struct Args {
    #[arg(long)]
    model: String,
    #[arg(long)]
    benchmark: bool,
    #[arg(long)]
    infer: bool,
    #[arg(long, default_value = "4096")]
    seq_len: usize,
    #[arg(long, default_value = "20")]
    n_runs: usize,
    #[arg(long)]
    cuda: bool,
}

fn main() {
    let args = Args::parse();

    let threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);

    eprintln!("모델 로딩: {}", args.model);

    let mut builder = ort::session::Session::builder()
        .expect("builder")
        .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level3)
        .expect("opt")
        .with_intra_threads(threads)
        .expect("threads");

    if args.cuda {
        let cuda_ep = ort::execution_providers::CUDAExecutionProvider::default();
        eprintln!("CUDA EP 요청");
        builder = builder
            .with_execution_providers([cuda_ep.into()])
            .expect("cuda");
    } else {
        eprintln!("CPU EP (threads={})", threads);
    }

    let mut session = builder
        .commit_from_file(&args.model)
        .expect("load model");
    eprintln!("로딩 완료");

    if args.benchmark {
        if args.seq_len == 4096 {
            for &sl in &[256, 512, 1024, 2048, 4096] {
                run_benchmark(&mut session, sl, args.n_runs);
            }
        } else {
            run_benchmark(&mut session, args.seq_len, args.n_runs);
        }
    } else if args.infer {
        run_infer(&mut session);
    }
}

fn make_tensor(ids: &[i64]) -> Tensor<i64> {
    let seq_len = ids.len();
    Tensor::from_array(([1usize, seq_len], ids.to_vec())).expect("tensor")
}

fn run_benchmark(session: &mut ort::session::Session, seq_len: usize, n_runs: usize) {
    let mut data = vec![42i64; seq_len];
    data[0] = 2;

    for _ in 0..5.min(n_runs) {
        let t = make_tensor(&data);
        let _ = session.run(ort::inputs![t]).unwrap();
    }

    let t0 = Instant::now();
    for _ in 0..n_runs {
        let t = make_tensor(&data);
        let _ = session.run(ort::inputs![t]).unwrap();
    }
    let elapsed = t0.elapsed();
    let avg_ms = elapsed.as_secs_f64() / n_runs as f64 * 1000.0;
    println!("seq={seq_len}: {avg_ms:.1}ms ({n_runs} runs)");
}

fn run_infer(session: &mut ort::session::Session) {
    let input_str = std::io::read_to_string(std::io::stdin()).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&input_str).unwrap();
    let ids: Vec<i64> = parsed["ids"]
        .as_array().unwrap()
        .iter()
        .map(|v| v.as_i64().unwrap())
        .collect();

    let seq_len = ids.len();
    let t = make_tensor(&ids);

    let t0 = Instant::now();
    let outputs = session.run(ort::inputs![t]).unwrap();
    let elapsed = t0.elapsed();

    // tag_logits → ndarray → argmax
    let logits = outputs[0]
        .try_extract_array::<f32>()
        .expect("extract");
    let shape = logits.shape();
    eprintln!("shape: {:?}, {:.1}ms", shape, elapsed.as_secs_f64() * 1000.0);

    let n_tags = shape[2];
    let data = logits.as_slice().unwrap();
    let mut tags = Vec::with_capacity(seq_len);
    for t in 0..seq_len {
        let base = t * n_tags;
        let mut best_idx = 0usize;
        let mut best_val = f32::NEG_INFINITY;
        for j in 0..n_tags {
            let v = data[base + j];
            if v > best_val {
                best_val = v;
                best_idx = j;
            }
        }
        tags.push(best_idx);
    }

    println!("{}", serde_json::json!({"tags": tags}));
}
