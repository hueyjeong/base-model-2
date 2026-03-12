mod config;
mod model;
mod tokenizer;

use anyhow::Result;
use clap::Parser;
use std::io::{self, BufRead, Write};
use std::time::Instant;

extern "C" { fn malloc_trim(pad: usize) -> i32; }

#[derive(Parser)]
#[command(name = "bitmamba-inference")]
#[command(about = "BitMamba Seq2Seq 한국어 문법 교정 추론 엔진")]
struct Args {
    /// export 디렉토리 경로 (config.json, model.safetensors 등)
    #[arg(short, long, default_value = "exported")]
    model_dir: String,

    /// 최대 생성 토큰 수
    #[arg(short = 'n', long, default_value_t = 128)]
    max_tokens: usize,

    /// 단일 입력 텍스트 (없으면 대화형 모드)
    #[arg(short, long)]
    input: Option<String>,
}

fn main() -> Result<()> {
    let args = Args::parse();

    // 설정 로드
    let config_path = format!("{}/config.json", args.model_dir);
    let cfg = config::ModelConfig::from_file(&config_path)?;
    eprintln!("설정 로드: d_model={}, enc={}층, dec={}층, vocab={}",
              cfg.d_model, cfg.n_encoder_layers, cfg.n_decoder_layers, cfg.vocab_size);

    // 토크나이저 로드
    let tok = tokenizer::KeyboardTokenizer::from_dir(&args.model_dir)?;
    eprintln!("토크나이저 로드: vocab_size={}", tok.vocab_size());

    // 모델 로드
    let t0 = Instant::now();
    let model_path = format!("{}/model.safetensors", args.model_dir);
    let mut model = model::BitMambaSeq2Seq::load(&model_path, &cfg)?;
    let load_ms = t0.elapsed().as_millis();
    eprintln!("모델 로드: {}ms", load_ms);

    if let Some(input) = args.input {
        run_inference(&mut model, &tok, &input, args.max_tokens, true)?;
    } else {
        eprintln!("\n대화형 모드 (Ctrl+D로 종료)");
        eprintln!("교정할 텍스트를 입력하세요:");

        let stdin = io::stdin();
        loop {
            eprint!("> ");
            io::stderr().flush()?;

            let mut line = String::new();
            if stdin.lock().read_line(&mut line)? == 0 {
                break;
            }
            let line = line.trim();
            if line.is_empty() { continue; }

            run_inference(&mut model, &tok, line, args.max_tokens, false)?;
            println!();
        }
    }

    Ok(())
}

fn run_inference(
    model: &mut model::BitMambaSeq2Seq,
    tok: &tokenizer::KeyboardTokenizer,
    input: &str,
    max_tokens: usize,
    drop_after_encode: bool,
) -> Result<()> {
    eprintln!("입력: {}", input);

    let src_ids = tok.encode(input);
    eprintln!("토큰 수: {}", src_ids.len());

    let t_total = Instant::now();

    // 인코딩
    let t0 = Instant::now();
    let encoder_out = model.encode(&src_ids)?;
    let enc_ms = t0.elapsed().as_millis();

    // KV 캐시 초기화
    let t0 = Instant::now();
    let mut dec_state = model.init_decoder_state(&encoder_out)?;
    let cache_ms = t0.elapsed().as_millis();

    // f32 Tensor 해제 — 디코딩에는 i8 weight만 사용
    drop(encoder_out);
    if drop_after_encode {
        model.drop_tensors();
        // glibc에 해제된 메모리를 OS에 반환하도록 요청
        unsafe { malloc_trim(0); }
        // RSS 측정 (drop 후 실제 사용 메모리)
        if let Ok(status) = std::fs::read_to_string("/proc/self/status") {
            for line in status.lines() {
                if line.starts_with("VmRSS:") {
                    eprintln!("  [메모리] drop 후 {}", line.trim());
                }
            }
        }
    }

    // 디코딩
    let t0 = Instant::now();
    let output_ids = model.generate_with_state(
        &src_ids,
        max_tokens,
        tok.bos_id,
        tok.eos_id,
        tok.pad_id,
        &mut dec_state,
    )?;
    let dec_ms = t0.elapsed().as_millis();
    let total_ms = t_total.elapsed().as_millis();

    let output = tok.decode(&output_ids);
    println!("{}", output);

    let n_gen = output_ids.len().saturating_sub(1); // BOS 제외
    let tok_per_s = if dec_ms > 0 { n_gen as f64 / dec_ms as f64 * 1000.0 } else { 0.0 };
    let step_ms = if n_gen > 0 { dec_ms as f64 / n_gen as f64 } else { 0.0 };

    eprintln!("──────────────────────────────────");
    eprintln!("  인코딩:    {:>6}ms (src {} tok)", enc_ms, src_ids.len());
    eprintln!("  KV캐시:    {:>6}ms", cache_ms);
    eprintln!("  디코딩:    {:>6}ms ({} tok, {:.1} tok/s, {:.1}ms/step)", dec_ms, n_gen, tok_per_s, step_ms);
    eprintln!("  합계:      {:>6}ms", total_ms);
    eprintln!("──────────────────────────────────");

    Ok(())
}
