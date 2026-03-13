mod bmmq;
mod config;
mod model;
mod tokenizer;

use anyhow::Result;
use clap::Parser;
use std::io::{self, BufRead, Write};
use std::time::Instant;

#[derive(Parser)]
#[command(name = "biteditor-inference")]
#[command(about = "BitEditor 한국어 문법 교정 추론 엔진")]
struct Args {
    /// export 디렉토리 경로 (config.json, model.bmmq 등)
    #[arg(short, long, default_value = "exported")]
    model_dir: String,

    /// 단일 입력 텍스트 (없으면 대화형 모드)
    #[arg(short, long)]
    input: Option<String>,
}

fn main() -> Result<()> {
    let args = Args::parse();

    // 설정 로드
    let config_path = format!("{}/config.json", args.model_dir);
    let cfg = config::ModelConfig::from_file(&config_path)?;
    eprintln!("설정 로드: d_model={}, layers={}, vocab={}, n_tags={}, experts={}×{}",
              cfg.d_model, cfg.n_rwkv_layers, cfg.vocab_size, cfg.n_tags, cfg.n_experts, cfg.top_k);

    // 토크나이저 로드
    let tok = tokenizer::KeyboardTokenizer::from_dir(&args.model_dir)?;
    eprintln!("토크나이저 로드: vocab_size={}", tok.vocab_size());

    // 모델 로드
    let t0 = Instant::now();
    let bmmq_path = format!("{}/model.bmmq", args.model_dir);
    let model = model::BitEditor::load_bmmq(&bmmq_path, &cfg)?;
    let load_ms = t0.elapsed().as_millis();
    eprintln!("모델 로드: {}ms", load_ms);

    if let Some(input) = args.input {
        run_inference(&model, &tok, &input)?;
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

            run_inference(&model, &tok, line)?;
            println!();
        }
    }

    Ok(())
}

fn run_inference(
    model: &model::BitEditor,
    tok: &tokenizer::KeyboardTokenizer,
    input: &str,
) -> Result<()> {
    eprintln!("입력: {}", input);

    let src_ids = tok.encode(input);
    eprintln!("토큰 수: {}", src_ids.len());

    let t0 = Instant::now();
    let output_ids = model.correct(&src_ids);
    let elapsed_ms = t0.elapsed().as_millis();

    let output = tok.decode(&output_ids);
    println!("{}", output);

    eprintln!("──────────────────────────────────");
    eprintln!("  입력 토큰:  {}", src_ids.len());
    eprintln!("  출력 토큰:  {}", output_ids.len());
    eprintln!("  처리 시간:  {}ms", elapsed_ms);
    eprintln!("──────────────────────────────────");

    Ok(())
}
