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

    /// 벤치마크 모드: N회 반복 추론 후 통계 출력
    #[arg(long)]
    bench: bool,

    /// 벤치마크 반복 횟수 (기본 10)
    #[arg(long, default_value_t = 10)]
    bench_n: usize,

    /// 벤치마크 입력 텍스트 (--bench 시, 없으면 기본 문장 사용)
    #[arg(long)]
    bench_text: Option<String>,

    /// 프로파일 모드: 구간별 시간 측정
    #[arg(long)]
    profile: bool,
}

fn backend_name() -> &'static str {
    #[cfg(feature = "avx2-only")]
    { "AVX2-only" }
    #[cfg(not(feature = "avx2-only"))]
    { "AVX-VNNI" }
}

fn main() -> Result<()> {
    let args = Args::parse();

    // 설정 로드
    let config_path = format!("{}/config.json", args.model_dir);
    let cfg = config::ModelConfig::from_file(&config_path)?;
    eprintln!("설정 로드: d_model={}, layers={}, vocab={}, n_tags={}, experts={}×{}",
              cfg.d_model, cfg.n_rwkv_layers, cfg.vocab_size, cfg.n_tags, cfg.n_experts, cfg.top_k);
    eprintln!("백엔드: {}", backend_name());

    // 토크나이저 로드
    let tok = tokenizer::KeyboardTokenizer::from_dir(&args.model_dir)?;
    eprintln!("토크나이저 로드: vocab_size={}", tok.vocab_size());

    // 모델 로드
    let t0 = Instant::now();
    let bmmq_path = format!("{}/model.bmmq", args.model_dir);
    let model = model::BitEditor::load_bmmq(&bmmq_path, &cfg)?;
    let load_ms = t0.elapsed().as_millis();
    eprintln!("모델 로드: {}ms", load_ms);

    if args.profile {
        let text = args.input.as_deref()
            .or(args.bench_text.as_deref())
            .unwrap_or("나는 어제 학교에 갔습니닼. 그런데 선생님이 숙게를 안내줬다.");
        let src_ids = tok.encode(text);
        eprintln!("프로파일: {} 토큰", src_ids.len());
        // warmup
        let _ = model.correct(&src_ids);
        // profiled run
        let output_ids = model.correct_profiled(&src_ids);
        let output = tok.decode(&output_ids);
        println!("{}", output);
    } else if args.bench {
        run_benchmark(&model, &tok, &args)?;
    } else if let Some(input) = args.input {
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

fn run_benchmark(
    model: &model::BitEditor,
    tok: &tokenizer::KeyboardTokenizer,
    args: &Args,
) -> Result<()> {
    let default_text = "나는 어제 학교에 갔습니닼. 그런데 선생님이 숙게를 안내줬다.";
    let text = args.bench_text.as_deref()
        .or(args.input.as_deref())
        .unwrap_or(default_text);

    let src_ids = tok.encode(text);
    let n = args.bench_n;

    eprintln!("\n══════ 벤치마크 시작 ══════");
    eprintln!("  백엔드:     {}", backend_name());
    eprintln!("  입력 텍스트: {}", text);
    eprintln!("  입력 토큰:  {}", src_ids.len());
    eprintln!("  반복 횟수:  {}", n);

    // Warmup (1회)
    let warmup_output = model.correct(&src_ids);
    let warmup_text = tok.decode(&warmup_output);
    eprintln!("  교정 결과:  {}", warmup_text);
    eprintln!();

    // 벤치마크 실행
    let mut times_ms = Vec::with_capacity(n);
    for i in 0..n {
        let t0 = Instant::now();
        let _ = model.correct(&src_ids);
        let elapsed = t0.elapsed().as_secs_f64() * 1000.0;
        times_ms.push(elapsed);
        eprint!("\r  진행: {}/{}", i + 1, n);
    }
    eprintln!();

    // 통계 계산
    let avg = times_ms.iter().sum::<f64>() / times_ms.len() as f64;
    let min = times_ms.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = times_ms.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let std_dev = (times_ms.iter().map(|t| (t - avg).powi(2)).sum::<f64>()
                   / times_ms.len() as f64).sqrt();
    let tok_per_sec = src_ids.len() as f64 / (avg / 1000.0);

    eprintln!("══════ 벤치마크 결과 ══════");
    eprintln!("  백엔드:     {}", backend_name());
    eprintln!("  반복 횟수:  {}", n);
    eprintln!("  입력 토큰:  {}", src_ids.len());
    eprintln!("  평균:       {:.2}ms", avg);
    eprintln!("  최소:       {:.2}ms", min);
    eprintln!("  최대:       {:.2}ms", max);
    eprintln!("  표준편차:   {:.2}ms", std_dev);
    eprintln!("  tok/s:      {:.1}", tok_per_sec);
    eprintln!("══════════════════════════");

    // stdout에 CSV 형식으로도 출력 (파이프 등에 유용)
    println!("backend,tokens,avg_ms,min_ms,max_ms,std_ms,tok_per_sec");
    println!("{},{},{:.2},{:.2},{:.2},{:.2},{:.1}",
             backend_name(), src_ids.len(), avg, min, max, std_dev, tok_per_sec);

    Ok(())
}
