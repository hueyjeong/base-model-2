"""외부 M2 벤치마크 자동 평가 파이프라인

체크포인트 → 교정 → KAGAS M2 생성 → M2 scorer → P/R/F0.5 출력

Usage:
    python eval_m2.py --ckpt checkpoints/dense_mamba2_d640_step_100000.pt
    python eval_m2.py --ckpt checkpoints/dense_mamba2_d640_step_100000.pt --n_samples 1000 --threshold 0.7
"""
import argparse
import os
import subprocess
import sys
import tempfile
import time

import torch

sys.path.insert(0, os.path.dirname(__file__))

PROJECT_ROOT = os.path.dirname(__file__)
KAGAS_ROOT = "/workspace/Standard_Korean_GEC"
KAGAS_VENV_PYTHON = os.path.join(KAGAS_ROOT, ".venv", "bin", "python")
KAGAS_SCRIPT = os.path.join(KAGAS_ROOT, "KAGAS", "parallel_to_m2_korean.py")
KAGAS_HUNSPELL = os.path.join(KAGAS_ROOT, "KAGAS", "aff-dic")
M2_SCORER = os.path.join(KAGAS_ROOT, "metric", "m2scorer", "scripts", "m2scorer.py")
M2_SCORER_DIR = os.path.join(KAGAS_ROOT, "metric", "m2scorer")

# 기본 평가 데이터
DEFAULT_ORIG = os.path.join(PROJECT_ROOT, "corpus", "nikl_para", "test_orig.txt")
DEFAULT_COR = os.path.join(PROJECT_ROOT, "corpus", "nikl_para", "test_cor.txt")


def load_and_correct(ckpt_path, orig_lines, device, batch_size=64, threshold=0.0):
    """모델 로드 → 교정"""
    from eval_kagas import load_model, load_tokenizer, correct_sentences
    model, config, step = load_model(ckpt_path, device)
    tokenizer = load_tokenizer()
    preds = correct_sentences(model, config, tokenizer, orig_lines, device,
                              batch_size=batch_size, threshold=threshold)
    return preds, step


def generate_m2(orig_path, cor_path, m2_path):
    """KAGAS parallel_to_m2_korean.py 실행"""
    cmd = [
        KAGAS_VENV_PYTHON, KAGAS_SCRIPT,
        "-orig", orig_path,
        "-cor", cor_path,
        "-out", m2_path,
        "-hunspell", KAGAS_HUNSPELL,
        "-noprint",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=KAGAS_ROOT)
    if result.returncode != 0:
        print(f"[WARN] KAGAS 일부 에러 (정상 동작 중 skip 발생 가능)")
    # skip 수 카운트
    skip_count = result.stdout.count("[SKIP]") + result.stderr.count("[SKIP]")
    return skip_count


def match_system_output(m2_path, orig_lines, sys_lines):
    """M2 S-line과 시스템 출력 매칭 (KAGAS skip 대응)"""
    # M2에서 S-line 추출
    m2_slines = []
    with open(m2_path) as f:
        for line in f:
            if line.startswith("S "):
                m2_slines.append(line[2:].strip())

    # orig → index 매핑 (공백 제거 비교)
    orig_norm_map = {}
    for i, line in enumerate(orig_lines):
        key = line.replace(" ", "")
        if key not in orig_norm_map:
            orig_norm_map[key] = i

    # 매칭
    matched = []
    miss = 0
    for s in m2_slines:
        s_norm = s.replace(" ", "")
        idx = orig_norm_map.get(s_norm)
        if idx is not None and idx < len(sys_lines):
            matched.append(sys_lines[idx])
        else:
            matched.append("")
            miss += 1

    return matched, len(m2_slines), miss


def run_m2_scorer(system_path, gold_m2_path):
    """M2 scorer 실행 → P/R/F0.5 파싱"""
    cmd = [
        KAGAS_VENV_PYTHON, "scripts/m2scorer.py",
        system_path, gold_m2_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=M2_SCORER_DIR)
    output = result.stdout + result.stderr

    p, r, f05 = 0.0, 0.0, 0.0
    for line in output.split("\n"):
        line = line.strip()
        if line.startswith("Precision"):
            p = float(line.split(":")[-1].strip())
        elif line.startswith("Recall"):
            r = float(line.split(":")[-1].strip())
        elif line.startswith("F_"):
            f05 = float(line.split(":")[-1].strip())

    return p, r, f05


def main():
    parser = argparse.ArgumentParser(description="외부 M2 벤치마크 자동 평가")
    parser.add_argument("--ckpt", required=True, help="체크포인트 경로")
    parser.add_argument("--orig", default=DEFAULT_ORIG, help="오류문 파일")
    parser.add_argument("--cor", default=DEFAULT_COR, help="교정문 파일")
    parser.add_argument("--n_samples", type=int, default=10000, help="평가 문장 수 (0=전체)")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--thresholds", type=str, default=None,
                        help="threshold 스윕 (쉼표 구분, 예: 0.0,0.5,0.7,0.8)")
    args = parser.parse_args()

    # 데이터 로드
    with open(args.orig) as f:
        all_orig = [l.strip() for l in f]
    with open(args.cor) as f:
        all_cor = [l.strip() for l in f]

    assert len(all_orig) == len(all_cor), \
        f"orig/cor 줄 수 불일치: {len(all_orig)} vs {len(all_cor)}"

    if args.n_samples > 0 and args.n_samples < len(all_orig):
        orig_lines = all_orig[:args.n_samples]
        cor_lines = all_cor[:args.n_samples]
    else:
        orig_lines = all_orig
        cor_lines = all_cor

    print(f"[Data] {len(orig_lines)} 문장 ({args.orig})")

    # threshold 목록
    if args.thresholds:
        thresholds = [float(t) for t in args.thresholds.split(",")]
    else:
        thresholds = [args.threshold]

    # KAGAS M2 gold 생성 (threshold와 무관, 한 번만)
    print("[M2 Gold] KAGAS M2 생성 중...")
    t0 = time.time()
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as fo, \
         tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as fc:
        for line in orig_lines:
            fo.write(line + "\n")
        for line in cor_lines:
            fc.write(line + "\n")
        tmp_orig = fo.name
        tmp_cor = fc.name

    gold_m2 = tempfile.NamedTemporaryFile(suffix=".m2", delete=False).name
    skip_count = generate_m2(tmp_orig, tmp_cor, gold_m2)
    print(f"[M2 Gold] 완료 ({time.time() - t0:.1f}s, {skip_count} skipped)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # threshold별 평가
    results = []
    for i, threshold in enumerate(thresholds):
        print(f"\n{'='*60}")
        print(f"[Eval] threshold={threshold}, ckpt={os.path.basename(args.ckpt)}")

        # 모델 교정 (첫 threshold에서만 로드, 이후 재사용)
        if i == 0:
            t0 = time.time()
            preds, step = load_and_correct(args.ckpt, orig_lines, device,
                                           batch_size=args.batch_size,
                                           threshold=threshold)
            print(f"[Correct] {len(preds)} 문장, {time.time() - t0:.1f}s (step={step})")
        else:
            # threshold만 바꿔서 재교정
            t0 = time.time()
            preds, step = load_and_correct(args.ckpt, orig_lines, device,
                                           batch_size=args.batch_size,
                                           threshold=threshold)
            print(f"[Correct] {len(preds)} 문장, {time.time() - t0:.1f}s")

        # M2 매칭
        matched, n_m2, n_miss = match_system_output(gold_m2, orig_lines, preds)
        print(f"[Match] M2 S-lines: {n_m2}, 매칭 실패: {n_miss}")

        # 시스템 출력 저장
        sys_path = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False).name
        with open(sys_path, "w") as f:
            for line in matched:
                f.write(line + "\n")

        # M2 scorer
        p, r, f05 = run_m2_scorer(sys_path, gold_m2)
        print(f"[Result] P={p:.4f}  R={r:.4f}  F0.5={f05:.4f}")
        results.append((threshold, p, r, f05))

        os.unlink(sys_path)

    # 요약
    if len(results) > 1:
        print(f"\n{'='*60}")
        print(f"[Summary] {os.path.basename(args.ckpt)} — {len(orig_lines)} sentences")
        print(f"{'Threshold':>10s} | {'Precision':>10s} | {'Recall':>10s} | {'F0.5':>10s}")
        print("-" * 50)
        for threshold, p, r, f05 in results:
            print(f"{threshold:>10.2f} | {p:>10.4f} | {r:>10.4f} | {f05:>10.4f}")

    # 임시 파일 정리
    for f in [tmp_orig, tmp_cor, gold_m2]:
        try:
            os.unlink(f)
        except OSError:
            pass


if __name__ == "__main__":
    main()
