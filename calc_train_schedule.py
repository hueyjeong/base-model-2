#!/usr/bin/env python3
"""
코퍼스 크기 기반 학습 파라미터 계산기

입력:
- pack_size
- batch_size
- lr
- lr이 min에 수렴하길 원하는 지점(0~1)

출력:
- max_steps
- cycle_steps
- lr_decay

주의:
- max_steps는 코퍼스 문자 수를 토큰 수로 근사해서 계산한다.
- 근사 정확도는 chars_per_token(문자/토큰 비율) 설정에 따라 달라진다.
"""

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import json
import math
import os
from pathlib import Path


def format_num(v: int) -> str:
    if v >= 1_000_000_000:
        return f"{v / 1_000_000_000:.2f}B"
    if v >= 1_000_000:
        return f"{v / 1_000_000:.2f}M"
    if v >= 1_000:
        return f"{v / 1_000:.2f}K"
    return str(v)


def _count_chunk(lines: list[str], is_jsonl: bool, text_key: str | None) -> tuple[int, int, int]:
    """라인 청크에서 (chars, total_lines, valid_lines) 계산."""
    total_chars = 0
    total_lines = 0
    valid_lines = 0

    for raw in lines:
        total_lines += 1
        line = raw.strip()
        if not line:
            continue

        if is_jsonl:
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            if text_key:
                text = obj.get(text_key, "")
            else:
                text = line
        else:
            text = line

        if not isinstance(text, str) or not text:
            continue

        total_chars += len(text)
        valid_lines += 1

    return total_chars, total_lines, valid_lines


def scan_total_chars(
    corpus_path: Path,
    text_key: str | None,
    scan_workers: int = 1,
    chunk_lines: int = 10000,
) -> tuple[int, int, int]:
    """코퍼스를 스트리밍 스캔하며 총 문자 수/라인 수/유효 라인 수를 계산.

    - 전체 파일을 메모리에 올리지 않음
    - chunk 단위로 worker 스레드에 분배 가능
    """
    total_chars = 0
    total_lines = 0
    valid_lines = 0

    is_jsonl = corpus_path.suffix in {".jsonl", ".json"}

    worker_count = max(1, int(scan_workers))
    chunk_size = max(1, int(chunk_lines))

    if worker_count == 1:
        with corpus_path.open("r", encoding="utf-8") as f:
            chunk: list[str] = []
            for raw in f:
                chunk.append(raw)
                if len(chunk) >= chunk_size:
                    c, t, v = _count_chunk(chunk, is_jsonl, text_key)
                    total_chars += c
                    total_lines += t
                    valid_lines += v
                    chunk = []

            if chunk:
                c, t, v = _count_chunk(chunk, is_jsonl, text_key)
                total_chars += c
                total_lines += t
                valid_lines += v
        return total_chars, total_lines, valid_lines

    max_pending = worker_count * 2
    with ThreadPoolExecutor(max_workers=worker_count) as ex:
        pending = set()
        with corpus_path.open("r", encoding="utf-8") as f:
            chunk: list[str] = []
            for raw in f:
                chunk.append(raw)
                if len(chunk) >= chunk_size:
                    pending.add(ex.submit(_count_chunk, chunk, is_jsonl, text_key))
                    chunk = []

                    if len(pending) >= max_pending:
                        done, pending = wait(pending, return_when=FIRST_COMPLETED)
                        for fut in done:
                            c, t, v = fut.result()
                            total_chars += c
                            total_lines += t
                            valid_lines += v

            if chunk:
                pending.add(ex.submit(_count_chunk, chunk, is_jsonl, text_key))

        for fut in pending:
            c, t, v = fut.result()
            total_chars += c
            total_lines += t
            valid_lines += v

    return total_chars, total_lines, valid_lines


def compute_schedule(
    total_chars: int,
    pack_size: int,
    batch_size: int,
    lr: float,
    min_lr: float,
    warmup_steps: int,
    converge_ratio: float,
    cycles_to_converge: int,
    chars_per_token: float,
    epochs: float,
) -> dict:
    """코퍼스 크기와 학습 설정으로 스케줄 파라미터 계산."""
    est_total_tokens = max(1, int(total_chars / max(chars_per_token, 1e-9)))
    tokens_per_step = max(1, pack_size * batch_size)

    steps_per_epoch = math.ceil(est_total_tokens / tokens_per_step)
    max_steps = max(1, math.ceil(steps_per_epoch * epochs))

    # warmup 이후 구간에서 converge_ratio 지점 계산
    post_warmup_total = max(1, max_steps - warmup_steps)
    target_step = int(round(warmup_steps + post_warmup_total * converge_ratio))
    target_step = min(max_steps, max(warmup_steps, target_step))

    steps_to_target = max(1, target_step - warmup_steps)
    cycles_to_target = max(1, cycles_to_converge)
    cycle_steps = max(1, round(steps_to_target / cycles_to_target))

    # pretrain.py SGDR와 동일하게 cycle_num마다 max_lr *= lr_decay
    # target_step 시점까지 cycle_max_lr가 min_lr에 근접하도록 decay 계산
    cycle_num_at_target = max(1, steps_to_target // cycle_steps)
    ratio = min_lr / max(lr, 1e-12)

    if ratio >= 1.0:
        lr_decay = 1.0
    else:
        lr_decay = math.exp(math.log(max(ratio, 1e-12)) / cycle_num_at_target)
        lr_decay = max(0.1, min(0.9999, lr_decay))

    return {
        "est_total_tokens": est_total_tokens,
        "tokens_per_step": tokens_per_step,
        "steps_per_epoch": steps_per_epoch,
        "max_steps": max_steps,
        "target_step": target_step,
        "cycle_steps": cycle_steps,
        "lr_decay": lr_decay,
        "cycle_num_at_target": cycle_num_at_target,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="코퍼스 기반 pretrain 스케줄 파라미터 계산기",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--corpus", default="corpus/sample_full.jsonl", help="코퍼스 경로")
    parser.add_argument("--text_key", default="text", help="JSONL 텍스트 필드명")
    parser.add_argument("--total_chars", type=int, default=None,
                        help="총 문자 수를 직접 지정(설정 시 코퍼스 스캔 생략)")
    parser.add_argument("--scan_workers", type=int, default=min(32, os.cpu_count() or 1),
                        help="코퍼스 문자 수 스캔 worker 스레드 수")
    parser.add_argument("--scan_chunk_lines", type=int, default=10000,
                        help="스캔 시 worker에 전달할 라인 청크 크기")

    parser.add_argument("--pack_size", type=int, default=4096,
                        help="패킹 목표 토큰 수")
    parser.add_argument("--batch_size", type=int, default=1, help="배치 크기")
    parser.add_argument("--lr", type=float, default=5e-4, help="최대 학습률")
    parser.add_argument("--min_lr", type=float, default=1e-5, help="최소 학습률")
    parser.add_argument("--warmup_steps", type=int, default=1000, help="워밍업 스텝")

    parser.add_argument("--converge_ratio", type=float, required=True,
                        help="lr이 min에 수렴하길 원하는 지점 (0.0~1.0)")
    parser.add_argument("--cycles_to_converge", type=int, default=4,
                        help="수렴 지점까지 사용할 SGDR cycle 개수")

    parser.add_argument("--chars_per_token", type=float, default=1.0,
                        help="문자/토큰 비율 근사치 (토크나이저별 튜닝 권장)")
    parser.add_argument("--epochs", type=float, default=1.0,
                        help="코퍼스 반복 횟수(1.0=1 epoch)")
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    if not (0.0 <= args.converge_ratio <= 1.0):
        raise ValueError("--converge_ratio 는 0.0~1.0 범위여야 합니다.")
    if args.pack_size <= 0 or args.batch_size <= 0:
        raise ValueError("--pack_size/--batch_size 는 1 이상이어야 합니다.")
    if args.lr <= 0 or args.min_lr <= 0:
        raise ValueError("--lr/--min_lr 는 0보다 커야 합니다.")
    if args.total_chars is not None and args.total_chars <= 0:
        raise ValueError("--total_chars 는 1 이상이어야 합니다.")
    if args.scan_workers <= 0:
        raise ValueError("--scan_workers 는 1 이상이어야 합니다.")
    if args.scan_chunk_lines <= 0:
        raise ValueError("--scan_chunk_lines 는 1 이상이어야 합니다.")

    corpus_path = Path(args.corpus)
    if args.total_chars is None:
        if not corpus_path.exists():
            raise FileNotFoundError(f"코퍼스를 찾을 수 없습니다: {corpus_path}")

        print(f"[1/2] 코퍼스 스캔 중: {corpus_path} (workers={args.scan_workers})")
        total_chars, total_lines, valid_lines = scan_total_chars(
            corpus_path,
            args.text_key,
            scan_workers=args.scan_workers,
            chunk_lines=args.scan_chunk_lines,
        )
    else:
        print(f"[1/2] 코퍼스 스캔 생략: --total_chars={args.total_chars:,}")
        total_chars = int(args.total_chars)
        total_lines = 0
        valid_lines = 0

    print("[2/2] 스케줄 계산 중")
    result = compute_schedule(
        total_chars=total_chars,
        pack_size=args.pack_size,
        batch_size=args.batch_size,
        lr=args.lr,
        min_lr=args.min_lr,
        warmup_steps=args.warmup_steps,
        converge_ratio=args.converge_ratio,
        cycles_to_converge=args.cycles_to_converge,
        chars_per_token=args.chars_per_token,
        epochs=args.epochs,
    )

    print("\n=== Corpus Stats ===")
    if args.total_chars is None:
        print(f"lines(total/valid): {format_num(total_lines)} / {format_num(valid_lines)}")
    else:
        print("lines(total/valid): skipped / skipped")
    print(f"total_chars       : {format_num(total_chars)} ({total_chars:,})")

    print("\n=== Estimated Schedule (SGDR) ===")
    print(f"est_total_tokens  : {format_num(result['est_total_tokens'])} ({result['est_total_tokens']:,})")
    print(f"tokens_per_step   : {format_num(result['tokens_per_step'])} ({result['tokens_per_step']:,})")
    print(f"steps_per_epoch   : {format_num(result['steps_per_epoch'])} ({result['steps_per_epoch']:,})")
    print(f"max_steps         : {result['max_steps']:,}")
    print(f"target_min_step   : {result['target_step']:,} (ratio={args.converge_ratio:.3f})")
    print(f"cycle_steps       : {result['cycle_steps']:,}")
    print(f"lr_decay          : {result['lr_decay']:.6f}")
    print(f"cycle_at_target   : {result['cycle_num_at_target']:,}")

    print("\n=== Suggested pretrain args ===")
    print(
        "--lr_schedule sgdr "
        f"--max_steps {result['max_steps']} "
        f"--cycle_steps {result['cycle_steps']} "
        f"--lr_decay {result['lr_decay']:.6f} "
        f"--warmup_steps {args.warmup_steps} "
        f"--min_lr {args.min_lr:g} "
        f"--pack_size {args.pack_size} "
        f"--batch_size {args.batch_size}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
