"""KoELECTRA 체크포인트 번들 업로드.

한 step의 체크포인트 = 본체 1개 + rank별 RNG sidecar N개.
로그 파일도 함께 업로드하고 로컬은 본체 기준으로 cleanup.
"""
from __future__ import annotations

import glob
import os
import shlex
import subprocess
import threading


def _rng_sidecars_for(ckpt_path: str, world_size: int) -> list[str]:
    base, ext = os.path.splitext(ckpt_path)  # (.../electra_step_N, .pt)
    return [f"{base}.rng_rank{r}{ext}" for r in range(world_size)]


def upload_checkpoint_bundle(
    ckpt_path: str,
    log_path: str | None,
    world_size: int,
    remote_dest: str,
    keep_latest_n: int = 3,
    blocking: bool = False,
) -> "threading.Thread | None":
    """체크포인트 번들 + 로그 업로드 & 로컬 cleanup.

    Args:
        ckpt_path: 본체 체크포인트 경로 (예: .../electra_step_10000.pt)
        log_path: 학습 로그 파일 경로 (None이면 로그 스킵)
        world_size: DDP rank 수 — 이 수만큼 sidecar를 찾아 업로드 시도
        remote_dest: rclone 원격지 (예: "gdrive:exp-jamo-codec-koelectra/small/")
        keep_latest_n: 로컬에 남길 최근 step 수 (final_*.pt는 항상 보존).
                       실제로는 race 방지 위해 `keep_latest_n + 1` 보존.
        blocking: True면 현재 스레드에서 즉시 실행 (학습 종료 시 권장).
                  False면 daemon 스레드로 백그라운드 실행.

    업로드 대상:
        - ckpt_path (본체)
        - ckpt_path.rng_rank{R}.pt for R in [0, world_size) (존재하는 것만)
        - log_path (존재할 때만)

    Cleanup 정책:
        - glob: `electra_step_*.pt` 중 `.rng_rank` 미포함 && `_final.pt` 아님
        - mtime 기준 최근 (keep_latest_n + 1) 개 본체 + 그에 속한 RNG sidecar 보존
          (+1 buffer는 아직 업로드 중인 이전 thread를 race에서 보호)
        - 나머지 본체와 대응 RNG sidecar는 삭제 (방금 저장한 ckpt_path는 삭제 안 함)

    Returns:
        blocking=False 시 생성된 Thread 객체 (join 가능). blocking=True 시 None.
    """
    def _task():
        try:
            # ── 업로드 파일 목록 ──
            files_to_upload: list[str] = [ckpt_path]
            for sidecar in _rng_sidecars_for(ckpt_path, world_size):
                if os.path.exists(sidecar):
                    files_to_upload.append(sidecar)
            if log_path and os.path.exists(log_path):
                files_to_upload.append(log_path)

            # ── 업로드 ──
            for f in files_to_upload:
                cmd = f"rclone copy {shlex.quote(f)} {shlex.quote(remote_dest)}"
                r = subprocess.run(
                    cmd, shell=True, check=False,
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                )
                if r.returncode != 0:
                    print(f"[Upload Error] {f}: rc={r.returncode}\n"
                          f"  stderr: {r.stderr.decode(errors='replace').strip()[:500]}")
                    # 나머지 파일은 계속 시도 (개별 실패가 전체 업로드를 막지 않음)

            # ── 로컬 cleanup (본체 기준) ──
            save_dir = os.path.dirname(ckpt_path)
            all_bodies = sorted(
                [
                    p for p in glob.glob(os.path.join(save_dir, "electra_step_*.pt"))
                    if ".rng_rank" not in os.path.basename(p)
                    and not p.endswith("_final.pt")
                ],
                key=os.path.getmtime,
            )
            # +1 buffer: 아직 업로드 진행 중인 이전 thread의 파일이 race로 사라지지 않도록
            protect_n = keep_latest_n + 1
            if len(all_bodies) > protect_n:
                to_delete_bodies = all_bodies[:-protect_n]
                for old_body in to_delete_bodies:
                    if old_body == ckpt_path:
                        continue  # 방금 저장한 것 보호
                    base, ext = os.path.splitext(old_body)
                    # 해당 step의 모든 RNG sidecar 삭제
                    for sidecar in glob.glob(f"{base}.rng_rank*{ext}"):
                        try:
                            os.remove(sidecar)
                        except Exception as e:
                            print(f"[Cleanup] {sidecar} 삭제 실패: {e}")
                    # 본체 삭제
                    try:
                        os.remove(old_body)
                        print(f"[Cleanup] 체크포인트 번들 삭제: {old_body}"
                              f" (+ RNG sidecars)")
                    except Exception as e:
                        print(f"[Cleanup] {old_body} 삭제 실패: {e}")

        except subprocess.CalledProcessError as e:
            print(f"\n[Upload Error] rclone 업로드 실패: {e}")
        except Exception as e:
            print(f"\n[Upload Error] 알 수 없는 오류: {e}")

    if blocking:
        _task()
        return None
    t = threading.Thread(target=_task, daemon=True)
    t.start()
    return t


if __name__ == "__main__":
    # 수동 테스트: 인자로 ckpt 경로 주면 파일 리스트만 출력
    import sys
    if len(sys.argv) >= 3:
        ckpt = sys.argv[1]
        ws = int(sys.argv[2])
        print("본체:", ckpt)
        print("RNG sidecars:")
        for s in _rng_sidecars_for(ckpt, ws):
            exist = "✓" if os.path.exists(s) else "✗"
            print(f"  [{exist}] {s}")
