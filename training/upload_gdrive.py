import os
import glob
import shlex
import subprocess
import threading
import time

def upload_and_cleanup(ckpt_path: str, log_path: str, remote_dest: str, keep_latest_n: int = 1):
    """
    백그라운드 스레드에서 rclone을 통해 체크포인트와 로그를 업로드하고,
    업로드 성공 시 이전 체크포인트들을 삭제합니다.
    
    Args:
        ckpt_path: 방금 저장된 최신 체크포인트 경로
        log_path: 현재 기록 중인 훈련 로그 파일 경로
        remote_dest: rclone 원격지 (예: 'gdrive:base-model-2-ckpts/')
        keep_latest_n: 로컬에 남겨둘 최신 체크포인트 수량
    """
    def _task():
        try:
            # 1. 체크포인트 업로드 (rclone copy)
            # 로그 출력을 최소화하여 메인 스레드 블로킹 방지
            cmd_ckpt = f"rclone copy {shlex.quote(ckpt_path)} {shlex.quote(remote_dest)}"
            subprocess.run(cmd_ckpt, shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            # 2. 로그 파일 업로드 (선택사항, 존재할 경우만)
            if log_path and os.path.exists(log_path):
                cmd_log = f"rclone copy {shlex.quote(log_path)} {shlex.quote(remote_dest)}"
                subprocess.run(cmd_log, shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            # 3. 로컬 체크포인트 정리 (업로드가 모두 성공한 경우에만)
            save_dir = os.path.dirname(ckpt_path)
            # step/editor 체크포인트 패턴 자동 감지 (final_*.pt 제외)
            all_ckpts = sorted(
                glob.glob(os.path.join(save_dir, "step_*.pt"))
                + glob.glob(os.path.join(save_dir, "editor_*_step*.pt")),
                key=os.path.getmtime,
            )
            if not all_ckpts:
                return

            if len(all_ckpts) > keep_latest_n:
                to_delete = all_ckpts[:-keep_latest_n]
                for old_ckpt in to_delete:
                    if old_ckpt != ckpt_path:
                        try:
                            os.remove(old_ckpt)
                            print(f"[Cleanup] 이전 체크포인트 삭제 완료: {old_ckpt}")
                        except Exception as e:
                            print(f"[Cleanup Error] {old_ckpt} 삭제 실패: {e}")

        except subprocess.CalledProcessError as e:
            print(f"\n[Upload Error] rclone 업로드 실패: {e}")
        except Exception as e:
            print(f"\n[Upload Error] 알 수 없는 오류 발생: {e}")

    # 백그라운드 스레드로 실행 (메인 학습 루프 블로킹 방지)
    t = threading.Thread(target=_task, daemon=True)
    t.start()
