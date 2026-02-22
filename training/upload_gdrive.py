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
            # step_*.pt 형태의 모든 체크포인트 검출 (final_*.pt 등 제외 가능)
            all_ckpts = glob.glob(os.path.join(save_dir, "step_*.pt"))
            if not all_ckpts:
                return
            
            # 구체적 list 변환을 통해 slice 문제 해결
            ckpt_list = [str(f) for f in all_ckpts]
            
            # 남길 갯수를 제외한 나머지 삭제 대상 선별
            if len(ckpt_list) > keep_latest_n:
                to_delete = ckpt_list[:-keep_latest_n]
                for old_ckpt in to_delete:
                    # 삭제 전 안전 확인 (현재 방금 저장한 파일과 일치하지 않는지)
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
