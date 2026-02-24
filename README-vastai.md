# Vast.ai 프로젝트 환경 구축 안내

이 저장소는 PyTorch 기반의 `Mamba` 및 `Causal Conv1d` 등 최신 최적화 모듈을 사용하므로, 환경을 구축할 때 적절한 CUDA 버전(12.1 또는 12.2)과 Python 3.12가 요구됩니다.

Vast.ai에서 본 파이썬 프로젝트를 설정하는 방법은 크게 두 가지가 있습니다. 
편한 방법을 선택해 주세요!

---

## 1. 커스텀 프라이빗 도커 이미지 사용 (Dockerfile 활용)
사용 중인 패키지의 버전 및 종속성에 변수를 두지 않고 언제나 동일한 환경을 불러오고 싶을 때 추천합니다.

1. **로컬(또는 빌드 전용 서버)에서 도커 빌드하기:**
   ```bash
   docker build -t 내아이디/base-model-2:latest .
   docker push 내아이디/base-model-2:latest
   ```
2. **Vast.ai 템플릿 설정하기:**
   - 인스턴스를 빌릴 때 **"Edit Image & Config"** 버튼을 클릭
   - **"Docker Image"** 란에 방금 푸시한 `내아이디/base-model-2:latest`를 입력 (만약 공개 이미지가 아니라면 레지스트리 로그인 정보를 추가로 입력)
   - 기타 On-start나 런타임 설정은 기본값으로 유지 (또는 SSH, Jupyter 설정에 맞게 변경)
   - 인스턴스에 접속한 뒤 `/workspace/base-model-2` 로 이동하면 코드가 들어 있으며, 이미 Python 3.12와 필수 패키지가 가상환경(`.venv`)에 빌드되어 있습니다.

---

## 2. Vast.ai 공식 PyTorch 이미지 + On-start Script 활용 (권장, 속도 빠름)
도커 허브에 별도로 푸시하기 번거롭다면, Vast.ai 콘솔에서 **On-start Script** 기능을 이용할 수 있습니다. 이미 GPU 환경이 완벽히 세팅된 이미지를 불러오고, 그 위에 필요한 라이브러리만 스크립트를 통해 자동으로 설치합니다.

1. **Vast.ai 템플릿의 베이스 이미지 선택하기:**
   - 추천 베이스 이미지: `nvidia/cuda:12.1.1-devel-ubuntu22.04` (Cuda 12.1이 명시된 Devel 이미지 필수) 
   - 또는 "Cuda 12.1 / PyTorch 2.x"가 쓰여 있는 Ubuntu 22.04 기반 인기 템플릿을 선택하세요.
2. **On-start Script 첨부:**
   - 인스턴스 셋팅 창 하단에 있는 **On-start Script** (또는 jupyter/ssh 진입 후 초기 구동 스크립트) 텍스트 박스에 현재 폴더 안의 `vastai_setup.sh`의 내용을 복사해서 붙여넣습니다.
3. 주의사항:
   - 본 스크립트는 Python 3.12를 설치하고 가상환경(`.venv`) 셋팅, Torch(cu121), Causal Conv1d 및 Mamba_SSM과 같은 까다로운 커널을 강제 빌드 격리 해제 옵션(`--no-build-isolation`)과 함께 설치합니다.
   - On-start Script가 돌아갈 동안 약간의 시간이 소요되며 백그라운드에서 진행됩니다. `tail -f /var/log/onstart.log` (환경마다 다름)를 통해 진행 상황을 모니터링할 수 있습니다.
---

## 3. 다중 GPU 통신 설정 및 학습 (멀티 GPU)
Vast.ai에서 2개, 4개 등의 다중 GPU 인스턴스를 빌렸다면, `DDP (Distributed Data Parallel)` 방식을 통해 전체 GPU를 활용하여 모델을 학습시킬 수 있습니다.

### 멀티 GPU 실행 방법 (`torchrun` 활용)
기존 `python -m training.pretrain ...` 명령어 대신 `torchrun`을 사용해 각 GPU마다 독립적인 데이터셋을 읽어 병렬로 학습합니다.

```bash
# 가상환경 활성화
source .venv/bin/activate

# 2개의 GPU를 사용할 경우 (--nproc_per_node에 GPU 갯수 지정)
torchrun --nproc_per_node=2 training/pretrain.py \
    --size 8M \
    --corpus corpus/sample_10g.jsonl \
    --batch_size 4 \
    --grad_accum_steps 8 \
    --bf16 \
    --fused_ce
```

### INT8 CUDA 백엔드 사용

`BitLinear`를 C++/CUDA 경로로 교체하려면 아래 옵션을 추가하세요.

```bash
torchrun --nproc_per_node=2 training/pretrain.py \
   --size 8M \
   --corpus corpus/sample_10g.jsonl \
   --batch_size 4 \
   --grad_accum_steps 8 \
   --bf16 \
   --fused_ce \
   --int8 \
   --int8_backend cuda

# 권장 환경변수 (non-graph 가속)
export BITLINEAR_CUDA_BACKWARD=bf16_tc
export BITLINEAR_CUDA_GRADW_LT=1
export BITLINEAR_CUDA_FUSED_ACT=1
export BITLINEAR_CUDA_FUSED_WEIGHT=1

# CUDA Graph는 실험용으로만 사용 권장
# (일부 배치/시퀀스 조합에서 오히려 감속 가능)
# --cuda_graph
```

- `--int8` 경로에서는 현재 `--compile`이 자동으로 건너뛰어집니다(custom autograd 제약).
- 메모리/처리량 균형 기준으로, 현재 환경의 안정 기본값은 `batch_size=1~2`입니다.

상세 사용법과 트러블슈팅은 `docs/int8_cuda_backend.md`를 참고하세요.

---

## 4. Google Drive 자동 업로드 설정 (선택 사항)
학습 중 생성되는 대용량 체크포인트 파일과 로그를 매 저장 주기마다 자동으로 구글 드라이브에 백업하고, 로컬 하드 디스크 용량 확보를 위해 가장 최신 체크포인트 1개만 남기고 삭제할 수 있습니다.

1. **rclone 설정하기:**
   Vast.ai 인스턴스에 접속한 뒤 터미널에서 아래 명령어로 구글 드라이브 연동을 진행합니다.
   ```bash
   rclone config
   # n (New remote) -> 이름 입력(예: gdrive) -> 18 (Google Drive) -> 이후 안내에 따라 웹 브라우저 인증 수행
   ```

2. **RCLONE 환경 변수 활용 (권장 - 완전 자동화):**
   rclone 설정 과정을 건너뛰고 싶다면, Vast.ai 인스턴스 생성 시 환경 변수로 직접 설정할 수 있습니다.
   
   이전에 `rclone config`로 한 번이라도 얻어낸 아래 형식의 토큰 정보가 필요합니다:
   `{"access_token":"ya29...","token_type":"Bearer","refresh_token":"1//0eJy...","expiry":"..."}`
   
   Vast.ai 인스턴스 설정의 `On-Start Script`나 `Extra Docker Options`에 환경변수를 추가하세요.
   **Extra Docker Options에 추가할 경우:**
   ```bash
   -e RCLONE_CONFIG_GDRIVE_TYPE=drive -e RCLONE_CONFIG_GDRIVE_SCOPE=drive -e RCLONE_CONFIG_GDRIVE_TOKEN='{"access_token":"ya29...","token_type":"Bearer","refresh_token":"1//...","expiry":"2026-..."}'
   ```
   이 설정이 주입되면, 별도의 `rclone config` 과정 없이 터미널 코드 상에서 곧바로 `gdrive:체크포인트폴더` 접근이 가능해집니다.

3. **학습 스크립트에 파라미터 추가:**
   연동된 rclone remote 경로와 함께 `--gdrive_remote` 옵션을 넘겨주세요.
   ```bash
   torchrun --nproc_per_node=2 training/pretrain.py \
       --corpus corpus/sample_10g.jsonl \
       --log_file training_run.log \
       --gdrive_remote "gdrive:base-model-2-checkpoints/"
   ```
   이 설정 시 학습 중 새로 저장되는 체크포인트(`step_*.pt`) 및 `training_run.log`가 방해 없이 백그라운드에서 드라이브로 복사되며, 업로드가 완료되면 과거 체크포인트들은 모두 삭제됩니다.
- **`torchrun` 역할:** `torchrun`이 자동으로 각 GPU 갯수만큼 프로세스를 띄우고, 각 프로세스에 사용 가능한 GPU ID(`LOCAL_RANK`)를 부여하여 그래디언트를 동기화합니다.
- **`dataset.py` 처리 로직:** 현재 프로젝트의 `StreamingPackedDataset`은 `world_size`(총 GPU 갯수)와 각 GPU의 `rank`를 인식해서, 한 개의 큰 코퍼스 파일에서 독립된 줄(데이터 스트림)을 교차로 할당받아 학습합니다. 따라서 메모리가 부족해지거나 중복된 데이터를 학습하는 현상이 발생하지 않습니다.
