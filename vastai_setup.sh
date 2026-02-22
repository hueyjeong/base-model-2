#!/bin/bash
# Vast.ai 초기 설정용 온스타트(On-start) 스크립트 
# (Vast.ai Instance 생성 시 'On-start Script'에 복사해서 붙여넣으세요.)

# 환경 변수 설정
export DEBIAN_FRONTEND=noninteractive
export PYTHONUNBUFFERED=1
export PIP_NO_CACHE_DIR=1

# CUDA 13.0 경로 설정 (Vast.ai 기본 이미지 등에 설치되어 있을 경우)
export PATH=/usr/local/cuda-13.0/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

# 한국어 미캡(Mecab) 및 시스템 패키지 설치
echo "1. 시스템 패키지 설치 시작..."
apt-get update && apt-get install -y \
    software-properties-common

# Python 3.12 저장소 추가 (Ubuntu 22.04 기본은 3.10이므로 필요할 수 있음)
add-apt-repository -y ppa:deadsnakes/ppa
apt-get update

apt-get install -y \
    python3.12 \
    python3.12-venv \
    python3.12-dev \
    python3-pip \
    git \
    wget \
    curl \
    build-essential \
    ninja-build \
    mecab \
    libmecab-dev \
    mecab-ipadic-utf8 \
    rclone

# 파이썬 3.12을 기본값으로 설정
update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1
update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1

# pip 업데이트
echo "2. Pip 업그레이드..."
# 3.12용 pip 확인
curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12


# (옵션) github 저장소가 있다면 이곳에서 git clone 하도록 수정하세요.
# cd /workspace
# git clone https://github.com/사용자계정/base-model-2.git
# cd base-model-2

echo "3. 가상환경 생성 및 의존성 패키지 설치..."
if [ -d "/workspace/base-model-2" ]; then
    cd /workspace/base-model-2
    
    # 1. Python 3.12 venv 생성 및 활성화
    python3.12 -m venv .venv
    source .venv/bin/activate
    
    # 2. PyTorch (CUDA 13.0)
    echo "Installing PyTorch..."
    pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu130

    # 3. 핵심 의존성
    echo "Installing Core Dependencies..."
    pip install wheel tokenizers transformers sentencepiece einops ninja packaging

    # 4. Mamba CUDA 커널 구축 (의존성 분리 방지)
    echo "Installing causal-conv1d and mamba-ssm from source..."
    TORCH_CUDA_ARCH_LIST="12.0" pip install causal-conv1d --no-build-isolation
    TORCH_CUDA_ARCH_LIST="12.0" pip install mamba_ssm --force-reinstall --no-build-isolation

    # 5. MeCab (한국어 토크나이저용)
    echo "Installing MeCab Python..."
    pip install mecab-python3 mecab-ko-dic
    
    # 기타 requirements.txt의 패키지가 필요하다면 추가 설치
    # pip install -r requirements.txt
else
    echo "경고: /workspace/base-model-2 디렉토리가 존재하지 않습니다."
    echo "GitHub 저장소를 클론하거나 코드를 업로드한 뒤, 위 설치 스크립트를 수동으로 실행하세요."
fi

echo "--- Vast.ai 설정 완료! ---"
