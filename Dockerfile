# 사용 하려는 PyTorch 버전 및 CUDA 12.1에 맵핑되는 도커 이미지.
# 우분투 버전을 기본으로 설정하되, python 3.12를 활용.
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

# 컨테이너 내 상호작용 프롬프트 무시
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

# CUDA 12.1 경로 설정
ENV PATH=/usr/local/cuda-12.1/bin${PATH:+:${PATH}}
ENV LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

# 시스템 종속성 설치용 저장소 추가 및 기본 패키지 설치
RUN apt-get update && apt-get install -y \
    software-properties-common

# Python 3.12 저장소(Ubuntu 22.04의 경우)
RUN add-apt-repository -y ppa:deadsnakes/ppa && apt-get update

RUN apt-get install -y \
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
    rclone \
    && rm -rf /var/lib/apt/lists/*

# python 및 pip를 최신 버전으로 심볼릭 링크
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1

# pip 업데이트
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12

# 작업 디렉토리 설정
WORKDIR /workspace/base-model-2

# 패키지 요구사항만 먼저 복사하여 도커 빌드 캐시 극대화
COPY requirements.txt .

# 1. 가상환경 생성 (Docker 내부라도 종속성을 확실하게 관리하기 위해 VEnv 사용 가능, 또는 System Python에 바로 설치)
# 아래는 VEnV를 활용하여 패키지를 설치하는 스크립트화 방식입니다. 
# RUN 명령어마다 VEnv를 활성화해야하므로 bash를 사용해서 일괄 실행.
RUN /bin/bash -c "python3.12 -m venv .venv && \
    source .venv/bin/activate && \
    echo 'Installing PyTorch...' && \
    pip install torch --index-url https://download.pytorch.org/whl/cu121 && \
    echo 'Installing Core Dependencies...' && \
    pip install wheel tokenizers transformers sentencepiece einops ninja packaging && \
    echo 'Installing Mamba CUDA Kernels...' && \
    TORCH_CUDA_ARCH_LIST=\"12.0\" pip install causal-conv1d --no-build-isolation && \
    TORCH_CUDA_ARCH_LIST=\"12.0\" pip install mamba_ssm --force-reinstall --no-build-isolation && \
    echo 'Installing Mecab...' && \
    pip install mecab-python3 mecab-ko-dic && \
    echo 'Installing Remaining Requirement...' && \
    pip install -r requirements.txt"

# 의존성 설치가 끝난 후 전체 코드 복사 (소스코드 변경 시 빌드 캐시가 깨지지 않도록)
COPY . .

# Venv 자동화를 위한 환경변수 PATH 수정
ENV PATH="/workspace/base-model-2/.venv/bin:$PATH"

# Vast.ai 등에서 접속 후 작업을 시작할 수 있도록 bash를 기본 커맨드로 지정
CMD ["/bin/bash"]
