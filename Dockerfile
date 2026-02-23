# 빌드 이미지를 CUDA 12.8 (Ubuntu 24.04 기반)으로 맞춰 mamba_ssm 및 pytorch 컴파일 버전과 동일하게 합니다.
FROM nvidia/cuda:12.8.0-devel-ubuntu24.04

# 컨테이너 내 상호작용 프롬프트 무시
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

# CUDA 12.8 경로 설정
ENV PATH=/usr/local/cuda-12.8/bin${PATH:+:${PATH}}
ENV LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

# 시스템 종속성 설치용 저장소 추가 및 기본 패키지 설치
RUN apt-get update && apt-get install -y \
    software-properties-common

# Python 3.12 저장소(Ubuntu 22.04의 경우)
# RUN add-apt-repository -y ppa:deadsnakes/ppa && apt-get update

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
    openssh-server \
    openssl \
    && rm -rf /var/lib/apt/lists/*

# python 및 pip를 최신 버전으로 심볼릭 링크
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1

# pip 업데이트
# RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12

# 작업 디렉토리 설정
WORKDIR /workspace/base-model-2


# 암호화 및 분할 압축된 corpus 데이터 및 압축 해제 스크립트 복사
# 도커 빌드 전 호스트에서 pack_corpus.sh를 실행해 압축 파일들을 생성해야 합니다.
COPY corpus/corpus.tar.gz.enc.aa ./corpus/
COPY corpus/corpus.tar.gz.enc.ab ./corpus/
COPY corpus/corpus.tar.gz.enc.ac ./corpus/
COPY corpus/corpus.tar.gz.enc.ad ./corpus/
COPY corpus/corpus.tar.gz.enc.ae ./corpus/
COPY corpus/corpus.tar.gz.enc.af ./corpus/
COPY corpus/corpus.tar.gz.enc.ag ./corpus/
COPY corpus/corpus.tar.gz.enc.ah ./corpus/
COPY corpus/corpus.tar.gz.enc.ai ./corpus/
COPY corpus/corpus.tar.gz.enc.aj ./corpus/
COPY corpus/corpus.tar.gz.enc.ak ./corpus/
COPY corpus/corpus.tar.gz.enc.al ./corpus/
COPY corpus/corpus.tar.gz.enc.am ./corpus/
COPY corpus/corpus.tar.gz.enc.an ./corpus/
COPY corpus/corpus.tar.gz.enc.ao ./corpus/
COPY corpus/corpus.tar.gz.enc.ap ./corpus/
COPY corpus/corpus.tar.gz.enc.aq ./corpus/
COPY corpus/corpus.tar.gz.enc.ar ./corpus/
COPY corpus/corpus.tar.gz.enc.as ./corpus/
COPY corpus/corpus.tar.gz.enc.at ./corpus/
COPY corpus/corpus.tar.gz.enc.au ./corpus/
COPY corpus/corpus.tar.gz.enc.av ./corpus/
COPY corpus/corpus.tar.gz.enc.aw ./corpus/
COPY corpus/corpus.tar.gz.enc.ax ./corpus/
COPY corpus/corpus.tar.gz.enc.ay ./corpus/
COPY corpus/corpus.tar.gz.enc.az ./corpus/
COPY corpus/corpus.tar.gz.enc.ba ./corpus/
COPY corpus/corpus.tar.gz.enc.bb ./corpus/
COPY corpus/corpus.tar.gz.enc.bc ./corpus/
COPY corpus/corpus.tar.gz.enc.bd ./corpus/
COPY corpus/corpus.tar.gz.enc.be ./corpus/
COPY corpus/corpus.tar.gz.enc.bf ./corpus/
COPY corpus/corpus.tar.gz.enc.bg ./corpus/
COPY corpus/corpus.tar.gz.enc.bh ./corpus/
COPY corpus/corpus.tar.gz.enc.bi ./corpus/
COPY corpus/corpus.tar.gz.enc.bj ./corpus/
COPY corpus/corpus.tar.gz.enc.bk ./corpus/
COPY corpus/corpus.tar.gz.enc.bl ./corpus/
COPY corpus/corpus.tar.gz.enc.bm ./corpus/
COPY corpus/corpus.tar.gz.enc.bn ./corpus/
COPY corpus/corpus.tar.gz.enc.bo ./corpus/
COPY corpus/corpus.tar.gz.enc.bp ./corpus/
COPY corpus/corpus.tar.gz.enc.bq ./corpus/
COPY corpus/corpus.tar.gz.enc.br ./corpus/
COPY corpus/corpus.tar.gz.enc.bs ./corpus/
COPY corpus/corpus.tar.gz.enc.bt ./corpus/
COPY corpus/corpus.tar.gz.enc.bu ./corpus/
COPY corpus/corpus.tar.gz.enc.bv ./corpus/
COPY unpack_corpus.sh .
RUN chmod +x unpack_corpus.sh

# 패키지 요구사항만 먼저 복사하여 도커 빌드 캐시 극대화
COPY requirements.txt .

# 1. 가상환경 생성 (Docker 내부라도 종속성을 확실하게 관리하기 위해 VEnv 사용 가능, 또는 System Python에 바로 설치)
RUN python3.12 -m venv .venv

# venv를 기본으로 사용하도록 PATH 환경변수를 먼저 설정 (이후 모든 RUN에서 자동으로 venv 적용)
ENV PATH="/workspace/base-model-2/.venv/bin:$PATH"

RUN echo 'Installing Requirement...' && \
    pip install -r requirements.txt

RUN echo 'Installing PyTorch...' && \
    pip install torch --index-url https://download.pytorch.org/whl/cu128

RUN echo 'Installing Core Dependencies...' && \
    pip install wheel tokenizers transformers sentencepiece einops ninja packaging

RUN echo 'Installing Mamba CUDA Kernels (causal-conv1d)...' && \
    pip install causal-conv1d

RUN echo 'Installing Mamba CUDA Kernels (mamba_ssm)...' && \
    pip install mamba_ssm

RUN echo 'Installing Mecab...' && \
    pip install mecab-python3 mecab-ko-dic

# 의존성 설치가 끝난 후 전체 코드 복사 (소스코드 변경 시 빌드 캐시가 깨지지 않도록)
COPY . .

# Entrypoint 준비
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# Vast.ai 등에서 접속 후 작업을 시작할 수 있도록 bash를 기본 커맨드로 지정
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["/bin/bash"]
