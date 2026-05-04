# ASGrasp Docker image
# - CUDA 11.1 + Python 3.8 + PyTorch 1.9.1
# - GSNet (pointnet2 + knn) と MinkowskiEngine を事前ビルド

FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Tokyo
# Turing (RTX 5000) = 7.5, もし他GPUで使うなら適宜拡張
ENV TORCH_CUDA_ARCH_LIST="7.5"
ENV FORCE_CUDA=1

# 基本パッケージ
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.8 python3.8-dev python3-pip python3.8-venv \
    git build-essential cmake ninja-build \
    libopenblas-dev libglu1-mesa-dev libosmesa6-dev \
    libassimp-dev libgomp1 \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
    ca-certificates wget curl && \
    rm -rf /var/lib/apt/lists/*

# python3 → python3.8 を既定に
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.8 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1 && \
    python -m pip install --upgrade "pip<24" "setuptools==59.5.0" wheel ninja

# PyTorch 1.9.1 + CUDA 11.1
RUN pip install --no-cache-dir \
    torch==1.9.1+cu111 torchvision==0.10.1+cu111 \
    --extra-index-url https://download.pytorch.org/whl/cu111

# ASGrasp 依存パッケージ
RUN pip install --no-cache-dir \
    numpy==1.23.5 scipy scikit-learn tqdm matplotlib \
    opencv-python==4.7.0.72 open3d==0.15.2 \
    tensorboard==2.10.1 transforms3d autolab-core \
    pillow==9.5.0

# MinkowskiEngine (GSNet が使用)
# --install-option は新しい pip で廃止されているため git clone して手動ビルド
RUN git clone --branch v0.5.4 --depth 1 https://github.com/NVIDIA/MinkowskiEngine.git /tmp/MinkowskiEngine && \
    cd /tmp/MinkowskiEngine && \
    python setup.py install --blas=openblas && \
    rm -rf /tmp/MinkowskiEngine

# 作業ディレクトリ
WORKDIR /asgrasp
CMD ["/bin/bash"]
