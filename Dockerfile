ARG PYTORCH="1.10.0"
ARG CUDA="11.3"
ARG CUDNN="8"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6+PTX" \
    TORCH_NVCC_FLAGS="-Xfatbin -compress-all" \
    CMAKE_PREFIX_PATH="$(dirname $(which conda))/../" \
    FORCE_CUDA="1"

# Avoid Public GPG key error
# https://github.com/NVIDIA/nvidia-docker/issues/1631
RUN rm /etc/apt/sources.list.d/cuda.list \
    && rm /etc/apt/sources.list.d/nvidia-ml.list \
    && apt-key del 7fa2af80 \
    && apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub \
    && apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

# (Optional, use Mirror to speed up downloads)
# RUN sed -i 's/http:\/\/archive.ubuntu.com\/ubuntu\//http:\/\/mirrors.aliyun.com\/ubuntu\//g' /etc/apt/sources.list && \
#    pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# Install the required packages
RUN apt-get update \
    && apt-get install -y git ninja-build \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install MMDetection3D
RUN conda clean --all
COPY . /ST3D
WORKDIR /ST3D

ENV FORCE_CUDA="1"
RUN pip install -r requirements.txt
RUN pip install cumm-cu113 spconv-cu113
RUN pip install -v -e .
# Add below to resolve binary incompatibility issues
# RUN pip install numpy==1.20.1


# FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04

# # Avoid Public GPG key error
# # https://github.com/NVIDIA/nvidia-docker/issues/1631
# RUN rm /etc/apt/sources.list.d/cuda.list \
#     && rm /etc/apt/sources.list.d/nvidia-ml.list \
#     && apt-key del 7fa2af80 \
#     && apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub \
#     && apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

# RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections

# # Install basics
# RUN apt-get update -y \
#     && apt-get install build-essential \
#     && apt-get install -y apt-utils git curl ca-certificates bzip2 tree htop wget \
#     && apt-get install -y libglib2.0-0 libsm6 libxext6 libxrender-dev bmon iotop g++ python3.7 python3.7-dev python3.7-distutils

# # Install cmake v3.13.2
# RUN apt-get purge -y cmake && \
#     mkdir /root/temp && \
#     cd /root/temp && \
#     wget https://github.com/Kitware/CMake/releases/download/v3.13.2/cmake-3.13.2.tar.gz && \
#     tar -xzvf cmake-3.13.2.tar.gz && \
#     cd cmake-3.13.2 && \
#     bash ./bootstrap && \
#     make && \
#     make install && \
#     cmake --version && \
#     rm -rf /root/temp

# # Install python
# RUN ln -sv /usr/bin/python3.7 /usr/bin/python
# RUN wget https://bootstrap.pypa.io/get-pip.py && \
# 	python get-pip.py && \
# 	rm get-pip.py

# # Install python packages
# RUN PIP_INSTALL="python -m pip --no-cache-dir install" && \
#     $PIP_INSTALL numpy==1.19.3 llvmlite numba 

# # Install torch and torchvision
# # See https://pytorch.org/ for other options if you use a different version of CUDA
# RUN pip install --user torch==1.6 torchvision==0.7.0 -f https://download.pytorch.org/whl/cu102/torch_stable.html

# # Install python packages
# RUN PIP_INSTALL="python -m pip --no-cache-dir install" && \
#     $PIP_INSTALL tensorboardX easydict pyyaml scikit-image tqdm SharedArray six

# WORKDIR /root

# # Install Boost geometry
# RUN wget https://jaist.dl.sourceforge.net/project/boost/boost/1.68.0/boost_1_68_0.tar.gz && \
#     tar xzvf boost_1_68_0.tar.gz && \
#     cp -r ./boost_1_68_0/boost /usr/include && \
#     rm -rf ./boost_1_68_0 && \
#     rm -rf ./boost_1_68_0.tar.gz 

# # A weired problem that hasn't been solved yet
# RUN pip uninstall -y SharedArray && \
#     pip install SharedArray

# RUN pip install spconv-cu102

# COPY . /ST3D
# WORKDIR /ST3D
