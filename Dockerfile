ARG PYTORCH="1.10.0"
ARG CUDA="11.3"
ARG CUDNN="8"

# Add CUDA support
FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel AS pytorch

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

# Install the required packages
RUN apt-get update \
    && apt-get install -y ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 lsb-core \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Minimal setup
RUN apt-get update \
 && apt-get install -y locales lsb-release
ARG DEBIAN_FRONTEND=noninteractive
RUN dpkg-reconfigure locales

# Install ROS Noetic
RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
RUN apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654
RUN apt-get update \
 && apt-get install -y --no-install-recommends ros-melodic-desktop-full
RUN apt-get install -y --no-install-recommends python3-rosdep
RUN rosdep init \
 && rosdep fix-permissions \
 && rosdep update
RUN echo "source /opt/ros/melodic/setup.bash" >> ~/.bashrc

# Install coda-models
RUN conda clean --all
COPY . /coda-models
WORKDIR /coda-models

ENV FORCE_CUDA="1"
RUN pip install -r requirements.txt
RUN pip install cumm-cu113 spconv-cu113 wandb
RUN python setup.py develop --user
# Add below to resolve binary incompatibility issues
# RUN pip install numpy==1.20.1

# Dataset specific installation
RUN pip install nuscenes-devkit==1.0.5 
RUN pip install waymo-open-dataset-tf-2-5-0 --user
