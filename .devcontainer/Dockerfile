
# For NVIDIA image from dockerhub, but currently using nvcr.io
# ARG UBUNTU_VERSION="22.04"
# ARG CUDA_VERSION="12.1.1"
# ARG CUDNN_VERSION="8.9.2.26"
# ARG CUDNN_V="8"
# ARG PYTHON_VERSION="3.11"

# FROM nvidia/cuda:${CUDA_VERSION}-cudnn${CUDNN_V}-devel-ubuntu${UBUNTU_VERSION} AS base
FROM nvcr.io/nvidia/pytorch:24.03-py3 as base

ARG DEBIAN_FRONTEND=noninteractive

# Do NOT modify UID from 1000
# vscode updates this UID to the user building the container
ARG USERNAME=developer
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Create the user
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME


# [Optional] Uncomment this section to install additional OS packages.
# RUN apt-get update &&  \
#     apt-get -y install --no-install-recommends <your-package-list-here>


# Cleanup
# pip3 cache purge
# apt-get autoremove -y
# apt-get clean

USER $USERNAME