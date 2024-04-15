
ARG UBUNTU_VERSION="22.04"
ARG CUDA_VERSION="12.1.1"
ARG CUDNN_VERSION="8.9.2.26"
ARG CUDNN_V="8"
ARG PYTHON_VERSION="3.11"

# FROM nvidia/cuda:${CUDA_VERSION}-cudnn${CUDNN_V}-devel-ubuntu${UBUNTU_VERSION} AS base
FROM nvcr.io/nvidia/pytorch:24.03-py3 as base
# FROM ubuntu:${UBUNTU_VERSION} AS base

ARG DEBIAN_FRONTEND=noninteractive



# [Optional] Uncomment this section to install additional OS packages.
# RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
#     && apt-get -y install --no-install-recommends <your-package-list-here>


# Cleanup
# pip3 cache purge
# apt-get autoremove -y
# apt-get clean