# Dev Container NVIDIA Torch

This is an example of how to set up an NVIDIA VSCode DevContainer with GPU support for PyTorch and profiling.

The `examples` folder contains several examples of how to compile CUDA kernels within PyTorch, showcasing profiling when possible.


## Prerequisites

- Docker engine (and set up .wslconfig to use more cores and memory than default)
- NVIDIA driver for the graphics card
- NVIDIA Container Toolkit (which may be already included in Windows' Docker Desktop; Linux users will need to install it)
- VSCode with the DevContainer extension installed
- Follow the instructions to enable hardware counters profiling presented [here](https://developer.nvidia.com/nvidia-development-tools-solutions-err_nvgpuctrperm-permission-issue-performance-counters)
  - On Windows, this involves updating settings on the NVIDIA Control Panel and **rebooting** the computer


## Start the DevContainer

- Clone this repo
- In VSCode, press `ctrl + shift + P` or `cmd + shift + P` to bring up the Command Palette.
- Enter and find `Dev Containers: Reopen in Container`.
- VSCode will download the Docker CUDA image, install the dependencies, and open the directory in the DevContainer
- The DevContainer will then run `nvidia-smi` to display the GPU that the container can access


## Navigate to an example folder

Use `make` to compile and profile the code.

```bash
cd examples/gemm-cuda
make # to compile (or run) the (python) code
make ncu # to profile the kernel
make nsys # to profile the whole application
```


## Visualize the profiling results

Install nsight-sys and nsight-compute on your local machine and open the files generated in `examples/<app>/prof` folder


## Setup details

We leverage the NVIDIA CUDA image, which contains CUDA and PyTorch dependencies installed. See the [Dockerfile](.devcontainer/Dockerfile) for more details.


### Additional Python packages
The file `.devcontainer/requirements.txt` contains all third party Python packages you wish to install. Modify the list as you like and uncoment the "updateContentCommand" line in [`.devcontainer/devcontainer.json`](.devcontainer/devcontainer.json) to install the packages.

```
# Example: Minimal deps for tensorflow
numpy
scikit-learn
matplotlib
tensorflow
autokeras
ipykernel
regex
```

### References

This project leveraged the following sources:

- [Nsisght Compute Docker Image](https://github.com/leimao/Nsight-Compute-Docker-Image)
- [Setup a NVIDIA DevContainer with GPU Support for Tensorflow/Keras on Windows](https://alankrantas.medium.com/setup-a-nvidia-devcontainer-with-gpu-support-for-tensorflow-keras-on-windows-d00e6e204630)
- [NVIDIA CUDA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- [profiling-cuda-in-torch](https://github.com/cuda-mode/profiling-cuda-in-torch)
