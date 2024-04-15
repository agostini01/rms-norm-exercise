# Dev Container NVIDIA Torch

That is an example of how to setup a NVIDIA DevContainer with GPU Support for PyTorch.

## Prerequisites

- Docker engine (and setup .wslconfig to use more cores and memory than default)
- NVIDIA driver for the graphic card
- NVIDIA Container Toolkit (which is already included in Windows’ Docker Desktop; Linux users have to install it)
- VS Code with DevContainer extension installed
- Follow the instructions to enable hardware counters profiling presented [here](https://developer.nvidia.com/nvidia-development-tools-solutions-err_nvgpuctrperm-permission-issue-performance-counters)
  - On windows this involves updating settings on the NVIDIA Control Panel and **rebooting** the computer

## Start the DevContainer
- Clone this repo.
- In VS Code press `Ctrl + Shift + P` to bring up the Command Palette. 
- Enter and find `Dev Containers: Reopen in Container`. 
- VS Code will starts to download the CUDA image, run the script and install everything, and finish opening the directory in DevContainer.
- The DevContainer would then run nvidia-smi to show what GPU can be seen by the container. Be noted that this works even without setting up cuDNN or any environment variables.

## Navigate to a example folder

Use `make` to compile and profile the code.

```bash
cd examples/gemm-cuda
make # to compile the code
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
