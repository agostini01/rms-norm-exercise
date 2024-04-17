# Uncomment the sections you need and add your own commands
# These are dependencies of tensorflow
# Note that cuda-pytorch image used already installs many of these tools

# # Update system
# apt-get update
# apt-get upgrade -y

# # Install Linux tools and Python 3
# apt-get install software-properties-common git wget curl \
#     python3-dev python3-pip python3-wheel python3-setuptools -y

# Install Python packages
python3 -m pip install --upgrade pip
pip3 install --user -r .devcontainer/requirements.txt

# # CUDA and Libraries
# # Update CUDA Linux GPG repository key
# wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
# dpkg -i cuda-keyring_1.0-1_all.deb
# rm cuda-keyring_1.0-1_all.deb
# # Install cuDNN
# wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
# mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
# apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub
# add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /" -y
# apt-get update
# apt-get install libcudnn8=8.9.0.*-1+cuda11.8
# apt-get install libcudnn8-dev=8.9.0.*-1+cuda11.8

# Install recommended packages
# Some of these support x11 forwarding on linux
# apt-get install zlib1g g++ freeglut3-dev \
#     libx11-dev libxmu-dev libxi-dev libglu1-mesa libglu1-mesa-dev libfreeimage-dev -y

# Clean up - for smaller image size
# pip3 cache purge # not needed since cache is already disabled
# sudo apt-get autoremove -y
# sudo apt-get clean
