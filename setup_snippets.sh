# Run command (windows)
$env:RUST_BACKTRACE=1; cargo run --release -- --env pointenv --gui --log warn
# Run command (linux)
RUST_BACKTRACE=1 cargo run --release -- --env pointenv --gui --log warn


# Cuda / nvcc
## TODO


# CuDNN
## (Assuming CUDA 12.2 and Ubuntu 20.04)
## Go to https://developer.nvidia.com/rdp/cudnn-download
## Download: Local Installer for Ubuntu20.04 x86_64 (Deb)
## Filename: cudnn-local-repo-ubuntu2004-8.9.5.29_1.0-1_amd64.deb
sudo apt-get install zlib1g
sudo dpkg -i cudnn-local-repo-ubuntu2004-8.9.5.29_1.0-1_amd64.deb
sudo cp /var/cudnn-local-repo-*/cudnn-local-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get install libcudnn8=8.9.5.29-1+cuda12.2
sudo apt-get install libcudnn8-dev=8.9.5.29-1+cuda12.2

# Verify CuDNN installation:
sudo apt-get install libfreeimage3 libfreeimage-dev
sudo apt-get install libcudnn8-samples=8.9.5.29-1+cuda12.2
cp -r /usr/src/cudnn_samples_v8/ $HOME # copy the samples to a writable path
cd  $HOME/cudnn_samples_v8/mnistCUDNN
make clean && make
./mnistCUDNN


# Libtorch
cd /home/travis
# Clicking to the right version from (https://pytorch.org/get-started/locally/): linux, libtorch, C++, cuda 12.1
wget https://download.pytorch.org/libtorch/cu121/libtorch-cxx11-abi-shared-with-deps-2.1.0%2Bcu121.zip
unzip libtorch-cxx11-abi-shared-with-deps-2.1.0+cu121.zip
rm libtorch-cxx11-abi-shared-with-deps-2.1.0+cu121.zip
export LIBTORCH=/home/travis/libtorch


# Tch: Gym / AutoROM
pip install gym[atari]==0.21.0  # gym version of last working tch(ddpg) commit Feb 3. 2022 (tch: #453)
export PATH="/home/travis/.local/bin:$PATH"
pip install --upgrade AutoROM
AutoROM --accept-license




# Setup Gymnasium:
## Install Python as a dynamic/shared library
env PYTHON_CONFIGURE_OPTS="--enable-shared" pyenv install 3.11
## Install missing libraries
sudo apt-get install libpython3.11-dev
## Setup virtualenv and install gymnasium
pyenv virtualenv 3.11 thesis
pyenv local thesis
pip install gymnasium



# Misc

# Watch GPU
watch --differences=permanent -n 0.3 nvidia-smi

# Check compute_cap
nvidia-smi --query-gpu=compute_cap --format=csv,noheader
