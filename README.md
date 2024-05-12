# On Combining the Fields of Graph-Based Planning and Deep Reinforcement Learning


# Run Experiments

Windows (powershell) and Linux (bash) commands are provided,
they each enable the `RUST_BACKTRACE` environment variable for easier
debugging in case something goes wrong, and the Linux experiment commands are
additionally run with the `nohup` command to run in the background
(because I usually run them on a remote server and they would
otherwise terminate when the connection is dropped).


## H-DDPG / HGB-DDPG on PointEnv-(Empty, OneLine, Hook)-(Close, Mid, Far)

```powershell
$REPS=50; `
$ALG="ddpg_hgb"; `
$ENV="pointenv"; `
$ENV_V1="hooks"; `
$ENV_V2="mid"; `
$EXAMPLE="${ENV}_${ALG}"; `
$NAME="${ALG}_${ENV}_${ENV_V1}_${ENV_V2}"; `
$ALG_CONFIG="${ALG}.ron"; `
$TRAIN_CONFIG="${ENV}_training.ron"; `
$ENV_CONFIG="${ENV}_${ENV_V1}_${ENV_V2}.ron"; `
$PRETRAIN_TRAIN_CONFIG="${ENV}_pretraining.ron"; `
$PRETRAIN_ENV_CONFIG="${ENV}_pretrain.ron"; `
$env:RUST_BACKTRACE=1; `
cargo run `
    --release `
    --example $EXAMPLE `
    -- `
    --log warn `
    --name $NAME `
    --pretrain-train-config ".\examples\configs\$PRETRAIN_TRAIN_CONFIG" `
    --pretrain-env-config ".\examples\configs\env_configs\$PRETRAIN_ENV_CONFIG" `
    --train-config ".\examples\configs\$TRAIN_CONFIG" `
    --env-config ".\examples\configs\env_configs\$ENV_CONFIG" `
    --alg-config ".\examples\configs\$ALG_CONFIG" `
    --n-repetitions $REPS `
```
```powershell
# GUI commands
    --load-model ".\data" "decent-ddpg-$ENV" `
    --gui `
```
```bash
REPS=50; \
ALG="ddpg"; \
ENV="pointenv"; \
ENV_V1="empty"; \
ENV_V2="far"; \
EXAMPLE="${ENV}_${ALG}"; \
NAME="${ALG}_${ENV}_${ENV_V1}_${ENV_V2}"; \
ALG_CONFIG="${ALG}.ron"; \
TRAIN_CONFIG="${ENV}_training.ron"; \
ENV_CONFIG="${ENV}_${ENV_V1}_${ENV_V2}.ron"; \
PRETRAIN_TRAIN_CONFIG="${ENV}_pretraining.ron"; \
PRETRAIN_ENV_CONFIG="${ENV}_pretrain.ron"; \
RUST_BACKTRACE=1; \
nohup \
cargo run \
    --release \
    --example ${EXAMPLE} \
    -- \
    --log warn \
    --name ${NAME} \
    --pretrain-train-config "./examples/configs/${PRETRAIN_TRAIN_CONFIG}" \
    --pretrain-env-config "./examples/configs/env_configs/${PRETRAIN_ENV_CONFIG}" \
    --train-config "./examples/configs/${TRAIN_CONFIG}" \
    --env-config "./examples/configs/env_configs/${ENV_CONFIG}" \
    --alg-config "./examples/configs/${ALG_CONFIG}" \
    --device cuda \
    --n-repetitions ${REPS} \
    &
```
```bash
# GUI commands
    --load-model "./data" "decent-ddpg-${ENV}" \
    --gui
```





# Plot

```powershell
python `
    ".\scripts\viz_data.py" `
    -d `
        ".\data\ddpg-pointenv-empty\" `
        ".\data\ddpg-pointenv-oneline\" `
        ".\data\sgm-pointenv-empty\" `
        ".\data\sgm-pointenv-oneline\" `
    -o "plot.png" `
    -t "DDPG with and without SGM on various PointEnv difficulties"
```
```bash
NAME="plot_3"; \
TITLE="H-DDPG vs HGB-DDPG on various PointEnv difficulties"; \
declare -a files=(); \
declare -a algs=("ddpg" "ddpg_hgb"); \
declare -a envs=("pointenv"); \
declare -a env_v1s=("oneline"); \
declare -a env_v2s=("far"); \
for alg in "${algs[@]}"; do \
    for env in "${envs[@]}"; do \
        for env_v1 in "${env_v1s[@]}"; do \
            for env_v2 in "${env_v2s[@]}"; do \
                files+=("./data/${alg}_${env}_${env_v1}_${env_v2}"); \
            done; \
        done; \
    done; \
done; \
python \
    "./scripts/viz_data.py" \
    -d "${files[@]}" \
    -o "${NAME}.png" \
    -t "${TITLE}"
```



# Misc

## View the Docs locally

Windows
```powershell
$env:RUSTDOCFLAGS="--html-in-header katex-header.html"; `
    cargo doc `
    --document-private-items `
    --no-deps `
    --open
```

Linux
```bash
RUSTDOCFLAGS="--html-in-header katex-header.html" \
    cargo doc \
    --document-private-items \
    --no-deps \
    --open
```


## Install & Setup Gymnasium

```bash
# Setup Gymnasium:
## Install Python as a dynamic/shared library
env PYTHON_CONFIGURE_OPTS="--enable-shared" pyenv install 3.11
## Install missing libraries
sudo apt-get install libpython3.11-dev
## Setup virtualenv and install gymnasium
pyenv virtualenv 3.11 thesis
pyenv local thesis
pip install gymnasium
```

## Might need to fix a Pyenv bug with this on Linux:

```bash
export LD_LIBRARY_PATH=/home/travis/.pyenv/versions/3.10.13/lib:$LD_LIBRARY_PATH
```

## Watch the GPU

```bash
watch --differences=permanent -n 0.3 nvidia-smi
```

## Check your GPU's compute cap
```bash
nvidia-smi --query-gpu=compute_cap --format=csv,noheader
```

## CuDNN Setup
```bash
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
```


## Libtorch Setup
```bash

# Libtorch
cd /home/travis
# Clicking to the right version from (https://pytorch.org/get-started/locally/): linux, libtorch, C++, cuda 12.1
wget https://download.pytorch.org/libtorch/cu121/libtorch-cxx11-abi-shared-with-deps-2.1.0%2Bcu121.zip
unzip libtorch-cxx11-abi-shared-with-deps-2.1.0+cu121.zip
rm libtorch-cxx11-abi-shared-with-deps-2.1.0+cu121.zip
export LIBTORCH=/home/travis/libtorch
```

## Install & Setup Gym (the old library, needed by tch-rs)
```bash
# Tch: Gym / AutoROM
pip install gym[atari]==0.21.0  # gym version of last working tch(ddpg) commit Feb 3. 2022 (tch: #453)
export PATH="/home/travis/.local/bin:$PATH"
pip install --upgrade AutoROM
AutoROM --accept-license
```
