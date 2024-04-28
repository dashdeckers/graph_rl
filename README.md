# On Combining the Fields of Graph-Based Planning and Deep Reinforcement Learning

## Todo's

---

Paper

- MBRL Diagram
- finish prior works section
- methods section (algorithms, environment, experiments, parameters)

- (GBRL Diagram --> maybe at the very end)

---

Experiments

- DDPG / SGM on PointEnv Easy / Hard
- Pretrain DDPG part
- Throw trained DDPG / SGM into new environments
- Show SGM graph (we can see & edit the graph and watch it change over time!)

---

Codebase

- pretraining? (--> just an examples for training & saving)

---

Publish Codebase

- cleanup scripts
- try candle HEAD
- remove polars & config-gate python/gymnasium
- check docs, then transfer to clean repo & publish

---


# Run Experiments

Windows (powershell) and Linux (bash) commands are provided,
they each enable the `RUST_BACKTRACE` environment variable for easier
debugging in case something goes wrong, and the Linux experiment commands are
additionally run with the `nohup` command to run in the background
(because I usually run them on a remote server and they would
otherwise terminate when the connection is dropped).


## GUI

Launch `DDPG_SGM` on the `GUI` and load model weights from file:

```powershell
$env:RUST_BACKTRACE=1; `
cargo run `
    --release `
    --example pointenv_sgm `
    -- `
    --load-model ".\data" "decent-ddpg-pointenv" `
    --gui `
    --log warn `
    --name sgm-pointenv-gui
```

```bash
RUST_BACKTRACE=1 \
cargo run \
    --release \
    --example pointenv_sgm \
    -- \
    --load-model "./data" "decent-ddpg-pointenv" \
    --gui \
    --log warn \
    --name sgm-pointenv-gui
```


## Empty

### Run `DDPG` on `PointEnv` with `PointEnvWalls::None`:

Windows
```powershell
$env:RUST_BACKTRACE=1; `
cargo run `
    --release `
    --example pointenv_ddpg `
    -- `
    --train-config ".\examples\configs\train_ddpg_pointenv.ron" `
    --env-config ".\examples\configs\pointenv_10x10_empty.ron" `
    --alg-config ".\examples\configs\ddpg.ron" `
    --n-repetitions 50 `
    --log warn `
    --name ddpg-pointenv-empty
```

Linux
```bash
RUST_BACKTRACE=1 \
nohup \
cargo run \
    --release \
    --example pointenv_ddpg \
    -- \
    --train-config "./examples/configs/train_ddpg_pointenv.ron" \
    --env-config "./examples/configs/pointenv_10x10_empty.ron" \
    --alg-config "./examples/configs/ddpg.ron" \
    --n-repetitions 50 \
    --log warn \
    --name ddpg-pointenv-empty \
    &
```

### Run `DDPG_SGM` on `PointEnv` with `PointEnvWalls::None`:

Windows
```powershell
$env:RUST_BACKTRACE=1; `
cargo run `
    --release `
    --example pointenv_sgm `
    -- `
    --train-config ".\examples\configs\train_ddpg_pointenv.ron" `
    --env-config ".\examples\configs\pointenv_10x10_one_line.ron" `
    --alg-config ".\examples\configs\ddpg_sgm.ron" `
    --n-repetitions 50 `
    --log warn `
    --name sgm-pointenv-empty
```

Linux
```bash
RUST_BACKTRACE=1 \
nohup \
cargo run \
    --release \
    --example pointenv_sgm \
    -- \
    --train-config "./examples/configs/train_ddpg_pointenv.ron" \
    --env-config "./examples/configs/pointenv_10x10_one_line.ron" \
    --alg-config "./examples/configs/ddpg_sgm.ron" \
    --n-repetitions 50 \
    --log warn \
    --name sgm-pointenv-empty \
    &
```


## OneLine

### Run `DDPG` on `PointEnv` with `PointEnvWalls::OneLine`:

Windows
```powershell
$env:RUST_BACKTRACE=1; `
cargo run `
    --release `
    --example pointenv_ddpg `
    -- `
    --train-config ".\examples\configs\train_ddpg_pointenv.ron" `
    --env-config ".\examples\configs\pointenv_10x10_one_line.ron" `
    --alg-config ".\examples\configs\ddpg.ron" `
    --n-repetitions 50 `
    --log warn `
    --name ddpg-pointenv-oneline
```

Linux
```bash
RUST_BACKTRACE=1 \
nohup \
cargo run \
    --release \
    --example pointenv_ddpg \
    -- \
    --train-config "./examples/configs/train_ddpg_pointenv.ron" \
    --env-config "./examples/configs/pointenv_10x10_one_line.ron" \
    --alg-config "./examples/configs/ddpg.ron" \
    --n-repetitions 50 \
    --log warn \
    --name ddpg-pointenv-oneline \
    &
```


### Run `DDPG_SGM` on `PointEnv` with `PointEnvWalls::OneLine`:

Windows
```powershell
$env:RUST_BACKTRACE=1; `
cargo run `
    --release `
    --example pointenv_sgm `
    -- `
    --train-config ".\examples\configs\train_ddpg_pointenv.ron" `
    --env-config ".\examples\configs\pointenv_10x10_one_line.ron" `
    --alg-config ".\examples\configs\ddpg_sgm.ron" `
    --n-repetitions 50 `
    --log warn `
    --name sgm-pointenv-oneline
```

Linux
```bash
RUST_BACKTRACE=1 \
nohup \
cargo run \
    --release \
    --example pointenv_sgm \
    -- \
    --train-config "./examples/configs/train_ddpg_pointenv.ron" \
    --env-config "./examples/configs/pointenv_10x10_one_line.ron" \
    --alg-config "./examples/configs/ddpg_sgm.ron" \
    --n-repetitions 50 \
    --log warn \
    --name sgm-pointenv-oneline \
    &
```


## TwoLine

### Run `DDPG` on `PointEnv` with `PointEnvWalls::TwoLine`:

Windows
```powershell
$env:RUST_BACKTRACE=1; `
cargo run `
    --release `
    --example pointenv_ddpg `
    -- `
    --train-config ".\examples\configs\train_ddpg_pointenv.ron" `
    --env-config ".\examples\configs\pointenv_10x10_two_line.ron" `
    --alg-config ".\examples\configs\ddpg.ron" `
    --n-repetitions 50 `
    --log warn `
    --name ddpg-pointenv-twoline
```

Linux
```bash
RUST_BACKTRACE=1 \
nohup \
cargo run \
    --release \
    --example pointenv_ddpg \
    -- \
    --train-config "./examples/configs/train_ddpg_pointenv.ron" \
    --env-config "./examples/configs/pointenv_10x10_two_line.ron" \
    --alg-config "./examples/configs/ddpg.ron" \
    --n-repetitions 50 \
    --log warn \
    --name ddpg-pointenv-twoline \
    &
```

### Run `DDPG_SGM` on `PointEnv` with `PointEnvWalls::TwoLine`:

Windows
```powershell
$env:RUST_BACKTRACE=1; `
cargo run `
    --release `
    --example pointenv_sgm `
    -- `
    --train-config ".\examples\configs\train_ddpg_pointenv.ron" `
    --env-config ".\examples\configs\pointenv_10x10_two_line.ron" `
    --alg-config ".\examples\configs\ddpg_sgm.ron" `
    --n-repetitions 50 `
    --log warn `
    --name sgm-pointenv-twoline
```

Linux
```bash
RUST_BACKTRACE=1 \
nohup \
cargo run \
    --release \
    --example pointenv_sgm \
    -- \
    --train-config "./examples/configs/train_ddpg_pointenv.ron" \
    --env-config "./examples/configs/pointenv_10x10_two_line.ron" \
    --alg-config "./examples/configs/ddpg_sgm.ron" \
    --n-repetitions 50 \
    --log warn \
    --name sgm-pointenv-twoline \
    &
```

# Visualize Results

## Single and Multiline Plots

### Single line.

Windows
```powershell
python `
    ".\scripts\viz_data.py" `
    -d ".\data\sgm-pointenv-empty\" `
    -o "plot.png" `
    -t "DDPG_SGM on PointEnv-Empty"
```

Linux
```bash
python \
    "./scripts/viz_data.py" \
    -d "./data/sgm-pointenv-empty" \
    -o "plot.png" \
    -t "DDPG_SGM on PointEnv-Empty"
```

### `DDPG` with and without `SGM` on Empty vs OneLine

Windows
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

Linux
```bash
python \
    "./scripts/viz_data.py" \
    -d \
        "./data/ddpg-pointenv-empty" \
        "./data/ddpg-pointenv-oneline" \
        "./data/sgm-pointenv-empty" \
        "./data/sgm-pointenv-oneline" \
    -o "plot.png" \
    -t "DDPG with and without SGM on various PointEnv difficulties"
```

### `DDPG` with and without `SGM` on all difficulties

Windows
```powershell
python `
    ".\scripts\viz_data.py" `
    -d `
        ".\data\ddpg-pointenv-empty\" `
        ".\data\ddpg-pointenv-oneline\" `
        ".\data\ddpg-pointenv-twoline\" `
        ".\data\sgm-pointenv-empty\" `
        ".\data\sgm-pointenv-oneline\" `
        ".\data\sgm-pointenv-twoline\" `
    -o "plot.png" `
    -t "DDPG with and without SGM on various PointEnv difficulties"
```

Linux
```bash
python \
    "./scripts/viz_data.py" \
    -d \
        "./data/ddpg-pointenv-empty" \
        "./data/ddpg-pointenv-oneline" \
        "./data/ddpg-pointenv-twoline" \
        "./data/sgm-pointenv-empty" \
        "./data/sgm-pointenv-oneline" \
        "./data/sgm-pointenv-twoline" \
    -o "plot.png" \
    -t "DDPG with and without SGM on various PointEnv difficulties"
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
