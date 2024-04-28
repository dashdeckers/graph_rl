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
- remove polars
- check docs
- transfer to clean repo & publish

---

## Run Experiments (Windows)

### GUI

Launch `DDPG_SGM` on the `GUI` and load model weights from file:

```powershell
cargo run --release `
    --example pointenv_sgm `
    -- `
    --load-model ".\data" "decent-ddpg-pointenv" `
    --gui `
    --log warn `
    --name sgm-pointenv-gui
```



### Empty

Run `DDPG` on `PointEnv` with `PointEnvWalls::None`:

```powershell
cargo run --release `
    --example pointenv_ddpg `
    -- `
    --train-config ".\examples\configs\train_ddpg_pointenv.ron" `
    --env-config ".\examples\configs\pointenv_10x10_empty.ron" `
    --alg-config ".\examples\configs\ddpg.ron" `
    --n-repetitions 50 `
    --log warn `
    --name ddpg-pointenv-empty
```

Run `DDPG_SGM` on `PointEnv` with `PointEnvWalls::None`:

```powershell
cargo run --release `
    --example pointenv_sgm `
    -- `
    --train-config ".\examples\configs\train_ddpg_pointenv.ron" `
    --env-config ".\examples\configs\pointenv_10x10_one_line.ron" `
    --alg-config ".\examples\configs\ddpg_sgm.ron" `
    --n-repetitions 50 `
    --log warn `
    --name sgm-pointenv-empty
```


### OneLine

Run `DDPG` on `PointEnv` with `PointEnvWalls::OneLine`:

```powershell
cargo run --release `
    --example pointenv_ddpg `
    -- `
    --train-config ".\examples\configs\train_ddpg_pointenv.ron" `
    --env-config ".\examples\configs\pointenv_10x10_one_line.ron" `
    --alg-config ".\examples\configs\ddpg.ron" `
    --n-repetitions 50 `
    --log warn `
    --name ddpg-pointenv-oneline
```

Run `DDPG_SGM` on `PointEnv` with `PointEnvWalls::OneLine`:

```powershell
cargo run --release `
    --example pointenv_sgm `
    -- `
    --train-config ".\examples\configs\train_ddpg_pointenv.ron" `
    --env-config ".\examples\configs\pointenv_10x10_one_line.ron" `
    --alg-config ".\examples\configs\ddpg_sgm.ron" `
    --n-repetitions 50 `
    --log warn `
    --name sgm-pointenv-oneline
```


### TwoLine

Run `DDPG` on `PointEnv` with `PointEnvWalls::TwoLine`:

```powershell
cargo run --release `
    --example pointenv_ddpg `
    -- `
    --train-config ".\examples\configs\train_ddpg_pointenv.ron" `
    --env-config ".\examples\configs\pointenv_10x10_two_line.ron" `
    --alg-config ".\examples\configs\ddpg.ron" `
    --n-repetitions 50 `
    --log warn `
    --name ddpg-pointenv-twoline
```

Run `DDPG_SGM` on `PointEnv` with `PointEnvWalls::TwoLine`:

```powershell
cargo run --release `
    --example pointenv_sgm `
    -- `
    --train-config ".\examples\configs\train_ddpg_pointenv.ron" `
    --env-config ".\examples\configs\pointenv_10x10_two_line.ron" `
    --alg-config ".\examples\configs\ddpg_sgm.ron" `
    --n-repetitions 50 `
    --log warn `
    --name sgm-pointenv-twoline
```

# Visualize Results

## Single and Multiline Plots

Single line.

```powershell
python `
    ".\scripts\viz_data.py" `
    -d ".\data\sgm-pointenv-empty\" `
    -o "plot.png" `
    -t "DDPG_SGM on PointEnv-Empty"
```

`DDPG` with and without `SGM` on Empty vs OneLine

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

`DDPG` with and without `SGM` on all difficulties

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