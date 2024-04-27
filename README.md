# On Combining the Fields of Graph-Based Planning and Deep Reinforcement Learning

## Todo's

---

Paper

- MBRL Diagram
- finish prior works section
- start methods section (algorithms, environment, experiments, parameters)

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

## Windows

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
    --log warn `
    --name sgm-pointenv-oneline
```

