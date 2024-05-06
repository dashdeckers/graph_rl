RUST_BACKTRACE=1

REPS=100

nohup \
cargo run \
    --release \
    --example pointenv_ddpg \
    -- \
    --train-config "./examples/configs/train_ddpg_pointenv.ron" \
    --env-config "./examples/configs/pointenv_10x10_empty.ron" \
    --alg-config "./examples/configs/ddpg.ron" \
    --n-repetitions $REPS \
    --device cuda \
    --log warn \
    --name ddpg-pointenv-empty \
    &


nohup \
cargo run \
    --release \
    --example pointenv_sgm \
    -- \
    --train-config "./examples/configs/train_ddpg_pointenv.ron" \
    --env-config "./examples/configs/pointenv_10x10_empty.ron" \
    --alg-config "./examples/configs/ddpg_sgm.ron" \
    --n-repetitions $REPS \
    --device cuda \
    --log warn \
    --name sgm-pointenv-empty \
    &


nohup \
cargo run \
    --release \
    --example pointenv_ddpg \
    -- \
    --train-config "./examples/configs/train_ddpg_pointenv.ron" \
    --env-config "./examples/configs/pointenv_10x10_one_line.ron" \
    --alg-config "./examples/configs/ddpg.ron" \
    --n-repetitions $REPS \
    --device cuda \
    --log warn \
    --name ddpg-pointenv-oneline \
    &


nohup \
cargo run \
    --release \
    --example pointenv_sgm \
    -- \
    --train-config "./examples/configs/train_ddpg_pointenv.ron" \
    --env-config "./examples/configs/pointenv_10x10_one_line.ron" \
    --alg-config "./examples/configs/ddpg_sgm.ron" \
    --n-repetitions $REPS \
    --device cuda \
    --log warn \
    --name sgm-pointenv-oneline \
    &