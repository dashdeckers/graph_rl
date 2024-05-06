RUST_BACKTRACE=1

REPS=10

nohup \
cargo run \
    --release \
    --example pointenv_ddpg \
    -- \
    --train-config "./examples/configs/train_ddpg_pointenv.ron" \
    --env-config "./examples/configs/pointenv_10x10_empty_mindist.ron" \
    --alg-config "./examples/configs/ddpg.ron" \
    --n-repetitions $REPS \
    --device cuda \
    --log warn \
    --name ddpg-pointenv-empty-mindist-y1-2ke-t10 \
    &


nohup \
cargo run \
    --release \
    --example pointenv_ddpg \
    -- \
    --train-config "./examples/configs/train_ddpg_pointenv.ron" \
    --env-config "./examples/configs/pointenv_10x10_empty_maxdist.ron" \
    --alg-config "./examples/configs/ddpg.ron" \
    --n-repetitions $REPS \
    --device cuda \
    --log warn \
    --name ddpg-pointenv-empty-maxdist-y1-2ke-t10 \
    &


nohup \
cargo run \
    --release \
    --example pointenv_ddpg \
    -- \
    --train-config "./examples/configs/train_ddpg_pointenv.ron" \
    --env-config "./examples/configs/pointenv_10x10_empty_fardist.ron" \
    --alg-config "./examples/configs/ddpg.ron" \
    --n-repetitions $REPS \
    --device cuda \
    --log warn \
    --name ddpg-pointenv-empty-fardist-y1-2ke-t10 \
    &

