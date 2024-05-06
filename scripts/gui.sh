RUST_BACKTRACE=1 \
cargo run \
    --release \
    --example pointenv_sgm \
    -- \
    --load-model "./data" "decent-ddpg-pointenv" \
    --gui \
    --log warn \
    --name sgm-pointenv-gui