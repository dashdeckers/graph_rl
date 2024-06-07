train_alg_on_env () {
    REPS=50;
    ALG="$1";
    ENV="pointenv";
    ENV_V1="$2";
    ENV_V2="$3";
    EXAMPLE="${ENV}_${ALG}";
    NAME="${ALG}_${ENV}_${ENV_V1}_${ENV_V2}${SUFFIX}";
    ALG_CONFIG="${ALG}.ron";
    TRAIN_CONFIG="${ENV}_training.ron";
    ENV_CONFIG="${ENV}_${ENV_V1}_${ENV_V2}.ron";
    PRETRAIN_TRAIN_CONFIG="${ENV}_pretraining.ron";
    PRETRAIN_ENV_CONFIG="${ENV}_pretrain.ron";
    RUST_BACKTRACE=1;
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
        --load-model "./data" "decent-ddpg-pointenv" \
        &
}


SUFFIX="-out-of-dist";

declare -a algs=("ddpg" "ddpg_hgb");
declare -a env_v1s=("empty" "oneline" "hooks");
declare -a env_v2s=("close" "mid" "far");

for alg in "${algs[@]}"; do
    for env_v1 in "${env_v1s[@]}"; do
        for env_v2 in "${env_v2s[@]}"; do
            train_alg_on_env "${alg}" "${env_v1}" "${env_v2}";
        done;
    done;
done;
