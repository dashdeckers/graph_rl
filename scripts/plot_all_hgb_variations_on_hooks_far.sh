plot_hgb_on_hooks () {
    declare -a files=();
    declare -a algs=("ddpg_hgb");
    declare -a envs=("pointenv");
    declare -a env_v1s=("hooks");
    declare -a env_v2s=("$@");
    for alg in "${algs[@]}"; do
        for env in "${envs[@]}"; do
            for env_v1 in "${env_v1s[@]}"; do
                for env_v2 in "${env_v2s[@]}"; do
                    files+=("./data/${alg}_${env}_${env_v1}_${env_v2}");
                done;
            done;
        done;
    done;
    python \
        "./scripts/viz_data.py" \
        -d "${files[@]}" \
        -o "plot-hgb-on-hooks.png" \
        -t "HGB-DDPG on PointEnv-Hooks-far with various parameter settings" \
        -s
}

declare -a hooks_variations=(
    # "far"
    # "far-double-tries"
    # "far-double-tries-no-reconstruct"
    "far-loaded-model"
    "far-loaded-model-no-reconstruct"
    # "far-no-reconstruct"
    # "far-replenish"
    # "far-replenish-100-double-tries"
    "far-replenish-100-loaded-model"
    # "far-replenish-loaded-model"
);
plot_hgb_on_hooks "${hooks_variations[@]}";
