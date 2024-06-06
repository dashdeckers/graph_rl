plot_ddpg_vs_hgb () {
    declare -a files=();
    declare -a algs=("ddpg_hgb" "ddpg");
    declare -a envs=("pointenv");
    declare -a env_v1s=("$1");
    declare -a env_v2s=("$2");
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
        -o "plot-ddpg-vs-hgb-${env_v1s[0]}-${env_v2s[0]}.png" \
        -t "H-DDPG vs HGB-DDPG on PointEnv-${env_v1s[0]}-${env_v2s[0]}" \
        -s
}

declare -a env_v1s=("empty" "oneline" "hooks");
declare -a env_v2s=("close" "mid" "far");

for env_v1 in "${env_v1s[@]}"; do
    for env_v2 in "${env_v2s[@]}"; do
        plot_ddpg_vs_hgb "${env_v1}" "${env_v2}";
    done;
done;
