python \
    "./scripts/viz_data.py" \
    -d "./data/ddpg-pointenv-empty_maxdist" \
    -o "plot_ddpg_empty_maxdist.png" \
    -t "DDPG on PointEnv-Empty with MaxDist"

python \
    "./scripts/viz_data.py" \
    -d "./data/ddpg-pointenv-empty_mindist" \
    -o "plot_ddpg_empty_mindist.png" \
    -t "DDPG on PointEnv-Empty with MinDist"

python \
    "./scripts/viz_data.py" \
    -d \
        "./data/ddpg-pointenv-empty-maxdist-y1-2ke-t10" \
        "./data/ddpg-pointenv-empty-mindist-y1-2ke-t10" \
    -o "plot_ddpg_empty_mindist_vs_maxdist.png" \
    -t "DDPG on PointEnv-Empty with MinDist vs MaxDist"





python \
    "./scripts/viz_data.py" \
    -d \
        "./data/ddpg-pointenv-empty" \
        "./data/ddpg-pointenv-oneline" \
        "./data/sgm-pointenv-empty" \
        "./data/sgm-pointenv-oneline" \
    -o "plot_empty_vs_oneline.png" \
    -t "DDPG with and without SGM on various PointEnv difficulties"


python \
    "./scripts/viz_data.py" \
    -d \
        "./data/ddpg-pointenv-empty" \
        "./data/ddpg-pointenv-oneline" \
        "./data/ddpg-pointenv-twoline" \
        "./data/sgm-pointenv-empty" \
        "./data/sgm-pointenv-oneline" \
        "./data/sgm-pointenv-twoline" \
    -o "plot_all.png" \
    -t "DDPG with and without SGM on various PointEnv difficulties"