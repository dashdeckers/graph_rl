import polars as pl
import seaborn as sns
import matplotlib.pyplot as plt

from pathlib import Path
import argparse
import sys

plt.rcParams["figure.dpi"] = 300
plt.rcParams["figure.figsize"] = (12, 8)

root_path = Path("data")
recognized_envs = [
    "pendulum",
    "pointmaze",
    "pointenv",
]

parser = argparse.ArgumentParser()
parser.add_argument('-d', "--dirs", nargs='+', default=[])
parser.add_argument('-s', "--successes", default=False, action=argparse.BooleanOptionalAction)
args = parser.parse_args()
if not args.dirs:
    print("No paths provided")
    sys.exit()

environment = [
    envname for envname in recognized_envs
    if all(envname in dirname for dirname in args.dirs)
]
if len(environment) != 1:
    print("All provided paths must contain the same (unique) environment name")
    sys.exit()
environment = environment[0]

suffix = "successes" if args.successes else "total_rewards"

episode_nums = []
for directory, color in zip(args.dirs, sns.color_palette(n_colors=len(args.dirs))):
    files = list((root_path / Path(directory)).glob("run_*_data.parquet"))

    df = (
        pl.concat(
            [
                pl.read_parquet(filepath)
                for filepath in files
            ],
            how="horizontal",
        )
        .with_columns(
            pl.concat_list(f"^run_.*_{suffix}$").list.eval(
                pl.element().std()
            ).list.first().alias("std"),

            pl.concat_list(f"^run_.*_{suffix}$").list.eval(
                pl.element().mean()
            ).list.first().alias("mean"),
        )
    )

    mean = df["mean"].to_numpy()
    std = df["std"].to_numpy()
    episode_nums.append(len(mean))

    sns.lineplot(
        x=range(1, len(mean) + 1),
        y=mean,
        color=color,
        label=f"{directory} (n_runs={len(files)})",
    )
    plt.fill_between(
        range(1, len(mean) + 1),
        mean - std,
        mean + std,
        color=color,
        alpha=0.3,
    )

plt.xlim((1, max(episode_nums)))
if args.successes:
    plt.ylim((0, 1))
else:
    if environment == "pendulum":
        plt.ylim((-1750, 0))

plt.xlabel("Episode")
plt.ylabel("Total reward" if not args.successes else "Success rate")
plt.grid()
plt.legend(loc="lower right")
plt.title(f"Learning Curves for {environment}")

stripped = [dirname.replace(f"{environment}_", "", 1) for dirname in args.dirs]
filename = f"{'s' if args.successes else 'r'}_{'_'.join(stripped)}.png"
if len(args.dirs) == 1:
    plt.savefig(root_path / args.dirs[0] / filename)
else:
    plt.savefig(root_path / filename)

print("Success")
