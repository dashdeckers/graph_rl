import polars as pl
import seaborn as sns
import matplotlib.pyplot as plt

from pathlib import Path
import argparse
import sys

plt.rcParams["figure.dpi"] = 300
plt.rcParams["figure.figsize"] = (12, 8)

root_path = Path.cwd() #("data")

parser = argparse.ArgumentParser()
parser.add_argument('-d', "--dirs", nargs='+', default=[])
parser.add_argument('-o', "--output", default="plot.png")
parser.add_argument('-t', "--title", default="")
parser.add_argument('-y', "--ylim", nargs='*', type=float, default=None)
parser.add_argument('-s', "--successes", default=False, action=argparse.BooleanOptionalAction)
args = parser.parse_args()
if not args.dirs:
    print("No paths provided")
    sys.exit()
if args.ylim is not None and len(args.ylim) != 2:
    print("Invalid nargs for ylim")

suffix = "successes" if args.successes else "total_rewards"
episode_nums = []

for directory, color in zip(args.dirs, sns.color_palette(n_colors=len(args.dirs))):

    directory = Path(directory)
    files = list((root_path / directory).glob("run_*_data.parquet"))

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
        label=f"{directory.parts[-1]} (n_runs={len(files)})",
    )
    plt.fill_between(
        range(1, len(mean) + 1),
        mean - std,
        mean + std,
        color=color,
        alpha=0.3,
    )

plt.xlim((1, max(episode_nums)))
plt.xlabel("Episode")

plt.ylim(args.ylim)
plt.ylabel("Total reward" if not args.successes else "Success rate")

plt.grid()
plt.legend(loc="lower right")
plt.title(f"Learning Curves for {args.title}")
plt.savefig(root_path / args.output)

print("Success")
