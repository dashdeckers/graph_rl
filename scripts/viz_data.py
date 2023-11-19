import polars as pl
import seaborn as sns
import matplotlib.pyplot as plt

from pathlib import Path
import argparse
import sys

parser = argparse.ArgumentParser()

parser.add_argument('-d', "--dir")
parser.add_argument('-n', "--runs")
args = parser.parse_args()
if args.dir is None:
    print("No path provided")
    sys.exit()

path = Path("data") / Path(args.dir)
runs = 100 if args.runs is None else int(args.runs)

df = (
    pl.concat(
        [
            pl.read_parquet(path / f"run_{i}_data.parquet")
            for i in range(runs)
        ],
        how="horizontal",
    )
    .with_columns(
        pl.concat_list("^run_.*_total_rewards$").list.eval(
            pl.element().std()
        ).list.first().alias("std"),

        pl.concat_list("^run_.*_total_rewards$").list.eval(
            pl.element().mean()
        ).list.first().alias("mean"),
    )
)

mean = df["mean"].to_numpy()
std = df["std"].to_numpy()

sns.lineplot(
    x=range(len(mean)),
    y=mean,
)
plt.fill_between(
    range(len(mean)),
    mean - std,
    mean + std,
    alpha=0.3,
)
plt.ylim((-1750, 0))
plt.xlabel("Episode")
plt.ylabel("Total reward")
plt.title(f"Learning Curve (n_runs={runs})")

plt.savefig(path / f"{args.dir}.png")
print("Success")
