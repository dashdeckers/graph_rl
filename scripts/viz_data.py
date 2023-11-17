import polars as pl
import seaborn as sns
import matplotlib.pyplot as plt

from pathlib import Path
import argparse
import sys

parser = argparse.ArgumentParser()

parser.add_argument('-d', "--dir",)
args = parser.parse_args()
if args.dir is None:
    print("No path provided")
    sys.exit()

path = Path(args.dir)

df = (
    pl.read_parquet(path / "data.parquet")
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
plt.xlabel("Episode")
plt.ylabel("Total reward")
plt.title("Learning Curve")

plt.savefig(path / f"{args.dir}.png")
print("Success")
