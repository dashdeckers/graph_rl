import polars as pl
import seaborn as sns
import matplotlib.pyplot as plt

from pathlib import Path
import argparse
import sys

plt.rcParams["figure.dpi"] = 300
plt.rcParams["figure.figsize"] = (12, 8)

root_path = Path("data")

parser = argparse.ArgumentParser()
parser.add_argument('-d', "--dirs", nargs='+', default=[])
args = parser.parse_args()


df_dict = dict()

if args.dirs:
    for directory in args.dirs:

        files = list((root_path / Path(directory)).glob("run_*_data.parquet"))

        df_dict[directory] = (
            pl.concat(
                [
                    pl.read_parquet(filepath)
                    for filepath in files
                ],
                how="horizontal",
            )
        )
