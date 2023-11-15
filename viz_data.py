import polars as pl
import seaborn as sns
import matplotlib.pyplot as plt

df = (
    pl.read_parquet("pendulum_data.parquet")
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
plt.title("Learning Curve for DDPG on Pendulum-v0")

plt.show()