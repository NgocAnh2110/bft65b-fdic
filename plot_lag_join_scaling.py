from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

folder = Path("outputs")
runs = pd.read_csv(folder / "lag_join_scaling.csv")
summary = runs.groupby(["engine", "quarters"], as_index=False).agg(
    seconds=("seconds", "median"),
    peak_mib=("peak_MiB", "median"),
)

fig, axes = plt.subplots(1, 2, figsize=(8, 3.2), constrained_layout=True)
styles = {
    "pandas": {"color": "#9B492C", "marker": "o"},
    "polars": {"color": "#205D81", "marker": "s"},
}

for ax, metric, title, unit in [
    (axes[0], "seconds", "Elapsed time", "Seconds"),
    (axes[1], "peak_mib", "Peak process memory", "MiB"),
]:
    for engine, style in styles.items():
        data = summary.loc[summary["engine"].eq(engine)].sort_values("quarters")
        ax.plot(data["quarters"], data[metric], label=engine, linewidth=2, **style)

    ax.set_title(title)
    ax.set_xlabel("Prediction window (quarters)")
    ax.set_ylabel(unit)
    ax.set_xticks([24, 48, 96])
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)

axes[0].legend(frameon=False)
path = folder / "lag_join_scaling.png"
fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Saved: {path.resolve()}")
