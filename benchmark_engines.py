from datetime import date
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np
import pandas as pd
import polars as pl

project = Path(__file__).resolve().parent
panel = project / "data_processed" / "financial_panel.parquet"
out = project / "outputs"
out.mkdir(exist_ok=True)
start, end = date(1984, 3, 31), date(2007, 12, 31)
columns = ["REPDTE", "EQ", "ASSET"]

def pandas_query():
    df = pd.read_parquet(
        panel, columns=columns,
        filters=[("REPDTE", ">=", start), ("REPDTE", "<=", end)],
    )
    df = df.loc[df["EQ"].notna() & df["ASSET"].gt(0)]
    result = df.groupby("REPDTE", as_index=False).agg(
        banks=("ASSET", "size"), equity=("EQ", "sum"), assets=("ASSET", "sum")
    )
    return result.assign(equity_assets=result["equity"] / result["assets"])

plan = (
    pl.scan_parquet(panel)
    .filter(pl.col("REPDTE").is_between(start, end))
    .filter(pl.col("EQ").is_not_null() & (pl.col("ASSET") > 0))
    .group_by("REPDTE")
    .agg(
        pl.len().alias("banks"),
        pl.col("EQ").sum().alias("equity"),
        pl.col("ASSET").sum().alias("assets"),
    )
    .with_columns((pl.col("equity") / pl.col("assets")).alias("equity_assets"))
    .sort("REPDTE")
)

# Warm-up and correctness check; conversion is outside the timed runs.
pandas_result = pandas_query().sort_values("REPDTE").reset_index(drop=True)
polars_result = plan.collect().to_pandas()
assert len(pandas_result) == len(polars_result) == 96
np.testing.assert_array_equal(
    pd.to_datetime(pandas_result["REPDTE"]).to_numpy(dtype="datetime64[D]"),
    pd.to_datetime(polars_result["REPDTE"]).to_numpy(dtype="datetime64[D]"),
)
np.testing.assert_array_equal(pandas_result["banks"], polars_result["banks"])
for name in ["equity", "assets", "equity_assets"]:
    np.testing.assert_allclose(
        pandas_result[name], polars_result[name], rtol=1e-10, atol=1e-12
    )
(out / "polars_quarterly_plan.txt").write_text(plan.explain(), encoding="utf-8")

queries = {"pandas": pandas_query, "Polars": plan.collect}
times = {name: [] for name in queries}
for repeat in range(4):
    order = ["pandas", "Polars"] if repeat % 2 == 0 else ["Polars", "pandas"]
    for name in order:
        t0 = perf_counter()
        queries[name]()
        times[name].append(perf_counter() - t0)

summary = pd.DataFrame([
    {"engine": name, **{f"run_{i}_s": t for i, t in enumerate(runs, 1)},
     "median_s": median(runs)}
    for name, runs in times.items()
])
summary.to_csv(out / "pandas_polars_benchmark.csv", index=False)
print("Matched quarterly results:", len(pandas_result))
print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
print("\nOptimized Polars plan:")
print(plan.explain())
