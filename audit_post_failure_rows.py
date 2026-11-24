from datetime import date
from pathlib import Path

import pandas as pd
import polars as pl

project = Path(__file__).resolve().parent
panel = project / "data_processed" / "financial_panel.parquet"
out = project / "outputs"

events = pd.read_parquet(
    project / "data_raw" / "fdic" / "failures.parquet",
    columns=["CERT", "FAILDATE", "RESTYPE"],
)
events = events.loc[
    events["RESTYPE"].astype("string").str.strip().eq("FAILURE")
    & events["CERT"].notna()
].copy()
events["CERT"] = pd.to_numeric(events["CERT"], errors="raise").astype("int64")
events["FAILDATE"] = pd.to_datetime(events["FAILDATE"], format="mixed", errors="raise")
events = events.loc[events["FAILDATE"] <= "2026-06-30"]

# One failure date per CERT keeps the join from multiplying panel rows.
first = events.groupby("CERT", as_index=False)["FAILDATE"].min()
first = pl.from_pandas(first).with_columns(pl.col("FAILDATE").cast(pl.Date))

affected = (
    pl.scan_parquet(panel)
    .select("CERT", "REPDTE")
    .join(first.lazy(), on="CERT", how="inner")
    .filter(pl.col("REPDTE") >= pl.col("FAILDATE"))
    .with_columns(
        pl.when(pl.col("REPDTE") == pl.col("FAILDATE"))
        .then(pl.lit("same_day"))
        .otherwise(pl.lit("after_failure"))
        .alias("timing")
    )
    .sort(["CERT", "REPDTE"])
    .collect()
)

affected.write_csv(out / "post_failure_quarters.csv")

for name, frame in [
    ("Full panel through 2026Q2", affected),
    ("Modelling quarters through 2025Q2",
     affected.filter(pl.col("REPDTE") <= date(2025, 6, 30))),
]:
    print(f"\n{name}")
    for timing in ["same_day", "after_failure"]:
        subset = frame.filter(pl.col("timing") == timing)
        print(f"{timing}: {subset.height:,} rows, {subset['CERT'].n_unique():,} banks")
