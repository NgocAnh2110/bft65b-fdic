from datetime import date
from pathlib import Path

import pandas as pd
import polars as pl

project = Path(__file__).resolve().parent
processed = project / "data_processed"
outputs = project / "outputs"
outputs.mkdir(exist_ok=True)

# One documented first failure per bank.
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
first = events.groupby("CERT", as_index=False)["FAILDATE"].min()
first = pl.from_pandas(first).with_columns(pl.col("FAILDATE").cast(pl.Date))

# Keep only quarters with a complete four-quarter outcome window.
keys = (
    pl.scan_parquet(processed / "financial_panel.parquet")
    .select("CERT", "REPDTE")
    .filter(pl.col("REPDTE") <= date(2025, 6, 30))
    .collect()
)
joined = keys.join(first, on="CERT", how="left", validate="m:1")

q = pl.col("REPDTE")
failure = pl.col("FAILDATE")
eligible = joined.filter(failure.is_null() | (q < failure))

# The same quarter-end one year later is the end of the fourth following quarter.
labels = eligible.with_columns(
    ((failure > q) & (failure <= q.dt.offset_by("1y")))
    .fill_null(False)
    .cast(pl.UInt8)
    .alias("y"),

    pl.when(q <= date(2007, 12, 31)).then(pl.lit("development"))
    .when(q <= date(2008, 12, 31)).then(pl.lit("gap"))
    .when(q <= date(2014, 12, 31)).then(pl.lit("test"))
    .otherwise(pl.lit("outside_split"))
    .alias("split"),
)

removed = keys.height - labels.height
if removed != 1211:
    raise ValueError(f"Expected 1,211 exclusions from the audit, found {removed:,}")

summary = (
    labels.group_by("split")
    .agg(
        pl.len().alias("bank_quarters"),
        pl.col("y").cast(pl.Int64).sum().alias("positive_quarters"),
        pl.col("CERT").filter(pl.col("y") == 1).n_unique()
        .alias("distinct_first_failures"),
    )
    .with_columns(
        (100 * pl.col("positive_quarters") / pl.col("bank_quarters"))
        .round(3).alias("positive_pct")
    )
    .sort("split")
)

# The event date was used to construct the target. Keep it out of the model table.
labels.select("CERT", "REPDTE", "split", "y").write_parquet(
    processed / "modelling_labels.parquet", compression="zstd"
)
summary.write_csv(outputs / "label_split_summary.csv")

print(f"Rows through 2025Q2 before exclusion: {keys.height:,}")
print(f"Rows excluded on or after first failure: {removed:,}")
print(f"Eligible labelled rows: {labels.height:,}")
print(summary.to_pandas().to_string(index=False))
