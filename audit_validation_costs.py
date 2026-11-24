from pathlib import Path
import pandas as pd

project = Path(__file__).resolve().parent
out = project / "outputs"

pred = pd.read_parquet(
    out / "development_predictions.parquet",
    columns=["CERT", "REPDTE", "y", "fold"],
)
positive = pred.loc[pred["y"].eq(1)].copy()
positive["REPDTE"] = pd.to_datetime(positive["REPDTE"])

events = pd.read_parquet(
    project / "data_raw" / "fdic" / "failures.parquet",
    columns=["CERT", "FAILDATE", "RESTYPE", "COST"],
)
events = events.loc[
    events["RESTYPE"].astype("string").str.strip().eq("FAILURE")
    & events["CERT"].notna()
].copy()
events["CERT"] = pd.to_numeric(events["CERT"], errors="raise").astype("int64")
events["FAILDATE"] = pd.to_datetime(events["FAILDATE"], format="mixed", errors="raise")

# Use the same first-failure definition as in build_labels.py.
first = (
    events.sort_values(["CERT", "FAILDATE"])
    .drop_duplicates("CERT", keep="first")
)
linked = positive.merge(first, on="CERT", how="left", validate="m:1")
horizon = linked["REPDTE"] + pd.offsets.QuarterEnd(4)
in_window = linked["FAILDATE"].gt(linked["REPDTE"]) & linked["FAILDATE"].le(horizon)
if not in_window.all():
    raise ValueError("A positive validation row does not match its first failure.")

unique = linked.drop_duplicates(["CERT", "FAILDATE"]).copy()
raw_cost = unique["COST"].astype("string").str.strip().replace("", pd.NA)
cost = pd.to_numeric(raw_cost, errors="coerce")
unique["numeric_cost"] = cost

print(f"Positive validation bank-quarters: {len(positive):,}")
print(f"Distinct validation failure events: {len(unique):,}")
print(f"Missing COST: {raw_cost.isna().sum():,}")
print(f"Non-empty but non-numeric COST: {(raw_cost.notna() & cost.isna()).sum():,}")
print(f"Negative COST: {cost.lt(0).sum():,}")
print(f"Zero COST: {cost.eq(0).sum():,}")
print(f"Positive COST: {cost.gt(0).sum():,}")
print("\nPositive COST quantiles, in source units:")
print(cost[cost.gt(0)].quantile([0.25, 0.50, 0.75, 0.95]).to_string())

unique[["CERT", "FAILDATE", "fold", "numeric_cost"]].to_csv(
    out / "validation_failure_cost_audit.csv", index=False
)
