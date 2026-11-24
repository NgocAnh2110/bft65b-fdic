from pathlib import Path

import pandas as pd
import polars as pl

project = Path(__file__).resolve().parent
raw = project / "data_raw" / "fdic"
panel = project / "data_processed" / "financial_panel.parquet"
output = project / "outputs"
output.mkdir(exist_ok=True)
cutoff = pd.Timestamp("2026-06-30")

# Find the last observed report for each bank.
banks = (
    pl.scan_parquet(panel)
    .group_by("CERT")
    .agg(pl.col("REPDTE").max().alias("last_report"))
    .collect()
    .to_pandas()
)
banks["last_report"] = pd.to_datetime(banks["last_report"])

# Attach institution records without multiplying bank rows.
inst = pd.read_parquet(raw / "institutions.parquet")
inst["CERT"] = pd.to_numeric(inst["CERT"], errors="raise").astype("Int64")
fields = ["CERT", "ACTIVE", "CHANGEC1", "ENDEFYMD"]
print("Unavailable institution fields:", [c for c in fields if c not in inst])
keep = [c for c in fields if c in inst]
audit = banks.merge(
    inst.loc[inst["CERT"].notna(), keep],
    on="CERT", how="left", validate="one_to_one", indicator=True,
)

# Match only documented failures on or after the last report, through cutoff.
events = pd.read_parquet(
    raw / "failures.parquet", columns=["CERT", "FAILDATE", "RESTYPE"]
)
events = events.loc[
    events["RESTYPE"].astype("string").str.strip().eq("FAILURE")
    & events["CERT"].notna()
].copy()
events["CERT"] = pd.to_numeric(events["CERT"], errors="raise").astype("Int64")
events["FAILDATE"] = pd.to_datetime(events["FAILDATE"], format="mixed", errors="raise")

linked = banks.merge(events[["CERT", "FAILDATE"]], on="CERT", how="inner")
matched = linked.loc[
    linked["FAILDATE"].ge(linked["last_report"])
    & linked["FAILDATE"].le(cutoff)
]
audit["matched_failure_date"] = audit["CERT"].map(
    matched.groupby("CERT")["FAILDATE"].min()
)

early = audit["last_report"].lt(cutoff)
failure = audit["matched_failure_date"].notna()
remaining = audit.loc[early & ~failure]

print(f"Panel banks: {len(audit):,}")
print(f"Banks unmatched to Institutions: {audit['_merge'].eq('left_only').sum():,}")
print(f"Banks reporting in 2026Q2: {(~early).sum():,}")
print(f"Banks whose last report was before 2026Q2: {early.sum():,}")
print(f"  With a documented failure on or after last report: {(early & failure).sum():,}")
print(f"  Without such a failure by cutoff: {len(remaining):,}")

if "CHANGEC1" in remaining.columns:
    print("\nCHANGEC1 among early-ending banks without a matched failure:")
    print(remaining["CHANGEC1"].value_counts(dropna=False).to_string())

audit.sort_values("CERT").to_csv(output / "bank_exit_audit.csv", index=False)
