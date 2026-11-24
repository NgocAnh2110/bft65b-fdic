from pathlib import Path
import pandas as pd

project = Path(__file__).resolve().parent
data = project / "data_raw" / "fdic"

# Select failure events in the study window.
events = pd.read_parquet(
    data / "failures.parquet", columns=["CERT", "FAILDATE", "RESTYPE"]
)
events["FAILDATE"] = pd.to_datetime(events["FAILDATE"], format="mixed", errors="raise")
events = events.loc[
    events["RESTYPE"].astype("string").str.strip().eq("FAILURE")
    & events["FAILDATE"].gt("1984-03-31")
    & events["FAILDATE"].le("2026-06-30"),
    ["CERT", "FAILDATE"],
].copy()
events["CERT"] = pd.to_numeric(events["CERT"], errors="raise").astype("int64")
events = events.reset_index(drop=True)
events.index.name = "event_id"

# Find reports for these banks, using only modelling quarters through 2025Q2.
failure_certs = set(events["CERT"])
parts = []
for path in sorted((data / "financials").glob("REPDTE=*.parquet")):
    quarter = path.stem.split("=")[1]
    if quarter > "20250630":
        continue
    keys = pd.read_parquet(path, columns=["CERT", "REPDTE"])
    keys["CERT"] = pd.to_numeric(keys["CERT"], errors="raise").astype("int64")
    parts.append(keys.loc[keys["CERT"].isin(failure_certs)])

reports = pd.concat(parts, ignore_index=True)
reports["REPDTE"] = pd.to_datetime(
    reports["REPDTE"].astype("string"), format="%Y%m%d", errors="raise"
)

# A report qualifies when REPDTE < FAILDATE <= end of the fourth following quarter.
pairs = events.reset_index().merge(reports, on="CERT", how="inner")
horizon_end = pairs["REPDTE"] + pd.offsets.QuarterEnd(4)
qualifies = pairs["REPDTE"].lt(pairs["FAILDATE"]) & pairs["FAILDATE"].le(horizon_end)

events["cert_in_model_panel"] = events["CERT"].isin(reports["CERT"])
counts = pairs.loc[qualifies].groupby("event_id").size()
events["eligible_reports"] = counts.reindex(events.index, fill_value=0)

covered = events["eligible_reports"].gt(0)
print(f"Failure events checked: {len(events):,}")
print(f"Events with CERT in modelling panel: {events['cert_in_model_panel'].sum():,}")
print(f"Events with at least one eligible prior report: {covered.sum():,}")
print(f"Events without an eligible prior report: {(~covered).sum():,}")

output = project / "outputs" / "failure_event_coverage.csv"
output.parent.mkdir(exist_ok=True)
events.to_csv(output)

# Count distinct failure events covered by each set of reporting quarters.
eligible = pairs.loc[qualifies, ["event_id", "CERT", "REPDTE"]].copy()
periods = {
    "Development": ("1984-03-31", "2007-12-31"),
    "Test": ("2009-03-31", "2014-12-31"),
}

rows = []
for name, (start, end) in periods.items():
    selected = eligible.loc[eligible["REPDTE"].between(start, end)]
    rows.append({
        "period": name,
        "first_report": start,
        "last_report": end,
        "positive_bank_quarters": len(selected.drop_duplicates(["CERT", "REPDTE"])),
        "distinct_failure_events": selected["event_id"].nunique(),
    })

split_summary = pd.DataFrame(rows)
print("\nSplit coverage:")
print(split_summary.to_string(index=False))
split_summary.to_csv(project / "outputs" / "split_failure_counts.csv", index=False)
