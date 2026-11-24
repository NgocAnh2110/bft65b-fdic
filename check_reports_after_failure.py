from pathlib import Path
import pandas as pd

project = Path(__file__).resolve().parent
out = project / "outputs"

banks = pd.read_csv(out / "bank_exit_audit.csv", usecols=["CERT", "last_report"])
banks["last_report"] = pd.to_datetime(banks["last_report"], errors="raise")

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

linked = banks.merge(events, on="CERT", how="inner", validate="one_to_many")
flagged = linked.loc[
    linked["last_report"].gt(linked["FAILDATE"]),
    ["CERT", "FAILDATE", "last_report"],
].sort_values(["CERT", "FAILDATE"])

flagged.to_csv(out / "reports_after_failure_review.csv", index=False)
print(f"Banks with a report after a failure date: {flagged['CERT'].nunique():,}")
print(flagged.head(5).to_string(index=False))
