from pathlib import Path
import pandas as pd

project = Path(__file__).resolve().parent
failures = pd.read_parquet(project / "data_raw" / "fdic" / "failures.parquet")

def clean_text(column):
    return column.astype("string").str.strip().replace("", pd.NA)

cert = clean_text(failures["CERT"])
date_raw = clean_text(failures["FAILDATE"])
date = pd.to_datetime(date_raw, format="mixed", errors="coerce")
event_type = clean_text(failures["RESTYPE"])

keys = pd.DataFrame({"CERT": cert, "FAILDATE": date}).dropna()
print(f"Failure-table rows: {len(failures):,}")
print(f"Missing CERT: {cert.isna().sum():,}")
print(f"Missing FAILDATE: {date_raw.isna().sum():,}")
print(f"Unparseable non-empty FAILDATE: {(date_raw.notna() & date.isna()).sum():,}")
print(f"Duplicate CERT-FAILDATE rows beyond the first: {keys.duplicated().sum():,}")

print("\nRESTYPE counts:")
print(event_type.fillna("(missing)").value_counts().to_string())

# Count missing bank identifiers by event type and date range.
events = pd.DataFrame({
    "RESTYPE": event_type,
    "FAILDATE": date,
    "missing_cert": cert.isna(),
})

def summarize_events(frame):
    result = frame.groupby("RESTYPE", dropna=False).agg(
        event_rows=("missing_cert", "size"),
        missing_cert=("missing_cert", "sum"),
    )
    result["with_cert"] = result["event_rows"] - result["missing_cert"]
    return result

print("\nAll dates:")
print(summarize_events(events).to_string())

# A failure must occur after the first observation to enter a future label.
in_window = events["FAILDATE"].gt("1984-03-31") & events["FAILDATE"].le("2026-06-30")
print("\nAfter 1984-03-31 through 2026-06-30:")
print(summarize_events(events.loc[in_window]).to_string())
