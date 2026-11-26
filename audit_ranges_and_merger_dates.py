from io import BytesIO
from pathlib import Path

import pandas as pd
import polars as pl
import requests

out = Path("outputs")
out.mkdir(exist_ok=True)

# 1. Count flagged bank-quarters and distinct banks in the original panel.
panel = pl.scan_parquet("data_processed/financial_panel.parquet")
rules = {
    "ASSET <= 0": pl.col("ASSET") <= 0,
    "EQ > ASSET": pl.col("EQ") > pl.col("ASSET"),
    "EQ < 0": pl.col("EQ") < 0,
    "LNLSGR <= 0": pl.col("LNLSGR") <= 0,
}

metrics = [pl.len().alias("total")]
for name, rule in rules.items():
    metrics.extend([
        rule.sum().alias(name + "_rows"),
        pl.col("CERT").filter(rule).n_unique().alias(name + "_banks"),
    ])

counts = panel.select(metrics).collect().row(0, named=True)
ranges = pd.DataFrame([
    {
        "check": name,
        "panel_rows": counts["total"],
        "flagged_rows": counts[name + "_rows"],
        "flagged_banks": counts[name + "_banks"],
    }
    for name in rules
])
ranges.to_csv(out / "financial_range_checks.csv", index=False)
print(ranges.to_string(index=False), flush=True)

# 2. Identify code-223 banks in the project's existing exit audit.
audit = pd.read_csv(out / "bank_exit_audit.csv")
audit["last_report"] = pd.to_datetime(audit["last_report"], errors="raise")
cutoff = pd.Timestamp("2026-06-30")
code = pd.to_numeric(audit["CHANGEC1"], errors="coerce")

group = audit.loc[
    audit["last_report"].lt(cutoff)
    & audit["matched_failure_date"].isna()
    & code.eq(223),
    ["CERT", "last_report"],
].copy()

if group["CERT"].isna().any() or not group["CERT"].is_unique:
    raise ValueError("Expected one identified bank per exit-audit row.")

group["window_end"] = group["last_report"] + pd.offsets.QuarterEnd(1)

# 3. Retrieve dated FDIC merger records and verify pagination.
url = "https://api.fdic.gov/banks/history"
with requests.Session() as session:
    def fetch(params):
        response = session.get(url, params=params, timeout=120)
        response.raise_for_status()
        return response

    count = fetch({
        "filters": "CHANGECODE:223", "fields": "CHANGECODE", "limit": 1
    })
    expected = int(count.json()["meta"]["total"])

    pages = []
    for offset in range(0, expected, 10000):
        response = fetch({
            "filters": "CHANGECODE:223",
            "fields": "CHANGECODE,EFFDATE,OUT_CERT,SUR_CERT",
            "sort_by": "EFFDATE", "sort_order": "ASC",
            "limit": 10000, "offset": offset, "format": "csv",
        })
        pages.append(pd.read_csv(BytesIO(response.content)))

history = pd.concat(pages, ignore_index=True)
if len(history) != expected or history["ID"].nunique() != expected:
    raise ValueError("FDIC history count or unique-record check failed.")

history["OUT_CERT"] = pd.to_numeric(history["OUT_CERT"], errors="coerce")
history["effective"] = pd.to_datetime(history["EFFDATE"], format="mixed")
history = history.loc[history["effective"].le(cutoff)]

# Count each outgoing bank once, even if it has several history records.
pairs = group.merge(history, left_on="CERT", right_on="OUT_CERT", how="left")
in_window = pairs["effective"].between(pairs["last_report"], pairs["window_end"])
with_history = pairs.loc[pairs["effective"].notna(), "CERT"].nunique()
matched = pairs.loc[in_window, "CERT"].nunique()

exits = pd.DataFrame([{
    "cutoff": cutoff.date(),
    "code_223_banks": len(group),
    "within_window": matched,
    "only_outside_window": with_history - matched,
    "no_dated_history": len(group) - with_history,
    "confirmed_pct": round(100 * matched / len(group), 2),
}])
exits.to_csv(out / "merger_date_checks.csv", index=False)
print()
print(exits.to_string(index=False))
print("Saved: outputs/financial_range_checks.csv")
print("Saved: outputs/merger_date_checks.csv")
