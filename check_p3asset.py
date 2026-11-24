from pathlib import Path
import pandas as pd

output_dir = Path(__file__).resolve().parent / "outputs"
audit = pd.read_csv(
    output_dir / "missing_by_quarter.csv",
    dtype={"report_date": str},
)

p3 = audit.loc[audit["column"] == "P3ASSET"].copy()
p3 = p3.sort_values("report_date")
p3["year"] = p3["report_date"].str[:4]
p3["all_missing"] = p3["missing_rows"] == p3["quarter_rows"]

yearly = p3.groupby("year").agg(
    quarters=("report_date", "count"),
    total_rows=("quarter_rows", "sum"),
    missing_rows=("missing_rows", "sum"),
    quarters_all_missing=("all_missing", "sum"),
)

# Tính trên tổng số dòng của năm vì các quý có thể khác số ngân hàng.
yearly["missing_pct"] = (
    100 * yearly["missing_rows"] / yearly["total_rows"]
).round(2)

print(yearly.to_string())

available = p3.loc[p3["missing_rows"] < p3["quarter_rows"], "report_date"]
if not available.empty:
    print(f"\nFirst quarter with any reported value: {available.iloc[0]}")

print(f"Quarters with all values missing: {p3['all_missing'].sum()} / {len(p3)}")
yearly.to_csv(output_dir / "p3asset_missing_by_year.csv")