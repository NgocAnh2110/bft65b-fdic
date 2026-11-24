from pathlib import Path

import pandas as pd

data_dir = Path(__file__).resolve().parent / "data_raw" / "fdic"

# Đọc một quý mẫu và kiểm tra số dòng của bảng Institutions.
quarter = pd.read_parquet(data_dir / "financials" / "REPDTE=20230630.parquet")
institutions = pd.read_parquet(data_dir / "institutions.parquet", columns=["CERT"])

print(f"Institutions rows: {len(institutions):,}")
print(f"2023Q2: {len(quarter):,} rows, {len(quarter.columns)} columns")

# Đếm giá trị thiếu theo cột và quy đổi ra phần trăm.
missing = quarter.isna().sum()
audit = pd.DataFrame({
    "missing_rows": missing,
    "missing_pct": (100 * missing / len(quarter)).round(2),
})

print("\nMissing values in 2023Q2:")
with_missing = audit.loc[audit["missing_rows"] > 0]
if with_missing.empty:
    print("No missing values in this quarter.")
else:
    print(with_missing.sort_values("missing_rows", ascending=False).to_string())
    # Audit missing values across all reporting quarters.
files = sorted((data_dir / "financials").glob("REPDTE=*.parquet"))
if len(files) != 170:
    raise ValueError(f"Expected 170 quarterly files, found {len(files)}")

records = []
total_rows = 0

for path in files:
    df = pd.read_parquet(path)
    report_date = path.stem.split("=")[1]
    total_rows += len(df)

    for column, count in df.isna().sum().items():
        records.append({
            "report_date": report_date,
            "column": column,
            "quarter_rows": len(df),
            "missing_rows": int(count),
        })

by_quarter = pd.DataFrame(records)
by_quarter["missing_pct"] = (
    100 * by_quarter["missing_rows"] / by_quarter["quarter_rows"]
).round(2)

summary = by_quarter.groupby("column").agg(
    quarters_present=("report_date", "nunique"),
    rows_in_present_quarters=("quarter_rows", "sum"),
    null_rows=("missing_rows", "sum"),
)

# Nếu một quý thiếu hẳn cột, mọi dòng của quý đó đều thiếu biến này.
summary["quarters_absent"] = len(files) - summary["quarters_present"]
summary["rows_in_absent_quarters"] = total_rows - summary["rows_in_present_quarters"]
summary["missing_rows"] = summary["null_rows"] + summary["rows_in_absent_quarters"]
summary["missing_pct"] = (100 * summary["missing_rows"] / total_rows).round(2)
summary = summary.sort_values("missing_rows", ascending=False)

output_dir = Path(__file__).resolve().parent / "outputs"
output_dir.mkdir(exist_ok=True)
by_quarter.to_csv(output_dir / "missing_by_quarter.csv", index=False)
summary.to_csv(output_dir / "missing_summary.csv")

print(f"\nAudited {len(files)} quarters and {total_rows:,} bank-quarter rows.")
print("\nTop 10 columns by missing values:")
print(summary[["missing_rows", "missing_pct", "quarters_absent"]].head(10))
print(f"\nResults saved to {output_dir}")