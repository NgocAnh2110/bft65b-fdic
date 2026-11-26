from pathlib import Path

import polars as pl

processed = Path("data_processed")
outputs = Path("outputs")
outputs.mkdir(exist_ok=True)

key = ["CERT", "q"]
features = [
    "noncurrent_loans_lag1",
    "tier1_leverage_lag1",
    "change_equity_assets_lag1",
]

def quarter_index():
    day = pl.col("REPDTE")
    return day.dt.year() * 4 + day.dt.quarter().cast(pl.Int32)

def ratio(numerator, denominator):
    return (
        pl.when(pl.col(denominator) > 0)
        .then(pl.col(numerator) / pl.col(denominator))
        .otherwise(None)
    )

# Start with exactly the development observations used in the original model.
dev = (
    pl.scan_parquet(processed / "feature_table.parquet")
    .filter(pl.col("split") == "development")
    .with_columns(quarter_index().alias("q"))
)

reports = (
    pl.scan_parquet(processed / "financial_panel.parquet")
    .select(
        "CERT",
        quarter_index().alias("source_q"),
        ratio("NCLNLS", "LNLSGR").alias("noncurrent_loans_lag1"),
        pl.col("RBC1AAJ").alias("tier1_leverage_lag1"),
        ratio("EQ", "ASSET").alias("capital_ratio"),
    )
)

# A report at q-1 supplies the two level features and the later capital ratio.
lag1 = reports.select(
    "CERT",
    (pl.col("source_q") + 1).alias("q"),
    "noncurrent_loans_lag1",
    "tier1_leverage_lag1",
    pl.col("capital_ratio").alias("capital_q1"),
)

# A report at q-2 supplies the earlier capital ratio.
lag2 = reports.select(
    "CERT",
    (pl.col("source_q") + 2).alias("q"),
    pl.col("capital_ratio").alias("capital_q2"),
)

extended = (
    dev.join(lag1, on=key, how="left", validate="1:1")
    .join(lag2, on=key, how="left", validate="1:1")
    .with_columns(
        (pl.col("capital_q1") - pl.col("capital_q2"))
        .alias("change_equity_assets_lag1")
    )
    .drop("q", "capital_q1", "capital_q2")
    .collect()
)

if extended.height != 1_228_552 or int(extended["y"].sum()) != 8_759:
    raise ValueError("Development sample differs from the audited sample.")

summary = []
for name in features:
    missing = extended[name].null_count()
    summary.append({
        "feature": name,
        "available_rows": extended.height - missing,
        "missing_rows": missing,
        "coverage_pct": round(100 * (1 - missing / extended.height), 2),
    })

coverage = pl.DataFrame(summary)
coverage.write_csv(outputs / "extended_feature_coverage.csv")
extended.write_parquet(
    processed / "development_features_extended.parquet", compression="zstd"
)

print(f"Development bank-quarters: {extended.height:,}")
print(f"Positive bank-quarters: {int(extended['y'].sum()):,}")
print(coverage.to_pandas().to_string(index=False))
