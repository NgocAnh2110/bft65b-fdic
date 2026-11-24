from pathlib import Path
import polars as pl

project = Path(__file__).resolve().parent
processed = project / "data_processed"
outputs = project / "outputs"
outputs.mkdir(exist_ok=True)

labels = pl.scan_parquet(processed / "modelling_labels.parquet")
financials = pl.scan_parquet(processed / "financial_panel.parquet")

# A calendar-quarter index increases by exactly one per quarter.
def quarter_id(column):
    return column.dt.year().cast(pl.Int32) * 4 + column.dt.quarter().cast(pl.Int32)

def safe_ratio(numerator, denominator):
    return (
        pl.when(pl.col(denominator) > 0)
        .then(pl.col(numerator) / pl.col(denominator))
        .otherwise(None)
    )

features = [
    "log_assets_lag1",
    "equity_assets_lag1",
    "deposits_assets_lag1",
    "reserves_loans_lag1",
    "roa_lag1",
]

# Each financial report supplies features for the next calendar quarter.
source = financials.select(
    "CERT",
    (quarter_id(pl.col("REPDTE")) + 1).alias("prediction_quarter"),
    pl.col("REPDTE").alias("source_report"),
    pl.when(pl.col("ASSET") > 0)
      .then(pl.col("ASSET").log())
      .otherwise(None)
      .alias("log_assets_lag1"),
    safe_ratio("EQ", "ASSET").alias("equity_assets_lag1"),
    safe_ratio("DEP", "ASSET").alias("deposits_assets_lag1"),
    safe_ratio("LNATRES", "LNLSGR").alias("reserves_loans_lag1"),
    pl.col("ROA").alias("roa_lag1"),
)

table = (
    labels.with_columns(quarter_id(pl.col("REPDTE")).alias("prediction_quarter"))
    .join(source, on=["CERT", "prediction_quarter"], how="left", validate="1:1")
    .drop("prediction_quarter")
    .collect()
)

# Verify the actual date of every matched financial report.
bad_dates = table.filter(
    pl.col("source_report").is_not_null()
    & (quarter_id(pl.col("source_report")) + 1 != quarter_id(pl.col("REPDTE")))
).height

if table.height != 1663998 or bad_dates != 0:
    raise ValueError(f"Unexpected feature table: {table.height} rows, {bad_dates} date errors")

dev = table.filter(pl.col("split") == "development")
coverage = pl.DataFrame({
    "feature": features,
    "development_rows": [dev.height] * len(features),
    "missing_rows": [dev[name].null_count() for name in features],
}).with_columns(
    (100 * pl.col("missing_rows") / pl.col("development_rows"))
    .round(3).alias("missing_pct")
)

table.write_parquet(processed / "feature_table.parquet", compression="zstd")
coverage.write_csv(outputs / "development_feature_coverage.csv")

print(f"Feature table rows: {table.height:,}")
print(f"Incorrect source quarters: {bad_dates:,}")
print(f"Development rows without a prior-quarter report: {dev['source_report'].null_count():,}")
print(coverage.to_pandas().to_string(index=False))
