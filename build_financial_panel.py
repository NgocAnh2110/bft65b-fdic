from pathlib import Path

import polars as pl
from fdic_download_starter import FIN_FIELDS

project = Path(__file__).resolve().parent
raw = project / "data_raw" / "fdic" / "financials"
output = project / "data_processed" / "financial_panel.parquet"
output.parent.mkdir(exist_ok=True)
temporary = output.with_suffix(".tmp.parquet")

files = sorted(raw.glob("REPDTE=*.parquet"))
if len(files) != 170:
    raise ValueError(f"Expected 170 quarterly files, found {len(files)}")

# One schema for all quarters, including fields that were entirely null.
schema = {name: pl.Float64 for name in FIN_FIELDS}
schema.update({"CERT": pl.Int64, "REPDTE": pl.Date, "STNAME": pl.String})

quarters = []
for path in files:
    source = pl.scan_parquet(path)
    available = set(source.collect_schema().names())
    missing_keys = {"CERT", "REPDTE"} - available
    if missing_keys:
        raise ValueError(f"{path.name}: missing key columns {missing_keys}")

    expressions = []
    for name in FIN_FIELDS:
        if name not in available:
            expr = pl.lit(None, dtype=schema[name])
        elif name == "REPDTE":
            expr = pl.col(name).cast(pl.String).str.strptime(
                pl.Date, format="%Y%m%d", strict=True
            )
        else:
            expr = pl.col(name).cast(schema[name], strict=True)
        expressions.append(expr.alias(name))

    quarters.append(source.select(expressions))

panel = pl.concat(quarters, how="vertical")
panel.sink_parquet(temporary, compression="zstd")

# Read back the written output and compare it with the raw-data audit.
checks = pl.scan_parquet(temporary).select(
    pl.len().alias("rows"),
    pl.col("REPDTE").n_unique().alias("quarters"),
    pl.col("CERT").n_unique().alias("banks"),
    pl.struct(["CERT", "REPDTE"]).n_unique().alias("unique_keys"),
    pl.col("P3ASSET").null_count().alias("p3asset_nulls"),
).collect().row(0, named=True)

expected = {
    "rows": 1682738,
    "quarters": 170,
    "banks": 24064,
    "unique_keys": 1682738,
    "p3asset_nulls": 972047,
}
if checks != expected:
    raise ValueError(f"Panel differs from the audited snapshot: {checks}")

temporary.replace(output)
for name, value in checks.items():
    print(f"{name}: {value:,}")
print(f"Saved: {output}")
