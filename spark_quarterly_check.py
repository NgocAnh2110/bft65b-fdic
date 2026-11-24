from contextlib import redirect_stdout
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
from pyspark.sql import SparkSession, functions as F

project = Path(__file__).resolve().parent
panel = project / "data_processed" / "financial_panel.parquet"
out = project / "outputs"
out.mkdir(exist_ok=True)
start = date(1984, 3, 31)
end = date(2007, 12, 31)

# Reproduce the Polars benchmark query.
reference = (
    pl.scan_parquet(panel)
    .filter(pl.col("REPDTE").is_between(start, end))
    .filter(pl.col("EQ").is_not_null() & (pl.col("ASSET") > 0))
    .group_by("REPDTE")
    .agg(
        pl.len().alias("banks"),
        pl.col("EQ").sum().alias("equity"),
        pl.col("ASSET").sum().alias("assets"),
    )
    .with_columns((pl.col("equity") / pl.col("assets")).alias("equity_assets"))
    .sort("REPDTE")
    .collect()
)

spark = (
    SparkSession.builder
    .master("local[2]")
    .appName("BFT65B-quarterly-comparison")
    .config("spark.sql.shuffle.partitions", "4")
    .getOrCreate()
)

try:
    query = (
        spark.read.parquet(panel.as_uri())
        .select("REPDTE", "EQ", "ASSET")
        .where(F.col("REPDTE").between(start, end))
        .where(F.col("EQ").isNotNull() & (F.col("ASSET") > 0))
        .groupBy("REPDTE")
        .agg(
            F.count("*").alias("banks"),
            F.sum("EQ").alias("equity"),
            F.sum("ASSET").alias("assets"),
        )
        .withColumn("equity_assets", F.col("equity") / F.col("assets"))
        .orderBy("REPDTE")
    )

    # collect() executes the Spark query before we record its plan.
    rows = query.collect()
    with (out / "spark_quarterly_plan.txt").open("w", encoding="utf-8") as f:
        with redirect_stdout(f):
            query.explain(mode="formatted")

    if len(rows) != 96 or reference.height != 96:
        raise ValueError(f"Unexpected quarter counts: Spark={len(rows)}, Polars={reference.height}")

    if [r["REPDTE"] for r in rows] != reference["REPDTE"].to_list():
        raise ValueError("Reporting dates differ between Spark and Polars.")

    np.testing.assert_array_equal(
        [r["banks"] for r in rows], reference["banks"].to_numpy()
    )
    for name in ["equity", "assets", "equity_assets"]:
        np.testing.assert_allclose(
            [r[name] for r in rows], reference[name].to_numpy(),
            rtol=1e-10, atol=1e-12,
        )

    pl.DataFrame([r.asDict() for r in rows]).write_csv(out / "spark_quarterly_results.csv")
    print(f"Matched quarters: {len(rows)}")
    print(f"First quarter: {rows[0]['REPDTE']}")
    print(f"Last quarter: {rows[-1]['REPDTE']}")
    print(f"Spark plan saved: {out / 'spark_quarterly_plan.txt'}")
finally:
    spark.stop()
