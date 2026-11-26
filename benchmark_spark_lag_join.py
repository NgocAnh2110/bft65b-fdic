import os
import sys
from pathlib import Path
from statistics import median
from time import perf_counter

import pandas as pd
from pyspark.sql import SparkSession, functions as F

from check_lag_join_engines import pandas_join

os.environ["PYSPARK_PYTHON"] = sys.executable
output = Path("outputs")
output.mkdir(exist_ok=True)

start = perf_counter()
spark = (
    SparkSession.builder
    .master("local[2]")
    .appName("FDIC prior-quarter join")
    .config("spark.sql.shuffle.partitions", "4")
    .config("spark.ui.enabled", "false")
    .getOrCreate()
)
startup = perf_counter() - start
spark.sparkContext.setLogLevel("WARN")

try:
    def run_join():
        day = F.col("REPDTE")
        quarter = F.year(day) * 4 + F.quarter(day)

        targets = (
            spark.read.parquet("data_processed/modelling_labels.parquet")
            .filter(day.between("2002-03-31", "2007-12-31"))
            .select("CERT", quarter.alias("q"))
        )
        source = (
            spark.read.parquet("data_processed/financial_panel.parquet")
            .filter(day.between("2001-12-31", "2007-09-30"))
            .select(
                "CERT",
                (quarter + 1).alias("q"),
                day.alias("source_date"),
                F.when(F.col("ASSET") > 0, F.col("EQ") / F.col("ASSET")).alias("ratio"),
            )
        )
        return targets.join(source, ["CERT", "q"], "left")

    timings = []
    for run in range(4):
        start = perf_counter()
        result = run_join().toPandas()
        elapsed = perf_counter() - start

        if run > 0:
            timings.append(elapsed)
        label = "Warm-up" if run == 0 else f"Run {run}"
        print(f"{label}: {elapsed:.3f} seconds", flush=True)
finally:
    spark.stop()

# Compare outside the timed runs.
expected = pandas_join()
columns = ["CERT", "q", "source_date", "ratio"]

def ordered(frame):
    frame = frame[columns].sort_values(["CERT", "q"]).reset_index(drop=True)
    frame["source_date"] = pd.to_datetime(frame["source_date"]).astype("datetime64[ns]")
    return frame

pd.testing.assert_frame_equal(
    ordered(expected), ordered(result),
    check_dtype=False, check_exact=False, rtol=1e-12, atol=1e-12
)

pd.DataFrame({
    "run": [1, 2, 3],
    "seconds": timings,
    "startup_seconds": startup,
    "bank_quarters": len(result),
}).to_csv(output / "spark_lag_join_24q.csv", index=False)

print(f"Spark startup: {startup:.3f} seconds")
print(f"Median query and collection: {median(timings):.3f} seconds")
print(f"Bank-quarters compared: {len(result):,}")
print("Spark vs pandas: PASS")
