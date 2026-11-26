import json
import subprocess
import sys
from pathlib import Path
from time import perf_counter

import pandas as pd
import polars as pl

PANEL = "data_processed/financial_panel.parquet"
EVENTS = "data_raw/fdic/failures.parquet"
LABELS = "data_processed/modelling_labels.parquet"
END = pd.Timestamp("2007-12-31")


def pandas_labels():
    reports = pd.read_parquet(PANEL, columns=["CERT", "REPDTE"])
    reports["REPDTE"] = pd.to_datetime(reports["REPDTE"])
    reports = reports.loc[reports["REPDTE"] <= END]

    events = pd.read_parquet(EVENTS, columns=["CERT", "RESTYPE", "FAILDATE"])
    events = events.loc[
        events["RESTYPE"].astype("string").str.strip().eq("FAILURE")
        & events["CERT"].notna()
    ].copy()
    events["CERT"] = pd.to_numeric(events["CERT"]).astype("int64")
    events["FAILDATE"] = pd.to_datetime(events["FAILDATE"], format="mixed")
    first = events.groupby("CERT", as_index=False)["FAILDATE"].min()

    joined = reports.merge(first, on="CERT", how="left", validate="many_to_one")
    eligible = joined.loc[
        joined["FAILDATE"].isna() | joined["REPDTE"].lt(joined["FAILDATE"])
    ].copy()
    horizon = eligible["REPDTE"] + pd.DateOffset(years=1)
    eligible["y"] = eligible["FAILDATE"].le(horizon).astype("uint8")
    return eligible[["CERT", "REPDTE", "y"]]


def polars_labels():
    reports = (
        pl.scan_parquet(PANEL)
        .select("CERT", "REPDTE")
        .filter(pl.col("REPDTE") <= pl.lit(END.date()))
    )
    # Parse the FDIC date strings consistently with the original project.
    events = pd.read_parquet(EVENTS, columns=["CERT", "RESTYPE", "FAILDATE"])
    events = events.loc[
        events["RESTYPE"].astype("string").str.strip().eq("FAILURE")
        & events["CERT"].notna()
    ].copy()
    events["CERT"] = pd.to_numeric(events["CERT"]).astype("int64")
    events["FAILDATE"] = pd.to_datetime(events["FAILDATE"], format="mixed")

    first = (
        pl.from_pandas(events[["CERT", "FAILDATE"]]).lazy()
        .group_by("CERT")
        .agg(pl.col("FAILDATE").min().cast(pl.Date))
    )
    day = pl.col("REPDTE")
    failure = pl.col("FAILDATE")
    return (
        reports.join(first, on="CERT", how="left", validate="m:1")
        .filter(failure.is_null() | (day < failure))
        .select(
            "CERT", "REPDTE",
            (failure <= day.dt.offset_by("1y")).fill_null(False).cast(pl.UInt8).alias("y")
        )
        .collect()
    )


def ordered(frame):
    frame = frame.copy()
    frame["REPDTE"] = pd.to_datetime(frame["REPDTE"]).astype("datetime64[ns]")
    return frame.sort_values(["CERT", "REPDTE"]).reset_index(drop=True)


def benchmark():
    expected = pd.read_parquet(LABELS, columns=["CERT", "REPDTE", "y"])
    expected = ordered(expected.loc[pd.to_datetime(expected["REPDTE"]) <= END])
    if expected.duplicated(["CERT", "REPDTE"]).any():
        raise ValueError("Duplicate keys in the reference labels")

    for result in (pandas_labels(), polars_labels().to_pandas()):
        pd.testing.assert_frame_equal(
            expected, ordered(result), check_dtype=False, check_exact=True
        )

    print(f"Labels checked: {len(expected):,}", flush=True)
    print(f"Positive bank-quarters: {int(expected['y'].sum()):,}", flush=True)
    print("Both engines vs saved labels: PASS", flush=True)

    records = []
    for run in range(4):
        order = ("pandas", "polars") if run % 2 == 0 else ("polars", "pandas")
        for engine in order:
            output = subprocess.check_output(
                [sys.executable, __file__, engine], text=True
            )
            if run > 0:
                records.append({"run": run, "engine": engine, **json.loads(output)})

    results = pd.DataFrame(records)
    path = Path("outputs/label_benchmark_96q.csv")
    results.to_csv(path, index=False)
    print(results.groupby("engine")["seconds"].median().round(3).to_string())
    print(f"Saved: {path}")


if __name__ == "__main__":
    if len(sys.argv) == 2:
        build = {"pandas": pandas_labels, "polars": polars_labels}[sys.argv[1]]
        start = perf_counter()
        result = build()
        print(json.dumps({"seconds": perf_counter() - start, "rows": len(result)}))
    else:
        benchmark()
