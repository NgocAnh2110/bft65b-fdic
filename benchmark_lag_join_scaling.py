import json
import subprocess
import sys
from datetime import date
from pathlib import Path
from time import perf_counter

import pandas as pd
import psutil

import check_lag_join_engines as joins


def set_window(quarters):
    first_year = 2008 - quarters // 4
    joins.START = date(first_year, 3, 31)
    joins.SOURCE_START = date(first_year - 1, 12, 31)
    # END and SOURCE_END stay at 2007Q4 and 2007Q3.


def verify(quarters):
    set_window(quarters)
    left = joins.pandas_join()
    right = joins.polars_join().to_pandas()

    for frame in (left, right):
        frame.sort_values(["CERT", "q"], inplace=True, ignore_index=True)
        frame["source_date"] = pd.to_datetime(frame["source_date"]).astype("datetime64[ns]")

    pd.testing.assert_frame_equal(
        left, right, check_dtype=False, check_exact=False,
        rtol=1e-12, atol=1e-12
    )
    if left["q"].nunique() != quarters:
        raise ValueError(f"Expected {quarters} prediction quarters")
    return len(left)


def measure(quarters, engine):
    set_window(quarters)
    run_join = {"pandas": joins.pandas_join, "polars": joins.polars_join}[engine]

    start = perf_counter()
    result = run_join()
    seconds = perf_counter() - start
    peak_mib = psutil.Process().memory_info().peak_wset / (1024 ** 2)

    return {"rows": len(result), "seconds": seconds, "peak_MiB": peak_mib}


def benchmark():
    rows = []
    for quarters in (48, 96):
        expected = verify(quarters)
        print(f"{quarters} quarters: PASS, {expected:,} bank-quarters", flush=True)

        for run in range(4):
            order = ("pandas", "polars") if run % 2 == 0 else ("polars", "pandas")
            for engine in order:
                output = subprocess.check_output(
                    [sys.executable, __file__, str(quarters), engine], text=True
                )
                result = json.loads(output)
                if result["rows"] != expected:
                    raise ValueError(f"Row count changed for {engine}")
                if run > 0:
                    rows.append({
                        "quarters": quarters, "engine": engine,
                        "run": run, **result
                    })

    old = pd.read_csv("outputs/lag_join_benchmark_24q.csv").assign(quarters=24)
    all_runs = pd.concat([old, pd.DataFrame(rows)], ignore_index=True)
    path = Path("outputs/lag_join_scaling.csv")
    all_runs.to_csv(path, index=False)

    summary = all_runs.groupby(["quarters", "engine"]).agg(
        bank_quarters=("rows", "first"),
        median_seconds=("seconds", "median"),
        median_peak_MiB=("peak_MiB", "median"),
    )
    print(summary.round(3).to_string())
    print(f"Saved: {path}")


if __name__ == "__main__":
    if len(sys.argv) == 3:
        print(json.dumps(measure(int(sys.argv[1]), sys.argv[2])))
    else:
        benchmark()
