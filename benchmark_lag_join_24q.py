import json
import subprocess
import sys
from pathlib import Path
from time import perf_counter

import pandas as pd
import psutil

from check_lag_join_engines import pandas_join, polars_join


def measure(engine):
    run_join = {"pandas": pandas_join, "polars": polars_join}[engine]
    start = perf_counter()
    result = run_join()
    seconds = perf_counter() - start
    peak_mib = psutil.Process().memory_info().peak_wset / (1024 ** 2)

    if len(result) != 218_364:
        raise ValueError(f"Unexpected row count: {len(result)}")

    return {
        "engine": engine,
        "seconds": seconds,
        "peak_MiB": peak_mib,
        "rows": len(result),
    }


def benchmark():
    records = []
    for run in range(4):
        order = ("pandas", "polars") if run % 2 == 0 else ("polars", "pandas")
        for engine in order:
            output = subprocess.check_output(
                [sys.executable, __file__, engine], text=True
            )
            result = json.loads(output)
            if run > 0:
                records.append({"run": run, **result})

    results = pd.DataFrame(records)
    destination = Path("outputs/lag_join_benchmark_24q.csv")
    destination.parent.mkdir(exist_ok=True)
    results.to_csv(destination, index=False)

    summary = results.groupby("engine")[["seconds", "peak_MiB"]].median()
    print("Median of 3 measured runs:")
    print(summary.round(3).to_string())
    print(f"Saved: {destination}")


if __name__ == "__main__":
    if len(sys.argv) == 2:
        print(json.dumps(measure(sys.argv[1])))
    else:
        benchmark()
