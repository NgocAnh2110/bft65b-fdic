from pathlib import Path
from statistics import median
from tempfile import TemporaryDirectory
from time import perf_counter

import pandas as pd

project = Path(__file__).resolve().parent
source = project / "data_raw" / "fdic" / "financials"
output = project / "outputs"
output.mkdir(exist_ok=True)

parquet_files = sorted(source.glob("REPDTE=*.parquet"))
if len(parquet_files) != 170:
    raise ValueError(f"Expected 170 quarterly files, found {len(parquet_files)}")

columns = ["CERT", "ASSET"]


def run_query(files, file_format):
    start = perf_counter()
    result = []
    for path in files:
        if file_format == "CSV":
            df = pd.read_csv(path, usecols=columns)
        else:
            df = pd.read_parquet(path, columns=columns)

        result.append((
            path.stem,
            len(df),
            int(df["CERT"].nunique()),
            int(df["ASSET"].count()),
            df["ASSET"].sum(min_count=1),
        ))

    elapsed = perf_counter() - start
    result = pd.DataFrame(
        result,
        columns=["quarter", "rows", "banks", "non_missing_assets", "total_assets"],
    )
    return result, elapsed


with TemporaryDirectory(prefix="fdic_csv_") as temporary:
    csv_dir = Path(temporary)
    print("Creating one uncompressed CSV per quarter...", flush=True)

    for index, path in enumerate(parquet_files, start=1):
        pd.read_parquet(path).to_csv(csv_dir / f"{path.stem}.csv", index=False)
        if index % 25 == 0 or index == len(parquet_files):
            print(f"Converted {index}/170 quarters", flush=True)

    files = {
        "CSV": sorted(csv_dir.glob("*.csv")),
        "Parquet": parquet_files,
    }

    # Warm up both readers and verify the quarterly results before timing.
    reference, _ = run_query(files["Parquet"], "Parquet")
    csv_result, _ = run_query(files["CSV"], "CSV")
    pd.testing.assert_frame_equal(
        reference, csv_result, check_dtype=False, check_exact=True
    )
    print(f"Quarterly query results match across {len(reference):,} quarters.")
    print(f"Rows verified: {reference['rows'].sum():,}", flush=True)

    timings = {"CSV": [], "Parquet": []}
    for repeat in range(1, 4):
        order = ["CSV", "Parquet"] if repeat % 2 else ["Parquet", "CSV"]
        for name in order:
            _, seconds = run_query(files[name], name)
            timings[name].append(seconds)
            print(f"Run {repeat} | {name}: {seconds:.2f} seconds", flush=True)

    rows = []
    for name, paths in files.items():
        rows.append({
            "format": name,
            "files": len(paths),
            "size_MB": sum(p.stat().st_size for p in paths) / 1_000_000,
            "run_1_s": timings[name][0],
            "run_2_s": timings[name][1],
            "run_3_s": timings[name][2],
            "median_query_s": median(timings[name]),
        })

summary = pd.DataFrame(rows).set_index("format")
summary.to_csv(output / "storage_benchmark.csv")

print("\nStorage benchmark:")
print(summary.round(2).to_string())
print(f"\nCSV / Parquet size ratio: "
      f"{summary.loc['CSV', 'size_MB'] / summary.loc['Parquet', 'size_MB']:.2f}x")
print(f"CSV / Parquet query-time ratio: "
      f"{summary.loc['CSV', 'median_query_s'] / summary.loc['Parquet', 'median_query_s']:.2f}x")
