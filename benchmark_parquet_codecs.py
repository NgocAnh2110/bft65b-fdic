from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
import pyarrow.parquet as pq

source = Path("data_raw/fdic/financials")
files = sorted(source.glob("REPDTE=*.parquet"))
if len(files) != 170:
    raise ValueError(f"Expected 170 quarterly files, found {len(files)}")

records = []
with TemporaryDirectory() as temporary:
    folder = Path(temporary)

    for index, file in enumerate(files, start=1):
        table = pq.ParquetFile(file).read()
        row = {"quarter": file.stem.split("=")[1], "rows": table.num_rows}

        for codec in ("snappy", "zstd"):
            target = folder / f"{codec}.parquet"
            pq.write_table(
                table, target,
                compression=codec,
                row_group_size=65536,
                use_dictionary=True,
            )
            restored = pq.ParquetFile(target).read()
            if not restored.equals(table):
                raise ValueError(f"Round-trip check failed: {file.name}, {codec}")
            row[f"{codec}_bytes"] = target.stat().st_size

        records.append(row)
        if index % 50 == 0:
            print(f"Checked {index}/170 quarters", flush=True)

results = pd.DataFrame(records)
total_rows = int(results["rows"].sum())
if total_rows != 1_682_738:
    raise ValueError(f"Snapshot row count differs: {total_rows:,}")

snappy = int(results["snappy_bytes"].sum())
zstd = int(results["zstd_bytes"].sum())
saving = 100 * (snappy - zstd) / snappy

output = Path("outputs/codec_benchmark.csv")
output.parent.mkdir(exist_ok=True)
results.to_csv(output, index=False)

print(f"Quarters checked: {len(results)}")
print(f"Rows checked: {total_rows:,}")
print(f"Snappy size: {snappy / 1_000_000:.2f} MB")
print(f"Zstandard size: {zstd / 1_000_000:.2f} MB")
print(f"Size reduction vs Snappy: {saving:.2f}%")
print("Round-trip checks: PASS")
