from pathlib import Path
import pandas as pd

project = Path(__file__).resolve().parent
folder = project / "data_raw" / "fdic" / "financials"
files = sorted(folder.glob("REPDTE=*.parquet"))
if not files:
    raise FileNotFoundError(f"No quarterly files found in {folder}")

checks = []
for path in files:
    expected_date = path.stem.split("=")[1]
    keys = pd.read_parquet(path, columns=["CERT", "REPDTE"])
    dates = keys["REPDTE"].astype("string").str.strip().replace("", pd.NA)
    certs = keys["CERT"].astype("string").str.strip().replace("", pd.NA)

    checks.append({
        "quarter": expected_date,
        "rows": len(keys),
        "missing_cert": int(certs.isna().sum()),
        "missing_repdte": int(dates.isna().sum()),
        "wrong_quarter": int(dates.ne(expected_date).sum()),
    })

audit = pd.DataFrame(checks)
output = project / "outputs" / "panel_key_checks.csv"
output.parent.mkdir(exist_ok=True)
audit.to_csv(output, index=False)

print(f"Quarters checked: {len(audit)}")
print(audit[["rows", "missing_cert", "missing_repdte", "wrong_quarter"]].sum())
