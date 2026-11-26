from pathlib import Path

import pandas as pd

features = [
    "noncurrent_loans_lag1",
    "tier1_leverage_lag1",
    "change_equity_assets_lag1",
]

data = pd.read_parquet(
    "data_processed/development_features_extended.parquet",
    columns=["REPDTE", *features],
)
outputs = Path("outputs")
outputs.mkdir(exist_ok=True)

# Summarise finite values and count missing or non-finite values separately.
rows = []
for name in features:
    observed = data[name].dropna()
    finite = observed.loc[~observed.isin([float("inf"), -float("inf")])]
    if finite.empty:
        raise ValueError(f"No finite observations for {name}")

    rows.append({
        "feature": name,
        "missing": int(data[name].isna().sum()),
        "nonfinite": len(observed) - len(finite),
        "negative": int(finite.lt(0).sum()),
        "zero": int(finite.eq(0).sum()),
        "min": finite.min(),
        "p0.5": finite.quantile(0.005),
        "median": finite.median(),
        "p99.5": finite.quantile(0.995),
        "max": finite.max(),
    })

profile = pd.DataFrame(rows)
profile.to_csv(outputs / "extended_feature_profile.csv", index=False)

# Group by the year of the financial report supplying the lagged ratio.
prediction_q = pd.to_datetime(data["REPDTE"]).dt.to_period("Q")
source_year = (prediction_q - 1).dt.year
tier1 = data["tier1_leverage_lag1"]
usable = tier1.notna() & ~tier1.isin([float("inf"), -float("inf")])

annual = (
    pd.DataFrame({
        "source_year": source_year,
        "observed": usable,
        "zero": usable & tier1.eq(0),
    })
    .groupby("source_year", as_index=False)[["observed", "zero"]]
    .sum()
)
annual = annual.loc[annual["observed"] > 0].copy()
annual["zero_pct"] = 100 * annual["zero"] / annual["observed"]
annual.to_csv(outputs / "tier1_zero_by_source_year.csv", index=False)

print("Development feature profile")
print(profile.round(6).to_string(index=False))
print("\nTier 1 leverage: zeros by source-report year")
print(annual.round(2).to_string(index=False))
