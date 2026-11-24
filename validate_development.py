from pathlib import Path

import pandas as pd
import polars as pl
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

project = Path(__file__).resolve().parent
outputs = project / "outputs"
outputs.mkdir(exist_ok=True)

features = [
    "log_assets_lag1",
    "equity_assets_lag1",
    "deposits_assets_lag1",
    "reserves_loans_lag1",
    "roa_lag1",
    "no_prior_report",
]

dev = (
    pl.scan_parquet(project / "data_processed" / "feature_table.parquet")
    .filter(pl.col("split") == "development")
    .with_columns(
        pl.col("source_report").is_null().cast(pl.UInt8).alias("no_prior_report")
    )
    .select(["CERT", "REPDTE", "y"] + features)
    .collect()
    .to_pandas()
)
dev["REPDTE"] = pd.to_datetime(dev["REPDTE"])

# Four full quarters separate each training cutoff from validation.
folds = [
    ("fold_1", "1988-12-31", "1990-03-31", "1993-12-31"),
    ("fold_2", "1994-12-31", "1996-03-31", "1999-12-31"),
    ("fold_3", "2001-12-31", "2003-03-31", "2007-12-31"),
]

results = []
predictions = []

for name, train_end, val_start, val_end in folds:
    train = dev.loc[dev["REPDTE"] <= train_end]
    valid = dev.loc[dev["REPDTE"].between(val_start, val_end)]

    if train["y"].nunique() != 2 or valid["y"].nunique() != 2:
        raise ValueError(f"{name}: check class counts before computing AP")

    # Missing baseline values use the training median for this fold.
    equity_median = train["equity_assets_lag1"].median()
    baseline_score = -valid["equity_assets_lag1"].fillna(equity_median)

    # Both imputation and scaling learn only from this training period.
    model = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(max_iter=1000, random_state=42),
    )
    model.fit(train[features], train["y"])
    model_score = model.predict_proba(valid[features])[:, 1]

    results.append({
        "fold": name,
        "train_end": train_end,
        "validation_start": val_start,
        "validation_end": val_end,
        "validation_rows": len(valid),
        "positive_rows": int(valid["y"].sum()),
        "prevalence": valid["y"].mean(),
        "AP_baseline": average_precision_score(valid["y"], baseline_score),
        "AP_logistic": average_precision_score(valid["y"], model_score),
    })

    scored = valid[["CERT", "REPDTE", "y"]].copy()
    scored["fold"] = name
    scored["baseline_score"] = baseline_score
    scored["logistic_score"] = model_score
    predictions.append(scored)

summary = pd.DataFrame(results)
summary.to_csv(outputs / "development_validation.csv", index=False)
pd.concat(predictions, ignore_index=True).to_parquet(
    outputs / "development_predictions.parquet", index=False
)

print(summary.to_string(index=False, float_format=lambda value: f"{value:.5f}"))
