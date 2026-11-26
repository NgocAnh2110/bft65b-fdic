from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

financial = [
    "log_assets_lag1",
    "equity_assets_lag1",
    "deposits_assets_lag1",
    "reserves_loans_lag1",
    "roa_lag1",
    "noncurrent_loans_lag1",
    "tier1_leverage_lag1",
    "change_equity_assets_lag1",
]
inputs = financial + ["no_prior_report"]


class TrainingQuantileClipper(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        self.lower_ = X[financial].quantile(0.005)
        self.upper_ = X[financial].quantile(0.995)
        return self

    def transform(self, X):
        result = X.copy()
        result[financial] = X[financial].clip(
            lower=self.lower_, upper=self.upper_, axis="columns"
        )
        return result


data = pd.read_parquet(
    "data_processed/development_features_extended.parquet",
    columns=["REPDTE", "source_report", "y"] + financial,
)
data["REPDTE"] = pd.to_datetime(data["REPDTE"])
data["no_prior_report"] = data["source_report"].isna().astype("uint8")

folds = [
    ("Fold 1", "1988-12-31", "1990-03-31", "1993-12-31"),
    ("Fold 2", "1994-12-31", "1996-03-31", "1999-12-31"),
    ("Fold 3", "2001-12-31", "2003-03-31", "2007-12-31"),
]

rows = []
for fold, train_end, valid_start, valid_end in folds:
    train = data.loc[data["REPDTE"] <= train_end]
    valid = data.loc[data["REPDTE"].between(valid_start, valid_end)]

    row = {
        "fold": fold,
        "validation_rows": len(valid),
        "positive_rows": int(valid["y"].sum()),
    }

    for name, clipper in [
        ("AP_unclipped", "passthrough"),
        ("AP_clipped", TrainingQuantileClipper()),
    ]:
        model = Pipeline([
            ("clip", clipper),
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("logistic", LogisticRegression(C=1.0, max_iter=1000, random_state=42)),
        ])
        model.fit(train[inputs], train["y"])
        scores = model.predict_proba(valid[inputs])[:, 1]
        row[name] = average_precision_score(valid["y"], scores)

    row["AP_difference"] = row["AP_clipped"] - row["AP_unclipped"]
    rows.append(row)

results = pd.DataFrame(rows)
path = Path("outputs/extended_logistic_clipping.csv")
path.parent.mkdir(exist_ok=True)
results.to_csv(path, index=False)
print(results.round(5).to_string(index=False))
print(f"Saved: {path}")
