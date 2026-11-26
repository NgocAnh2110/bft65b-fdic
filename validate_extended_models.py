from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

financial = [
    "log_assets_lag1", "equity_assets_lag1", "deposits_assets_lag1",
    "reserves_loans_lag1", "roa_lag1", "noncurrent_loans_lag1",
    "tier1_leverage_lag1", "change_equity_assets_lag1",
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
            self.lower_, self.upper_, axis="columns"
        )
        return result


def make_models():
    models = {}
    for c in (0.01, 1.0, 100.0):
        models[f"Logistic C={c:g}"] = Pipeline([
            ("clip", TrainingQuantileClipper()),
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", LogisticRegression(C=c, max_iter=1000, random_state=42)),
        ])

    # The tree handles missing inputs directly. Disable random early-stopping splits.
    models["Gradient boosting"] = Pipeline([
        ("clip", TrainingQuantileClipper()),
        ("model", HistGradientBoostingClassifier(
            max_iter=80, max_leaf_nodes=15, min_samples_leaf=50,
            learning_rate=0.08, early_stopping=False, random_state=42,
        )),
    ])
    return models


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

out = Path("outputs")
previous = pd.read_csv(out / "extended_logistic_clipping.csv").set_index("fold")
rows = []

for fold, train_end, valid_start, valid_end in folds:
    train = data.loc[data["REPDTE"] <= train_end]
    valid = data.loc[data["REPDTE"].between(valid_start, valid_end)]
    X_train, X_valid = train[inputs], valid[inputs]

    for name, model in make_models().items():
        model.fit(X_train, train["y"])
        scores = model.predict_proba(X_valid)[:, 1]
        ap = average_precision_score(valid["y"], scores)
        auc = roc_auc_score(valid["y"], scores)

        if name == "Logistic C=1":
            np.testing.assert_allclose(
                ap, previous.loc[fold, "AP_clipped"], rtol=1e-6, atol=1e-9
            )

        rows.append({
            "fold": fold, "model": name, "validation_rows": len(valid),
            "positive_rows": int(valid["y"].sum()), "AP": ap, "ROC_AUC": auc,
        })
    print(f"{fold} complete", flush=True)

results = pd.DataFrame(rows)
path = out / "extended_model_validation.csv"
results.to_csv(path, index=False)

print("\nAverage precision")
print(results.pivot(index="model", columns="fold", values="AP").round(5))
print("\nROC-AUC")
print(results.pivot(index="model", columns="fold", values="ROC_AUC").round(5))
print("\nLogistic C=1 vs Step 17: PASS")
print(f"Saved: {path}")
