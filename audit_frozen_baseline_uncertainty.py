from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

outputs = Path("outputs")
field = "equity_assets_lag1"
data = pd.read_parquet(
    "data_processed/feature_table.parquet",
    columns=["CERT", "split", "y", field],
)

dev_median = data.loc[data["split"].eq("development"), field].median()
test = data.loc[data["split"].eq("test")].copy()
test["score"] = -test[field].fillna(dev_median)

y = test["y"].to_numpy()
score = test["score"].to_numpy()
ap = average_precision_score(y, score)
auc = roc_auc_score(y, score)

original = pd.read_csv(outputs / "final_test_results.csv").iloc[0]
assert len(test) == int(original["test_rows"])
assert int(y.sum()) == int(original["positive_quarters"])
np.testing.assert_allclose(ap, original["AP_baseline"], rtol=1e-10)
print("Frozen baseline AP vs original test result: PASS", flush=True)

bank_index, banks = pd.factorize(test["CERT"])
assert (bank_index >= 0).all()
n_banks = len(banks)
rng = np.random.default_rng(42)
bootstrap_ap = []

for draw in range(500):
    sampled = rng.integers(0, n_banks, size=n_banks)
    bank_counts = np.bincount(sampled, minlength=n_banks)
    weights = bank_counts[bank_index]
    if weights[y == 1].sum() == 0:
        continue
    bootstrap_ap.append(average_precision_score(y, score, sample_weight=weights))

lower, upper = np.percentile(bootstrap_ap, [2.5, 97.5])
result = pd.DataFrame([{
    "test_banks": n_banks,
    "test_rows": len(test),
    "positive_rows": int(y.sum()),
    "AP": ap,
    "ROC_AUC": auc,
    "AP_95_low": lower,
    "AP_95_high": upper,
    "valid_bootstrap_draws": len(bootstrap_ap),
    "seed": 42,
}])

path = outputs / "frozen_baseline_uncertainty.csv"
result.to_csv(path, index=False)
print(result.round(5).to_string(index=False))
print(f"Saved: {path}")
