from pathlib import Path
import pandas as pd
import polars as pl
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

project = Path(__file__).resolve().parent
out = project / "outputs"

policy = pd.read_csv(out / "selected_threshold.csv", float_precision="round_trip")
if len(policy) != 1 or policy.iloc[0]["scenario"] != "main":
    raise ValueError("Expected one development-selected main threshold.")
threshold = float(policy.iloc[0]["score_threshold"])
fa_cost = float(policy.iloc[0]["fa_unit_cost"])

features = [
    "log_assets_lag1", "equity_assets_lag1", "deposits_assets_lag1",
    "reserves_loans_lag1", "roa_lag1", "no_prior_report",
]
table = (
    pl.scan_parquet(project / "data_processed" / "feature_table.parquet")
    .filter(pl.col("split").is_in(["development", "test"]))
    .with_columns(
        pl.col("source_report").is_null().cast(pl.UInt8).alias("no_prior_report")
    )
    .select(["CERT", "REPDTE", "split", "y"] + features)
    .collect()
    .to_pandas()
)
dev = table.loc[table["split"].eq("development")]
test = table.loc[table["split"].eq("test")].copy()
if len(test) != 178642 or int(test["y"].sum()) != 1650:
    raise ValueError("Test sample differs from the audited split.")

# Freeze preprocessing on development.
equity_median = dev["equity_assets_lag1"].median()
baseline = -test["equity_assets_lag1"].fillna(equity_median)

model = make_pipeline(
    SimpleImputer(strategy="median"),
    StandardScaler(),
    LogisticRegression(max_iter=1000, random_state=42),
)
model.fit(dev[features], dev["y"])
logistic = model.predict_proba(test[features])[:, 1]

# Quarter-level alarm metrics for the selected baseline.
alarm = baseline.ge(threshold)
positive = test["y"].eq(1)
tp = int((alarm & positive).sum())
fp = int((alarm & ~positive).sum())
fn = int((~alarm & positive).sum())

# build_labels.py targets the first failure per CERT: count each event once.
event_detection = (
    test.loc[positive, ["CERT"]]
    .assign(detected=alarm.loc[positive])
    .groupby("CERT")["detected"].any()
)
if len(event_detection) != 469:
    raise ValueError("Distinct test failures differ from the audited split.")

# Use the same cost rule and development-derived imputation as in validation.
validation_costs = pd.read_csv(out / "validation_failure_cost_audit.csv")
positive_costs = validation_costs.loc[
    validation_costs["numeric_cost"] > 0, "numeric_cost"
]
median_cost = positive_costs.median()

events = pd.read_parquet(
    project / "data_raw" / "fdic" / "failures.parquet",
    columns=["CERT", "FAILDATE", "RESTYPE", "COST"],
)
events = events.loc[
    events["RESTYPE"].astype("string").str.strip().eq("FAILURE")
    & events["CERT"].notna()
].copy()
events["CERT"] = pd.to_numeric(events["CERT"], errors="raise").astype("int64")
events["FAILDATE"] = pd.to_datetime(events["FAILDATE"], format="mixed", errors="raise")
first = events.sort_values("FAILDATE").drop_duplicates("CERT").set_index("CERT")
if not event_detection.index.isin(first.index).all():
    raise ValueError("A test failure could not be linked to its cost record.")
cost = pd.to_numeric(first.loc[event_detection.index, "COST"], errors="coerce")
penalty = cost.clip(lower=0).fillna(median_cost)
miss_cost = float(penalty.loc[~event_detection].sum())

summary = pd.DataFrame([{
    "test_rows": len(test),
    "positive_quarters": int(positive.sum()),
    "AP_baseline": average_precision_score(positive, baseline),
    "AP_logistic": average_precision_score(positive, logistic),
    "alarms": int(alarm.sum()),
    "false_alarms": fp,
    "true_positive_quarters": tp,
    "missed_positive_quarters": fn,
    "quarter_precision_pct": 100 * tp / max(tp + fp, 1),
    "quarter_recall_pct": 100 * tp / (tp + fn),
    "failure_events": len(event_detection),
    "detected_events": int(event_detection.sum()),
    "missed_events": int((~event_detection).sum()),
    "event_recall_pct": 100 * event_detection.mean(),
    "total_proxy_cost": fp * fa_cost + miss_cost,
}])
summary.to_csv(out / "final_test_results.csv", index=False)
print(summary.T.to_string(header=False, float_format=lambda x: f"{x:,.5f}"))
