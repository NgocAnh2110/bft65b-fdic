from pathlib import Path
import numpy as np
import pandas as pd

project = Path(__file__).resolve().parent
out = project / "outputs"

pred = pd.read_parquet(
    out / "development_predictions.parquet",
    columns=["CERT", "REPDTE", "y", "baseline_score"],
)
costs = pd.read_csv(out / "validation_failure_cost_audit.csv")
if pred.duplicated(["CERT", "REPDTE"]).any():
    raise ValueError("Duplicate validation bank-quarters.")
if not np.isfinite(pred["baseline_score"]).all():
    raise ValueError("Baseline scores contain missing or infinite values.")
if costs["CERT"].duplicated().any():
    raise ValueError("Expected one first-failure event per CERT.")

median_positive = costs.loc[costs["numeric_cost"] > 0, "numeric_cost"].median()
if not np.isfinite(median_positive):
    raise ValueError("No valid positive cost median.")
costs["miss_penalty"] = costs["numeric_cost"].clip(lower=0).fillna(median_positive)

# An event is detected if at least one eligible positive quarter raises an alarm.
event_scores = (
    pred.loc[pred["y"].eq(1)]
    .groupby("CERT")["baseline_score"].max()
    .rename("event_score")
)
if set(event_scores.index) != set(costs["CERT"]):
    raise ValueError("Validation events and cost records do not match.")
events = costs.join(event_scores, on="CERT", validate="one_to_one")

# Sweep scores from high risk to low risk. The alarm rule is score >= threshold.
by_score = (
    pred.assign(false_alarm=pred["y"].eq(0).astype(int))
    .groupby("baseline_score")
    .agg(new_alarms=("y", "size"), new_false_alarms=("false_alarm", "sum"))
    .sort_index(ascending=False)
)
by_event = events.groupby("event_score").agg(
    new_events=("CERT", "size"),
    new_detected_penalty=("miss_penalty", "sum"),
)
curve = by_score.join(by_event).fillna(0)
curve.index.name = "threshold"
curve = curve.reset_index()

# Include the policy that never raises an alarm.
no_alarm = pd.DataFrame([{
    "threshold": np.inf, "new_alarms": 0, "new_false_alarms": 0,
    "new_events": 0, "new_detected_penalty": 0.0,
}])
curve = pd.concat([no_alarm, curve], ignore_index=True)
curve["alarms"] = curve["new_alarms"].cumsum().astype(int)
curve["false_alarms"] = curve["new_false_alarms"].cumsum().astype(int)
curve["detected_events"] = curve["new_events"].cumsum().astype(int)
curve["missed_events"] = len(events) - curve["detected_events"]
curve["miss_cost"] = (
    events["miss_penalty"].sum() - curve["new_detected_penalty"].cumsum()
).clip(lower=0)

rows = []
for fraction in [0.001, 0.01, 0.05]:
    fa_cost = fraction * median_positive
    loss = curve["miss_cost"] + fa_cost * curve["false_alarms"]
    # Scores are descending, so an exact tie selects the policy with fewer alarms.
    best = curve.loc[loss.idxmin()]
    rows.append({
        "scenario": "main" if fraction == 0.01 else "sensitivity",
        "fa_fraction": fraction,
        "fa_unit_cost": fa_cost,
        "score_threshold": best["threshold"],
        "alarms": int(best["alarms"]),
        "false_alarms": int(best["false_alarms"]),
        "detected_events": int(best["detected_events"]),
        "missed_events": int(best["missed_events"]),
        "event_recall_pct": 100 * best["detected_events"] / len(events),
        "total_proxy_cost": loss.min(),
    })

result = pd.DataFrame(rows)
result.to_csv(out / "threshold_sensitivity.csv", index=False)
result.loc[result["scenario"].eq("main")].to_csv(
    out / "selected_threshold.csv", index=False
)

print(f"Validation events: {len(events):,}")
print(result.to_string(index=False, float_format=lambda x: f"{x:,.5f}"))
