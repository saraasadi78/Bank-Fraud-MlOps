# Monitor a new month against a reference month. Label-free signals (available immediately): feature drift (PSI), alert rate
# Label-based signals (fraud labels arrive weeks later): recall, FPR, fairness


import argparse
import json
import os
from pathlib import Path

import joblib
import numpy as np

from 1-config-and-data import AGE_COL, ARTIFACT_DIR, LABEL,load_data, months, make_features
from 3-metrics import evaluate

DRIFT_FEATURES = ["credit_risk_score", "proposed_credit_limit", "name_email_similarity",
                  "velocity_6h", "customer_age", "income"]
PSI_ALERT = 0.25


def psi(ref, cur, bins=10) -> float:
    ref, cur = np.asarray(ref, float), np.asarray(cur, float)
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:
        return 0.0
    ref_p = np.histogram(np.clip(ref, edges[0], edges[-1]), edges)[0] / len(ref)
    cur_p = np.histogram(np.clip(cur, edges[0], edges[-1]), edges)[0] / len(cur)
    ref_p, cur_p = np.clip(ref_p, 1e-6, None), np.clip(cur_p, 1e-6, None)
    return float(np.sum((cur_p - ref_p) * np.log(cur_p / ref_p)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--reference-month", required=True, type=int)
    p.add_argument("--current-month", required=True, type=int)
    args = p.parse_args()

    df = load_data()
    ref, cur = months(df, [args.reference_month]), months(df, [args.current_month])
    model = joblib.load(ARTIFACT_DIR / "model.joblib")
    threshold = json.loads((ARTIFACT_DIR / "model_meta.json").read_text())["threshold"]

    def perf(part):
        scores = model.predict_proba(make_features(part))[:, 1]
        return evaluate(part[LABEL].values, scores, threshold, part[AGE_COL].values)

    r, c = perf(ref), perf(cur)
    drift = {f: psi(ref[f], cur[f]) for f in DRIFT_FEATURES if f in df.columns}

    alerts = []
    for feat, value in drift.items():
        if value > PSI_ALERT:
            alerts.append(f"PSI {feat}={value:.2f}")
    if abs(c["alert_rate"] - r["alert_rate"]) > 0.5 * r["alert_rate"]:
        alerts.append(f"Alert rate moved {r['alert_rate']:.1%} -> {c['alert_rate']:.1%}")
    if c["recall"] < 0.9 * r["recall"]:
        alerts.append(f"Recall dropped {r['recall']:.3f} -> {c['recall']:.3f}")
    if c["fpr_ratio_age"] < r["fpr_ratio_age"] - 0.10:
        alerts.append(f"Age fairness ratio fell {r['fpr_ratio_age']:.2f} -> {c['fpr_ratio_age']:.2f}")

    report = {"reference_month": args.reference_month, "current_month": args.current_month,
              "psi": drift, "reference": r, "current": c,
              "alerts": alerts, "drift_detected": bool(alerts)}
    Path("reports").mkdir(exist_ok=True)
    (Path("reports") / f"monitor_month{args.current_month}.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))

    if "GITHUB_OUTPUT" in os.environ:
        with open(os.environ["GITHUB_OUTPUT"], "a") as f:
            f.write(f"drift={str(bool(alerts)).lower()}\n")



if __name__ == "__main__":
    main()
