# Champion vs challenger on a month neither model has seen. Challenger must catch more fraud AND not be meaningfully less fair.

import argparse
import json

import joblib
import mlflow
from mlflow.tracking import MlflowClient

from 1-config-and-data import AGE_COL, ARTIFACT_DIR, FAIRNESS_TOLERANCE, LABEL, MODEL_NAME, TARGET_FPR, TRACKING_URI, load_data, months, make_features
from 3-metrics import evaluate, recall_at_fpr


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--eval-month", required=True, type=int)
    args = p.parse_args()

    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient()

    ev = months(load_data(), [args.eval_month])
    X, y, age = make_features(ev), ev[LABEL].values, ev[AGE_COL].values

    def assess(version):
        model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/{version.version}")
        threshold = client.get_run(version.run_id).data.metrics["threshold"]
        scores = model.predict_proba(X)[:, 1]
        report = evaluate(y, scores, threshold, age)
        report["recall_at_target_fpr"], _ = recall_at_fpr(y, scores)
        print(f"v{version.version}: recall@{TARGET_FPR:.0%}FPR={report['recall_at_target_fpr']:.3f}  "
              f"age FPR ratio={report['fpr_ratio_age']:.2f}")
        return model, threshold, report

    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    challenger = max(versions, key=lambda v: int(v.version))
    model, threshold, ch = assess(challenger)

    try:
        champion = client.get_model_version_by_alias(MODEL_NAME, "champion")
    except Exception:
        champion = None

    if champion is not None:
        if champion.version == challenger.version:
            print("Challenger is already champion.")
            return
        _, _, cp = assess(champion)
        better = ch["recall_at_target_fpr"] > cp["recall_at_target_fpr"]
        fair = ch["fpr_ratio_age"] >= cp["fpr_ratio_age"] - FAIRNESS_TOLERANCE
        if not (better and fair):
            reason = "does not catch more fraud" if not better else "fails fairness check"
            print(f"Challenger {reason}. Keeping champion v{champion.version}.")
            return

    client.set_registered_model_alias(MODEL_NAME, "champion", challenger.version)
    ARTIFACT_DIR.mkdir(exist_ok=True)
    joblib.dump(model, ARTIFACT_DIR / "model.joblib")
    meta = {"version": challenger.version, "threshold": threshold, "target_fpr": TARGET_FPR,
            "features": list(model.feature_names_in_)}
    (ARTIFACT_DIR / "model_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Promoted v{challenger.version} to champion and exported for serving.")


if __name__ == "__main__":
    main()
  
