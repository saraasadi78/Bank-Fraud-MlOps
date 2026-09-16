import argparse
import sys

import mlflow
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from 1-config-and-data import AGE_COL, LABEL, MODEL_NAME, TARGET_FPR, TRACKING_URI, load_data, months, parse_months, build_pipeline, make_features
from 3-metrics import evaluate, recall_at_fpr
from 4-validate import validate


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train-months", required=True)
    p.add_argument("--val-month", required=True, type=int)
    args = p.parse_args()

    df = load_data()
    train, val = months(df, parse_months(args.train_months)), months(df, [args.val_month])
    for split, part in {"train": train, "val": val}.items():
        errors = validate(part)
        if errors:
            print(f"Validation failed on {split}: {errors}")
            sys.exit(1)

    X_train, y_train = make_features(train), train[LABEL].values
    X_val, y_val = make_features(val), val[LABEL].values

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(MODEL_NAME)

    candidates = {
        "logreg_baseline": (LogisticRegression(max_iter=1000, class_weight="balanced"), True),
        "xgboost": (XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.05,
                                  subsample=0.8, colsample_bytree=0.8,
                                  eval_metric="aucpr", n_jobs=-1), False),
    }
    best_run, best_recall = None, -1.0
    for name, (estimator, scale) in candidates.items():
        with mlflow.start_run(run_name=name) as run:
            pipe = build_pipeline(X_train, estimator, scale)
            pipe.fit(X_train, y_train)
            scores = pipe.predict_proba(X_val)[:, 1]

            recall, threshold = recall_at_fpr(y_val, scores)
            report = evaluate(y_val, scores, threshold, val[AGE_COL].values)

            mlflow.log_params({"model": name, "train_months": args.train_months,
                               "val_month": args.val_month, "target_fpr": TARGET_FPR,
                               "n_train": len(train)})
            mlflow.log_metrics({"recall_at_target_fpr": recall, "threshold": threshold,
                                "pr_auc": report["pr_auc"],
                                "fpr_ratio_age": report["fpr_ratio_age"]})
            mlflow.sklearn.log_model(pipe, "model", serialization_format="cloudpickle")
            print(f"{name}: recall@{TARGET_FPR:.0%}FPR={recall:.3f}  "
                  f"PR-AUC={report['pr_auc']:.3f}  age FPR ratio={report['fpr_ratio_age']:.2f}")
            if recall > best_recall:
                best_run, best_recall = run.info.run_id, recall

    version = mlflow.register_model(f"runs:/{best_run}/model", MODEL_NAME)
    print(f"Registered {MODEL_NAME} v{version.version} (recall {best_recall:.3f})")


if __name__ == "__main__":
    main()
