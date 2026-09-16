# Fraud metrics used by banks. Accuracy is useless at ~1% fraud (predicting 'never fraud' is 99% accurate)
# using recall at a fixed false-positive rate (how much fraud we catch at a review budget)
# using PR-AUC (threshold-free ranking quality on rare positives)
#using FPR ratio between age groups (are legitimate older/younger applicants flagged unequally?)


import numpy as np
from sklearn.metrics import average_precision_score, roc_curve

from 1-config-and-data import TARGET_FPR


def recall_at_fpr(y, scores, target_fpr=TARGET_FPR) -> tuple[float, float]:
    """Best recall with FPR <= target. Returns (recall, threshold)."""
    fpr, tpr, thresholds = roc_curve(y, scores)
    idx = np.where(fpr <= target_fpr)[0][-1]
    return float(tpr[idx]), float(thresholds[idx])


def rates(y, scores, threshold) -> dict:
    y, flagged = np.asarray(y), np.asarray(scores) >= threshold
    tp = np.sum(flagged & (y == 1)); fn = np.sum(~flagged & (y == 1))
    fp = np.sum(flagged & (y == 0)); tn = np.sum(~flagged & (y == 0))
    return {"recall": float(tp / max(tp + fn, 1)),
            "fpr": float(fp / max(fp + tn, 1)),
            "alert_rate": float(flagged.mean())}


def fpr_ratio_by_age(y, scores, threshold, age) -> float:
    """min(FPR)/max(FPR) for applicants >=50 vs <50. 1.0 = equal treatment."""
    y, scores, older = np.asarray(y), np.asarray(scores), np.asarray(age) >= 50
    f_old = rates(y[older], scores[older], threshold)["fpr"]
    f_young = rates(y[~older], scores[~older], threshold)["fpr"]
    low, high = sorted([f_old, f_young])
    return float(low / high) if high > 0 else 1.0


def evaluate(y, scores, threshold, age) -> dict:
    report = rates(y, scores, threshold)
    report["pr_auc"] = float(average_precision_score(y, scores))
    report["fpr_ratio_age"] = fpr_ratio_by_age(y, scores, threshold, age)
    report["fraud_rate"] = float(np.mean(y))
    return report
