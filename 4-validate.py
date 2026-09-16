import pandas as pd
from 1-config-and-data import AGE_COL, CATEGORICAL, LABEL, TIME_COL

REQUIRED = [LABEL, TIME_COL, AGE_COL, "income", "credit_risk_score", "proposed_credit_limit", *CATEGORICAL]


def validate(df: pd.DataFrame, min_rows: int = 1000, max_null_rate: float = 0.01) -> list[str]:
    # Return a list of problems. Empty list = safe to train on
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        return [f"Missing columns: {missing}"]

    errors = []
    if len(df) < min_rows:
        errors.append(f"Only {len(df)} rows (min {min_rows})")
    if not set(df[LABEL].unique()) <= {0, 1}:
        errors.append(f"{LABEL} must be 0/1")
    fraud_rate = df[LABEL].mean()
    if not 0.001 <= fraud_rate <= 0.10:
        errors.append(f"Fraud rate {fraud_rate:.2%} outside expected 0.1%-10%")
    for col, rate in df[REQUIRED].isna().mean().items():
        if rate > max_null_rate:
            errors.append(f"{col}: {rate:.1%} nulls")
    if not df[AGE_COL].between(10, 90).all():
        errors.append(f"{AGE_COL} outside 10-90")
    return errors
