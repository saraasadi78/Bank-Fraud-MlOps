import os
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

#Config

DATA_PATH = Path(os.getenv("DATA_PATH", "data/Base.csv"))
ARTIFACT_DIR = Path("artifacts")
MODEL_NAME = "bank-fraud"
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")

LABEL = "fraud_bool"
TIME_COL = "month"
AGE_COL = "customer_age"
CATEGORICAL = ["payment_type", "employment_status", "housing_status", "source", "device_os"]

TARGET_FPR = 0.05            # bank tolerates flagging 5% of legitimate applicants for review
FAIRNESS_TOLERANCE = 0.05    



# Loading the BAF dataset and normalising column names
def load_data(path=DATA_PATH) -> pd.DataFrame:
    path = str(path)
    df = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)
    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")])
    # The datasheet and the files name this column slightly differently
    if "device_distinct_emails" in df.columns:
        df = df.rename(columns={"device_distinct_emails": "device_distinct_emails_8w"})
    return df


def months(df: pd.DataFrame, month_list) -> pd.DataFrame:
    return df[df["month"].isin(month_list)].reset_index(drop=True)


def parse_months(text: str) -> list[int]:
    """'0-3' -> [0, 1, 2, 3];  '5' -> [5]"""
    start, _, end = text.partition("-")
    return list(range(int(start), int(end or start) + 1))


#features

# Per the datasheet, -1 means "missing" in these columns
MINUS_ONE_IS_MISSING = ["prev_address_months_count", "current_address_months_count",
                        "bank_months_count", "session_length_in_minutes",
                        "device_distinct_emails_8w"]


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """Single source of truth for features, used by training AND the API.
    `month` is dropped: a model should not learn from the calendar itself."""
    X = df.drop(columns=[c for c in (LABEL, TIME_COL) if c in df.columns]).copy()
    for col in MINUS_ONE_IS_MISSING:
        if col in X.columns:
            X[col] = X[col].astype(float).replace(-1, np.nan)
    if "intended_balcon_amount" in X.columns:          # negatives = missing
        X["intended_balcon_amount"] = X["intended_balcon_amount"].astype(float)
        X.loc[X["intended_balcon_amount"] < 0, "intended_balcon_amount"] = np.nan
    return X


def build_pipeline(X: pd.DataFrame, estimator, scale: bool) -> Pipeline:
    cat = [c for c in CATEGORICAL if c in X.columns]
    num = [c for c in X.columns if c not in cat]
    num_steps = [("impute", SimpleImputer(strategy="median", add_indicator=True))]
    if scale:
        num_steps.append(("scale", StandardScaler()))
    prep = ColumnTransformer([
        ("num", Pipeline(num_steps), num),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
    ])
    return Pipeline([("prep", prep), ("model", estimator)])


