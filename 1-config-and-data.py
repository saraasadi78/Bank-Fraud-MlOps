import os
from pathlib import Path
import pandas as pd

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




