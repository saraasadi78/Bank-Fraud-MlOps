# Generating a small SYNTHETIC file with the BAF schema so CI and tests can run without downloading from Kaggle. Not for real results.


from pathlib import Path
import numpy as np
import pandas as pd

def generate(rows_per_month: int = 5000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    parts = []
    for month in range(8):
        n, shift = rows_per_month, 0.3 if month >= 6 else 0.0   # drift in months 6-7
        risk = rng.normal(130 + 40 * shift, 60, n).round()
        sim = rng.uniform(0, 1, n)
        foreign = rng.binomial(1, 0.03, n)
        age = rng.choice([20, 30, 40, 50, 60, 70], n, p=[.2, .25, .2, .15, .12, .08])
        logit = -4.2 + 0.012 * (risk - 130) - 1.5 * sim + 1.8 * foreign + 0.01 * (age - 40)
        fraud = rng.binomial(1, 1 / (1 + np.exp(-logit)))
        parts.append(pd.DataFrame({
            "fraud_bool": fraud,
            "income": rng.choice(np.round(np.arange(0.1, 1.0, 0.1), 1), n),
            "name_email_similarity": sim,
            "prev_address_months_count": np.where(rng.random(n) < .6, -1, rng.integers(0, 380, n)),
            "current_address_months_count": rng.integers(-1, 429, n),
            "customer_age": age,
            "days_since_request": rng.exponential(1, n),
            "intended_balcon_amount": rng.normal(10, 20, n),
            "payment_type": rng.choice(["AA", "AB", "AC", "AD", "AE"], n),
            "zip_count_4w": rng.integers(1, 6830, n),
            "velocity_6h": rng.normal(5600, 3000, n),
            "velocity_24h": rng.normal(4800, 1500, n),
            "velocity_4w": rng.normal(4900, 900, n),
            "bank_branch_count_8w": rng.integers(0, 2404, n),
            "date_of_birth_distinct_emails_4w": rng.integers(0, 39, n),
            "employment_status": rng.choice(["CA", "CB", "CC", "CD", "CE", "CF", "CG"], n),
            "credit_risk_score": risk,
            "email_is_free": rng.binomial(1, .5, n),
            "housing_status": rng.choice(["BA", "BB", "BC", "BD", "BE", "BF", "BG"], n),
            "phone_home_valid": rng.binomial(1, .4, n),
            "phone_mobile_valid": rng.binomial(1, .9, n),
            "bank_months_count": rng.integers(-1, 32, n),
            "has_other_cards": rng.binomial(1, .2, n),
            "proposed_credit_limit": rng.choice([200, 500, 1000, 1500, 2000], n),
            "foreign_request": foreign,
            "source": rng.choice(["INTERNET", "TELEAPP"], n, p=[.99, .01]),
            "session_length_in_minutes": rng.exponential(8, n),
            "device_os": rng.choice(["windows", "macintosh", "linux", "x11", "other"], n),
            "keep_alive_session": rng.binomial(1, .6, n),
            "device_distinct_emails_8w": rng.choice([-1, 0, 1, 2], n, p=[.01, .02, .95, .02]),
            "device_fraud_count": np.zeros(n, dtype=int),
            "month": month,
        }))
    return pd.concat(parts, ignore_index=True)


if __name__ == "__main__":
    Path("data").mkdir(exist_ok=True)
    generate().to_parquet("data/sample.parquet")
    print("Wrote data/sample.parquet")

