"""Feature definitions shared by training and the scoring API.

Keeping one `build_features` function used by both prevents training/serving skew.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TARGET = "bad_loan"

# Deliberately excluded from the model:
#   int_rate   - LendingClub's own risk price. It already encodes their credit model, so
#                using it would measure their underwriting, not ours. Kept as a benchmark.
#   addr_state - geography can act as a proxy for protected classes (redlining risk).
EXCLUDED = ["int_rate", "addr_state"]

CATEGORIES = {
    "term": ["36 months", "60 months"],
    "home_ownership": ["MORTGAGE", "OWN", "RENT", "OTHER"],
    "verification_status": ["verified", "not verified"],
    "purpose": [
        "car", "credit_card", "debt_consolidation", "educational", "home_improvement",
        "house", "major_purchase", "medical", "moving", "other", "renewable_energy",
        "small_business", "vacation", "wedding",
    ],
}

# Direction of risk we require the model to respect (+1: higher value, higher risk).
# A constraint like this is what makes a reason code defensible: a borrower can never be
# told a *higher* income hurt them because of a quirk in a tree split.
MONOTONE = {
    "loan_amnt": 0,
    "emp_length": 0,
    "emp_length_missing": 0,
    "log_income": -1,
    "loan_to_income": 1,
    "dti": 1,
    "delinq_2yrs": 1,
    "revol_util": 1,
    "total_acc": 0,
    "longest_credit_length": -1,
}

NUMERIC = list(MONOTONE)
CATEGORICAL = list(CATEGORIES)
FEATURES = NUMERIC + CATEGORICAL


def build_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Turn raw application fields into model features. Must not use any post-origination data."""
    out = pd.DataFrame(index=raw.index)
    income = pd.to_numeric(raw["annual_inc"], errors="coerce").clip(lower=1)

    out["loan_amnt"] = pd.to_numeric(raw["loan_amnt"], errors="coerce")
    out["emp_length"] = pd.to_numeric(raw["emp_length"], errors="coerce")
    out["emp_length_missing"] = out["emp_length"].isna().astype(int)
    out["log_income"] = np.log(income)
    out["loan_to_income"] = out["loan_amnt"] / income
    for col in ["dti", "delinq_2yrs", "revol_util", "total_acc", "longest_credit_length"]:
        out[col] = pd.to_numeric(raw[col], errors="coerce")

    home = raw["home_ownership"].where(
        raw["home_ownership"].isin(CATEGORIES["home_ownership"]), "OTHER"
    )
    out["home_ownership"] = pd.Categorical(home, categories=CATEGORIES["home_ownership"])
    for col in ["term", "verification_status", "purpose"]:
        out[col] = pd.Categorical(raw[col], categories=CATEGORIES[col])
    return out[FEATURES]


def total_interest(amount: pd.Series, annual_rate_pct: pd.Series, term: pd.Series) -> pd.Series:
    """Total interest paid over the life of a fully amortizing loan."""
    months = term.astype(str).str.extract(r"(\d+)")[0].astype(float)
    r = annual_rate_pct / 100 / 12
    payment = amount * r / (1 - (1 + r) ** -months)
    return payment * months - amount
