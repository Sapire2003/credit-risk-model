"""Load the trained artifact, score applications, and produce adverse-action reasons."""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from features import FEATURES, build_features

ARTIFACT = Path(__file__).resolve().parents[1] / "artifacts" / "model.joblib"

# Plain-language reason text, modeled on the style of the sample notices in Regulation B
# (12 CFR 1002, Appendix C). Only factors a person can understand and act on should appear.
REASON_TEXT = {
    "log_income": "Income insufficient for amount of credit requested",
    "loan_to_income": "Amount requested is high relative to income",
    "dti": "Excessive obligations in relation to income",
    "delinq_2yrs": "Delinquent past or present credit obligations",
    "revol_util": "Proportion of revolving balances to credit limits is too high",
    "longest_credit_length": "Length of credit history is too short",
    "total_acc": "Number of credit accounts on file",
    "emp_length": "Length of employment",
    "emp_length_missing": "Unable to verify employment",
    "loan_amnt": "Amount of credit requested",
    "term": "Requested repayment term",
    "home_ownership": "Housing status",
    "purpose": "Purpose of the loan",
    "verification_status": "Income verification status",
}


class Scorer:
    def __init__(self, artifact_path: Path = ARTIFACT):
        artifact = joblib.load(artifact_path)
        self.booster = artifact["booster"]
        self.calibrator = artifact["calibrator"]
        self.cutoff = artifact["cutoff"]
        self.version = artifact["version"]

    def predict_pd(self, features: pd.DataFrame) -> np.ndarray:
        raw = self.booster.predict(features[FEATURES])
        return self.calibrator.predict(raw) if self.calibrator is not None else raw

    def top_reasons(self, features: pd.DataFrame, k: int = 4) -> list[list[str]]:
        """Up to k risk-increasing factors per applicant, ranked by SHAP contribution."""
        contrib = self.booster.predict(features[FEATURES], pred_contrib=True)[:, :-1]
        results = []
        for row in contrib:
            order = np.argsort(row)[::-1]
            results.append([REASON_TEXT[FEATURES[i]] for i in order[:k] if row[i] > 0])
        return results

    def score_application(self, application: dict) -> dict:
        features = build_features(pd.DataFrame([application]))
        pd_value = float(self.predict_pd(features)[0])
        approved = pd_value <= self.cutoff
        return {
            "probability_of_default": round(pd_value, 4),
            "decision": "approve" if approved else "decline",
            "cutoff": round(self.cutoff, 4),
            "adverse_action_reasons": [] if approved else self.top_reasons(features)[0],
            "model_version": self.version,
        }
