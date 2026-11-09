"""Run from the project root after training:  python -m pytest"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from api import app  # noqa: E402
from features import EXCLUDED, FEATURES, build_features, total_interest  # noqa: E402
from metrics import psi  # noqa: E402
from scoring import REASON_TEXT, Scorer  # noqa: E402

client = TestClient(app)
GOOD = {
    "loan_amnt": 8000, "term": "36 months", "annual_inc": 160000, "dti": 6, "home_ownership": "MORTGAGE",
    "purpose": "car", "verification_status": "not verified", "emp_length": 10, "delinq_2yrs": 0,
    "revol_util": 8, "total_acc": 30, "longest_credit_length": 25,
}
RISKY = {
    "loan_amnt": 35000, "term": "60 months", "annual_inc": 30000, "dti": 38, "home_ownership": "RENT",
    "purpose": "small_business", "verification_status": "verified", "emp_length": 0, "delinq_2yrs": 4,
    "revol_util": 98, "total_acc": 6, "longest_credit_length": 2,
}


def test_no_excluded_fields_in_features():
    assert not set(EXCLUDED) & set(FEATURES)


def test_every_feature_has_reason_text():
    assert set(FEATURES) == set(REASON_TEXT)


def test_unseen_category_is_handled():
    frame = build_features(pd.DataFrame([{**GOOD, "home_ownership": "SPACESHIP"}]))
    assert frame["home_ownership"].iloc[0] == "OTHER"


def test_total_interest_matches_known_amortization():
    # $10,000 at 12% for 36 months has a $332.14 payment, so about $1,957 of interest.
    result = total_interest(pd.Series([10000.0]), pd.Series([12.0]), pd.Series(["36 months"])).iloc[0]
    assert result == pytest.approx(1957.15, abs=1)


def test_psi_detects_shift_and_ignores_noise():
    rng = np.random.default_rng(0)
    base = rng.normal(size=50_000)
    assert psi(base, rng.normal(size=50_000)) < 0.01
    assert psi(base, rng.normal(1.0, size=50_000)) > 0.25


@pytest.mark.parametrize("feature,direction", [("dti", 1), ("revol_util", 1), ("delinq_2yrs", 1), ("annual_inc", -1)])
def test_monotone_constraints_hold_in_the_model(feature, direction):
    scorer = Scorer()
    lows = {"dti": 1, "revol_util": 1, "delinq_2yrs": 0, "annual_inc": 25000}
    highs = {"dti": 39, "revol_util": 110, "delinq_2yrs": 8, "annual_inc": 400000}
    values = np.linspace(lows[feature], highs[feature], 40)
    frame = build_features(pd.DataFrame([{**RISKY, feature: v} for v in values]))
    # For income, loan_to_income moves too, and its constraint points the same way.
    steps = np.diff(scorer.booster.predict(frame[FEATURES]))
    assert (steps * direction >= -1e-12).all()


def test_health():
    assert client.get("/health").json()["status"] == "ok"


def test_risky_applicant_scores_above_safe_applicant():
    good = client.post("/score", json=GOOD).json()
    risky = client.post("/score", json=RISKY).json()
    assert risky["probability_of_default"] > good["probability_of_default"]
    assert good["decision"] == "approve" and good["adverse_action_reasons"] == []


def test_decline_includes_reasons():
    body = client.post("/score", json=RISKY).json()
    assert body["decision"] == "decline"
    assert 1 <= len(body["adverse_action_reasons"]) <= 4


def test_missing_optional_fields_still_score():
    payload = {k: v for k, v in GOOD.items() if k not in {"emp_length", "revol_util"}}
    assert client.post("/score", json=payload).status_code == 200


@pytest.mark.parametrize("patch", [{"annual_inc": -5}, {"loan_amnt": 0}, {"term": "12 months"}, {"dti": 500}])
def test_invalid_input_is_rejected(patch):
    assert client.post("/score", json={**GOOD, **patch}).status_code == 422
