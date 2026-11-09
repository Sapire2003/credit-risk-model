"""Scoring service.  Run from the project root:  uvicorn --app-dir src api:app --port 8000

Settings (environment variables, both optional):
  DATABASE_URL   PostgreSQL connection string. Without it, decisions go to a local SQLite file.
  MODEL_S3_URI   s3://bucket/key of the model. Without it, the local artifacts/model.joblib is used.
"""
from __future__ import annotations

import logging
from enum import Enum

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.exc import SQLAlchemyError

import db
from model_store import ensure_model
from scoring import Scorer

log = logging.getLogger("credit_api")

app = FastAPI(title="Credit Default Scoring API", version="1.1.0")
scorer = Scorer(ensure_model())
engine = db.get_engine()


class Term(str, Enum):
    short = "36 months"
    long = "60 months"


class Purpose(str, Enum):
    car = "car"
    credit_card = "credit_card"
    debt_consolidation = "debt_consolidation"
    educational = "educational"
    home_improvement = "home_improvement"
    house = "house"
    major_purchase = "major_purchase"
    medical = "medical"
    moving = "moving"
    other = "other"
    renewable_energy = "renewable_energy"
    small_business = "small_business"
    vacation = "vacation"
    wedding = "wedding"


class Verification(str, Enum):
    verified = "verified"
    not_verified = "not verified"


class Application(BaseModel):
    """Only fields known at application time. Protected characteristics are not accepted."""

    loan_amnt: float = Field(gt=0, le=100_000)
    term: Term
    annual_inc: float = Field(gt=0)
    dti: float = Field(ge=0, le=100)
    home_ownership: str = Field(max_length=20)
    purpose: Purpose
    verification_status: Verification
    emp_length: float | None = Field(default=None, ge=0, le=60)
    delinq_2yrs: float = Field(default=0, ge=0)
    revol_util: float | None = Field(default=None, ge=0)
    total_acc: float | None = Field(default=None, ge=0)
    longest_credit_length: float | None = Field(default=None, ge=0)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_version": scorer.version, "database": engine.dialect.name}


@app.post("/score")
def score(application: Application) -> dict:
    payload = application.model_dump(mode="json")
    result = scorer.score_application(payload)
    # A lending decision that isn't recorded can't be audited or explained later, so if the
    # database is down we refuse to decide rather than return an unrecorded decision.
    try:
        result["application_id"] = db.log_score(engine, payload, result)
    except SQLAlchemyError:
        log.exception("Could not record decision")
        raise HTTPException(status_code=503, detail="Decision could not be recorded. Try again shortly.")
    return result


@app.get("/reports/daily")
def daily_report(days: int = Query(default=30, ge=1, le=730)) -> list[dict]:
    frame = db.daily_summary(engine, db.start_of_window(days))
    frame["scored_on"] = frame["scored_on"].astype(str)
    return frame.round(4).to_dict(orient="records")


@app.get("/reports/decline-reasons")
def decline_reason_report(days: int = Query(default=30, ge=1, le=730)) -> list[dict]:
    return db.top_decline_reasons(engine, db.start_of_window(days)).round(4).to_dict(orient="records")
