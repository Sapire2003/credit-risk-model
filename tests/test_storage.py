"""Database and S3 tests. Run from the project root after training:  python -m pytest

The database tests run on SQLite always, and also on PostgreSQL when TEST_POSTGRES_URL is set.
"""
import io
import os
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import boto3
import pytest
import sqlalchemy as sa
from botocore.response import StreamingBody
from botocore.stub import Stubber
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import api  # noqa: E402
import db  # noqa: E402
from features import build_features  # noqa: E402
from model_store import LOCAL_MODEL, ensure_model, parse_s3_uri  # noqa: E402
from scoring import Scorer  # noqa: E402
from test_model import GOOD, RISKY  # noqa: E402
from upload_model import upload  # noqa: E402

client = TestClient(api.app)
DATABASES = ["sqlite", pytest.param("postgres", marks=pytest.mark.skipif(
    not os.environ.get("TEST_POSTGRES_URL"), reason="set TEST_POSTGRES_URL to also test PostgreSQL"))]


@pytest.fixture(params=DATABASES)
def engine(request, tmp_path):
    if request.param == "sqlite":
        url = f"sqlite:///{(tmp_path / 'db.sqlite').as_posix()}"
    else:
        os.environ["DATABASE_URL"], saved = os.environ["TEST_POSTGRES_URL"], os.environ["DATABASE_URL"]
        url = db.database_url()  # also checks the postgres:// to postgresql+psycopg:// rewrite
        os.environ["DATABASE_URL"] = saved
    engine = db.get_engine(url)
    yield engine
    db.metadata.drop_all(engine)
    engine.dispose()


def decision(pd_value, approved, reasons=()):
    return {"probability_of_default": pd_value, "decision": "approve" if approved else "decline",
            "adverse_action_reasons": list(reasons), "model_version": "test"}


def at(day):
    return datetime(2026, 11, day, 12, tzinfo=timezone.utc)


def test_database_url_accepts_common_postgres_forms(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgres://u:p@host:5432/loans")
    assert db.database_url() == "postgresql+psycopg://u:p@host:5432/loans"
    monkeypatch.setenv("DATABASE_URL", "")
    assert db.database_url().startswith("sqlite:///")


def test_daily_summary_and_rolling_rate(engine):
    db.log_score(engine, GOOD, decision(0.1, True), now=at(1))
    db.log_score(engine, GOOD, decision(0.2, True), now=at(1))
    db.log_score(engine, GOOD, decision(0.3, True), now=at(2))
    db.log_score(engine, RISKY, decision(0.9, False, ["A"]), now=at(2))
    db.log_score(engine, RISKY, decision(0.8, False, ["A"]), now=at(3))  # outside the window below

    summary = db.daily_summary(engine, since=date(2026, 11, 1))
    assert list(summary["applications"]) == [2, 2, 1]
    assert list(summary["approval_rate"]) == pytest.approx([1.0, 0.5, 0.0])
    assert list(summary["approval_rate_rolling_7"]) == pytest.approx([1.0, 3 / 4, 3 / 5])
    assert summary["avg_probability_of_default"].iloc[1] == pytest.approx(0.6)
    assert len(db.daily_summary(engine, since=date(2026, 11, 3))) == 1


def test_decline_reasons_are_counted_per_decline(engine):
    db.log_score(engine, RISKY, decision(0.9, False, ["High debt", "Short history"]), now=at(1))
    db.log_score(engine, RISKY, decision(0.8, False, ["High debt"]), now=at(1))
    db.log_score(engine, GOOD, decision(0.1, True), now=at(1))

    reasons = db.top_decline_reasons(engine, since=date(2026, 11, 1))
    assert list(reasons["reason"]) == ["High debt", "Short history"]
    assert list(reasons["declines"]) == [2, 1]
    assert list(reasons["share_of_declines"]) == pytest.approx([1.0, 0.5])
    assert db.top_decline_reasons(engine, since=date(2026, 12, 1)).empty  # no division-by-zero error


def test_stored_applications_can_be_rescored(engine):
    db.log_score(engine, {k: v for k, v in GOOD.items() if k != "emp_length"}, decision(0.1, True), now=at(1))
    db.log_score(engine, RISKY, decision(0.9, False, ["A"]), now=at(1))
    batch = db.recent_applications(engine, since=date(2026, 11, 1))
    assert list(batch.columns) == db.APPLICATION_FIELDS
    assert batch["emp_length"].isna().iloc[0]  # missing values stay missing, not zero
    assert Scorer().predict_pd(build_features(batch))[1] > 0.5


def test_bad_decision_value_is_rejected_by_the_database(engine):
    with pytest.raises(sa.exc.IntegrityError):
        db.log_score(engine, GOOD, {**decision(0.1, True), "decision": "maybe"}, now=at(1))


def test_api_records_each_decision_with_reasons():
    body = client.post("/score", json=RISKY).json()
    with api.engine.connect() as conn:
        stored = conn.execute(sa.select(db.applications).where(db.applications.c.id == body["application_id"])).one()
        reasons = conn.execute(sa.select(db.decline_reasons.c.reason)
                               .where(db.decline_reasons.c.application_id == body["application_id"])
                               .order_by(db.decline_reasons.c.rank)).scalars().all()
    assert stored.decision == "decline" and stored.loan_amnt == RISKY["loan_amnt"]
    assert reasons == body["adverse_action_reasons"]


def test_api_reports():
    client.post("/score", json=GOOD)
    client.post("/score", json=RISKY)
    daily = client.get("/reports/daily?days=7").json()
    assert daily[-1]["applications"] >= 2
    reasons = client.get("/reports/decline-reasons?days=7").json()
    assert reasons and {"reason", "declines", "share_of_declines"} <= set(reasons[0])


def test_api_refuses_to_decide_if_it_cannot_record(monkeypatch):
    def broken(*args, **kwargs):
        raise sa.exc.OperationalError("INSERT", {}, Exception("database is down"))

    monkeypatch.setattr(db, "log_score", broken)
    assert client.post("/score", json=GOOD).status_code == 503


def test_parse_s3_uri():
    assert parse_s3_uri("s3://bucket/models/m.joblib") == ("bucket", "models/m.joblib")
    for bad in ["https://bucket/m", "s3://bucket", "s3:///key"]:
        with pytest.raises(ValueError):
            parse_s3_uri(bad)


def test_no_s3_uri_means_local_model():
    assert ensure_model(uri="") == LOCAL_MODEL


def test_model_downloads_from_s3_once(tmp_path):
    s3 = boto3.client("s3", region_name="us-east-1", aws_access_key_id="test", aws_secret_access_key="test")
    data = LOCAL_MODEL.read_bytes()
    with Stubber(s3) as stub:
        stub.add_response("get_object", {"Body": StreamingBody(io.BytesIO(data), len(data))},
                          {"Bucket": "bucket", "Key": "models/m.joblib"})
        path = ensure_model("s3://bucket/models/m.joblib", cache_dir=tmp_path, client=s3)
        again = ensure_model("s3://bucket/models/m.joblib", cache_dir=tmp_path, client=s3)  # no second call
        stub.assert_no_pending_responses()
    assert path == again and Scorer(path).version == Scorer().version


def test_upload_uses_a_versioned_key():
    s3 = boto3.client("s3", region_name="us-east-1", aws_access_key_id="test", aws_secret_access_key="test")
    key = f"models/credit-default-{Scorer().version}.joblib"
    with Stubber(s3) as stub:
        stub.add_response("put_object", {}, {"Bucket": "bucket", "Key": key, "Body": LOCAL_MODEL.read_bytes()})
        assert upload("s3://bucket/models", client=s3) == f"s3://bucket/{key}"
