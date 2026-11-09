"""Record every scored application in a SQL database and report on them with SQL.

Production uses PostgreSQL (Amazon RDS). Locally, with no DATABASE_URL set, it falls back to a
SQLite file so the project runs with no setup. The same queries run on both.

Why store every decision: lenders must keep application records (Regulation B requires 25 months),
and the stored applications are what the drift monitor compares against the training data.
"""
from __future__ import annotations

import os
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import sqlalchemy as sa

LOCAL_DB = Path(__file__).resolve().parents[1] / "data" / "scoring.db"

APPLICATION_FIELDS = [
    "loan_amnt", "term", "annual_inc", "dti", "home_ownership", "purpose", "verification_status",
    "emp_length", "delinq_2yrs", "revol_util", "total_acc", "longest_credit_length",
]

metadata = sa.MetaData()

applications = sa.Table(
    "scored_applications", metadata,
    sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
    sa.Column("scored_at", sa.DateTime(timezone=True), nullable=False),
    sa.Column("scored_on", sa.Date, nullable=False, index=True),
    sa.Column("model_version", sa.String(20), nullable=False),
    sa.Column("loan_amnt", sa.Float, nullable=False),
    sa.Column("term", sa.String(20), nullable=False),
    sa.Column("annual_inc", sa.Float, nullable=False),
    sa.Column("dti", sa.Float, nullable=False),
    sa.Column("home_ownership", sa.String(20), nullable=False),
    sa.Column("purpose", sa.String(30), nullable=False),
    sa.Column("verification_status", sa.String(20), nullable=False),
    sa.Column("emp_length", sa.Float),
    sa.Column("delinq_2yrs", sa.Float, nullable=False),
    sa.Column("revol_util", sa.Float),
    sa.Column("total_acc", sa.Float),
    sa.Column("longest_credit_length", sa.Float),
    sa.Column("probability_of_default", sa.Float, nullable=False),
    sa.Column("decision", sa.String(10), nullable=False),
    sa.CheckConstraint("decision IN ('approve', 'decline')", name="valid_decision"),
)

# One row per reason, so reasons can be counted with GROUP BY instead of parsing text.
decline_reasons = sa.Table(
    "decline_reasons", metadata,
    sa.Column("application_id", sa.Integer, sa.ForeignKey("scored_applications.id", ondelete="CASCADE"),
              primary_key=True),
    sa.Column("rank", sa.SmallInteger, primary_key=True),
    sa.Column("reason", sa.String(120), nullable=False),
)

SINCE = sa.bindparam("since", type_=sa.Date)

# Approval rate by day, plus a rolling rate over the last 7 days that had applications.
DAILY_SUMMARY = sa.text("""
WITH daily AS (
    SELECT scored_on,
           COUNT(*) AS applications,
           SUM(CASE WHEN decision = 'approve' THEN 1 ELSE 0 END) AS approvals,
           AVG(probability_of_default) AS avg_probability_of_default
    FROM scored_applications
    WHERE scored_on >= :since
    GROUP BY scored_on
)
SELECT scored_on,
       applications,
       approvals,
       1.0 * approvals / applications AS approval_rate,
       avg_probability_of_default,
       1.0 * SUM(approvals) OVER last_7 / SUM(applications) OVER last_7 AS approval_rate_rolling_7
FROM daily
WINDOW last_7 AS (ORDER BY scored_on ROWS BETWEEN 6 PRECEDING AND CURRENT ROW)
ORDER BY scored_on
""").bindparams(SINCE)

# How often each reason appears among declines. Shares can sum past 1: a decline lists up to 4 reasons.
TOP_DECLINE_REASONS = sa.text("""
SELECT r.reason,
       COUNT(*) AS declines,
       1.0 * COUNT(*) / NULLIF((SELECT COUNT(*) FROM scored_applications
                                WHERE decision = 'decline' AND scored_on >= :since), 0) AS share_of_declines
FROM decline_reasons r
JOIN scored_applications a ON a.id = r.application_id
WHERE a.scored_on >= :since
GROUP BY r.reason
ORDER BY declines DESC, r.reason
""").bindparams(SINCE)

RECENT_APPLICATIONS = sa.text(
    f"SELECT {', '.join(APPLICATION_FIELDS)} FROM scored_applications WHERE scored_on >= :since ORDER BY id"
).bindparams(SINCE)


def database_url() -> str:
    """DATABASE_URL if set, else the local SQLite file. Accepts the usual postgres:// forms."""
    url = os.environ.get("DATABASE_URL", "").strip()
    if not url:
        return f"sqlite:///{LOCAL_DB.as_posix()}"
    for prefix in ("postgres://", "postgresql://"):
        if url.startswith(prefix):
            return "postgresql+psycopg://" + url[len(prefix):]
    return url


def get_engine(url: str | None = None) -> sa.Engine:
    url = url or database_url()
    if url.startswith("sqlite:///") and not url.endswith(":memory:"):
        Path(url.removeprefix("sqlite:///")).parent.mkdir(parents=True, exist_ok=True)
    engine = sa.create_engine(url, pool_pre_ping=True)
    metadata.create_all(engine)
    return engine


def log_score(engine: sa.Engine, application: dict, result: dict, now: datetime | None = None) -> int:
    """Save one application and its decision in a single transaction. Returns the new row id."""
    now = now or datetime.now(timezone.utc)
    row = {field: application.get(field) for field in APPLICATION_FIELDS}
    row.update(
        scored_at=now, scored_on=now.date(), model_version=result["model_version"],
        probability_of_default=result["probability_of_default"], decision=result["decision"],
    )
    with engine.begin() as conn:
        app_id = conn.execute(applications.insert().values(**row)).inserted_primary_key[0]
        reasons = [{"application_id": app_id, "rank": i, "reason": text}
                   for i, text in enumerate(result.get("adverse_action_reasons", []), start=1)]
        if reasons:
            conn.execute(decline_reasons.insert(), reasons)
    return int(app_id)


def start_of_window(days: int) -> date:
    """First day of a window covering the last `days` days, today included, in UTC like scored_on."""
    return datetime.now(timezone.utc).date() - timedelta(days=days - 1)


def _query(engine: sa.Engine, query: sa.TextClause, since: date) -> pd.DataFrame:
    with engine.connect() as conn:
        return pd.read_sql(query, conn, params={"since": since})


def daily_summary(engine: sa.Engine, since: date) -> pd.DataFrame:
    frame = _query(engine, DAILY_SUMMARY, since)
    frame["scored_on"] = pd.to_datetime(frame["scored_on"]).dt.date
    return frame


def top_decline_reasons(engine: sa.Engine, since: date) -> pd.DataFrame:
    return _query(engine, TOP_DECLINE_REASONS, since)


def recent_applications(engine: sa.Engine, since: date) -> pd.DataFrame:
    return _query(engine, RECENT_APPLICATIONS, since)
