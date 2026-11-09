"""Compare recent applications to the training population and report drift.

Usage:
  python src/monitor.py path/to/applications.csv     check a CSV file
  python src/monitor.py --db --days 30               check applications the API scored in the last 30 days

With --db it reads from DATABASE_URL (or the local SQLite file).
Exits with status 2 if the score or any feature needs action, so a scheduler can alert on it.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import NUMERIC, build_features  # noqa: E402
from metrics import psi, psi_status  # noqa: E402
from scoring import Scorer  # noqa: E402

REFERENCE = Path(__file__).resolve().parents[1] / "artifacts" / "reference.joblib"
MIN_BATCH = 500  # below this, PSI swings too much from noise alone to act on


def drift_report(batch: pd.DataFrame, scorer: Scorer, reference: pd.DataFrame) -> pd.DataFrame:
    features = build_features(batch)
    rows = [{"metric": c, "psi": psi(reference[c].to_numpy(float), features[c].to_numpy(float))} for c in NUMERIC]
    rows.append({"metric": "MODEL_SCORE", "psi": psi(scorer.predict_pd(reference), scorer.predict_pd(features))})
    report = pd.DataFrame(rows).sort_values("psi", ascending=False)
    report["status"] = report["psi"].map(psi_status)
    return report


def load_batch(args: argparse.Namespace) -> pd.DataFrame:
    if args.db:
        import db

        return db.recent_applications(db.get_engine(), db.start_of_window(args.days))
    return pd.read_csv(args.csv)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv", nargs="?", help="CSV of applications")
    parser.add_argument("--db", action="store_true", help="read applications from the scoring database")
    parser.add_argument("--days", type=int, default=30, help="with --db, how many days back to look")
    args = parser.parse_args()
    if bool(args.csv) == args.db:
        parser.error("give either a CSV path or --db")

    batch = load_batch(args)
    if batch.empty:
        print("No applications to check.")
        return 1
    if len(batch) < MIN_BATCH:
        print(f"Warning: only {len(batch)} applications. PSI is unreliable below {MIN_BATCH}.\n")
    report = drift_report(batch, Scorer(), joblib.load(REFERENCE))
    print(report.round(4).to_string(index=False))
    return 2 if (report["status"] == "act").any() else 0


if __name__ == "__main__":
    raise SystemExit(main())
