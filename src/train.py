"""Train, compare, calibrate, and validate the default model; write artifacts and reports.

Run from the project root:  python src/train.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import lightgbm as lgb
import matplotlib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import CATEGORICAL, FEATURES, MONOTONE, NUMERIC, TARGET, build_features, total_interest  # noqa: E402
from metrics import bootstrap_auc_ci, evaluate, paired_bootstrap_auc_diff, psi, psi_status  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw" / "loan.csv"
REPORTS = ROOT / "reports"
ARTIFACTS = ROOT / "artifacts"
SEED = 42

# Assumptions behind the policy analysis. These are not in the data, so they get a sensitivity test.
LGD_BASE = 0.65
LGD_SENSITIVITY = [0.50, 0.65, 0.80]

BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#898781", "#e1e0d9"
plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 10, "axes.edgecolor": "#c3c2b7", "axes.labelcolor": "#52514e",
    "xtick.color": "#52514e", "ytick.color": "#52514e", "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.dpi": 160,
})

BASE_PARAMS = {
    "objective": "binary", "metric": "auc", "learning_rate": 0.03, "num_leaves": 31, "min_data_in_leaf": 200,
    "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 5.0, "seed": SEED,
    "verbosity": -1,
}


def fit_lgbm(x_tr, y_tr, x_val, y_val, monotone: bool) -> lgb.Booster:
    params = dict(BASE_PARAMS)
    if monotone:
        params["monotone_constraints"] = [MONOTONE.get(col, 0) for col in FEATURES]
    train_set = lgb.Dataset(x_tr, y_tr, categorical_feature=CATEGORICAL)
    valid_set = lgb.Dataset(x_val, y_val, categorical_feature=CATEGORICAL, reference=train_set)
    return lgb.train(params, train_set, 2000, valid_sets=[valid_set],
                     callbacks=[lgb.early_stopping(100, verbose=False)])


def fit_logistic(x_tr, y_tr):
    prep = ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])
    frame = x_tr.copy()
    frame[CATEGORICAL] = frame[CATEGORICAL].astype(object)
    return make_pipeline(prep, LogisticRegression(max_iter=2000, C=1.0)).fit(frame, y_tr)


def logistic_predict(model, x) -> np.ndarray:
    frame = x.copy()
    frame[CATEGORICAL] = frame[CATEGORICAL].astype(object)
    return model.predict_proba(frame)[:, 1]


def realized_value(raw: pd.DataFrame, approved: np.ndarray, lgd: float) -> float:
    """Dollars earned on approved loans, using each loan's actual outcome.

    Good loan: earns its contractual interest. Bad loan: loses lgd x principal.
    A simplification: it ignores funding and servicing costs, and interest collected before default.
    """
    interest = total_interest(raw["loan_amnt"], raw["int_rate"], raw["term"])
    value = np.where(raw[TARGET] == 1, -lgd * raw["loan_amnt"], interest)
    return float(value[approved].sum())


def policy_table(raw: pd.DataFrame, p: np.ndarray, lgd: float) -> pd.DataFrame:
    rows = []
    for rate in np.round(np.arange(0.30, 1.001, 0.01), 2):
        cutoff = np.quantile(p, rate) if rate < 1 else 1.0
        approved = p <= cutoff
        rows.append({
            "approval_rate": float(approved.mean()),
            "cutoff": float(cutoff),
            "bad_rate": float(raw.loc[approved, TARGET].mean()),
            "value_per_application": realized_value(raw, approved, lgd) / len(raw),
        })
    return pd.DataFrame(rows)


def main() -> None:
    REPORTS.mkdir(exist_ok=True)
    ARTIFACTS.mkdir(exist_ok=True)
    raw = pd.read_csv(DATA)
    x = build_features(raw)
    y = raw[TARGET].to_numpy()
    idx = np.arange(len(raw))

    # 60% train / 10% early stopping / 10% calibration and cutoff choice / 20% untouched test.
    tr, test = train_test_split(idx, test_size=0.20, random_state=SEED, stratify=y)
    tr, rest = train_test_split(tr, test_size=0.25, random_state=SEED, stratify=y[tr])
    stop, calib = train_test_split(rest, test_size=0.50, random_state=SEED, stratify=y[rest])

    logistic = fit_logistic(x.iloc[tr], y[tr])
    free = fit_lgbm(x.iloc[tr], y[tr], x.iloc[stop], y[stop], monotone=False)
    mono = fit_lgbm(x.iloc[tr], y[tr], x.iloc[stop], y[stop], monotone=True)

    # Calibrate on data the booster never saw; keep the calibrator only if it helps on that same set.
    iso = IsotonicRegression(out_of_bounds="clip").fit(mono.predict(x.iloc[calib]), y[calib])
    raw_calib = evaluate(y[calib], mono.predict(x.iloc[calib]))
    # Fit on half, score on the other half so the comparison isn't flattered by in-sample isotonic fit.
    half_a, half_b = train_test_split(calib, test_size=0.5, random_state=SEED, stratify=y[calib])
    iso_check = IsotonicRegression(out_of_bounds="clip").fit(mono.predict(x.iloc[half_a]), y[half_a])
    use_iso = (
        evaluate(y[half_b], iso_check.predict(mono.predict(x.iloc[half_b])))["ece"]
        < evaluate(y[half_b], mono.predict(x.iloc[half_b]))["ece"]
    )
    calibrator = iso if use_iso else None

    def final_pd(frame: pd.DataFrame) -> np.ndarray:
        score = mono.predict(frame)
        return calibrator.predict(score) if calibrator is not None else score

    y_test = y[test]
    x_test = x.iloc[test]
    p_logit = logistic_predict(logistic, x_test)
    p_free = free.predict(x_test)
    p_mono_raw = mono.predict(x_test)
    p_final = final_pd(x_test)
    rate_benchmark = raw["int_rate"].to_numpy()[test]

    models = {
        "logistic_regression": evaluate(y_test, p_logit),
        "lightgbm_unconstrained": evaluate(y_test, p_free),
        "lightgbm_monotone": evaluate(y_test, p_mono_raw),
        "lightgbm_monotone_final": evaluate(y_test, p_final),
    }
    models["lightgbm_monotone_final"]["auc_95ci"] = bootstrap_auc_ci(y_test, p_final)
    # int_rate isn't a probability, so only its rank-based metric is meaningful.
    from sklearn.metrics import roc_auc_score
    benchmark_auc = float(roc_auc_score(y_test, rate_benchmark))
    comparisons = {
        "final_vs_logistic": paired_bootstrap_auc_diff(y_test, p_final, p_logit),
        "final_vs_unconstrained": paired_bootstrap_auc_diff(y_test, p_final, p_free),
        "final_vs_lendingclub_rate": paired_bootstrap_auc_diff(y_test, p_final, rate_benchmark),
    }

    # File-order holdout: train on the first 80% of rows, test on the last 20%. The file has no dates,
    # but default rates and rates shift in blocks along it, so this is a harsher test of population shift.
    # It's a stress test, NOT a verified out-of-time test.
    cut = int(len(raw) * 0.8)
    order_tr, order_stop = train_test_split(idx[:cut], test_size=0.1, random_state=SEED, stratify=y[:cut])
    order_model = fit_lgbm(x.iloc[order_tr], y[order_tr], x.iloc[order_stop], y[order_stop], monotone=True)
    order_eval = evaluate(y[cut:], order_model.predict(x.iloc[cut:]))

    # Choose the cutoff on the calibration split, then report its effect once on the test split.
    calib_policy = policy_table(raw.iloc[calib], final_pd(x.iloc[calib]), LGD_BASE)
    best = calib_policy.loc[calib_policy["value_per_application"].idxmax()]
    cutoff = float(best["cutoff"])
    test_raw = raw.iloc[test]
    approved = p_final <= cutoff
    everyone = np.ones(len(test), dtype=bool)
    policy = {
        "chosen_on": "calibration split, LGD 65%",
        "cutoff_pd": cutoff,
        "test_approval_rate": float(approved.mean()),
        "test_bad_rate_approved": float(y_test[approved].mean()),
        "test_bad_rate_all": float(y_test.mean()),
        "note": "Applied to LendingClub-funded loans only; declined applicants have no observed outcome.",
        "lgd_sensitivity": {},
    }
    for lgd in LGD_SENSITIVITY:
        value_all = realized_value(test_raw, everyone, lgd) / len(test)
        value_cut = realized_value(test_raw, approved, lgd) / len(test)
        policy["lgd_sensitivity"][f"{lgd:.2f}"] = {
            "value_per_loan_approve_all": value_all,
            "value_per_loan_with_cutoff": value_cut,
            "lift_per_loan": value_cut - value_all,
        }

    # Population drift, relative to training data.
    drift_rows = {}
    train_x = x.iloc[tr]
    stress = raw.iloc[test].copy()
    stress["annual_inc"] *= 0.80
    stress["dti"] = (stress["dti"] + 5).clip(upper=60)
    stress["revol_util"] = (stress["revol_util"] + 10).clip(upper=150)
    x_stress = build_features(stress)
    scenarios = {"random_test_split": x_test, "file_order_holdout": x.iloc[cut:], "simulated_downturn": x_stress}
    ref_score = final_pd(train_x)
    for name, frame in scenarios.items():
        cols = {c: psi(train_x[c].to_numpy(dtype=float), frame[c].to_numpy(dtype=float)) for c in NUMERIC}
        score_psi = psi(ref_score, final_pd(frame))
        top = max(cols, key=cols.get)
        drift_rows[name] = {
            "score_psi": score_psi, "score_status": psi_status(score_psi),
            "largest_feature": top, "largest_feature_psi": cols[top], "largest_feature_status": psi_status(cols[top]),
            "mean_predicted_pd": float(final_pd(frame).mean()),
        }

    # Global importance: mean absolute SHAP on the test split.
    contrib = mono.predict(x_test, pred_contrib=True)[:, :-1]
    importance = pd.Series(np.abs(contrib).mean(axis=0), index=FEATURES).sort_values()

    # Performance by subgroup available in the data. This dataset has no protected-class fields.
    segments = {}
    for col in ["term", "home_ownership", "purpose"]:
        segments[col] = {}
        for value, group in pd.DataFrame({"v": test_raw[col].to_numpy(), "y": y_test, "p": p_final}).groupby("v"):
            if len(group) >= 500 and group["y"].nunique() == 2:
                segments[col][str(value)] = {
                    "n": int(len(group)), "auc": float(roc_auc_score(group["y"], group["p"])),
                    "observed": float(group["y"].mean()), "predicted": float(group["p"].mean()),
                }

    metrics = {
        "data": {"rows": int(len(raw)), "bad_rate": float(y.mean()), "train": len(tr), "early_stop": len(stop),
                 "calibration": len(calib), "test": len(test)},
        "models_on_test": models,
        "benchmark_lendingclub_rate_auc": benchmark_auc,
        "auc_differences": {k: {"diff": v[0], "ci95": [v[1], v[2]]} for k, v in comparisons.items()},
        "isotonic_kept": bool(use_iso),
        "calibration_split_raw": raw_calib,
        "file_order_holdout": order_eval,
        "policy": policy,
        "drift": drift_rows,
        "segments": segments,
        "shap_importance": importance.sort_values(ascending=False).round(4).to_dict(),
        "best_iterations": {"unconstrained": free.best_iteration, "monotone": mono.best_iteration},
    }
    (REPORTS / "metrics.json").write_text(json.dumps(metrics, indent=2))
    joblib.dump({"booster": mono, "calibrator": calibrator, "cutoff": cutoff, "version": "1.0.0"},
                ARTIFACTS / "model.joblib")
    joblib.dump(train_x, ARTIFACTS / "reference.joblib")

    # Chart 1: calibration.
    bins = pd.DataFrame({"y": y_test, "p": p_final})
    bins["bin"] = pd.qcut(bins["p"], 10, duplicates="drop")
    curve = bins.groupby("bin", observed=True).agg(pred=("p", "mean"), obs=("y", "mean"))
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    limit = max(curve.max()) * 1.1
    ax.plot([0, limit], [0, limit], color=MUTED, lw=1, ls="--")
    ax.annotate("Perfect calibration", (limit * 0.56, limit * 0.47), color=MUTED, fontsize=9)
    ax.plot(curve["pred"], curve["obs"], color=BLUE, lw=2, marker="o", ms=5)
    ax.set(title="Predicted default rates track observed rates", xlabel="Predicted default probability (test decile)",
           ylabel="Observed default rate", xlim=(0, limit), ylim=(0, limit))
    fig.tight_layout(); fig.savefig(REPORTS / "calibration.png"); plt.close(fig)

    # Chart 2: approval rate vs. realized value.
    test_policy = policy_table(test_raw, p_final, LGD_BASE)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(test_policy["approval_rate"] * 100, test_policy["value_per_application"], color=BLUE, lw=2)
    point = (approved.mean() * 100, policy["lgd_sensitivity"]["0.65"]["value_per_loan_with_cutoff"])
    ax.plot(*point, marker="o", ms=8, color=ORANGE, mec="#fcfcfb", mew=2, ls="none")
    ax.annotate(f"Chosen cutoff: approve {point[0]:.0f}%", point, xytext=(-10, -18), textcoords="offset points",
                ha="right", color=INK, fontsize=9)
    ax.set(title="Value peaks near full approval, then falls as volume is cut", xlabel="Approval rate among funded loans (%)",
           ylabel="Realized value per loan reviewed ($)")
    fig.tight_layout(); fig.savefig(REPORTS / "policy_curve.png"); plt.close(fig)

    # Chart 3: SHAP importance.
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.barh(importance.index, importance.values, color=BLUE, height=0.6)
    ax.grid(axis="y", visible=False)
    ax.set(title="What drives the model", xlabel="Mean |SHAP value| (log-odds of default)")
    fig.tight_layout(); fig.savefig(REPORTS / "feature_importance.png"); plt.close(fig)

    print(json.dumps({k: metrics[k] for k in
                      ["models_on_test", "benchmark_lendingclub_rate_auc", "auc_differences", "isotonic_kept",
                       "file_order_holdout", "policy", "drift", "best_iterations"]}, indent=1))
    print(importance.sort_values(ascending=False).round(3).to_string())


if __name__ == "__main__":
    main()
