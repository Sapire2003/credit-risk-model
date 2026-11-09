"""Credit-model metrics, written out by hand so each one can be explained in an interview."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import brier_score_loss, roc_auc_score


def ks_statistic(y: np.ndarray, p: np.ndarray) -> float:
    """Maximum gap between the score distributions of bad and good loans."""
    return float(ks_2samp(p[y == 1], p[y == 0]).statistic)


def expected_calibration_error(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    """Weighted mean |predicted - observed| across equal-population score bins."""
    frame = pd.DataFrame({"y": y, "p": p})
    frame["bin"] = pd.qcut(frame["p"], bins, duplicates="drop")
    grouped = frame.groupby("bin", observed=True).agg(p=("p", "mean"), y=("y", "mean"), n=("y", "size"))
    return float((grouped["n"] / len(frame) * (grouped["p"] - grouped["y"]).abs()).sum())


def evaluate(y: np.ndarray, p: np.ndarray) -> dict:
    auc = float(roc_auc_score(y, p))
    return {
        "auc": auc,
        "gini": 2 * auc - 1,
        "ks": ks_statistic(y, p),
        "brier": float(brier_score_loss(y, p)),
        "ece": expected_calibration_error(y, p),
        "mean_predicted": float(np.mean(p)),
        "observed_rate": float(np.mean(y)),
        "n": int(len(y)),
    }


def bootstrap_auc_ci(y: np.ndarray, p: np.ndarray, n_boot: int = 500, seed: int = 7) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        scores.append(roc_auc_score(y[idx], p[idx]))
    low, high = np.percentile(scores, [2.5, 97.5])
    return float(low), float(high)


def paired_bootstrap_auc_diff(
    y: np.ndarray, a: np.ndarray, b: np.ndarray, n_boot: int = 500, seed: int = 7
) -> tuple[float, float, float]:
    """AUC(a) - AUC(b) with a CI from resampling the same loans for both models."""
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        diffs.append(roc_auc_score(y[idx], a[idx]) - roc_auc_score(y[idx], b[idx]))
    low, high = np.percentile(diffs, [2.5, 97.5])
    return float(roc_auc_score(y, a) - roc_auc_score(y, b)), float(low), float(high)


def psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index. Common rule of thumb: <0.10 stable, 0.10-0.25 watch, >0.25 act."""
    reference = reference[~np.isnan(reference)]
    current = current[~np.isnan(current)]
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:  # near-constant feature: fall back to a two-sided split around the value
        edges = np.array([reference.min() - 1, reference.mean(), reference.max() + 1])
    edges[0], edges[-1] = -np.inf, np.inf
    ref = np.histogram(reference, edges)[0] / len(reference)
    cur = np.histogram(current, edges)[0] / len(current)
    ref, cur = np.clip(ref, 1e-4, None), np.clip(cur, 1e-4, None)
    return float(np.sum((cur - ref) * np.log(cur / ref)))


def psi_status(value: float) -> str:
    return "stable" if value < 0.10 else "watch" if value < 0.25 else "act"
