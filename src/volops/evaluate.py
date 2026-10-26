"""Scoring a model against the constant-variance benchmark on origins whose outcome is already known.

The same function serves two jobs: the holdout check at training time and the rolling production
check in the monitor. In production the label of a forecast made at t arrives h days later, so the
newest h origins can never be scored yet; `scoreable_origins` enforces that.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from volops import model as M


@dataclass(frozen=True)
class Score:
    n: int
    qlike_model: float
    qlike_benchmark: float
    gain: float  # benchmark - model; positive means the model is better
    dm_p: float  # Diebold-Mariano p-value (two-sided, HLN-corrected)


def dm_test(loss_a: np.ndarray, loss_b: np.ndarray, h: int) -> float:
    """Two-sided DM p-value with a Bartlett long-run variance (h-1 lags: h-step errors overlap) and the HLN correction."""
    d = np.asarray(loss_a) - np.asarray(loss_b)
    n = d.size
    if n < 10:
        return 1.0
    dc = d - d.mean()
    lrv = float(dc @ dc) / n
    for k in range(1, h):
        lrv += 2.0 * (1.0 - k / h) * float(dc[k:] @ dc[:-k]) / n
    if lrv <= 0:
        return 1.0
    stat = d.mean() / np.sqrt(lrv / n) * np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    return float(2.0 * stats.t.sf(abs(stat), df=n - 1))


def scoreable_origins(r: pd.Series, h: int) -> pd.Index:
    """Origins with a fully observed h-day future and complete features."""
    feats = M.features(r)
    y = M.target(r, h)
    return r.index[feats.notna().all(axis=1) & y.notna()]


def score(model: M.SeriesModel, r: pd.Series, origins: pd.Index) -> Score:
    h = model.horizon
    feats, y, bench = M.features(r).loc[origins], M.target(r, h).loc[origins], M.expanding_benchmark(r).loc[origins]
    lm, lb = M.qlike(model.predict(feats), y.to_numpy()), M.qlike(bench.to_numpy(), y.to_numpy())
    return Score(len(origins), float(lm.mean()), float(lb.mean()), float(lb.mean() - lm.mean()), dm_test(lm, lb, h))
