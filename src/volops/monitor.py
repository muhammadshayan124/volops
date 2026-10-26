"""Production monitoring: is the model still good, and do its inputs still look like its training data?

Two signals, because they fail differently.

Performance (the one that matters). A forecast made at t is scored once its h-day future has happened,
so the monitor re-scores the last `window` origins that are by now scoreable. Because the model is a
fixed function of observable data, re-computing those forecasts is exactly what the service would have
said at the time. Positive gain over the constant-variance benchmark means the model is still earning its place.
Only origins AFTER the model's `trained_through` date count as evidence: earlier ones are in the model's own
training data and flatter it. Right after a retrain there are none, and the report says so instead of
quoting an in-sample number as if it were skill.

Input drift (advisory only; see below). Distribution of the model's inputs over the last `drift_window` days
against the sample the model was trained on, tested with a two-sample Kolmogorov-Smirnov test. Variance
features are heavy-tailed and strongly autocorrelated, so a plain p-value alarms far too often; a flag
therefore needs both a small p-value AND a KS statistic above `min_effect`. This is a heuristic, not a
calibrated test.

Drift does NOT trigger retraining. Replaying a model trained 14 months earlier on real ECB data showed 79% of
its inputs "drifting" while it still beat the benchmark on every series (p < 0.001): variance levels move
between regimes by nature, so a drifted input is not a degraded model. Drift is reported as a warning; only
measured out-of-sample accuracy recommends a retrain.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from volops import data
from volops import evaluate as E
from volops import model as M

WINDOW = 250
DRIFT_WINDOW = 120
P_THRESHOLD = 0.01
MIN_EFFECT = 0.25
MIN_OUT_OF_SAMPLE = 20  # fewer scored origins than this is not evidence either way
RETRAIN_SHARE = 0.5
DRIFT_WARN_SHARE = 0.5


def drift_for(series: M.SeriesModel, r: pd.Series, window: int = DRIFT_WINDOW) -> dict[str, dict]:
    current = np.log10(np.maximum(M.features(r).dropna().iloc[-window:], M.FLOOR))
    out = {}
    for f in M.FEATURES:
        res = stats.ks_2samp(np.asarray(series.reference[f]), current[f].to_numpy())
        out[f] = {"ks": float(res.statistic), "p": float(res.pvalue), "drifted": bool(res.pvalue < P_THRESHOLD and res.statistic > MIN_EFFECT)}
    return out


def run(bundle: M.ModelBundle, rates: pd.DataFrame, today: pd.Timestamp | None = None, window: int = WINDOW, drift_window: int = DRIFT_WINDOW) -> dict:
    today = today if today is not None else pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
    perf, drift_flags, judged, all_gains = {}, [], [], []
    seen_ccy: set[str] = set()
    for key, sm in bundle.series.items():
        r = data.log_returns(rates, sm.currency)
        origins = E.scoreable_origins(r, sm.horizon)[-window:]
        s = E.score(sm, r, origins)
        fresh = origins[origins > pd.Timestamp(sm.trained_through)]  # genuinely out-of-sample for this model
        oos = E.score(sm, r, fresh) if len(fresh) >= MIN_OUT_OF_SAMPLE else None
        perf[key] = {
            "basis": "out_of_sample" if oos else "overlaps_training", "n_out_of_sample": int(len(fresh)),
            "n": (oos or s).n, "qlike_gain": (oos or s).gain, "dm_p": (oos or s).dm_p,
            "qlike_model": (oos or s).qlike_model, "qlike_benchmark": (oos or s).qlike_benchmark,
        }
        all_gains.append(perf[key]["qlike_gain"])
        if oos:
            judged.append(oos.gain)
        if sm.currency not in seen_ccy:  # features do not depend on the horizon, so test each currency once
            seen_ccy.add(sm.currency)
            drift_flags += [(f"{sm.currency}/{f}", d) for f, d in drift_for(sm, r, drift_window).items()]
    drift_share = sum(d["drifted"] for _, d in drift_flags) / len(drift_flags)
    negative_share = (sum(g <= 0 for g in judged) / len(judged)) if judged else 0.0
    reasons, warnings = [], []
    if judged and negative_share > RETRAIN_SHARE:
        reasons.append(f"{negative_share:.0%} of series are not beating the benchmark on origins after the model's training date")
    if drift_share > DRIFT_WARN_SHARE:
        warnings.append(f"{drift_share:.0%} of input features drifted from the training reference (advisory: accuracy, not drift, decides retraining)")
    last = rates.index.max()
    return {
        "as_of": str(last.date()), "data_age_days": int((today - last).days), "window": window,
        "model": bundle.provenance, "performance": perf, "mean_qlike_gain": float(np.mean(all_gains)), "series_judged_out_of_sample": len(judged),
        "share_series_not_beating_benchmark": negative_share,
        "drift": {k: d for k, d in drift_flags}, "drift_share": drift_share,
        "retrain_recommended": bool(reasons), "reasons": reasons, "warnings": warnings,
    }


def write(report: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(report, indent=2), encoding="utf-8")
    tmp.replace(path)


def evidently_html(bundle: M.ModelBundle, rates: pd.DataFrame, path: Path, currency: str = "USD", window: int = DRIFT_WINDOW) -> bool:
    """Optional human-readable drift report (Evidently >= 0.7). Returns False, writing nothing, if it is unavailable."""
    try:
        from evidently import Report
        from evidently.presets import DataDriftPreset
    except ImportError:
        return False
    sm = bundle.get(currency, M.HORIZONS[0])
    r = data.log_returns(rates, currency)
    reference = pd.DataFrame({f: sm.reference[f] for f in M.FEATURES})
    current = np.log10(np.maximum(M.features(r).dropna().iloc[-window:], M.FLOOR)).reset_index(drop=True)
    snapshot = Report([DataDriftPreset()]).run(current, reference)
    path.parent.mkdir(parents=True, exist_ok=True)
    snapshot.save_html(str(path))
    return True
