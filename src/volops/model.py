"""The model: a HAR-style regression of future average daily variance on trailing mean squared returns.

It is deliberately the simplest model that beat the constant-variance benchmark in the companion
backtest (the `fxcast` project): linear, four inputs, no tuning. The MLOps around it is the point.

Everything the service and the monitor need travels in one JSON document (coefficients, the
reference distribution of the inputs, provenance), so a model version is a small, inspectable,
non-executable artifact: no pickles.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

HORIZONS = (5, 20)
WINDOWS = (1, 5, 22, 66)
FEATURES = [f"rv_{w}" for w in WINDOWS]
SCALE = 1e4  # variances are ~1e-5; scaling keeps the regression well conditioned
FLOOR = 1e-10
ANNUALISE = 252


def features(r: pd.Series) -> pd.DataFrame:
    """Trailing mean squared returns over 1, 5, 22 and 66 days: causal, right-aligned windows."""
    sq = r**2
    return pd.DataFrame({f"rv_{w}": sq.rolling(w).mean() for w in WINDOWS})


def target(r: pd.Series, h: int) -> pd.Series:
    """mean(r_{t+1}^2 .. r_{t+h}^2), indexed by the origin t. NaN for origins whose future is not yet observed."""
    sq = r**2
    return sum(sq.shift(-k) for k in range(1, h + 1)) / h


def expanding_benchmark(r: pd.Series) -> pd.Series:
    """'Volatility is constant': the mean of all squared returns seen so far."""
    return (r**2).expanding().mean()


def qlike(forecast: np.ndarray, realised: np.ndarray) -> np.ndarray:
    """QLIKE loss (Patton 2011), robust to the noise in a squared-return variance proxy. Lower is better."""
    f = np.maximum(np.asarray(forecast, dtype=float), 1e-14)
    return np.log(f) + np.asarray(realised, dtype=float) / f


@dataclass
class SeriesModel:
    currency: str
    horizon: int
    intercept: float
    coef: list[float]
    trained_through: str
    n_train: int
    reference: dict[str, list[float]] = field(default_factory=dict)  # per feature: sorted sample of log10 values for drift tests

    def predict(self, x: pd.DataFrame | np.ndarray) -> np.ndarray:
        a = x[FEATURES].to_numpy() if isinstance(x, pd.DataFrame) else np.asarray(x)
        return np.maximum((self.intercept + a * SCALE @ np.asarray(self.coef)) / SCALE, FLOOR)


def fit_series(r: pd.Series, currency: str, h: int, train_end: pd.Timestamp | None = None, ref_size: int = 400) -> SeriesModel:
    """Fit on origins whose h-day label is fully realised by `train_end` (the purge: label of row s needs data up to s+h)."""
    train_end = train_end if train_end is not None else r.index[-1]
    r = r.loc[:train_end]  # nothing after train_end may be seen, not even to build the labels of earlier rows
    feats, y = features(r), target(r, h)
    ok = feats.notna().all(axis=1) & y.notna()  # y is NaN for the last h origins of the truncated series: that is the purge
    x, t = feats[ok].to_numpy() * SCALE, y[ok].to_numpy() * SCALE
    if len(t) < 500:
        raise ValueError(f"{currency}: need at least 500 training rows, have {len(t)}")
    design = np.column_stack([np.ones(len(x)), x])
    beta, *_ = np.linalg.lstsq(design, t, rcond=None)
    sample = np.log10(np.maximum(feats[ok].iloc[-ref_size:], FLOOR))
    return SeriesModel(currency, h, float(beta[0]), [float(b) for b in beta[1:]], str(feats[ok].index[-1].date()), int(ok.sum()),
                       {c: sorted(sample[c].round(4).tolist()) for c in FEATURES})


@dataclass
class ModelBundle:
    series: dict[str, SeriesModel]  # key "USD/5"
    provenance: dict

    @staticmethod
    def key(currency: str, h: int) -> str:
        return f"{currency}/{h}"

    def get(self, currency: str, h: int) -> SeriesModel:
        try:
            return self.series[self.key(currency, h)]
        except KeyError:
            raise KeyError(f"no model for {currency} at horizon {h}") from None

    def to_json(self) -> str:
        return json.dumps({"provenance": self.provenance, "series": {k: asdict(v) for k, v in self.series.items()}})

    @staticmethod
    def from_json(text: str) -> ModelBundle:
        d = json.loads(text)
        return ModelBundle({k: SeriesModel(**v) for k, v in d["series"].items()}, d["provenance"])
