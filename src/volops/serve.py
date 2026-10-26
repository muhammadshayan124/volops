"""The forecast service.

Stateless apart from two files it re-reads when they change: the rate store (written by the ingest
job) and the monitor report (written by the monitor job). Model changes are picked up by restarting
the pods (a Kubernetes rollout), so a model version never changes underneath a running process.
"""

from __future__ import annotations

import json
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, Counter, Gauge, Histogram, generate_latest

from volops import data, registry
from volops import model as M


class State:
    """Everything the endpoints read, refreshed when the underlying files change on disk."""

    def __init__(self, store: Path, monitor_path: Path, model_path: Path | None):
        self.store, self.monitor_path, self.model_path = store, monitor_path, model_path
        self.bundle: M.ModelBundle | None = None
        self.version = "none"
        self.rates: pd.DataFrame | None = None
        self._rates_mtime = 0.0
        self.monitor: dict | None = None
        self._monitor_mtime = 0.0

    def load_model(self) -> None:
        self.bundle, self.version = registry.load_bundle(self.model_path)

    def refresh(self) -> None:
        if self.store.exists() and (m := self.store.stat().st_mtime) != self._rates_mtime:
            self.rates, self._rates_mtime = data.load_rates(self.store), m
        if self.monitor_path.exists() and (m := self.monitor_path.stat().st_mtime) != self._monitor_mtime:
            self.monitor, self._monitor_mtime = json.loads(self.monitor_path.read_text(encoding="utf-8")), m

    @property
    def ready(self) -> bool:
        return self.bundle is not None and self.rates is not None


def forecast_one(state: State, currency: str, horizon: int) -> dict:
    assert state.bundle is not None and state.rates is not None
    try:
        sm = state.bundle.get(currency, horizon)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e)) from None
    r = data.log_returns(state.rates, currency)
    x = M.features(r).iloc[[-1]]
    if x.isna().any(axis=None):
        raise HTTPException(status_code=503, detail="not enough history to compute features")
    var = float(sm.predict(x)[0])
    bench = float(M.expanding_benchmark(r).iloc[-1])
    return {
        "currency": currency, "horizon_days": horizon, "as_of": str(r.index[-1].date()),
        "daily_vol_pct": round(float(np.sqrt(var)) * 100, 4), "annualised_vol_pct": round(float(np.sqrt(var * M.ANNUALISE)) * 100, 3),
        "benchmark_annualised_vol_pct": round(float(np.sqrt(bench * M.ANNUALISE)) * 100, 3),
        "model_version": state.version, "trained_through": sm.trained_through,
    }


def create_app(store: Path | None = None, monitor_path: Path | None = None, model_path: Path | None = None) -> FastAPI:
    state = State(store or Path(os.environ.get("VOLOPS_STORE", "data/rates.csv")),
                  monitor_path or Path(os.environ.get("VOLOPS_MONITOR", "data/monitor.json")), model_path)
    reg = CollectorRegistry()
    requests = Counter("volops_requests_total", "HTTP requests", ["route", "status"], registry=reg)
    latency = Histogram("volops_request_seconds", "Request latency", ["route"], registry=reg, buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5))
    forecast_g = Gauge("volops_forecast_daily_vol_pct", "Latest forecast daily volatility (%)", ["currency", "horizon"], registry=reg)
    age_g = Gauge("volops_data_age_days", "Days since the last stored fixing", registry=reg)
    info_g = Gauge("volops_model_info", "Loaded model (value is always 1)", ["version", "trained_through"], registry=reg)
    drift_g = Gauge("volops_monitor_drift_share", "Share of input features flagged as drifted by the last monitor run", registry=reg)
    retrain_g = Gauge("volops_monitor_retrain_recommended", "1 if the last monitor run recommends retraining", registry=reg)
    gain_g = Gauge("volops_monitor_qlike_gain", "Rolling QLIKE gain over the benchmark (positive is better)", ["series"], registry=reg)
    oos_g = Gauge("volops_monitor_out_of_sample_origins", "Scored origins after the model's training date (the gain above is only evidence when this is at least 20)", ["series"], registry=reg)
    monitor_age_g = Gauge("volops_monitor_age_seconds", "Seconds since the monitor report was written", registry=reg)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        try:
            state.load_model()
        except Exception as exc:  # noqa: BLE001 - stay up (not ready) so the failure is visible on /ready and in logs
            print(f"model load failed: {exc!r}", flush=True)
        state.refresh()
        yield

    app = FastAPI(title="volops", version="0.1.0", lifespan=lifespan)

    @app.middleware("http")
    async def _instrument(request: Request, call_next):
        t0 = time.perf_counter()
        response = await call_next(request)
        path = getattr(request.scope.get("route"), "path", "unmatched")  # set by the router while the request ran
        latency.labels(path).observe(time.perf_counter() - t0)
        requests.labels(path, str(response.status_code)).inc()
        return response

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.get("/ready")
    def ready(response: Response) -> dict:
        state.refresh()
        if not state.ready:
            response.status_code = 503
        return {"ready": state.ready, "model": state.version, "has_data": state.rates is not None}

    @app.get("/forecast/{currency}")
    def forecast(currency: str, horizon: int = 5) -> dict:
        state.refresh()
        if not state.ready:
            raise HTTPException(status_code=503, detail="service not ready")
        out = forecast_one(state, currency.upper(), horizon)
        forecast_g.labels(out["currency"], str(horizon)).set(out["daily_vol_pct"])
        return out

    @app.get("/forecast")
    def forecast_all(horizon: int = 5) -> list[dict]:
        state.refresh()
        if not state.ready:
            raise HTTPException(status_code=503, detail="service not ready")
        out = [forecast_one(state, c, horizon) for c in sorted({s.currency for s in state.bundle.series.values()})]
        for o in out:
            forecast_g.labels(o["currency"], str(horizon)).set(o["daily_vol_pct"])
        return out

    @app.get("/metrics")
    def metrics() -> Response:
        state.refresh()
        if state.rates is not None:
            age_g.set((pd.Timestamp.now(tz="UTC").tz_localize(None).normalize() - state.rates.index.max()).days)
        if state.bundle is not None:
            sm = next(iter(state.bundle.series.values()))
            info_g.labels(state.version, sm.trained_through).set(1)
        if state.monitor:
            drift_g.set(state.monitor["drift_share"])
            retrain_g.set(1 if state.monitor["retrain_recommended"] else 0)
            for k, v in state.monitor["performance"].items():
                gain_g.labels(k).set(v["qlike_gain"])
                oos_g.labels(k).set(v["n_out_of_sample"])
            monitor_age_g.set(max(0.0, time.time() - state._monitor_mtime))
        return Response(generate_latest(reg), media_type=CONTENT_TYPE_LATEST)

    return app


def app_factory() -> FastAPI:  # uvicorn --factory volops.serve:app_factory
    return create_app()
