from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from volops import data, monitor, registry, serve
from volops import evaluate as E
from volops import model as M


def synthetic_rates(n: int = 2600, seed: int = 0, shock_at: int | None = None) -> pd.DataFrame:
    """Volatility-clustered random walks (GARCH-like) so a variance model has something to find."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2012-01-02", periods=n)
    cols = {}
    for c in ("USD", "GBP"):
        h = np.empty(n)
        h[0] = 4e-5
        eps = rng.standard_normal(n)
        r = np.empty(n)
        r[0] = np.sqrt(h[0]) * eps[0]
        for t in range(1, n):
            h[t] = 1e-6 + 0.08 * r[t - 1] ** 2 + 0.9 * h[t - 1]
            r[t] = np.sqrt(h[t]) * eps[t]
        if shock_at is not None:
            r[shock_at:] *= 4.0  # a regime the model never saw
        cols[c] = 1.1 * np.exp(np.cumsum(r))
    return pd.DataFrame(cols, index=idx).rename_axis("date")


@pytest.fixture(scope="module")
def rates():
    return synthetic_rates()


@pytest.fixture(scope="module")
def bundle(rates):
    from volops import train

    series, _ = train.build(rates, ("USD", "GBP"), (5,), holdout=300)
    return M.ModelBundle(series, {"git_sha": "test"})


def test_features_are_causal(rates):
    r = data.log_returns(rates, "USD")
    f1 = M.features(r)
    r2 = r.copy()
    r2.iloc[1500:] = 9.0
    pd.testing.assert_frame_equal(f1.iloc[:1500], M.features(r2).iloc[:1500])


def test_target_is_nan_exactly_where_the_future_is_unobserved(rates):
    r = data.log_returns(rates, "USD")
    y = M.target(r, 5)
    assert y.iloc[-5:].isna().all() and y.iloc[:-5].notna().all()
    assert y.iloc[10] == pytest.approx((r.iloc[11 : 16] ** 2).mean())


def test_training_never_uses_unrealised_labels(rates):
    """Fitting with train_end=t must give the same model whether or not data after t exists."""
    r = data.log_returns(rates, "USD")
    cut = r.index[1800]
    a = M.fit_series(r.loc[:cut], "USD", 5, train_end=cut)
    b = M.fit_series(r, "USD", 5, train_end=cut)
    assert a.coef == pytest.approx(b.coef) and a.intercept == pytest.approx(b.intercept)


def test_model_beats_the_benchmark_on_clustered_data_and_serialises(bundle, rates):
    r = data.log_returns(rates, "USD")
    sm = bundle.get("USD", 5)
    score = E.score(sm, r, E.scoreable_origins(r, 5)[-300:])
    assert score.gain > 0
    again = M.ModelBundle.from_json(bundle.to_json())
    assert again.get("USD", 5).coef == sm.coef
    with pytest.raises(KeyError):
        bundle.get("JPY", 5)


def test_qlike_minimised_at_truth_and_dm_calibration():
    grid = np.linspace(0.5, 4, 200)
    assert grid[np.argmin([M.qlike(np.array([g]), np.array([2.0]))[0] for g in grid])] == pytest.approx(2.0, abs=0.03)
    rng = np.random.default_rng(0)
    h, n, rejections = 5, 800, 0
    for _ in range(200):
        a = np.convolve(rng.normal(0, 1, n + h), np.ones(h), mode="valid")[:n] ** 2
        b = np.convolve(rng.normal(0, 1, n + h), np.ones(h), mode="valid")[:n] ** 2
        rejections += E.dm_test(a, b, h) < 0.05
    assert rejections / 200 < 0.10


@pytest.mark.parametrize(
    ("cand", "champ", "ok"),
    [
        ({"a": 0.2, "b": 0.1}, None, True),
        ({"a": 0.2, "b": -0.1}, None, False),  # only half the series beat the benchmark
        ({"a": 0.01, "b": 0.01}, {"a": 0.5, "b": 0.5}, False),  # regression versus champion
        ({"a": 0.45, "b": 0.5}, {"a": 0.5, "b": 0.5}, True),  # within tolerance
        ({}, None, False),
    ],
)
def test_gate(cand, champ, ok):
    d = registry.gate(cand, champ)
    assert d.promote is ok and d.reasons


def test_monitor_quiet_on_stationary_data_and_warns_on_a_regime_shift(bundle, rates):
    today = rates.index[-1] + pd.Timedelta(days=1)
    calm = monitor.run(bundle, rates, today=today, window=200)
    assert calm["data_age_days"] == 1 and calm["drift_share"] <= 0.5 and not calm["retrain_recommended"] and not calm["warnings"]
    shocked = synthetic_rates(shock_at=len(rates) - 150)
    hot = monitor.run(bundle, shocked, today=today, window=200)
    assert hot["drift_share"] > calm["drift_share"]
    assert any("drifted" in w for w in hot["warnings"])  # flagged...
    assert not hot["retrain_recommended"]  # ...but drift alone is advisory: accuracy decides


def test_monitor_does_not_count_in_sample_origins_as_evidence(bundle, rates):
    rep = monitor.run(bundle, rates, today=rates.index[-1], window=200)
    assert all(p["basis"] == "overlaps_training" and p["n_out_of_sample"] == 0 for p in rep["performance"].values())
    assert rep["series_judged_out_of_sample"] == 0 and not any("benchmark" in r for r in rep["reasons"])


def test_monitor_judges_out_of_sample_and_flags_a_broken_model(rates):

    cut = rates.index[-400]
    series = {}
    for c in ("USD", "GBP"):
        series[M.ModelBundle.key(c, 5)] = M.fit_series(data.log_returns(rates, c), c, 5, train_end=cut)
    good = M.ModelBundle(series, {})
    rep = monitor.run(good, rates, today=rates.index[-1], window=300)
    assert all(p["basis"] == "out_of_sample" and p["n_out_of_sample"] > 200 for p in rep["performance"].values())
    assert rep["series_judged_out_of_sample"] == 2

    broken = M.ModelBundle({k: M.SeriesModel(v.currency, v.horizon, -50.0, [0.0] * 4, v.trained_through, v.n_train, v.reference)
                            for k, v in series.items()}, {})  # forecasts the variance floor forever
    bad = monitor.run(broken, rates, today=rates.index[-1], window=300)
    assert bad["share_series_not_beating_benchmark"] == 1.0 and bad["retrain_recommended"]
    assert any("not beating the benchmark" in r for r in bad["reasons"])


def test_merge_appends_only_new_rows_and_refuses_changed_history():
    old = pd.DataFrame({"USD": [1.1, 1.2]}, index=pd.to_datetime(["2024-01-02", "2024-01-03"]))
    new = pd.DataFrame({"USD": [1.2, 1.3]}, index=pd.to_datetime(["2024-01-03", "2024-01-04"]))
    assert data.merge(old, new)["USD"].tolist() == [1.1, 1.2, 1.3]
    with pytest.raises(ValueError):
        data.merge(old, pd.DataFrame({"USD": [1.5, 1.3]}, index=new.index))


def test_load_rates_rejects_non_positive(tmp_path):
    p = tmp_path / "r.csv"
    p.write_text("date,USD\n2024-01-02,1.1\n2024-01-03,0\n", encoding="utf-8")
    with pytest.raises(ValueError):
        data.load_rates(p)


@pytest.fixture()
def client(tmp_path, bundle, rates):
    store = tmp_path / "rates.csv"
    rates.to_csv(store)
    model = tmp_path / "model.json"
    model.write_text(bundle.to_json(), encoding="utf-8")
    mon = tmp_path / "monitor.json"
    monitor.write(monitor.run(bundle, rates, today=rates.index[-1]), mon)
    with TestClient(serve.create_app(store, mon, model)) as c:
        yield c


def test_forecast_endpoint(client):
    body = client.get("/forecast/usd?horizon=5").json()
    assert body["currency"] == "USD" and body["horizon_days"] == 5
    assert body["annualised_vol_pct"] > 0 and body["model_version"].startswith("file:")
    assert client.get("/forecast/JPY").status_code == 404
    assert client.get("/forecast/USD?horizon=3").status_code == 404
    assert len(client.get("/forecast").json()) == 2


def test_ready_and_not_ready(tmp_path, client):
    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/ready").json()["ready"] is True
    with TestClient(serve.create_app(tmp_path / "none.csv", tmp_path / "none.json", tmp_path / "none-model.json")) as c:
        assert c.get("/health").status_code == 200  # alive...
        assert c.get("/ready").status_code == 503  # ...but not ready, and says so
        assert c.get("/forecast/USD").status_code == 503


def test_metrics_exposes_forecast_monitor_and_request_series(client):
    client.get("/forecast/USD")
    text = client.get("/metrics").text
    for name in ("volops_requests_total", "volops_request_seconds_bucket", "volops_forecast_daily_vol_pct", "volops_data_age_days",
                 "volops_model_info", "volops_monitor_drift_share", "volops_monitor_retrain_recommended", "volops_monitor_qlike_gain"):
        assert name in text, name
    assert 'route="/forecast/{currency}"' in text


def test_service_rereads_the_store_when_it_changes(tmp_path, bundle, rates):
    store = tmp_path / "rates.csv"
    rates.iloc[:-5].to_csv(store)
    model = tmp_path / "model.json"
    model.write_text(bundle.to_json(), encoding="utf-8")
    with TestClient(serve.create_app(store, tmp_path / "m.json", model)) as c:
        first = c.get("/forecast/USD").json()["as_of"]
        import os
        import time

        time.sleep(0.05)
        rates.to_csv(store)
        os.utime(store, (time.time() + 5, time.time() + 5))
        assert c.get("/forecast/USD").json()["as_of"] > first


def _mlflow_env(tmp_path, monkeypatch):
    mlflow = pytest.importorskip("mlflow")
    uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    mlflow.set_tracking_uri(uri)


def test_mlflow_end_to_end_train_register_gate_load(tmp_path, monkeypatch, rates):
    from volops import train

    _mlflow_env(tmp_path, monkeypatch)
    store = tmp_path / "rates.csv"
    rates.to_csv(store)
    res = train.run(store, experiment="t", holdout=300, currencies=("USD",), horizons=(5,))
    assert res["promoted"] and res["version"] == "1"
    bundle, version = registry.load_bundle()
    assert version == "har-vol@champion v1" and bundle.get("USD", 5).currency == "USD"
    assert bundle.provenance["data_sha256_16"] == train.data_fingerprint(store)
    assert registry.champion_metrics() == pytest.approx(res["gains"])


def test_a_candidate_that_fails_the_gate_is_registered_but_not_promoted(tmp_path, monkeypatch, rates):
    """On the noisy GBP series the model does not beat the benchmark on the holdout, so there is no champion."""
    from mlflow import MlflowClient
    from mlflow.exceptions import MlflowException

    from volops import train

    _mlflow_env(tmp_path, monkeypatch)
    store = tmp_path / "rates.csv"
    rates.to_csv(store)
    res = train.run(store, experiment="t", holdout=300, currencies=("GBP",), horizons=(5,))
    assert not res["promoted"] and "benchmark" in res["reasons"][0]
    client = MlflowClient()
    assert str(client.get_model_version_by_alias(registry.MODEL_NAME, "candidate").version) == "1"
    with pytest.raises(MlflowException):
        client.get_model_version_by_alias(registry.MODEL_NAME, registry.CHAMPION)
    assert registry.champion_metrics() is None
