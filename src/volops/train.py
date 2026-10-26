"""Training run: fit, evaluate on a holdout, log everything to MLflow, register, gate, maybe promote.

Two fits per series, on purpose:
  * a *holdout* fit trained only on data before the last HOLDOUT origins, scored on those origins.
    This is the honest quality estimate and it is what the gate looks at.
  * the *production* fit trained on everything available. It is what gets registered and served.
The production fit is the holdout procedure applied to more data; its own quality can only be known later,
which is exactly what the monitor measures once labels arrive.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path

import pandas as pd

from volops import data, registry
from volops import evaluate as E
from volops import model as M

HOLDOUT = 500  # origins (about two years of trading days)


def data_fingerprint(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def _git_sha() -> str:
    if os.environ.get("GIT_SHA"):
        return os.environ["GIT_SHA"]
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True, timeout=5).stdout.strip()
    except Exception:  # noqa: BLE001 - provenance is best-effort
        return "unknown"


def build(rates: pd.DataFrame, currencies=data.CURRENCIES, horizons=M.HORIZONS, holdout: int = HOLDOUT):
    """Fit both versions of every series. Returns (production bundle, holdout scores {key: Score})."""
    series, scores = {}, {}
    for ccy in currencies:
        r = data.log_returns(rates, ccy)
        for h in horizons:
            origins = E.scoreable_origins(r, h)
            if len(origins) < holdout + 600:
                raise ValueError(f"{ccy}/{h}: not enough history for a {holdout}-origin holdout")
            test_origins = origins[-holdout:]
            cutoff = origins[-holdout - 1]  # last origin the holdout fit may use; its label is realised before the holdout starts
            holdout_fit = M.fit_series(r, ccy, h, train_end=cutoff)
            scores[M.ModelBundle.key(ccy, h)] = E.score(holdout_fit, r, test_origins)
            series[M.ModelBundle.key(ccy, h)] = M.fit_series(r, ccy, h)
    return series, scores


def run(store: Path = data.DEFAULT_STORE, experiment: str = "volops", holdout: int = HOLDOUT, currencies=data.CURRENCIES, horizons=M.HORIZONS) -> dict:
    import mlflow
    from mlflow import MlflowClient

    rates = data.load_rates(store)
    series, scores = build(rates, currencies, horizons, holdout)
    gains = {k: s.gain for k, s in scores.items()}
    champion = registry.champion_metrics()
    decision = registry.gate(gains, champion)

    mlflow.set_experiment(experiment)
    with mlflow.start_run() as run:
        provenance = {"data_sha256_16": data_fingerprint(store), "data_last_date": str(rates.index[-1].date()), "git_sha": _git_sha(),
                      "holdout_origins": holdout, "horizons": list(horizons), "currencies": list(currencies)}
        mlflow.log_params({k: str(v) for k, v in provenance.items()})
        for k, s in scores.items():
            tag = k.replace("/", "_h")
            mlflow.log_metrics({f"{tag}_qlike_gain": s.gain, f"{tag}_dm_p": s.dm_p, f"{tag}_qlike_model": s.qlike_model})
        mlflow.log_metric("mean_qlike_gain", sum(gains.values()) / len(gains))
        mlflow.set_tags({"holdout_gains": json.dumps(gains), "gate_promote": str(decision.promote), "gate_reasons": "; ".join(decision.reasons)})
        bundle = M.ModelBundle(series, provenance)
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / registry.ARTIFACT).write_text(bundle.to_json(), encoding="utf-8")
            mlflow.log_artifact(str(Path(tmp) / registry.ARTIFACT), artifact_path="model")
        client = MlflowClient()
        try:
            client.create_registered_model(registry.MODEL_NAME)
        except mlflow.exceptions.MlflowException:
            pass  # already exists
        mv = client.create_model_version(registry.MODEL_NAME, source=f"{run.info.artifact_uri}/model", run_id=run.info.run_id)
        client.set_registered_model_alias(registry.MODEL_NAME, "candidate", mv.version)
        if decision.promote:
            client.set_registered_model_alias(registry.MODEL_NAME, registry.CHAMPION, mv.version)
        return {"run_id": run.info.run_id, "version": str(mv.version), "promoted": decision.promote, "reasons": decision.reasons,
                "mean_gain": sum(gains.values()) / len(gains), "gains": gains}
