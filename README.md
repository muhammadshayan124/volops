# volops

The operational side of machine learning, built around one model that actually works.

The model is a small regression that forecasts **FX volatility** (the one thing the companion
[`fxcast`](../fxcast) backtest found to be reliably predictable; forecasting the exchange rate itself is
not). Everything around it is what a production service needs: experiment tracking, a model registry
with a promotion gate, an API, monitoring that catches both input drift and real accuracy loss,
dashboards and alerts, containers, Kubernetes manifests and CI/CD.

> **Scope, stated plainly.** This runs end to end on a laptop under Docker Compose (verified: MLflow container,
> training run promoted through the gate, API serving the champion, Prometheus scraping it with all 7 alert rules
> loading, Grafana dashboard provisioned), and the Kubernetes manifests are schema-validated. It has **not** been deployed to a cloud cluster, has no users, and
> carries no traffic. The point is a working, tested pipeline and the engineering judgement in it, not a
> claim of production scale.

## The lifecycle

```
ECB rates ──ingest──▶ rate store (CSV, atomic writes, never rewrites history)
                         │
              train ◀────┘            weekly CronJob
                │  fit on all data (production model) + fit on data before a 500-origin holdout (quality estimate)
                ▼
        MLflow run: params, per-series holdout metrics, data hash, git sha, model.json
                ▼
        registry version ──gate──▶ pass: alias `champion`      fail: alias `candidate` only (still auditable)
                                         │
                      API loads `champion` at startup ◀─────────┘      (rollout restart after a promotion)
                                         │
   monitor (daily): re-score the last 250 scoreable forecasts against realised variance + KS drift on inputs
                                         ▼
                 monitor.json ──▶ /metrics ──▶ Prometheus ──▶ Grafana dashboard + alert rules
```

## What the model is

HAR-style regression: next-`h`-day average daily variance on trailing mean squared returns over 1, 5,
22 and 66 days, one set of coefficients per currency and horizon (6 currencies x horizons 5 and 20).
It is evaluated with **QLIKE** loss against a constant-variance benchmark, with Diebold-Mariano tests
(overlap-aware, small-sample corrected). A model version is a small JSON document (coefficients, a
reference sample of the inputs for drift tests, provenance): no pickles, so it is inspectable and cannot
execute code when loaded.

Latest training run on the real ECB history (see `volops train`): the production model's honest
estimate is its **holdout** fit, trained only on data before the last 500 origins (about two years):

<!-- holdout:start -->
| series | holdout QLIKE gain | DM p-value |
|---|---|---|
| AUD h=20 | +0.186 | 0.001 |
| AUD h=5 | +0.308 | 0.000 |
| CHF h=20 | +0.039 | 0.000 |
| CHF h=5 | +0.055 | 0.000 |
| CNY h=20 | +0.285 | 0.000 |
| CNY h=5 | +0.308 | 0.000 |
| GBP h=20 | +0.379 | 0.000 |
| GBP h=5 | +0.461 | 0.000 |
| JPY h=20 | +0.230 | 0.000 |
| JPY h=5 | +0.245 | 0.000 |
| USD h=20 | +0.160 | 0.001 |
| USD h=5 | +0.189 | 0.000 |

Mean holdout gain +0.237; champion is version 1 (data hash ad7ab81132d98c3c, last fixing 2026-10-26). Positive = better than constant variance.
<!-- holdout:end -->

## The gate

A candidate is promoted to `champion` only if (1) at least 90% of series beat the benchmark on the
holdout, (2) the mean gain is positive, and (3) the mean gain has not fallen more than 0.05 below the
current champion's recorded gain. Be clear about (3): the champion's holdout covers a different period,
so it is a **regression guard** (quality did not collapse), not a head-to-head. A rejected candidate is
still registered under the alias `candidate` with the reasons attached to its run. Both outcomes are
covered by tests against a real (SQLite-backed) MLflow.

## Monitoring: two signals, because they fail differently

- **Performance (the one that decides).** A forecast's label arrives `h` days later, so the monitor
  re-scores the most recent origins whose future has now happened. The model is a fixed function of
  observable data, so recomputing those forecasts reproduces what the service said at the time. Only
  origins **after the model's training date** count as evidence (earlier ones are in its own training data
  and flatter it); with fewer than 20 of them the report says "overlaps_training" and abstains.
- **Input drift (advisory).** KS test of the last 120 days of inputs against the training reference, with an
  effect-size floor because variance features are heavy-tailed and autocorrelated. It is a heuristic, not a
  calibrated test, and it **does not trigger retraining** (see the replay below). An Evidently drift report is
  written alongside (`data/drift.html`).

<!-- monitor:start -->
**Replay on real data** (`scripts/replay.py`): a model fitted only on data through 2025-08-22, monitored on the ~250 trading days that followed (all genuinely out-of-sample for it). Result: it beats the benchmark on **12 of 12 series** (mean QLIKE gain +0.362, every DM p < 0.001), yet **79% of its input features (19 of 24) count as drifted**. My first design retrained on drift; this replay showed that would have fired on a perfectly healthy model, because volatility levels move between regimes by nature. So drift is now an advisory warning and only measured out-of-sample accuracy recommends a retrain. (A model that was *just* trained has no out-of-sample origins yet; the monitor reports that explicitly rather than quoting an in-sample number as skill.)
<!-- monitor:end -->

## Run it

```bash
pip install -e ".[train,dev]"
pytest                                   # 21 tests incl. MLflow registry/gate end to end and the API
export MLFLOW_TRACKING_URI=sqlite:///mlflow.db
volops ingest                            # top up the rate store from the ECB
volops train                             # fit, log, register, gate
volops monitor                           # rolling out-of-sample accuracy + drift -> data/monitor.json
python scripts/replay.py 300             # what the monitor says about a model trained 300 days ago
volops serve                             # http://127.0.0.1:8000/forecast/USD?horizon=5
```

```bash
docker compose up -d                                  # mlflow, api, prometheus, grafana (loopback only)
docker compose run --rm seed && docker compose run --rm train && docker compose run --rm monitor
docker compose restart api                            # pick up the new champion
# Grafana http://127.0.0.1:3000 (dashboard "volops"), Prometheus http://127.0.0.1:9090, MLflow http://127.0.0.1:5000
```

API: `GET /forecast/{currency}?horizon=5|20`, `GET /forecast?horizon=`, `/health` (liveness), `/ready`
(503 until a model and data are loaded), `/metrics` (Prometheus).

## Decisions worth knowing about

- **A test caught a real leak in my own first version.** The holdout fit used `train_end` to select
  rows but computed their labels from the full series, so the last `h` training rows had labels built from
  returns *after* the cut. `test_training_never_uses_unrealised_labels` failed; `fit_series` now truncates
  the series at `train_end` before building labels.
- **Two image targets.** The API image has only `mlflow-skinny` and runs as a non-root user; the
  training image carries MLflow, Evidently and the server. The serving path stays small.
- **The API never changes model underneath itself.** It loads the champion at startup; a promotion is
  followed by a rollout restart. No unauthenticated "reload" endpoint exists.
- **Store writes are atomic and history is immutable.** `ingest` writes a temp file and renames it, and
  refuses to merge if the source disagrees with stored history (ECB reference rates are not revised, so
  a changed past value is an alarm, not an update).
- **Kubernetes hygiene:** non-root, read-only root filesystem on the API, dropped capabilities, probes
  (startup/readiness/liveness), resource requests and limits, HPA, PodDisruptionBudget, NetworkPolicy,
  `concurrencyPolicy: Forbid` on every CronJob.
- **CI/CD:** CI runs lint and tests, validates manifests with kubeconform and the Prometheus rules with
  `promtool`, builds both images and smoke-tests that the API image starts and reports not-ready without a
  model. The release workflow (on a version tag) builds, vulnerability-scans, pushes to GHCR and renders
  manifests pinned to the release. It does not deploy.

## Limitations

- **Not deployed anywhere.** Manifests are schema-validated (`kubeconform`, 14 resources), not run on a cluster here.
  Compose was run for real; MLflow 3 needs `--allowed-hosts` for in-network Host headers (found by running it, now in both Compose and the Deployment).
- **No authentication** on the API, and the Compose Grafana password is a throwaway for a loopback-only stack.
- **SQLite-backed MLflow** (fine for one writer; the Deployment uses `Recreate` for that reason). A shared
  Postgres backend and object-store artifacts would be the next step for more than one trainer.
- **ReadWriteOnce volume** shared by the API and the jobs works on a single node; multi-node needs RWX
  storage or the store moved to object storage.
- **Retraining is scheduled, not automatic on a monitor alert.** The monitor raises the flag; a human or
  a follow-up automation decides.
- **The replay is one model on one history.** It shows the monitor behaving sensibly on real data; it is not a
  statistical study of how often accuracy decays.
- **The monitor re-scores with the current champion**, so it measures that model's skill on recent data,
  not the skill of earlier versions that actually served those dates.
- **The reference sample for drift is the last 400 training rows**, a choice, not a principled constant.
- **One model family and daily data.** No A/B serving, no shadow deployment, no feature store.

MIT licence. Data: European Central Bank euro foreign exchange reference rates.
