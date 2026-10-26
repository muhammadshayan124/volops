from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from volops import data, monitor, registry

app = typer.Typer(help="FX volatility forecasting: ingest, train, monitor, serve.", no_args_is_help=True)
console = Console(highlight=False)


@app.command("ingest")
def ingest_cmd(store: Annotated[Path, typer.Option(help="Rate store CSV")] = data.DEFAULT_STORE) -> None:
    """Append new ECB fixings to the store."""
    added, last = data.ingest(store)
    console.print(f"added {added} rows; last fixing {last.date()}")


@app.command("train")
def train_cmd(
    store: Annotated[Path, typer.Option(help="Rate store CSV")] = data.DEFAULT_STORE,
    holdout: int = 500,
) -> None:
    """Train, log to MLflow (MLFLOW_TRACKING_URI), register and gate."""
    from volops import train

    result = train.run(store, holdout=holdout)
    verdict = "PROMOTED to champion" if result["promoted"] else "registered as candidate only"
    console.print(f"run {result['run_id']} version {result['version']}: mean holdout QLIKE gain {result['mean_gain']:+.4f}; {verdict}")
    for r in result["reasons"]:
        console.print(f"  - {r}")


@app.command("monitor")
def monitor_cmd(
    store: Annotated[Path, typer.Option(help="Rate store CSV")] = data.DEFAULT_STORE,
    out: Annotated[Path, typer.Option(help="Where to write the report JSON")] = Path("data/monitor.json"),
    html: Annotated[Path, typer.Option(help="Optional Evidently drift report")] = Path("data/drift.html"),
    model: Annotated[Path, typer.Option(help="Model JSON (default: champion from the registry)")] = None,
) -> None:
    """Score recent forecasts against realised outcomes and check input drift."""
    bundle, version = registry.load_bundle(model)
    rates = data.load_rates(store)
    report = monitor.run(bundle, rates)
    monitor.write(report, out)
    wrote_html = monitor.evidently_html(bundle, rates, html)
    console.print(f"model {version}; mean rolling QLIKE gain {report['mean_qlike_gain']:+.4f} "
                  f"({report['series_judged_out_of_sample']}/{len(report['performance'])} series have enough out-of-sample origins to count); "
                  f"drift share {report['drift_share']:.0%}; "
                  f"retrain recommended: {report['retrain_recommended']}" + (f"; drift report {html}" if wrote_html else ""))
    for r in report["reasons"]:
        console.print(f"  - {r}")


@app.command("forecast")
def forecast_cmd(
    currency: str,
    horizon: int = 5,
    store: Annotated[Path, typer.Option(help="Rate store CSV")] = data.DEFAULT_STORE,
    model: Annotated[Path, typer.Option(help="Model JSON (default: champion from the registry)")] = None,
) -> None:
    """Print one forecast without starting the API."""
    from volops import serve

    state = serve.State(store, Path("data/monitor.json"), model)
    state.load_model()
    state.refresh()
    console.print(json.dumps(serve.forecast_one(state, currency.upper(), horizon), indent=2))


@app.command("serve")
def serve_cmd(host: str = "127.0.0.1", port: int = 8000) -> None:
    """Run the API."""
    import uvicorn

    uvicorn.run("volops.serve:app_factory", host=host, port=port, factory=True)


if __name__ == "__main__":
    app()
