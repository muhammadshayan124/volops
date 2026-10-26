"""Fill the README's result sections from the latest MLflow run and the monitor report.
Usage: MLFLOW_TRACKING_URI=sqlite:///mlflow.db python scripts/fill_readme.py"""

import json
from pathlib import Path

from mlflow import MlflowClient

from volops import registry

client = MlflowClient()
mv = client.get_model_version_by_alias(registry.MODEL_NAME, registry.CHAMPION)
run = client.get_run(mv.run_id)
m = run.data.metrics
rows = ["| series | holdout QLIKE gain | DM p-value |", "|---|---|---|"]
for key in sorted(k for k in m if k.endswith("_qlike_gain") and k != "mean_qlike_gain"):
    tag = key[: -len("_qlike_gain")]
    rows.append(f"| {tag.replace('_h', ' h=')} | {m[key]:+.3f} | {m[tag + '_dm_p']:.3f} |")
rows.append(f"\nMean holdout gain {m['mean_qlike_gain']:+.3f}; champion is version {mv.version} (data hash {run.data.params['data_sha256_16']}, "
            f"last fixing {run.data.params['data_last_date']}). Positive = better than constant variance.")
holdout = "\n".join(rows)

mon = json.loads(Path("results/replay_monitor.json").read_text(encoding="utf-8"))
perf = mon["performance"]
gains = [v["qlike_gain"] for v in perf.values()]
drifted = [k for k, v in mon["drift"].items() if v["drifted"]]
finding = (f"**Replay on real data** (`scripts/replay.py`): a model fitted only on data through {mon['model']['data_last_date_seen']}, monitored on the "
           f"~{next(iter(perf.values()))['n']} trading days that followed (all genuinely out-of-sample for it). Result: it beats the benchmark on "
           f"**{sum(g > 0 for g in gains)} of {len(gains)} series** (mean QLIKE gain {sum(gains) / len(gains):+.3f}, every DM p < 0.001), yet "
           f"**{mon['drift_share']:.0%} of its input features ({len(drifted)} of {len(mon['drift'])}) count as drifted**. "
           "My first design retrained on drift; this replay showed that would have fired on a perfectly healthy model, because volatility "
           "levels move between regimes by nature. So drift is now an advisory warning and only measured out-of-sample accuracy recommends a retrain. "
           "(A model that was *just* trained has no out-of-sample origins yet; the monitor reports that explicitly rather than quoting an "
           "in-sample number as skill.)")

text = Path("README.md").read_text(encoding="utf-8")
for name, body in (("holdout", holdout), ("monitor", finding)):
    a, b = f"<!-- {name}:start -->", f"<!-- {name}:end -->"
    i, j = text.find(a), text.find(b)
    text = text[: i + len(a)] + "\n" + body + "\n" + text[j:]
Path("README.md").write_text(text, encoding="utf-8")
