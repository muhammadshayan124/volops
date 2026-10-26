"""Replay: what would the monitor have said about a model trained N trading days ago?

Fits the production model on data up to N rows before the end of the store (no later data is visible to
the fit), writes it to results/replay_model.json, and runs the monitor on the FULL store. Every scored
origin is then genuinely out-of-sample for that model. Usage:
    python scripts/replay.py [N=300]
    volops monitor --model results/replay_model.json --out results/replay_monitor.json --html results/replay_drift.html
"""

import sys
from pathlib import Path

from volops import data, monitor, train

n = int(sys.argv[1]) if len(sys.argv) > 1 else 300
rates = data.load_rates()
past = rates.iloc[:-n]
series, _ = train.build(past)
bundle = train.M.ModelBundle(series, {"replay_trained_through_row": len(past), "data_last_date_seen": str(past.index[-1].date())})
Path("results").mkdir(exist_ok=True)
Path("results/replay_model.json").write_text(bundle.to_json(), encoding="utf-8")
report = monitor.run(bundle, rates)
monitor.write(report, Path("results/replay_monitor.json"))
print(f"model saw data through {past.index[-1].date()}; scored {n} trading days later")
for k, v in report["performance"].items():
    print(f"  {k}: basis {v['basis']}, n={v['n']}, QLIKE gain {v['qlike_gain']:+.3f}, DM p={v['dm_p']:.3f}")
print(f"drift share {report['drift_share']:.0%}; retrain recommended: {report['retrain_recommended']} {report['reasons']}")
