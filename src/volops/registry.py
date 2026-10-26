"""Promotion gate and model loading.

Gate. A candidate becomes the champion only if it clears absolute checks on its holdout and has not
regressed against the champion's recorded holdout quality. Be clear about what this is: the champion's
holdout covers a different period, so the second check is a regression guard (quality did not collapse),
not a head-to-head. A rejected candidate is still registered and tagged, so the decision is auditable.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from volops.model import ModelBundle

MODEL_NAME = "har-vol"
CHAMPION = "champion"
ARTIFACT = "model.json"


@dataclass(frozen=True)
class GateDecision:
    promote: bool
    reasons: list[str]


def gate(candidate: dict[str, float], champion: dict[str, float] | None, min_share_positive: float = 0.9, tolerance: float = 0.05) -> GateDecision:
    """`candidate` / `champion` map series key -> holdout QLIKE gain over the benchmark (positive is better)."""
    reasons = []
    if not candidate:
        return GateDecision(False, ["no holdout metrics"])
    share = sum(g > 0 for g in candidate.values()) / len(candidate)
    if share < min_share_positive:
        reasons.append(f"only {share:.0%} of series beat the benchmark on the holdout (need {min_share_positive:.0%})")
    mean = sum(candidate.values()) / len(candidate)
    if mean <= 0:
        reasons.append(f"mean holdout gain {mean:+.4f} is not positive")
    if champion:
        cmean = sum(champion.values()) / len(champion)
        if mean < cmean - tolerance:
            reasons.append(f"mean holdout gain {mean:+.4f} is more than {tolerance} below the champion's {cmean:+.4f}")
    return GateDecision(not reasons, reasons or ["passed all checks"])


def load_bundle(path: Path | None = None) -> tuple[ModelBundle, str]:
    """Load the serving model. A local file wins (used in tests and in a plain container);
    otherwise the `champion` alias is resolved through the MLflow registry. Returns (bundle, version label)."""
    local = path or (Path(os.environ["VOLOPS_MODEL_PATH"]) if "VOLOPS_MODEL_PATH" in os.environ else None)
    if local:
        return ModelBundle.from_json(Path(local).read_text(encoding="utf-8")), f"file:{Path(local).name}"
    import mlflow
    from mlflow import MlflowClient

    client = MlflowClient()
    mv = client.get_model_version_by_alias(MODEL_NAME, CHAMPION)
    local_dir = mlflow.artifacts.download_artifacts(artifact_uri=mv.source)
    text = (Path(local_dir) / ARTIFACT).read_text(encoding="utf-8") if (Path(local_dir) / ARTIFACT).exists() else \
        next(Path(local_dir).rglob(ARTIFACT)).read_text(encoding="utf-8")
    return ModelBundle.from_json(text), f"{MODEL_NAME}@{CHAMPION} v{mv.version}"


def champion_metrics() -> dict[str, float] | None:
    """Holdout gains recorded on the current champion's run, or None if there is no champion yet."""
    from mlflow import MlflowClient
    from mlflow.exceptions import MlflowException

    client = MlflowClient()
    try:
        mv = client.get_model_version_by_alias(MODEL_NAME, CHAMPION)
    except MlflowException:
        return None
    tag = client.get_run(mv.run_id).data.tags.get("holdout_gains")
    return json.loads(tag) if tag else None
