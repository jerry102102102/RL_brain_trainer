"""Evaluate a trained dock-readiness classifier on a labeled dataset."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from ..training.policy_config import write_json
from .handoff_dataset import read_jsonl
from .handoff_features import feature_vector
from .readiness_model import load_readiness_model, predict_readiness_score


def evaluate_classifier(*, dataset_path: Path, model_path: Path, artifact_root: Path) -> dict[str, object]:
    model, normalizer, threshold, metadata = load_readiness_model(model_path)
    records = read_jsonl(dataset_path)
    scores = np.asarray(
        [predict_readiness_score(model=model, normalizer=normalizer, features=feature_vector(record)) for record in records],
        dtype=float,
    )
    labels = np.asarray([bool(record.get("dock_success_from_here", False)) for record in records], dtype=bool)
    pred = scores >= threshold
    grouped: dict[str, list[float]] = defaultdict(list)
    for record, score in zip(records, scores):
        grouped[str(record.get("position_error_bucket"))].append(float(score))
    summary = {
        "dataset_path": str(dataset_path),
        "model_path": str(model_path),
        "threshold": threshold,
        "metadata": metadata,
        "sample_count": len(records),
        "positive_rate": float(np.mean(labels)) if len(labels) else 0.0,
        "predicted_positive_rate": float(np.mean(pred)) if len(pred) else 0.0,
        "accuracy": float(np.mean(pred == labels)) if len(labels) else 0.0,
        "mean_score": float(np.mean(scores)) if len(scores) else 0.0,
        "mean_score_positive": float(np.mean(scores[labels])) if np.any(labels) else None,
        "mean_score_negative": float(np.mean(scores[~labels])) if np.any(~labels) else None,
        "score_by_position_bucket": {
            name: {"count": len(vals), "mean_score": float(np.mean(vals)) if vals else 0.0}
            for name, vals in sorted(grouped.items())
        },
    }
    scored_records = [dict(record, readiness_score=float(score), readiness_pred=bool(score >= threshold)) for record, score in zip(records, scores)]
    artifact_root.mkdir(parents=True, exist_ok=True)
    (artifact_root / "dock_readiness_scored_dataset.jsonl").write_text(
        "\n".join(json.dumps(record) for record in scored_records) + ("\n" if scored_records else "")
    )
    write_json(artifact_root / "dock_readiness_eval_summary.json", summary)
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a Phase 1C dock-readiness classifier.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--artifact-root", required=True)
    return parser


def main() -> None:  # pragma: no cover
    args = build_arg_parser().parse_args()
    summary = evaluate_classifier(dataset_path=Path(args.dataset), model_path=Path(args.model), artifact_root=Path(args.artifact_root))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()
