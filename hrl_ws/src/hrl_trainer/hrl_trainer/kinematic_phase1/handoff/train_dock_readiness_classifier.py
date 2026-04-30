"""Train a dock-readiness classifier from labeled handoff states."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ..training.policy_config import deep_merge, load_yaml_file, write_json
from .handoff_dataset import read_jsonl
from .handoff_features import feature_vector
from .readiness_model import DockReadinessMLP, FeatureNormalizer, save_readiness_model


def readiness_default_config_path() -> Path:
    return Path(__file__).resolve().parents[1] / "configs" / "dock_readiness_default.yaml"


def _load_config(explicit_path: str | None) -> dict:
    cfg = load_yaml_file(readiness_default_config_path())
    if explicit_path:
        cfg = deep_merge(cfg, load_yaml_file(Path(explicit_path)))
    return cfg


def _binary_auc(y: np.ndarray, score: np.ndarray) -> float | None:
    positives = score[y == 1]
    negatives = score[y == 0]
    if len(positives) == 0 or len(negatives) == 0:
        return None
    comparisons = (positives[:, None] > negatives[None, :]).mean()
    ties = 0.5 * (positives[:, None] == negatives[None, :]).mean()
    return float(comparisons + ties)


def _pr_auc(y: np.ndarray, score: np.ndarray) -> float | None:
    if np.sum(y == 1) == 0:
        return None
    order = np.argsort(-score)
    y_sorted = y[order]
    tp = np.cumsum(y_sorted == 1)
    fp = np.cumsum(y_sorted == 0)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / max(np.sum(y == 1), 1)
    precision = np.concatenate([[1.0], precision])
    recall = np.concatenate([[0.0], recall])
    return float(np.trapz(precision, recall))


def _calibration_bins(y: np.ndarray, score: np.ndarray, bins: int = 10) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    edges = np.linspace(0.0, 1.0, bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (score >= lo) & (score < hi if hi < 1.0 else score <= hi)
        if not np.any(mask):
            rows.append({"bin_start": float(lo), "bin_end": float(hi), "count": 0, "mean_score": 0.0, "success_rate": 0.0})
            continue
        rows.append(
            {
                "bin_start": float(lo),
                "bin_end": float(hi),
                "count": int(np.sum(mask)),
                "mean_score": float(np.mean(score[mask])),
                "success_rate": float(np.mean(y[mask])),
            }
        )
    return rows


def _metrics(y: np.ndarray, score: np.ndarray, threshold: float) -> dict[str, object]:
    pred = score >= threshold
    labels = y.astype(bool)
    tp = int(np.sum(pred & labels))
    tn = int(np.sum(~pred & ~labels))
    fp = int(np.sum(pred & ~labels))
    fn = int(np.sum(~pred & labels))
    return {
        "roc_auc": _binary_auc(y, score),
        "pr_auc": _pr_auc(y, score),
        "threshold": float(threshold),
        "confusion_matrix": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
        "accuracy": float((tp + tn) / max(len(y), 1)),
        "precision": float(tp / max(tp + fp, 1)),
        "recall": float(tp / max(tp + fn, 1)),
        "positive_rate": float(np.mean(labels)) if len(labels) else 0.0,
        "mean_score": float(np.mean(score)) if len(score) else 0.0,
        "calibration_bins": _calibration_bins(y, score),
    }


def train_classifier(*, dataset_path: Path, artifact_root: Path, config: dict) -> dict[str, object]:
    records = read_jsonl(dataset_path)
    if not records:
        raise ValueError("Cannot train dock-readiness classifier on an empty dataset.")
    x = np.asarray([feature_vector(r) for r in records], dtype=np.float32)
    y = np.asarray([float(bool(r.get("dock_success_from_here", False))) for r in records], dtype=np.float32)
    cfg = config.get("readiness", {})
    seed = int(cfg.get("seed", 17))
    rng = np.random.default_rng(seed)
    indices = np.arange(len(records))
    rng.shuffle(indices)
    val_count = max(1, int(round(len(indices) * float(cfg.get("val_fraction", 0.2)))))
    val_idx = indices[:val_count]
    train_idx = indices[val_count:] if len(indices) > val_count else indices

    normalizer = FeatureNormalizer.fit(x[train_idx])
    x_norm = normalizer.transform(x)
    hidden_sizes = tuple(int(v) for v in cfg.get("hidden_sizes", [128, 64]))
    model = DockReadinessMLP(input_dim=x.shape[1], hidden_sizes=hidden_sizes)
    pos_count = float(np.sum(y[train_idx] == 1.0))
    neg_count = float(np.sum(y[train_idx] == 0.0))
    pos_weight_cfg = cfg.get("positive_weight", "auto")
    one_class_warning = pos_count == 0.0 or neg_count == 0.0
    pos_weight = neg_count / max(pos_count, 1.0) if pos_weight_cfg == "auto" and not one_class_warning else 1.0
    if pos_weight_cfg != "auto":
        pos_weight = float(pos_weight_cfg)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, dtype=torch.float32))
    opt = torch.optim.Adam(model.parameters(), lr=float(cfg.get("learning_rate", 1e-3)))
    loader = DataLoader(
        TensorDataset(torch.as_tensor(x_norm[train_idx]), torch.as_tensor(y[train_idx])),
        batch_size=int(cfg.get("batch_size", 64)),
        shuffle=True,
    )
    history: list[dict[str, float]] = []
    epochs = int(cfg.get("epochs", 80))
    for epoch in range(epochs):
        model.train()
        losses: list[float] = []
        for xb, yb in loader:
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu().item()))
        model.eval()
        with torch.no_grad():
            train_logits = model(torch.as_tensor(x_norm[train_idx], dtype=torch.float32))
            val_logits = model(torch.as_tensor(x_norm[val_idx], dtype=torch.float32))
            train_loss = float(loss_fn(train_logits, torch.as_tensor(y[train_idx])).cpu().item())
            val_loss = float(loss_fn(val_logits, torch.as_tensor(y[val_idx])).cpu().item())
        history.append({"epoch": epoch + 1, "batch_loss": float(np.mean(losses)) if losses else 0.0, "train_loss": train_loss, "val_loss": val_loss})

    threshold = float(cfg.get("threshold", 0.5))
    with torch.no_grad():
        train_score = torch.sigmoid(model(torch.as_tensor(x_norm[train_idx], dtype=torch.float32))).cpu().numpy()
        val_score = torch.sigmoid(model(torch.as_tensor(x_norm[val_idx], dtype=torch.float32))).cpu().numpy()
    model_path = artifact_root / "dock_readiness_model.pt"
    save_readiness_model(
        path=model_path,
        model=model,
        normalizer=normalizer,
        hidden_sizes=hidden_sizes,
        threshold=threshold,
        metadata={"dataset_path": str(dataset_path), "config": config},
    )
    summary = {
        "dataset_path": str(dataset_path),
        "model_path": str(model_path),
        "sample_count": len(records),
        "feature_dim": int(x.shape[1]),
        "train_count": int(len(train_idx)),
        "val_count": int(len(val_idx)),
        "positive_count": int(np.sum(y == 1.0)),
        "positive_rate": float(np.mean(y)),
        "pos_weight": float(pos_weight),
        "one_class_warning": bool(one_class_warning),
        "warning": "Dataset contains only one class; classifier is a gate baseline, not a discriminative readiness model."
        if one_class_warning
        else None,
        "history": history,
        "train_metrics": _metrics(y[train_idx], train_score, threshold),
        "val_metrics": _metrics(y[val_idx], val_score, threshold),
    }
    write_json(artifact_root / "dock_readiness_training_summary.json", summary)
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a Phase 1C dock-readiness classifier.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--config")
    return parser


def main() -> None:  # pragma: no cover
    args = build_arg_parser().parse_args()
    summary = train_classifier(dataset_path=Path(args.dataset), artifact_root=Path(args.artifact_root), config=_load_config(args.config))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()
