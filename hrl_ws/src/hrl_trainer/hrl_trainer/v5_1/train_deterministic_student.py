"""Train a deterministic student policy from extracted teacher datasets."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
import json
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .deterministic_student import (
    DeterministicStudentConfig,
    DeterministicStudentPolicy,
    load_student_checkpoint,
    save_student_checkpoint,
)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text:
            continue
        rows.append(json.loads(text))
    return rows


class WeightedTeacherDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows
        self.obs = torch.as_tensor(np.asarray([row["obs"] for row in rows], dtype=np.float32), dtype=torch.float32)
        self.target = torch.as_tensor(
            np.asarray([row["target_action_exec"] for row in rows], dtype=np.float32),
            dtype=torch.float32,
        )
        self.weight = torch.as_tensor(
            np.asarray([float(row.get("sample_weight", 1.0)) for row in rows], dtype=np.float32).reshape(-1, 1),
            dtype=torch.float32,
        )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        return self.obs[index], self.target[index], self.weight[index]


def _group_split(rows: list[dict[str, Any]], *, val_fraction: float, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if len(rows) <= 1 or float(val_fraction) <= 0.0:
        return list(rows), []
    grouped: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = f"{row.get('source_run_id', 'run')}::{row.get('episode_id', row.get('episode', 0))}"
        grouped[key].append(row)
    keys = list(grouped.keys())
    rng = np.random.default_rng(int(seed))
    rng.shuffle(keys)
    val_group_count = min(len(keys) - 1, max(1, int(round(len(keys) * float(val_fraction)))))
    val_keys = set(keys[:val_group_count])
    train_rows: list[dict[str, Any]] = []
    val_rows: list[dict[str, Any]] = []
    for key, grouped_rows in grouped.items():
        if key in val_keys:
            val_rows.extend(grouped_rows)
        else:
            train_rows.extend(grouped_rows)
    if not train_rows:
        train_rows, val_rows = val_rows, train_rows
    return train_rows, val_rows


def _split_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    tier_counts: Counter[str] = Counter()
    zone_counts: Counter[str] = Counter()
    tier_weight_sums: defaultdict[str, float] = defaultdict(float)
    zone_weight_sums: defaultdict[str, float] = defaultdict(float)
    for row in rows:
        tier = str(row.get("tier", "unknown"))
        zone = str(row.get("true_zone", "outside"))
        weight = float(row.get("sample_weight", 1.0))
        tier_counts[tier] += 1
        zone_counts[zone] += 1
        tier_weight_sums[tier] += weight
        zone_weight_sums[zone] += weight
    return {
        "count": int(len(rows)),
        "tier_counts": dict(tier_counts),
        "zone_counts": dict(zone_counts),
        "tier_weight_sums": {k: float(v) for k, v in tier_weight_sums.items()},
        "zone_weight_sums": {k: float(v) for k, v in zone_weight_sums.items()},
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_history(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _plot_training_curves(path: Path, history: list[dict[str, Any]]) -> str | None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return None

    if not history:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    epochs = [int(row["epoch"]) for row in history]
    train_weighted = [float(row["train_weighted_loss"]) for row in history]
    train_unweighted = [float(row["train_unweighted_loss"]) for row in history]
    val_weighted = [float(row["val_weighted_loss"]) for row in history]
    mean_action = [float(row["train_mean_action_l2"]) for row in history]
    target_action = [float(row["train_target_action_l2"]) for row in history]

    fig, axes = plt.subplots(2, 1, figsize=(9, 7))
    axes[0].plot(epochs, train_weighted, label="train_weighted_loss")
    axes[0].plot(epochs, train_unweighted, label="train_unweighted_loss")
    if any(v > 0.0 for v in val_weighted):
        axes[0].plot(epochs, val_weighted, label="val_weighted_loss")
    axes[0].set_title("Student Training Loss")
    axes[0].grid(alpha=0.2)
    axes[0].legend()

    axes[1].plot(epochs, mean_action, label="train_mean_action_l2")
    axes[1].plot(epochs, target_action, label="train_target_action_l2")
    axes[1].set_title("Student Action Magnitude")
    axes[1].grid(alpha=0.2)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)


def train_deterministic_student(
    *,
    dataset_path: Path,
    run_id: str,
    artifact_root: Path,
    seed: int = 0,
    hidden_dim: int = 128,
    epochs: int = 120,
    batch_size: int = 128,
    learning_rate: float = 3e-4,
    weight_decay: float = 1e-5,
    val_fraction: float = 0.15,
    early_stop_patience: int = 20,
    device: str = "cpu",
    resume_checkpoint: Path | None = None,
) -> dict[str, Any]:
    artifact_root = Path(artifact_root)
    artifact_root.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(int(seed))
    np.random.seed(int(seed))

    all_rows = _load_jsonl(Path(dataset_path))
    if not all_rows:
        raise ValueError(f"dataset is empty: {dataset_path}")

    train_rows, val_rows = _group_split(all_rows, val_fraction=float(val_fraction), seed=int(seed))
    if not train_rows:
        raise ValueError("no training samples after split")

    train_ds = WeightedTeacherDataset(train_rows)
    val_ds = WeightedTeacherDataset(val_rows) if val_rows else None
    train_loader = DataLoader(train_ds, batch_size=max(1, int(batch_size)), shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=max(1, int(batch_size)), shuffle=False) if val_ds is not None else None

    obs_dim = int(train_ds.obs.shape[-1])
    action_dim = int(train_ds.target.shape[-1])
    action_scale = float(np.max(np.abs(train_ds.target.numpy())))
    action_scale = max(action_scale, 0.08)

    policy = DeterministicStudentPolicy(
        DeterministicStudentConfig(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=int(hidden_dim),
            action_scale=float(action_scale),
            mu_limit=1.5,
            device=str(device),
        )
    )
    optimizer = torch.optim.Adam(policy.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay))

    start_epoch = 1
    if resume_checkpoint is not None and Path(resume_checkpoint).exists():
        loaded_policy, payload = load_student_checkpoint(Path(resume_checkpoint), device=str(device))
        policy = loaded_policy
        if "optimizer_state_dict" in payload:
            optimizer.load_state_dict(payload["optimizer_state_dict"])
        start_epoch = int((payload.get("extra_state", {}) or {}).get("epoch", 0)) + 1

    best_val = float("inf")
    best_epoch = 0
    no_improve = 0
    history: list[dict[str, Any]] = []

    checkpoint_best = artifact_root / "checkpoint_best.pt"
    checkpoint_latest = artifact_root / "checkpoint_latest.pt"

    for epoch in range(start_epoch, start_epoch + max(1, int(epochs))):
        policy.train()
        train_weighted_losses: list[float] = []
        train_unweighted_losses: list[float] = []
        train_mean_action_l2: list[float] = []
        train_target_action_l2: list[float] = []

        for obs, target, weight in train_loader:
            obs = obs.to(policy.device)
            target = target.to(policy.device)
            weight = weight.to(policy.device)
            pred = policy.forward(obs)
            per_sample = ((pred - target) ** 2).mean(dim=-1, keepdim=True)
            normalized_weight = weight / (weight.mean() + 1e-6)
            weighted_loss = (normalized_weight * per_sample).mean()
            unweighted_loss = per_sample.mean()

            optimizer.zero_grad(set_to_none=True)
            weighted_loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()

            train_weighted_losses.append(float(weighted_loss.detach().cpu().item()))
            train_unweighted_losses.append(float(unweighted_loss.detach().cpu().item()))
            train_mean_action_l2.append(float(torch.linalg.norm(pred, dim=-1).mean().detach().cpu().item()))
            train_target_action_l2.append(float(torch.linalg.norm(target, dim=-1).mean().detach().cpu().item()))

        policy.eval()
        val_weighted_losses: list[float] = []
        val_unweighted_losses: list[float] = []
        with torch.no_grad():
            if val_loader is not None:
                for obs, target, weight in val_loader:
                    obs = obs.to(policy.device)
                    target = target.to(policy.device)
                    weight = weight.to(policy.device)
                    pred = policy.forward(obs)
                    per_sample = ((pred - target) ** 2).mean(dim=-1, keepdim=True)
                    normalized_weight = weight / (weight.mean() + 1e-6)
                    val_weighted_losses.append(float((normalized_weight * per_sample).mean().detach().cpu().item()))
                    val_unweighted_losses.append(float(per_sample.mean().detach().cpu().item()))

        val_metric = float(np.mean(val_weighted_losses)) if val_weighted_losses else float(np.mean(train_weighted_losses))
        epoch_row = {
            "epoch": int(epoch),
            "train_weighted_loss": float(np.mean(train_weighted_losses)) if train_weighted_losses else 0.0,
            "train_unweighted_loss": float(np.mean(train_unweighted_losses)) if train_unweighted_losses else 0.0,
            "val_weighted_loss": float(np.mean(val_weighted_losses)) if val_weighted_losses else 0.0,
            "val_unweighted_loss": float(np.mean(val_unweighted_losses)) if val_unweighted_losses else 0.0,
            "train_mean_action_l2": float(np.mean(train_mean_action_l2)) if train_mean_action_l2 else 0.0,
            "train_target_action_l2": float(np.mean(train_target_action_l2)) if train_target_action_l2 else 0.0,
        }
        history.append(epoch_row)

        save_student_checkpoint(
            checkpoint_path=checkpoint_latest,
            policy=policy,
            run_id=run_id,
            metadata={"checkpoint_kind": "latest", "dataset_path": str(dataset_path)},
            optimizer=optimizer,
            extra_state={"epoch": int(epoch)},
        )

        if val_metric < best_val:
            best_val = float(val_metric)
            best_epoch = int(epoch)
            no_improve = 0
            save_student_checkpoint(
                checkpoint_path=checkpoint_best,
                policy=policy,
                run_id=run_id,
                metadata={"checkpoint_kind": "best", "dataset_path": str(dataset_path), "best_val_weighted_loss": float(best_val)},
                optimizer=optimizer,
                extra_state={"epoch": int(epoch)},
            )
        else:
            no_improve += 1
            if no_improve >= max(1, int(early_stop_patience)):
                break

    history_path = artifact_root / "train_history.jsonl"
    _write_history(history_path, history)

    split_summary = {
        "train": _split_summary(train_rows),
        "validation": _split_summary(val_rows),
    }
    summary = {
        "run_id": str(run_id),
        "dataset_path": str(dataset_path),
        "seed": int(seed),
        "epochs_requested": int(epochs),
        "epochs_completed": int(len(history)),
        "best_epoch": int(best_epoch),
        "best_val_weighted_loss": float(best_val),
        "checkpoint_best": str(checkpoint_best),
        "checkpoint_latest": str(checkpoint_latest),
        "student_config": {
            "obs_dim": int(obs_dim),
            "action_dim": int(action_dim),
            "hidden_dim": int(hidden_dim),
            "action_scale": float(action_scale),
            "device": str(device),
        },
        "optimizer": {
            "learning_rate": float(learning_rate),
            "weight_decay": float(weight_decay),
            "batch_size": int(batch_size),
            "val_fraction": float(val_fraction),
            "early_stop_patience": int(early_stop_patience),
        },
        "splits": split_summary,
        "history_tail": history[-10:],
    }

    plot_path = _plot_training_curves(artifact_root / "plots" / "training_curves.png", history)
    if plot_path is not None:
        summary["training_curves_plot"] = plot_path

    summary_path = artifact_root / "training_summary.json"
    _write_json(summary_path, summary)

    md_lines = [
        "# Deterministic Student Training",
        "",
        f"- run_id: `{run_id}`",
        f"- dataset_path: `{dataset_path}`",
        f"- epochs_completed: `{len(history)}`",
        f"- best_epoch: `{best_epoch}`",
        f"- best_val_weighted_loss: `{best_val:.8f}`",
        "",
        "## Split Counts",
        f"- train samples: `{split_summary['train']['count']}`",
        f"- validation samples: `{split_summary['validation']['count']}`",
        "",
        "## Train Zone Counts",
    ]
    for key in ["outside", "outer", "inner", "dwell"]:
        md_lines.append(f"- `{key}`: `{int(split_summary['train']['zone_counts'].get(key, 0))}`")
    md_lines.extend(["", "## Validation Zone Counts"])
    for key in ["outside", "outer", "inner", "dwell"]:
        md_lines.append(f"- `{key}`: `{int(split_summary['validation']['zone_counts'].get(key, 0))}`")
    (artifact_root / "training_report.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    return {
        "run_id": str(run_id),
        "checkpoint_best": str(checkpoint_best),
        "checkpoint_latest": str(checkpoint_latest),
        "training_summary": str(summary_path),
        "train_history": str(history_path),
        "training_report": str(artifact_root / "training_report.md"),
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a deterministic student from teacher dataset")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--early-stop-patience", type=int, default=20)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--resume-checkpoint", default="")
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()
    out = train_deterministic_student(
        dataset_path=Path(args.dataset),
        run_id=str(args.run_id),
        artifact_root=Path(args.artifact_root),
        seed=int(args.seed),
        hidden_dim=int(args.hidden_dim),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        learning_rate=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
        val_fraction=float(args.val_fraction),
        early_stop_patience=int(args.early_stop_patience),
        device=str(args.device),
        resume_checkpoint=Path(args.resume_checkpoint) if str(args.resume_checkpoint).strip() else None,
    )
    print(json.dumps(out, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
