"""Small MLP classifier for dock-readiness prediction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


class DockReadinessMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_sizes: tuple[int, ...] = (128, 64)) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_dim
        for hidden in hidden_sizes:
            layers.extend([nn.Linear(prev, hidden), nn.ReLU()])
            prev = hidden
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


@dataclass(frozen=True)
class FeatureNormalizer:
    mean: list[float]
    std: list[float]

    @classmethod
    def fit(cls, x: np.ndarray) -> "FeatureNormalizer":
        mean = np.mean(x, axis=0)
        std = np.std(x, axis=0)
        std = np.where(std < 1e-6, 1.0, std)
        return cls(mean=mean.astype(float).tolist(), std=std.astype(float).tolist())

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - np.asarray(self.mean, dtype=np.float32)) / np.asarray(self.std, dtype=np.float32)


def save_readiness_model(
    *,
    path: Path,
    model: DockReadinessMLP,
    normalizer: FeatureNormalizer,
    hidden_sizes: tuple[int, ...],
    threshold: float,
    metadata: dict[str, Any] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_dim": int(model.net[0].in_features),
            "hidden_sizes": list(hidden_sizes),
            "normalizer": {"mean": normalizer.mean, "std": normalizer.std},
            "threshold": float(threshold),
            "metadata": metadata or {},
        },
        path,
    )


def load_readiness_model(path: Path, *, device: str = "cpu") -> tuple[DockReadinessMLP, FeatureNormalizer, float, dict[str, Any]]:
    payload = torch.load(path, map_location=device)
    hidden_sizes = tuple(int(x) for x in payload.get("hidden_sizes", [128, 64]))
    model = DockReadinessMLP(input_dim=int(payload["input_dim"]), hidden_sizes=hidden_sizes)
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    model.eval()
    normalizer = FeatureNormalizer(**payload["normalizer"])
    return model, normalizer, float(payload.get("threshold", 0.5)), dict(payload.get("metadata", {}))


def predict_readiness_score(
    *,
    model: DockReadinessMLP,
    normalizer: FeatureNormalizer,
    features: list[float],
    device: str = "cpu",
) -> float:
    x = np.asarray([features], dtype=np.float32)
    x = normalizer.transform(x)
    with torch.no_grad():
        logits = model(torch.as_tensor(x, dtype=torch.float32, device=device))
        return float(torch.sigmoid(logits).cpu().numpy()[0])


__all__ = [
    "DockReadinessMLP",
    "FeatureNormalizer",
    "load_readiness_model",
    "predict_readiness_score",
    "save_readiness_model",
]
