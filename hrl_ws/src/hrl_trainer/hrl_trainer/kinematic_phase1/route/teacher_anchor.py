"""Teacher-anchor imitation callback for route curriculum PPO fine-tuning."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from stable_baselines3.common.callbacks import BaseCallback


@dataclass(frozen=True)
class TeacherAnchorConfig:
    enabled: bool = False
    dataset_path: str = ""
    loss_weight: float = 0.02
    batch_size: int = 256
    gradient_steps: int = 1
    every_rollouts: int = 1
    max_route_index: int = 120


class RouteTeacherAnchorCallback(BaseCallback):  # pragma: no cover - exercised in training
    """Applies a small deterministic-action imitation loss after PPO rollouts.

    This is intentionally lightweight. It does not replace PPO or control the
    route; it only anchors candidate policies to a verified teacher on the
    protected prefix so fine-tuning does not wash out early-route behavior.
    """

    def __init__(self, config: TeacherAnchorConfig) -> None:
        super().__init__()
        if not config.dataset_path:
            raise ValueError("TeacherAnchorConfig.dataset_path is required when enabled")
        self.config = config
        self._rng = np.random.default_rng(0)
        self._rollout_count = 0
        self._obs: dict[str, np.ndarray] = {}
        self._actions: np.ndarray | None = None

    def _on_training_start(self) -> None:
        payload = np.load(Path(self.config.dataset_path), allow_pickle=False)
        route_index = np.asarray(payload["route_index"], dtype=np.int32)
        keep = route_index <= int(self.config.max_route_index)
        actions = np.asarray(payload["actions"], dtype=np.float32)[keep]
        if actions.size == 0:
            raise ValueError(f"No teacher-anchor samples left after max_route_index={self.config.max_route_index}")
        obs: dict[str, np.ndarray] = {}
        for key in payload.files:
            if not key.startswith("obs__"):
                continue
            obs_key = key.removeprefix("obs__")
            obs[obs_key] = np.asarray(payload[key], dtype=np.float32)[keep]
        self._obs = obs
        self._actions = actions

    def _sample_batch(self) -> tuple[dict[str, np.ndarray], np.ndarray]:
        assert self._actions is not None
        batch_size = min(int(self.config.batch_size), len(self._actions))
        idx = self._rng.integers(0, len(self._actions), size=batch_size)
        obs = {key: value[idx] for key, value in self._obs.items()}
        actions = self._actions[idx]
        return obs, actions

    def _on_rollout_end(self) -> None:
        self._rollout_count += 1
        if self._rollout_count % max(int(self.config.every_rollouts), 1) != 0:
            return
        device = self.model.device
        policy = self.model.policy
        policy.train()
        last_loss = 0.0
        for _ in range(max(int(self.config.gradient_steps), 1)):
            obs_np, actions_np = self._sample_batch()
            obs_tensor, _ = policy.obs_to_tensor(obs_np)
            teacher_actions = torch.as_tensor(actions_np, dtype=torch.float32, device=device)
            pred_actions = policy._predict(obs_tensor, deterministic=True)  # noqa: SLF001 - SB3 policy API
            loss = F.mse_loss(pred_actions, teacher_actions) * float(self.config.loss_weight)
            policy.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=0.5)
            policy.optimizer.step()
            last_loss = float(loss.detach().cpu().item())
        self.logger.record("train/teacher_anchor_loss", last_loss)

    def _on_step(self) -> bool:
        return True

    def summary(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "dataset_path": self.config.dataset_path,
            "loss_weight": float(self.config.loss_weight),
            "batch_size": int(self.config.batch_size),
            "gradient_steps": int(self.config.gradient_steps),
            "every_rollouts": int(self.config.every_rollouts),
            "max_route_index": int(self.config.max_route_index),
            "sample_count": 0 if self._actions is None else int(len(self._actions)),
        }
