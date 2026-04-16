"""Deterministic student policy utilities for teacher-student extraction."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn


DEFAULT_DATASET_SOURCE_RUNS = [
    "main_reward_bonly_baseline_s0b_tc0_act008_001",
    "main_reward_bonly_solidify_s0b_tc0_act008_001",
]

DEFAULT_TEACHER_COMPARE_RUNS = [
    "main_reward_bonly_baseline_s0b_tc0_act008_001",
    "main_reward_bonly_solidify_s0b_tc0_act008_001",
    "main_reward_bonly_solidify_v2_s0b_tc0_act008_001",
]


def parse_csv_list(spec: str | None, default: list[str] | None = None) -> list[str]:
    raw = str(spec or "").strip()
    if not raw:
        return list(default or [])
    out = [item.strip() for item in raw.split(",") if item.strip()]
    return out or list(default or [])


@dataclass(frozen=True)
class DeterministicStudentConfig:
    obs_dim: int = 27
    action_dim: int = 7
    hidden_dim: int = 128
    action_scale: float = 0.08
    mu_limit: float = 1.5
    device: str = "cpu"


class DeterministicStudentModel(nn.Module):
    def __init__(self, cfg: DeterministicStudentConfig) -> None:
        super().__init__()
        hidden = int(cfg.hidden_dim)
        self.net = nn.Sequential(
            nn.Linear(int(cfg.obs_dim), hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, int(cfg.action_dim)),
        )
        self.mu_limit = float(cfg.mu_limit)
        self.action_scale = float(cfg.action_scale)

    def forward_mu_raw(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)

    def forward_components(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mu_raw = self.forward_mu_raw(obs)
        mu = torch.clamp(mu_raw, min=-self.mu_limit, max=self.mu_limit)
        return mu, mu_raw

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        mu, _mu_raw = self.forward_components(obs)
        return torch.tanh(mu) * self.action_scale


class DeterministicStudentPolicy:
    def __init__(self, cfg: DeterministicStudentConfig) -> None:
        self.cfg = cfg
        self.device = torch.device(str(cfg.device))
        self.model = DeterministicStudentModel(cfg).to(self.device)

    def eval(self) -> "DeterministicStudentPolicy":
        self.model.eval()
        return self

    def train(self) -> "DeterministicStudentPolicy":
        self.model.train()
        return self

    def parameters(self):
        return self.model.parameters()

    def state_dict(self) -> dict[str, Any]:
        return self.model.state_dict()

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.model.load_state_dict(state_dict)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.model(obs)

    def _act_tensor(
        self,
        obs_tensor: torch.Tensor,
        *,
        stochastic: bool,
        exploration_std_scale: float,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        mu, mu_raw = self.model.forward_components(obs_tensor)
        std = torch.ones_like(mu)
        noise = torch.zeros_like(mu)
        if stochastic and float(exploration_std_scale) > 0.0:
            noise = torch.randn_like(mu)
        pre_tanh = mu + float(exploration_std_scale) * noise
        post_tanh = torch.tanh(pre_tanh)
        action = post_tanh * float(self.cfg.action_scale)
        diagnostics = {
            "mode": "stochastic" if stochastic and float(exploration_std_scale) > 0.0 else "deterministic",
            "mu_raw": mu_raw.detach().cpu().numpy(),
            "mu": mu.detach().cpu().numpy(),
            "log_std": np.zeros_like(mu.detach().cpu().numpy()),
            "std": std.detach().cpu().numpy(),
            "std_scaled": (float(exploration_std_scale) * std).detach().cpu().numpy(),
            "noise": noise.detach().cpu().numpy(),
            "pre_tanh": pre_tanh.detach().cpu().numpy(),
            "post_tanh": post_tanh.detach().cpu().numpy(),
            "final_action": action.detach().cpu().numpy(),
            "action_scale": float(self.cfg.action_scale),
            "exploration_std_scale": float(exploration_std_scale),
            "mu_limit": float(self.cfg.mu_limit),
            "pre_tanh_abs_max": float(pre_tanh.abs().max().detach().cpu().item()),
            "post_tanh_abs_max": float(post_tanh.abs().max().detach().cpu().item()),
            "saturated_dims": int((post_tanh.abs() > 0.98).sum().detach().cpu().item()),
            "saturated_fraction": float((post_tanh.abs() > 0.98).float().mean().detach().cpu().item()),
        }
        return action, diagnostics

    def act(self, obs: np.ndarray, *, stochastic: bool = False, exploration_std_scale: float = 0.0) -> np.ndarray:
        action, _diagnostics = self.act_with_diagnostics(
            obs,
            stochastic=stochastic,
            exploration_std_scale=exploration_std_scale,
        )
        return action

    def act_with_diagnostics(
        self,
        obs: np.ndarray,
        *,
        stochastic: bool = False,
        exploration_std_scale: float = 0.0,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        obs_arr = np.asarray(obs, dtype=np.float32)
        obs_tensor = torch.as_tensor(obs_arr, dtype=torch.float32, device=self.device).unsqueeze(0)
        self.model.eval()
        with torch.no_grad():
            action_tensor, diagnostics = self._act_tensor(
                obs_tensor,
                stochastic=bool(stochastic),
                exploration_std_scale=float(exploration_std_scale),
            )
        action = action_tensor.squeeze(0).detach().cpu().numpy().astype(np.float32)
        flat = {}
        for key, value in diagnostics.items():
            if isinstance(value, np.ndarray):
                flat[key] = value.squeeze(0).tolist()
            else:
                flat[key] = value
        return action, flat


def save_student_checkpoint(
    *,
    checkpoint_path: Path,
    policy: DeterministicStudentPolicy,
    run_id: str,
    metadata: dict[str, Any] | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    extra_state: dict[str, Any] | None = None,
) -> str:
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_id": str(run_id),
        "student_config": asdict(policy.cfg),
        "model_state_dict": policy.state_dict(),
        "metadata": dict(metadata or {}),
        "extra_state": dict(extra_state or {}),
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    torch.save(payload, checkpoint_path)
    return str(checkpoint_path)


def load_student_checkpoint(
    checkpoint_path: str | Path,
    *,
    device: str = "cpu",
) -> tuple[DeterministicStudentPolicy, dict[str, Any]]:
    checkpoint_path = Path(checkpoint_path)
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    raw_cfg = dict(payload.get("student_config", {}) or {})
    raw_cfg["device"] = str(device)
    policy = DeterministicStudentPolicy(DeterministicStudentConfig(**raw_cfg))
    policy.load_state_dict(payload["model_state_dict"])
    policy.eval()
    return policy, payload
