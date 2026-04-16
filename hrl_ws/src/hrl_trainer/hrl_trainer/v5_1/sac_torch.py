"""Torch SAC mainline implementation for V5.1.

This is the primary SAC path used by pipeline_e2e (policy_mode=sac_torch).
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


@dataclass(frozen=True)
class SACTorchConfig:
    obs_dim: int
    action_dim: int
    hidden_dim: int = 128
    gamma: float = 0.99
    tau: float = 0.01
    lr_actor: float = 2e-4
    lr_critic: float = 3e-4
    lr_alpha: float = 3e-4
    init_alpha: float = 0.2
    target_entropy: float | None = None
    replay_capacity: int = 50_000
    batch_size: int = 32
    action_scale: float = 0.05
    mu_limit: float = 1.5
    executor_dt: float = 0.1
    joint_min: tuple[float, ...] = (-0.5, -2.8, -1.6, -2.8, -3.0, -2.8, -6.0)
    joint_max: tuple[float, ...] = (0.5, 2.8, 1.6, 2.8, 3.0, 2.8, 6.0)
    rate_limit_per_sec: tuple[float, ...] = (0.30, 0.30, 0.20, 0.30, 0.30, 0.30, 0.40)
    log_std_min: float = -5.0
    log_std_max: float = 2.0
    actor_update_delay: int = 2
    actor_grad_clip: float = 1.0
    bc_lambda: float = 0.05
    bc_outer_dpos_m: float = 0.08
    bc_inner_dpos_m: float = 0.04
    bc_topk: int = 3
    distill_lambda: float = 0.0
    distill_interval: int = 20
    distill_steps: int = 1
    distill_batch_size: int = 0
    distill_candidate_multiplier: int = 8
    distill_min_good_count: int = 8
    distill_outer_dpos_m: float = 0.08
    distill_support_dpos_m: float = 0.07
    distill_inner_dpos_m: float = 0.04
    distill_dwell_dpos_m: float = 0.025
    distill_min_progress_m: float = 0.003
    distill_max_delta_norm: float = 0.75
    distill_quality_threshold: float = 0.0
    distill_advantage_beta: float = 0.0
    distill_advantage_clip: float = 5.0
    distill_grad_clip: float = 1.0
    distill_exclude_rejected: bool = True
    distill_exclude_clamped: bool = True
    distill_exclude_projected: bool = True
    device: str = "cpu"
    param_hash_interval: int = 10


class TorchReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int, action_dim: int, device: torch.device) -> None:
        self.capacity = int(capacity)
        self.device = device
        self.obs = torch.zeros((capacity, obs_dim), dtype=torch.float32, device=device)
        self.raw_actions = torch.zeros((capacity, action_dim), dtype=torch.float32, device=device)
        self.exec_actions = torch.zeros((capacity, action_dim), dtype=torch.float32, device=device)
        self.rewards = torch.zeros((capacity, 1), dtype=torch.float32, device=device)
        self.next_obs = torch.zeros((capacity, obs_dim), dtype=torch.float32, device=device)
        self.dones = torch.zeros((capacity, 1), dtype=torch.float32, device=device)
        self.prev_q_des = torch.zeros((capacity, action_dim), dtype=torch.float32, device=device)
        self.next_prev_q_des = torch.zeros((capacity, action_dim), dtype=torch.float32, device=device)
        self.delta_limits = torch.zeros((capacity, action_dim), dtype=torch.float32, device=device)
        self.delta_norm = torch.zeros((capacity, 1), dtype=torch.float32, device=device)
        self.raw_norm = torch.zeros((capacity, 1), dtype=torch.float32, device=device)
        self.exec_norm = torch.zeros((capacity, 1), dtype=torch.float32, device=device)
        self.clamp_triggered = torch.zeros((capacity, 1), dtype=torch.float32, device=device)
        self.projection_triggered = torch.zeros((capacity, 1), dtype=torch.float32, device=device)
        self.rejected = torch.zeros((capacity, 1), dtype=torch.float32, device=device)
        self.success = torch.zeros((capacity, 1), dtype=torch.float32, device=device)
        self.dwell_count = torch.zeros((capacity, 1), dtype=torch.float32, device=device)
        self.ptr = 0
        self.size = 0

    def add(
        self,
        obs: np.ndarray,
        raw_action: np.ndarray,
        exec_action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
        info: dict[str, Any] | None = None,
    ) -> None:
        i = self.ptr
        info = info or {}
        self.obs[i] = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        self.raw_actions[i] = torch.as_tensor(raw_action, dtype=torch.float32, device=self.device)
        self.exec_actions[i] = torch.as_tensor(exec_action, dtype=torch.float32, device=self.device)
        self.rewards[i] = float(reward)
        self.next_obs[i] = torch.as_tensor(next_obs, dtype=torch.float32, device=self.device)
        self.dones[i] = float(done)
        self.prev_q_des[i] = torch.as_tensor(info.get("prev_q_des", np.zeros_like(exec_action)), dtype=torch.float32, device=self.device)
        self.next_prev_q_des[i] = torch.as_tensor(info.get("next_prev_q_des", np.zeros_like(exec_action)), dtype=torch.float32, device=self.device)
        self.delta_limits[i] = torch.as_tensor(info.get("delta_limits", np.zeros_like(exec_action)), dtype=torch.float32, device=self.device)
        self.delta_norm[i] = float(info.get("delta_norm", 0.0))
        self.raw_norm[i] = float(info.get("raw_norm", 0.0))
        self.exec_norm[i] = float(info.get("exec_norm", 0.0))
        self.clamp_triggered[i] = float(bool(info.get("clamp_triggered", False)))
        self.projection_triggered[i] = float(bool(info.get("projection_triggered", False)))
        self.rejected[i] = float(bool(info.get("rejected", False)))
        self.success[i] = float(bool(info.get("success", False)))
        self.dwell_count[i] = float(info.get("dwell_count", 0.0))
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> tuple[torch.Tensor, ...]:
        idx = torch.randint(0, self.size, (batch_size,), device=self.device)
        return (
            self.obs[idx],
            self.raw_actions[idx],
            self.exec_actions[idx],
            self.rewards[idx],
            self.next_obs[idx],
            self.dones[idx],
            self.prev_q_des[idx],
            self.next_prev_q_des[idx],
            self.delta_limits[idx],
            self.delta_norm[idx],
            self.raw_norm[idx],
            self.exec_norm[idx],
            self.clamp_triggered[idx],
            self.projection_triggered[idx],
            self.rejected[idx],
            self.success[idx],
            self.dwell_count[idx],
        )

    def __len__(self) -> int:
        return self.size


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GaussianActor(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int,
        log_std_min: float,
        log_std_max: float,
        mu_limit: float,
    ) -> None:
        super().__init__()
        self.backbone = MLP(obs_dim, hidden_dim, hidden_dim)
        self.mu = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.mu_limit = float(mu_limit)

    def forward_components(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.backbone(obs)
        mu_raw = self.mu(h)
        if self.mu_limit > 0.0:
            mu = self.mu_limit * torch.tanh(mu_raw / self.mu_limit)
        else:
            mu = mu_raw
        log_std = torch.clamp(self.log_std(h), self.log_std_min, self.log_std_max)
        return mu, log_std, mu_raw

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mu, log_std, _mu_raw = self.forward_components(obs)
        return mu, log_std

    def sample(self, obs: torch.Tensor, deterministic: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, log_std = self(obs)
        std = log_std.exp()
        dist = Normal(mu, std)
        z = mu if deterministic else dist.rsample()
        action = torch.tanh(z)
        # tanh correction
        log_prob = dist.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob, mu


class QCritic(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.q = MLP(obs_dim + action_dim, 1, hidden_dim)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.q(torch.cat([obs, action], dim=-1))


class SACTorchAgent:
    def __init__(self, config: SACTorchConfig, seed: int = 0) -> None:
        self.cfg = config
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.device = torch.device(self.cfg.device)
        target_entropy = self.cfg.target_entropy
        if target_entropy is None:
            target_entropy = -float(self.cfg.action_dim)
        self.target_entropy = float(target_entropy)

        self.actor = GaussianActor(
            self.cfg.obs_dim,
            self.cfg.action_dim,
            self.cfg.hidden_dim,
            self.cfg.log_std_min,
            self.cfg.log_std_max,
            self.cfg.mu_limit,
        ).to(self.device)
        self.q1 = QCritic(self.cfg.obs_dim, self.cfg.action_dim, self.cfg.hidden_dim).to(self.device)
        self.q2 = QCritic(self.cfg.obs_dim, self.cfg.action_dim, self.cfg.hidden_dim).to(self.device)
        self.q1_target = QCritic(self.cfg.obs_dim, self.cfg.action_dim, self.cfg.hidden_dim).to(self.device)
        self.q2_target = QCritic(self.cfg.obs_dim, self.cfg.action_dim, self.cfg.hidden_dim).to(self.device)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=self.cfg.lr_actor)
        self.q1_optim = torch.optim.Adam(self.q1.parameters(), lr=self.cfg.lr_critic)
        self.q2_optim = torch.optim.Adam(self.q2.parameters(), lr=self.cfg.lr_critic)

        self.log_alpha = torch.tensor(np.log(self.cfg.init_alpha), dtype=torch.float32, device=self.device, requires_grad=True)
        self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=self.cfg.lr_alpha)

        self.replay = TorchReplayBuffer(self.cfg.replay_capacity, self.cfg.obs_dim, self.cfg.action_dim, self.device)

        self.env_steps_collected = 0
        self.updates_applied = 0
        self.batch_draw_count = 0
        self.actor_update_count = 0
        self.critic_update_count = 0
        self.alpha_update_count = 0
        self.distill_update_count = 0
        self.distill_skip_count = 0
        self.last_actor_hash: str | None = None
        self.last_critic_hash: str | None = None
        self.active_distill_lambda = float(self.cfg.distill_lambda)
        self.distill_stage_name = "A"

    @property
    def alpha(self) -> float:
        return float(self.log_alpha.detach().exp().cpu().item())

    def set_target_entropy(self, target_entropy: float) -> None:
        self.target_entropy = float(target_entropy)

    def set_distill_mode(self, *, lambda_value: float, stage_name: str | None = None) -> None:
        self.active_distill_lambda = max(0.0, float(lambda_value))
        if stage_name is not None:
            self.distill_stage_name = str(stage_name)

    def act_with_diagnostics(
        self,
        obs: np.ndarray,
        stochastic: bool = True,
        exploration_std_scale: float = 1.0,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            mu, log_std, mu_raw = self.actor.forward_components(obs_t)
            std = log_std.exp()
            scale = max(0.0, float(exploration_std_scale))
            deterministic = (not stochastic) or scale <= 0.0

            if deterministic:
                noise = torch.zeros_like(mu)
                std_scaled = torch.zeros_like(std)
                pre_tanh = mu
                sample_mode = "deterministic"
            else:
                std_scaled = std * scale
                dist = Normal(mu, std_scaled)
                pre_tanh = dist.rsample()
                safe_std = torch.where(std_scaled > 0.0, std_scaled, torch.ones_like(std_scaled))
                noise = torch.where(std_scaled > 0.0, (pre_tanh - mu) / safe_std, torch.zeros_like(std_scaled))
                sample_mode = "stochastic"

            post_tanh = torch.tanh(pre_tanh)
            action = post_tanh * self.cfg.action_scale

        post_tanh_abs = post_tanh.abs()
        saturated_mask = post_tanh_abs >= 0.98
        diagnostics = {
            "mode": sample_mode,
            "exploration_std_scale": float(scale),
            "action_scale": float(self.cfg.action_scale),
            "mu_limit": float(self.cfg.mu_limit),
            "mu_raw": mu_raw.squeeze(0).cpu().numpy().astype(float).tolist(),
            "mu": mu.squeeze(0).cpu().numpy().astype(float).tolist(),
            "log_std": log_std.squeeze(0).cpu().numpy().astype(float).tolist(),
            "std": std.squeeze(0).cpu().numpy().astype(float).tolist(),
            "std_scaled": std_scaled.squeeze(0).cpu().numpy().astype(float).tolist(),
            "noise": noise.squeeze(0).cpu().numpy().astype(float).tolist(),
            "pre_tanh": pre_tanh.squeeze(0).cpu().numpy().astype(float).tolist(),
            "post_tanh": post_tanh.squeeze(0).cpu().numpy().astype(float).tolist(),
            "final_action": action.squeeze(0).cpu().numpy().astype(float).tolist(),
            "pre_tanh_abs_max": float(pre_tanh.abs().max().cpu().item()),
            "post_tanh_abs_max": float(post_tanh_abs.max().cpu().item()),
            "saturated_dims": int(saturated_mask.sum().cpu().item()),
            "saturated_fraction": float(saturated_mask.float().mean().cpu().item()),
        }
        return action.squeeze(0).cpu().numpy(), diagnostics

    def act(self, obs: np.ndarray, stochastic: bool = True, exploration_std_scale: float = 1.0) -> np.ndarray:
        action, _ = self.act_with_diagnostics(
            obs,
            stochastic=stochastic,
            exploration_std_scale=exploration_std_scale,
        )
        return action

    def remember(
        self,
        obs: np.ndarray,
        action_raw: np.ndarray,
        action_exec: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
        info: dict[str, Any] | None = None,
    ) -> None:
        self.replay.add(obs, action_raw, action_exec, reward, next_obs, done, info=info)
        self.env_steps_collected += 1

    @staticmethod
    def _grad_norm(module: nn.Module) -> float:
        grads = [p.grad.detach().reshape(-1) for p in module.parameters() if p.grad is not None]
        if not grads:
            return 0.0
        flat = torch.cat(grads)
        return float(torch.linalg.norm(flat).detach().cpu().item())

    @staticmethod
    def _param_hash(modules: list[nn.Module]) -> str:
        h = hashlib.sha256()
        with torch.no_grad():
            for module in modules:
                for p in module.parameters():
                    h.update(p.detach().cpu().numpy().tobytes())
        return h.hexdigest()

    def _soft_update(self) -> None:
        tau = self.cfg.tau
        with torch.no_grad():
            for p, tp in zip(self.q1.parameters(), self.q1_target.parameters(), strict=True):
                tp.data.mul_(1.0 - tau).add_(tau * p.data)
            for p, tp in zip(self.q2.parameters(), self.q2_target.parameters(), strict=True):
                tp.data.mul_(1.0 - tau).add_(tau * p.data)

    def _executor_proxy_torch(
        self,
        *,
        q_current: torch.Tensor,
        delta_q_cmd: torch.Tensor,
        prev_q_des: torch.Tensor,
        delta_limits: torch.Tensor,
    ) -> torch.Tensor:
        q_min = torch.as_tensor(self.cfg.joint_min, dtype=torch.float32, device=self.device).view(1, -1)
        q_max = torch.as_tensor(self.cfg.joint_max, dtype=torch.float32, device=self.device).view(1, -1)
        rate_limit = torch.as_tensor(self.cfg.rate_limit_per_sec, dtype=torch.float32, device=self.device).view(1, -1)
        dt = float(self.cfg.executor_dt)
        delta_clamped = torch.clamp(delta_q_cmd, -delta_limits, delta_limits)
        pre_rate_q_des = q_current + delta_clamped
        max_step = rate_limit * dt
        limited_step = torch.clamp(pre_rate_q_des - prev_q_des, -max_step, max_step)
        limited_q_des = prev_q_des + limited_step
        projected_q_des = torch.clamp(limited_q_des, q_min, q_max)
        return projected_q_des - q_current

    def _dpos_from_obs(self, obs: torch.Tensor) -> torch.Tensor:
        start = self.cfg.action_dim * 2
        ee_pos_err = obs[:, start : start + 3]
        return torch.linalg.norm(ee_pos_err, dim=-1, keepdim=True)

    def _sample_distill_batch(self, batch_size: int) -> tuple[dict[str, torch.Tensor], dict[str, float]] | None:
        if len(self.replay) <= 0:
            return None

        candidate_count = max(int(batch_size), int(batch_size) * max(1, int(self.cfg.distill_candidate_multiplier)))
        idx = torch.randint(0, self.replay.size, (candidate_count,), device=self.device)

        obs = self.replay.obs[idx]
        exec_act = self.replay.exec_actions[idx]
        nxt = self.replay.next_obs[idx]
        prev_q_des = self.replay.prev_q_des[idx]
        delta_limits = self.replay.delta_limits[idx]
        delta_norm = self.replay.delta_norm[idx]
        clamp_triggered = self.replay.clamp_triggered[idx]
        projection_triggered = self.replay.projection_triggered[idx]
        rejected = self.replay.rejected[idx]
        success = self.replay.success[idx]
        dwell_count = self.replay.dwell_count[idx]

        prev_dpos = self._dpos_from_obs(obs)
        next_dpos = self._dpos_from_obs(nxt)
        progress = prev_dpos - next_dpos

        outer_limit = float(self.cfg.distill_outer_dpos_m)
        support_limit = min(outer_limit, float(self.cfg.distill_support_dpos_m))
        inner_limit = float(self.cfg.distill_inner_dpos_m)
        outer = next_dpos <= outer_limit
        inner = next_dpos <= inner_limit
        dwell = (next_dpos <= float(self.cfg.distill_dwell_dpos_m)) | (dwell_count > 0.0)
        progressed = progress >= float(self.cfg.distill_min_progress_m)
        safe = torch.ones_like(next_dpos, dtype=torch.bool)
        if bool(self.cfg.distill_exclude_rejected):
            safe = safe & (rejected < 0.5)
        if bool(self.cfg.distill_exclude_clamped):
            safe = safe & (clamp_triggered < 0.5)
        if bool(self.cfg.distill_exclude_projected):
            safe = safe & (projection_triggered < 0.5)
        if float(self.cfg.distill_max_delta_norm) > 0.0:
            safe = safe & (delta_norm <= float(self.cfg.distill_max_delta_norm))

        success_hit = success > 0.5
        elite = success_hit | dwell | inner
        support = outer & progressed & (next_dpos <= support_limit) & (~elite)
        eligible = safe & (elite | support)
        progress_scale = max(float(self.cfg.distill_min_progress_m), 1e-6)
        progress_score = torch.clamp(progress / progress_scale, min=0.0, max=1.0)
        drift_score = torch.clamp((-progress) / progress_scale, min=0.0, max=2.0)
        support_span = max(support_limit - inner_limit, 1e-6)
        support_depth_score = torch.clamp((support_limit - next_dpos) / support_span, min=0.0, max=1.0)
        support_depth_score = support_depth_score * support.float()
        delta_scale = max(float(self.cfg.distill_max_delta_norm), 1e-6)
        delta_score = torch.clamp(delta_norm / delta_scale, min=0.0, max=2.0)
        quality = (
            8.0 * success_hit.float()
            + 4.0 * dwell.float()
            + 2.0 * inner.float()
            + 0.5 * support.float()
            + 0.75 * support_depth_score
            + 0.15 * progress_score
            - 1.25 * drift_score
            - 1.0 * clamp_triggered
            - 1.0 * projection_triggered
            - 1.0 * delta_score
        )
        eligible = eligible & (quality >= float(self.cfg.distill_quality_threshold))

        eligible_count = int(eligible.sum().detach().cpu().item())
        min_good = max(1, int(self.cfg.distill_min_good_count))
        stats = {
            "candidate_count": float(candidate_count),
            "eligible_count": float(eligible_count),
            "eligible_fraction": float(eligible.float().mean().detach().cpu().item()),
            "next_dpos_mean": float(next_dpos.mean().detach().cpu().item()),
            "progress_mean": float(progress.mean().detach().cpu().item()),
            "quality_mean": float(quality.mean().detach().cpu().item()),
            "safe_fraction": float(safe.float().mean().detach().cpu().item()),
        }
        if eligible_count < min_good:
            stats["skip_reason"] = 1.0
            return {"stats_only": torch.empty(0, device=self.device)}, stats

        k = min(int(batch_size), eligible_count)
        masked_quality = torch.where(eligible, quality, torch.full_like(quality, -1.0e9))
        top_idx = torch.topk(masked_quality.reshape(-1), k=k, largest=True).indices
        selected_quality = quality.reshape(-1)[top_idx]
        selected_next_dpos = next_dpos.reshape(-1)[top_idx]
        selected_progress = progress.reshape(-1)[top_idx]

        stats.update(
            {
                "selected_count": float(k),
                "selected_fraction": float(k) / float(max(1, candidate_count)),
                "selected_quality_mean": float(selected_quality.mean().detach().cpu().item()),
                "selected_quality_min": float(selected_quality.min().detach().cpu().item()),
                "selected_next_dpos_mean": float(selected_next_dpos.mean().detach().cpu().item()),
                "selected_progress_mean": float(selected_progress.mean().detach().cpu().item()),
                "selected_success_fraction": float(success.reshape(-1)[top_idx].mean().detach().cpu().item()),
                "selected_support_fraction": float(support.float().reshape(-1)[top_idx].mean().detach().cpu().item()),
                "selected_inner_fraction": float(inner.float().reshape(-1)[top_idx].mean().detach().cpu().item()),
                "selected_dwell_fraction": float(dwell.float().reshape(-1)[top_idx].mean().detach().cpu().item()),
            }
        )
        return (
            {
                "obs": obs[top_idx],
                "exec_act": exec_act[top_idx],
                "prev_q_des": prev_q_des[top_idx],
                "delta_limits": delta_limits[top_idx],
            },
            stats,
        )

    def _run_distill_step(self, update_index: int) -> dict[str, float]:
        active_lambda = float(getattr(self, "active_distill_lambda", self.cfg.distill_lambda))
        stage_name = str(getattr(self, "distill_stage_name", "A"))
        if active_lambda <= 0.0:
            return {
                "distill_enabled": 0.0,
                "distill_active_lambda": 0.0,
                "distill_stage_name": stage_name,
            }
        interval = max(1, int(self.cfg.distill_interval))
        if int(update_index) % interval != 0:
            return {
                "distill_enabled": 1.0,
                "distill_triggered": 0.0,
                "distill_active_lambda": active_lambda,
                "distill_stage_name": stage_name,
            }

        batch_size = int(self.cfg.distill_batch_size) if int(self.cfg.distill_batch_size) > 0 else int(self.cfg.batch_size)
        batch_size = max(1, batch_size)
        steps = max(1, int(self.cfg.distill_steps))

        losses: list[float] = []
        good_counts: list[float] = []
        good_fracs: list[float] = []
        quality_means: list[float] = []
        quality_mins: list[float] = []
        next_dpos_means: list[float] = []
        progress_means: list[float] = []
        advantage_means: list[float] = []
        mean_action_l2s: list[float] = []
        target_action_l2s: list[float] = []
        skipped = 0

        for _ in range(steps):
            sample = self._sample_distill_batch(batch_size)
            if sample is None:
                skipped += 1
                continue
            batch, stats = sample
            if "stats_only" in batch:
                skipped += 1
                good_counts.append(float(stats.get("eligible_count", 0.0)))
                good_fracs.append(float(stats.get("eligible_fraction", 0.0)))
                quality_means.append(float(stats.get("quality_mean", 0.0)))
                continue

            obs = batch["obs"]
            exec_act = batch["exec_act"].detach()
            prev_q_des = batch["prev_q_des"]
            delta_limits = batch["delta_limits"]

            mu, _log_std, _mu_raw = self.actor.forward_components(obs)
            mean_action = torch.tanh(mu) * self.cfg.action_scale
            per_sample_loss = ((mean_action - exec_act) ** 2).mean(dim=-1, keepdim=True)

            weights = torch.ones_like(per_sample_loss)
            advantage = torch.zeros_like(per_sample_loss)
            if float(self.cfg.distill_advantage_beta) > 0.0:
                with torch.no_grad():
                    q_current = obs[:, : self.cfg.action_dim]
                    mean_exec = self._executor_proxy_torch(
                        q_current=q_current,
                        delta_q_cmd=mean_action.detach(),
                        prev_q_des=prev_q_des,
                        delta_limits=delta_limits,
                    )
                    q_target = torch.min(self.q1(obs, exec_act), self.q2(obs, exec_act))
                    q_mean = torch.min(self.q1(obs, mean_exec), self.q2(obs, mean_exec))
                    advantage = torch.clamp(
                        q_target - q_mean,
                        min=-float(self.cfg.distill_advantage_clip),
                        max=float(self.cfg.distill_advantage_clip),
                    )
                    weights = torch.exp(float(self.cfg.distill_advantage_beta) * advantage)
                    weights = weights / (weights.mean() + 1e-6)

            distill_loss = active_lambda * (weights * per_sample_loss).mean()
            self.actor_optim.zero_grad(set_to_none=True)
            distill_loss.backward()
            grad_clip = float(self.cfg.distill_grad_clip)
            if grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), grad_clip)
            self.actor_optim.step()
            self.distill_update_count += 1

            losses.append(float(distill_loss.detach().cpu().item()))
            good_counts.append(float(stats.get("selected_count", 0.0)))
            good_fracs.append(float(stats.get("eligible_fraction", 0.0)))
            quality_means.append(float(stats.get("selected_quality_mean", 0.0)))
            quality_mins.append(float(stats.get("selected_quality_min", 0.0)))
            next_dpos_means.append(float(stats.get("selected_next_dpos_mean", 0.0)))
            progress_means.append(float(stats.get("selected_progress_mean", 0.0)))
            advantage_means.append(float(advantage.mean().detach().cpu().item()))
            mean_action_l2s.append(float(torch.linalg.norm(mean_action, dim=-1).mean().detach().cpu().item()))
            target_action_l2s.append(float(torch.linalg.norm(exec_act, dim=-1).mean().detach().cpu().item()))

        self.distill_skip_count += skipped
        return {
            "distill_enabled": 1.0,
            "distill_triggered": 1.0,
            "distill_active_lambda": active_lambda,
            "distill_stage_name": stage_name,
            "distill_update_count": float(self.distill_update_count),
            "distill_skip_count": float(self.distill_skip_count),
            "distill_loss": float(np.mean(losses)) if losses else 0.0,
            "distill_good_count": float(np.mean(good_counts)) if good_counts else 0.0,
            "distill_good_fraction": float(np.mean(good_fracs)) if good_fracs else 0.0,
            "distill_quality_mean": float(np.mean(quality_means)) if quality_means else 0.0,
            "distill_quality_min": float(np.mean(quality_mins)) if quality_mins else 0.0,
            "distill_next_dpos_mean": float(np.mean(next_dpos_means)) if next_dpos_means else 0.0,
            "distill_progress_mean": float(np.mean(progress_means)) if progress_means else 0.0,
            "distill_advantage_mean": float(np.mean(advantage_means)) if advantage_means else 0.0,
            "distill_mean_action_l2": float(np.mean(mean_action_l2s)) if mean_action_l2s else 0.0,
            "distill_target_action_l2": float(np.mean(target_action_l2s)) if target_action_l2s else 0.0,
            "distill_skipped_steps": float(skipped),
        }

    def train_step(self) -> dict[str, float] | None:
        if len(self.replay) < self.cfg.batch_size:
            return None

        (
            obs,
            raw_act,
            exec_act,
            rew,
            nxt,
            done,
            prev_q_des,
            next_prev_q_des,
            delta_limits,
            delta_norm,
            raw_norm,
            exec_norm,
            clamp_triggered,
            projection_triggered,
            rejected,
            success,
            dwell_count,
        ) = self.replay.sample(self.cfg.batch_size)

        with torch.no_grad():
            next_a, next_logp, _ = self.actor.sample(nxt)
            next_a_scaled = next_a * self.cfg.action_scale
            next_q_current = nxt[:, : self.cfg.action_dim]
            next_exec_a = self._executor_proxy_torch(
                q_current=next_q_current,
                delta_q_cmd=next_a_scaled,
                prev_q_des=next_prev_q_des,
                delta_limits=delta_limits,
            )
            q1_t = self.q1_target(nxt, next_exec_a)
            q2_t = self.q2_target(nxt, next_exec_a)
            q_t = torch.min(q1_t, q2_t) - self.log_alpha.exp() * next_logp
            y = rew + (1.0 - done) * self.cfg.gamma * q_t

        q1 = self.q1(obs, exec_act)
        q2 = self.q2(obs, exec_act)
        critic_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)

        self.q1_optim.zero_grad(set_to_none=True)
        self.q2_optim.zero_grad(set_to_none=True)
        critic_loss.backward()
        critic_grad_norm = float(np.sqrt(self._grad_norm(self.q1) ** 2 + self._grad_norm(self.q2) ** 2))
        self.q1_optim.step()
        self.q2_optim.step()
        self.critic_update_count += 1

        actor_loss_value = 0.0
        actor_loss_sac_value = 0.0
        actor_bc_loss_value = 0.0
        alpha_loss_value = 0.0
        actor_grad_norm = 0.0
        actor_updated = False
        alpha_updated = False
        bc_good_fraction = 0.0
        bc_good_count = 0.0
        distill_metrics: dict[str, float] = {
            "distill_enabled": float(float(getattr(self, "active_distill_lambda", self.cfg.distill_lambda)) > 0.0),
            "distill_triggered": 0.0,
            "distill_active_lambda": float(getattr(self, "active_distill_lambda", self.cfg.distill_lambda)),
            "distill_stage_name": str(getattr(self, "distill_stage_name", "A")),
            "distill_update_count": float(self.distill_update_count),
            "distill_skip_count": float(self.distill_skip_count),
            "distill_loss": 0.0,
            "distill_good_count": 0.0,
            "distill_good_fraction": 0.0,
            "distill_quality_mean": 0.0,
            "distill_quality_min": 0.0,
            "distill_next_dpos_mean": 0.0,
            "distill_progress_mean": 0.0,
            "distill_advantage_mean": 0.0,
            "distill_mean_action_l2": 0.0,
            "distill_target_action_l2": 0.0,
            "distill_skipped_steps": 0.0,
        }
        policy_delay = max(1, int(self.cfg.actor_update_delay))
        if (self.critic_update_count % policy_delay) == 0:
            pi, logp, mu = self.actor.sample(obs)
            pi_scaled = pi * self.cfg.action_scale
            q_current = obs[:, : self.cfg.action_dim]
            pi_exec = self._executor_proxy_torch(
                q_current=q_current,
                delta_q_cmd=pi_scaled,
                prev_q_des=prev_q_des,
                delta_limits=delta_limits,
            )
            q_pi = torch.min(self.q1(obs, pi_exec), self.q2(obs, pi_exec))
            actor_loss_sac = (self.log_alpha.exp().detach() * logp - q_pi).mean()

            # Distill good stochastic executed actions into the deterministic mean policy.
            next_ee_pos_err = nxt[:, self.cfg.action_dim * 2 : self.cfg.action_dim * 2 + 3]
            next_dpos = torch.linalg.norm(next_ee_pos_err, dim=-1)
            good_mask = next_dpos <= float(self.cfg.bc_outer_dpos_m)
            if int(self.cfg.bc_topk) > 0 and next_dpos.numel() > 0:
                k = min(int(self.cfg.bc_topk), int(next_dpos.numel()))
                topk_idx = torch.topk(next_dpos, k=k, largest=False).indices
                good_mask = good_mask.clone()
                good_mask[topk_idx] = True
            mean_action = torch.tanh(mu) * self.cfg.action_scale
            if bool(good_mask.any().detach().cpu().item()):
                bc_loss = F.mse_loss(mean_action[good_mask], exec_act[good_mask].detach())
                bc_good_fraction = float(good_mask.float().mean().detach().cpu().item())
                bc_good_count = float(good_mask.sum().detach().cpu().item())
            else:
                bc_loss = torch.zeros((), dtype=torch.float32, device=self.device)
            actor_loss = actor_loss_sac + (float(self.cfg.bc_lambda) * bc_loss)

            self.actor_optim.zero_grad(set_to_none=True)
            actor_loss.backward()
            if float(self.cfg.actor_grad_clip) > 0.0:
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), float(self.cfg.actor_grad_clip))
            actor_grad_norm = self._grad_norm(self.actor)
            self.actor_optim.step()
            self.actor_update_count += 1
            actor_loss_value = float(actor_loss.detach().cpu().item())
            actor_loss_sac_value = float(actor_loss_sac.detach().cpu().item())
            actor_bc_loss_value = float(bc_loss.detach().cpu().item())
            actor_updated = True

            alpha_loss = -(self.log_alpha * (logp.detach() + self.target_entropy)).mean()
            self.alpha_optim.zero_grad(set_to_none=True)
            alpha_loss.backward()
            self.alpha_optim.step()
            self.alpha_update_count += 1
            alpha_loss_value = float(alpha_loss.detach().cpu().item())
            alpha_updated = True

        distill_metrics.update(self._run_distill_step(self.updates_applied + 1))
        self._soft_update()

        self.updates_applied += 1
        self.batch_draw_count += self.cfg.batch_size
        if self.updates_applied % max(1, int(self.cfg.param_hash_interval)) == 0 or self.last_actor_hash is None:
            self.last_actor_hash = self._param_hash([self.actor])
            self.last_critic_hash = self._param_hash([self.q1, self.q2])

        entropy = float((-logp).mean().detach().cpu().item()) if actor_updated else 0.0
        return {
            "global_step": float(self.env_steps_collected),
            "env_steps_collected": float(self.env_steps_collected),
            "updates_applied": float(self.updates_applied),
            "batch_draw_count": float(self.batch_draw_count),
            "actor_update_count": float(self.actor_update_count),
            "critic_update_count": float(self.critic_update_count),
            "alpha_update_count": float(self.alpha_update_count),
            "gradient_norm_actor": float(actor_grad_norm),
            "gradient_norm_critic": float(critic_grad_norm),
            "actor_updated": float(actor_updated),
            "alpha_updated": float(alpha_updated),
            "actor_update_delay": float(policy_delay),
            "param_hash_actor": self.last_actor_hash,
            "param_hash_critic": self.last_critic_hash,
            "actor_loss": float(actor_loss_value),
            "actor_loss_sac": float(actor_loss_sac_value),
            "actor_bc_loss": float(actor_bc_loss_value),
            "critic_loss": float(critic_loss.detach().cpu().item()),
            "alpha_loss": float(alpha_loss_value),
            "alpha": self.alpha,
            "target_entropy": float(self.target_entropy),
            "entropy": entropy,
            "replay_size": float(len(self.replay)),
            "bc_lambda": float(self.cfg.bc_lambda),
            "bc_good_fraction": float(bc_good_fraction),
            "bc_good_count": float(bc_good_count),
            **distill_metrics,
            "delta_norm_mean": float(delta_norm.mean().detach().cpu().item()),
            "raw_norm_mean": float(raw_norm.mean().detach().cpu().item()),
            "exec_norm_mean": float(exec_norm.mean().detach().cpu().item()),
            "clamp_trigger_rate": float(clamp_triggered.mean().detach().cpu().item()),
            "projection_trigger_rate": float(projection_triggered.mean().detach().cpu().item()),
            "reject_rate": float(rejected.mean().detach().cpu().item()),
        }

    def export_state(self) -> dict[str, Any]:
        return {
            "actor": {k: v.detach().cpu().tolist() for k, v in self.actor.state_dict().items()},
            "q1": {k: v.detach().cpu().tolist() for k, v in self.q1.state_dict().items()},
            "q2": {k: v.detach().cpu().tolist() for k, v in self.q2.state_dict().items()},
            "alpha": self.alpha,
            "target_entropy": float(self.target_entropy),
        }
