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
    lr_actor: float = 3e-4
    lr_critic: float = 3e-4
    lr_alpha: float = 3e-4
    init_alpha: float = 0.2
    target_entropy: float | None = None
    replay_capacity: int = 50_000
    batch_size: int = 32
    action_scale: float = 0.05
    log_std_min: float = -5.0
    log_std_max: float = 2.0
    device: str = "cpu"
    param_hash_interval: int = 10


class TorchReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int, action_dim: int, device: torch.device) -> None:
        self.capacity = int(capacity)
        self.device = device
        self.obs = torch.zeros((capacity, obs_dim), dtype=torch.float32, device=device)
        self.actions = torch.zeros((capacity, action_dim), dtype=torch.float32, device=device)
        self.rewards = torch.zeros((capacity, 1), dtype=torch.float32, device=device)
        self.next_obs = torch.zeros((capacity, obs_dim), dtype=torch.float32, device=device)
        self.dones = torch.zeros((capacity, 1), dtype=torch.float32, device=device)
        self.ptr = 0
        self.size = 0

    def add(self, obs: np.ndarray, action: np.ndarray, reward: float, next_obs: np.ndarray, done: bool) -> None:
        i = self.ptr
        self.obs[i] = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        self.actions[i] = torch.as_tensor(action, dtype=torch.float32, device=self.device)
        self.rewards[i] = float(reward)
        self.next_obs[i] = torch.as_tensor(next_obs, dtype=torch.float32, device=self.device)
        self.dones[i] = float(done)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> tuple[torch.Tensor, ...]:
        idx = torch.randint(0, self.size, (batch_size,), device=self.device)
        return self.obs[idx], self.actions[idx], self.rewards[idx], self.next_obs[idx], self.dones[idx]

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
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int, log_std_min: float, log_std_max: float) -> None:
        super().__init__()
        self.backbone = MLP(obs_dim, hidden_dim, hidden_dim)
        self.mu = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.backbone(obs)
        mu = self.mu(h)
        log_std = torch.clamp(self.log_std(h), self.log_std_min, self.log_std_max)
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
        self.last_actor_hash: str | None = None
        self.last_critic_hash: str | None = None

    @property
    def alpha(self) -> float:
        return float(self.log_alpha.detach().exp().cpu().item())

    def act(self, obs: np.ndarray, stochastic: bool = True) -> np.ndarray:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            a, _, _ = self.actor.sample(obs_t, deterministic=not stochastic)
            a = a.squeeze(0) * self.cfg.action_scale
        return a.cpu().numpy()

    def remember(self, obs: np.ndarray, action: np.ndarray, reward: float, next_obs: np.ndarray, done: bool) -> None:
        self.replay.add(obs, action, reward, next_obs, done)
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

    def train_step(self) -> dict[str, float] | None:
        if len(self.replay) < self.cfg.batch_size:
            return None

        obs, act, rew, nxt, done = self.replay.sample(self.cfg.batch_size)

        with torch.no_grad():
            next_a, next_logp, _ = self.actor.sample(nxt)
            q1_t = self.q1_target(nxt, next_a)
            q2_t = self.q2_target(nxt, next_a)
            q_t = torch.min(q1_t, q2_t) - self.log_alpha.exp() * next_logp
            y = rew + (1.0 - done) * self.cfg.gamma * q_t

        q1 = self.q1(obs, act)
        q2 = self.q2(obs, act)
        critic_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)

        self.q1_optim.zero_grad(set_to_none=True)
        self.q2_optim.zero_grad(set_to_none=True)
        critic_loss.backward()
        critic_grad_norm = float(np.sqrt(self._grad_norm(self.q1) ** 2 + self._grad_norm(self.q2) ** 2))
        self.q1_optim.step()
        self.q2_optim.step()
        self.critic_update_count += 1

        pi, logp, _ = self.actor.sample(obs)
        q_pi = torch.min(self.q1(obs, pi), self.q2(obs, pi))
        actor_loss = (self.log_alpha.exp().detach() * logp - q_pi).mean()

        self.actor_optim.zero_grad(set_to_none=True)
        actor_loss.backward()
        actor_grad_norm = self._grad_norm(self.actor)
        self.actor_optim.step()
        self.actor_update_count += 1

        alpha_loss = -(self.log_alpha * (logp.detach() + self.target_entropy)).mean()
        self.alpha_optim.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_optim.step()
        self.alpha_update_count += 1

        self._soft_update()

        self.updates_applied += 1
        self.batch_draw_count += self.cfg.batch_size
        if self.updates_applied % max(1, int(self.cfg.param_hash_interval)) == 0 or self.last_actor_hash is None:
            self.last_actor_hash = self._param_hash([self.actor])
            self.last_critic_hash = self._param_hash([self.q1, self.q2])

        entropy = float((-logp).mean().detach().cpu().item())
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
            "param_hash_actor": self.last_actor_hash,
            "param_hash_critic": self.last_critic_hash,
            "actor_loss": float(actor_loss.detach().cpu().item()),
            "critic_loss": float(critic_loss.detach().cpu().item()),
            "alpha": self.alpha,
            "entropy": entropy,
            "replay_size": float(len(self.replay)),
        }

    def export_state(self) -> dict[str, Any]:
        return {
            "actor": {k: v.detach().cpu().tolist() for k, v in self.actor.state_dict().items()},
            "q1": {k: v.detach().cpu().tolist() for k, v in self.q1.state_dict().items()},
            "q2": {k: v.detach().cpu().tolist() for k, v in self.q2.state_dict().items()},
            "alpha": self.alpha,
        }
