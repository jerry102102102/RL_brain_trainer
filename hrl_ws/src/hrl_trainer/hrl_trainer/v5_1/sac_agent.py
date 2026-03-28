"""Legacy NumPy SAC implementation for V5.1.

Kept for backward compatibility with older tests; mainline now uses `sac_torch.py`.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections import deque
from typing import Any

import numpy as np


@dataclass(frozen=True)
class SACConfig:
    obs_dim: int
    action_dim: int
    gamma: float = 0.99
    tau: float = 0.01
    lr_actor: float = 1e-3
    lr_critic: float = 1e-3
    lr_alpha: float = 1e-3
    init_alpha: float = 0.2
    target_entropy: float = -6.0
    replay_capacity: int = 50_000
    batch_size: int = 32
    action_scale: float = 0.05
    log_std_min: float = -2.0
    log_std_max: float = 1.0


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self._buf: deque[tuple[np.ndarray, np.ndarray, float, np.ndarray, float]] = deque(maxlen=capacity)

    def add(self, obs: np.ndarray, action: np.ndarray, reward: float, next_obs: np.ndarray, done: bool) -> None:
        self._buf.append((obs.astype(float), action.astype(float), float(reward), next_obs.astype(float), float(done)))

    def sample(self, batch_size: int, rng: np.random.Generator) -> tuple[np.ndarray, ...]:
        idx = rng.choice(len(self._buf), size=batch_size, replace=False)
        obs, act, rew, nxt, done = zip(*(self._buf[int(i)] for i in idx), strict=False)
        return np.stack(obs), np.stack(act), np.asarray(rew), np.stack(nxt), np.asarray(done)

    def __len__(self) -> int:
        return len(self._buf)


class SACAgent:
    def __init__(self, config: SACConfig, seed: int = 0) -> None:
        self.cfg = config
        self.rng = np.random.default_rng(seed)

        self.actor_w = self.rng.normal(0.0, 0.05, size=(self.cfg.obs_dim, self.cfg.action_dim))
        self.actor_b = np.zeros(self.cfg.action_dim, dtype=float)
        self.actor_log_std = np.full(self.cfg.action_dim, -0.5, dtype=float)

        feat_dim = self.cfg.obs_dim + self.cfg.action_dim
        self.q1_w = self.rng.normal(0.0, 0.05, size=feat_dim)
        self.q1_b = 0.0
        self.q2_w = self.rng.normal(0.0, 0.05, size=feat_dim)
        self.q2_b = 0.0

        self.q1_t_w = self.q1_w.copy()
        self.q1_t_b = self.q1_b
        self.q2_t_w = self.q2_w.copy()
        self.q2_t_b = self.q2_b

        self.log_alpha = float(np.log(self.cfg.init_alpha))
        self.replay = ReplayBuffer(self.cfg.replay_capacity)

    @property
    def alpha(self) -> float:
        return float(np.exp(self.log_alpha))

    def _policy_dist(self, obs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mu = obs @ self.actor_w + self.actor_b
        log_std = np.clip(self.actor_log_std, self.cfg.log_std_min, self.cfg.log_std_max)
        std = np.exp(log_std)
        return mu, std

    def act(self, obs: np.ndarray, stochastic: bool = True) -> np.ndarray:
        obs = np.asarray(obs, dtype=float)
        mu, std = self._policy_dist(obs)
        if stochastic:
            eps = self.rng.normal(size=self.cfg.action_dim)
            raw = mu + std * eps
        else:
            raw = mu
        return np.clip(raw, -1.0, 1.0) * self.cfg.action_scale

    def remember(self, obs: np.ndarray, action: np.ndarray, reward: float, next_obs: np.ndarray, done: bool) -> None:
        self.replay.add(obs, action, reward, next_obs, done)

    def _q(self, obs: np.ndarray, action: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
        x = np.concatenate([obs, action], axis=-1)
        return x @ w + b

    def train_step(self) -> dict[str, float] | None:
        if len(self.replay) < self.cfg.batch_size:
            return None

        obs, act, rew, nxt, done = self.replay.sample(self.cfg.batch_size, self.rng)

        # Critic target
        mu_n, std_n = self._policy_dist(nxt)
        eps_n = self.rng.normal(size=mu_n.shape)
        next_a = np.clip(mu_n + std_n * eps_n, -1.0, 1.0) * self.cfg.action_scale
        log_prob_n = -0.5 * np.sum(eps_n**2 + 2.0 * np.log(std_n) + np.log(2.0 * np.pi), axis=1)

        q1_t = self._q(nxt, next_a, self.q1_t_w, self.q1_t_b)
        q2_t = self._q(nxt, next_a, self.q2_t_w, self.q2_t_b)
        target_v = np.minimum(q1_t, q2_t) - self.alpha * log_prob_n
        y = rew + (1.0 - done) * self.cfg.gamma * target_v

        # Critic update (MSE)
        x = np.concatenate([obs, act], axis=1)
        q1 = x @ self.q1_w + self.q1_b
        q2 = x @ self.q2_w + self.q2_b
        err1 = q1 - y
        err2 = q2 - y
        critic_loss = float(0.5 * (np.mean(err1**2) + np.mean(err2**2)))

        grad_q1_w = (x.T @ err1) / len(obs)
        grad_q1_b = float(np.mean(err1))
        grad_q2_w = (x.T @ err2) / len(obs)
        grad_q2_b = float(np.mean(err2))

        self.q1_w -= self.cfg.lr_critic * grad_q1_w
        self.q1_b -= self.cfg.lr_critic * grad_q1_b
        self.q2_w -= self.cfg.lr_critic * grad_q2_w
        self.q2_b -= self.cfg.lr_critic * grad_q2_b

        # Actor update (reparameterized, linear actor)
        mu, std = self._policy_dist(obs)
        eps = self.rng.normal(size=mu.shape)
        act_sample = np.clip(mu + std * eps, -1.0, 1.0) * self.cfg.action_scale
        log_prob = -0.5 * np.sum(eps**2 + 2.0 * np.log(std) + np.log(2.0 * np.pi), axis=1)

        q1_pi = self._q(obs, act_sample, self.q1_w, self.q1_b)
        q2_pi = self._q(obs, act_sample, self.q2_w, self.q2_b)
        q_pi = 0.5 * (q1_pi + q2_pi)
        actor_loss = float(np.mean(self.alpha * log_prob - q_pi))

        w1_a = self.q1_w[self.cfg.obs_dim :]
        w2_a = self.q2_w[self.cfg.obs_dim :]
        dq_da = 0.5 * (w1_a + w2_a)
        dqda_batch = np.broadcast_to(dq_da, mu.shape)

        # dL/dmu = -dQ/da * da/dmu (ignoring clip saturation for smoke)
        grad_mu = -dqda_batch * self.cfg.action_scale
        grad_w = (obs.T @ grad_mu) / len(obs)
        grad_b = np.mean(grad_mu, axis=0)

        # dL/dlog_std = alpha * dlogpi/dlogstd - dQ/da * da/dlogstd
        # with reparam: dlogpi/dlogstd = -1
        grad_log_std = self.alpha * (-1.0) - np.mean(dqda_batch * (std * eps) * self.cfg.action_scale, axis=0)

        self.actor_w -= self.cfg.lr_actor * grad_w
        self.actor_b -= self.cfg.lr_actor * grad_b
        self.actor_log_std = np.clip(
            self.actor_log_std - self.cfg.lr_actor * grad_log_std,
            self.cfg.log_std_min,
            self.cfg.log_std_max,
        )

        # Alpha / temperature update
        entropy = float(-np.mean(log_prob))
        alpha_loss = -self.alpha * (entropy - self.cfg.target_entropy)
        grad_log_alpha = -self.alpha * (entropy - self.cfg.target_entropy)
        self.log_alpha -= self.cfg.lr_alpha * grad_log_alpha

        # Target update
        tau = self.cfg.tau
        self.q1_t_w = (1.0 - tau) * self.q1_t_w + tau * self.q1_w
        self.q1_t_b = (1.0 - tau) * self.q1_t_b + tau * self.q1_b
        self.q2_t_w = (1.0 - tau) * self.q2_t_w + tau * self.q2_w
        self.q2_t_b = (1.0 - tau) * self.q2_t_b + tau * self.q2_b

        return {
            "actor_loss": actor_loss,
            "critic_loss": critic_loss,
            "alpha": self.alpha,
            "alpha_loss": float(alpha_loss),
            "entropy": entropy,
            "replay_size": float(len(self.replay)),
        }

    def export_state(self) -> dict[str, Any]:
        return {
            "actor_w": self.actor_w.tolist(),
            "actor_b": self.actor_b.tolist(),
            "actor_log_std": self.actor_log_std.tolist(),
            "q1_w": self.q1_w.tolist(),
            "q2_w": self.q2_w.tolist(),
            "alpha": self.alpha,
        }
