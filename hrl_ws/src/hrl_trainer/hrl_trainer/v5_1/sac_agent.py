from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle

import numpy as np

from .replay_buffer import ReplayBuffer


@dataclass(frozen=True)
class SACConfig:
    obs_dim: int
    action_dim: int
    hidden_dim: int = 128
    gamma: float = 0.99
    tau: float = 0.005
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    batch_size: int = 256
    replay_capacity: int = 200_000
    warmup_steps: int = 5000
    updates_per_step: float = 1.0
    gradient_clip_norm: float = 5.0
    target_entropy: float | None = None
    device: str = "cpu"


class SACAgent:
    """Numpy SAC-like core for environments without torch.

    Keeps actor + twin-Q + alpha autotune interfaces and artifacts stable.
    """

    def __init__(self, config: SACConfig, seed: int = 0):
        self.cfg = config
        self.rng = np.random.default_rng(seed)
        self.actor_w = self.rng.normal(0.0, 0.05, size=(config.action_dim, config.obs_dim)).astype(np.float32)
        self.q1_w = self.rng.normal(0.0, 0.05, size=(config.obs_dim + config.action_dim,)).astype(np.float32)
        self.q2_w = self.rng.normal(0.0, 0.05, size=(config.obs_dim + config.action_dim,)).astype(np.float32)
        self.q1_t = self.q1_w.copy()
        self.q2_t = self.q2_w.copy()
        self.log_alpha = 0.0
        self.target_entropy = -float(config.action_dim) if config.target_entropy is None else float(config.target_entropy)

        self.replay = ReplayBuffer(config.obs_dim, config.action_dim, capacity=config.replay_capacity)
        self._update_budget_carry = 0.0

    @property
    def alpha(self) -> float:
        return float(np.exp(self.log_alpha))

    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        obs_v = np.asarray(obs, dtype=np.float32)
        mu = self.actor_w @ obs_v
        if deterministic:
            return np.tanh(mu).astype(np.float32)
        noise = self.rng.normal(0.0, 0.15, size=mu.shape).astype(np.float32)
        return np.tanh(mu + noise).astype(np.float32)

    def remember(self, obs: np.ndarray, action: np.ndarray, reward: float, next_obs: np.ndarray, done: bool, truncated: bool, info: dict[str, object] | None = None) -> None:
        self.replay.add(obs, action, reward, next_obs, done, truncated, info)

    def _soft_update(self) -> None:
        self.q1_t = (1.0 - self.cfg.tau) * self.q1_t + self.cfg.tau * self.q1_w
        self.q2_t = (1.0 - self.cfg.tau) * self.q2_t + self.cfg.tau * self.q2_w

    def _q(self, w: np.ndarray, obs: np.ndarray, act: np.ndarray) -> np.ndarray:
        sa = np.concatenate([obs, act], axis=1)
        return (sa * w[None, :]).sum(axis=1, keepdims=True)

    def _run_single_update(self) -> dict[str, float] | None:
        if len(self.replay) < self.cfg.batch_size:
            return None
        b = self.replay.sample(self.cfg.batch_size, rng=self.rng)

        pi_next = np.tanh((b.next_obs @ self.actor_w.T)).astype(np.float32)
        q1_t = self._q(self.q1_t, b.next_obs, pi_next)
        q2_t = self._q(self.q2_t, b.next_obs, pi_next)
        q_tgt = np.minimum(q1_t, q2_t)
        entropy_proxy = np.mean(np.abs(pi_next), axis=1, keepdims=True)
        y = b.rew + (1.0 - b.done) * self.cfg.gamma * (q_tgt - self.alpha * entropy_proxy)

        q1 = self._q(self.q1_w, b.obs, b.act)
        q2 = self._q(self.q2_w, b.obs, b.act)
        err1 = q1 - y
        err2 = q2 - y
        q1_loss = float(np.mean(err1**2))
        q2_loss = float(np.mean(err2**2))

        sa = np.concatenate([b.obs, b.act], axis=1)
        grad_q1 = np.mean(2.0 * err1 * sa, axis=0)
        grad_q2 = np.mean(2.0 * err2 * sa, axis=0)
        self.q1_w -= self.cfg.critic_lr * np.clip(grad_q1, -self.cfg.gradient_clip_norm, self.cfg.gradient_clip_norm)
        self.q2_w -= self.cfg.critic_lr * np.clip(grad_q2, -self.cfg.gradient_clip_norm, self.cfg.gradient_clip_norm)

        pi = np.tanh((b.obs @ self.actor_w.T)).astype(np.float32)
        q_pi = np.minimum(self._q(self.q1_w, b.obs, pi), self._q(self.q2_w, b.obs, pi))
        actor_loss = float(np.mean(self.alpha * np.abs(pi) - q_pi))
        actor_grad = np.outer(np.mean((self.alpha * np.sign(pi) - 0.1), axis=0), np.mean(b.obs, axis=0))
        self.actor_w -= self.cfg.actor_lr * np.clip(actor_grad, -self.cfg.gradient_clip_norm, self.cfg.gradient_clip_norm)

        entropy = float(np.mean(np.abs(pi)))
        alpha_loss = float(-self.log_alpha * (entropy + self.target_entropy))
        self.log_alpha -= self.cfg.alpha_lr * np.clip(-(entropy + self.target_entropy), -5.0, 5.0)

        self._soft_update()
        return {
            "critic_loss_1": q1_loss,
            "critic_loss_2": q2_loss,
            "actor_loss": actor_loss,
            "alpha": self.alpha,
            "alpha_loss": alpha_loss,
            "q_target_mean": float(np.mean(y)),
            "q_target_std": float(np.std(y)),
            "replay_size": float(len(self.replay)),
        }

    def update(self) -> list[dict[str, float]]:
        self._update_budget_carry += float(self.cfg.updates_per_step)
        n_updates = int(self._update_budget_carry)
        self._update_budget_carry -= n_updates
        n_updates = max(1, n_updates)
        out: list[dict[str, float]] = []
        for _ in range(n_updates):
            row = self._run_single_update()
            if row is not None:
                out.append(row)
        return out

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": self.cfg.__dict__,
            "actor_w": self.actor_w,
            "q1_w": self.q1_w,
            "q2_w": self.q2_w,
            "q1_t": self.q1_t,
            "q2_t": self.q2_t,
            "log_alpha": self.log_alpha,
        }
        with p.open("wb") as f:
            pickle.dump(payload, f)

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> "SACAgent":
        with Path(path).open("rb") as f:
            payload = pickle.load(f)
        cfg = SACConfig(**payload["config"])
        cfg = SACConfig(**{**cfg.__dict__, "device": device})
        agent = cls(cfg)
        agent.actor_w = np.asarray(payload["actor_w"], dtype=np.float32)
        agent.q1_w = np.asarray(payload["q1_w"], dtype=np.float32)
        agent.q2_w = np.asarray(payload["q2_w"], dtype=np.float32)
        agent.q1_t = np.asarray(payload["q1_t"], dtype=np.float32)
        agent.q2_t = np.asarray(payload["q2_t"], dtype=np.float32)
        agent.log_alpha = float(payload.get("log_alpha", 0.0))
        return agent
