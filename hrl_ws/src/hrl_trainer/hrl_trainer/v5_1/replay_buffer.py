from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TransitionBatch:
    obs: np.ndarray
    act: np.ndarray
    rew: np.ndarray
    next_obs: np.ndarray
    done: np.ndarray
    truncated: np.ndarray


class ReplayBuffer:
    def __init__(self, obs_dim: int, action_dim: int, capacity: int = 200_000, n_step: int = 1):
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.act = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rew = np.zeros((capacity, 1), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.done = np.zeros((capacity, 1), dtype=np.float32)
        self.truncated = np.zeros((capacity, 1), dtype=np.float32)
        self.info: list[dict[str, object]] = [{} for _ in range(capacity)]
        self.capacity = int(capacity)
        self.n_step = int(n_step)
        self.ptr = 0
        self.size = 0

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
        truncated: bool,
        info: Mapping[str, object] | None = None,
    ) -> None:
        i = self.ptr
        self.obs[i] = np.asarray(obs, dtype=np.float32)
        self.act[i] = np.asarray(action, dtype=np.float32)
        self.rew[i, 0] = float(reward)
        self.next_obs[i] = np.asarray(next_obs, dtype=np.float32)
        self.done[i, 0] = float(done)
        self.truncated[i, 0] = float(truncated)
        self.info[i] = dict(info or {})
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, rng: np.random.Generator | None = None) -> TransitionBatch:
        if self.size <= 0:
            raise ValueError("ReplayBuffer is empty")
        random = rng or np.random.default_rng()
        idx = random.integers(0, self.size, size=int(batch_size))
        return TransitionBatch(
            obs=self.obs[idx],
            act=self.act[idx],
            rew=self.rew[idx],
            next_obs=self.next_obs[idx],
            done=self.done[idx],
            truncated=self.truncated[idx],
        )

    def __len__(self) -> int:
        return self.size
