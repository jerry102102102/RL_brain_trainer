r"""Hierarchical reinforcement learning controller module (v2).

This module implements a hierarchical controller with:

* A Deep Q-Network concept layer for discrete option selection.
* Dynamic Movement Primitives (DMPs) following ``\tau \dot{s} = -\alpha_s s``
  (``s(0)=1``) and joint dynamics ``\tau \dot{v} = \alpha_z (\beta_z (g-x) - v)
  + (g-x_0) f(s)`` and ``\tau \dot{x} = v`` with forcing term
  ``f(s) = (\sum_i \psi_i(s) w_i) s / (\sum_i \psi_i(s) + \varepsilon)``.
* A Control Barrier Function (CBF) safety layer enforcing
  ``\nabla h(x)^\top (f(x) + g(x) u) + \alpha h(x) \ge 0`` via quadratic
  programs with per-constraint slack penalties.
"""

from __future__ import annotations

import json
import logging
import math
import random
import time
from collections import deque
from dataclasses import asdict, dataclass
from typing import Callable, Deque, Dict, List, Literal, Optional, Protocol, Tuple

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover - required dependency
    raise ImportError("numpy is required for hrl_control.py") from exc

try:
    import torch
    from torch import nn, optim
except ImportError as exc:  # pragma: no cover - required dependency
    raise ImportError("PyTorch >= 2.0 is required for hrl_control.py") from exc

try:
    from scipy import sparse
except ImportError as exc:  # pragma: no cover - required dependency
    raise ImportError("scipy is required for hrl_control.py") from exc

try:  # pragma: no cover - optional dependency
    import osqp  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    osqp = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from qpsolvers import solve_qp  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    solve_qp = None  # type: ignore

EPS = 1e-8
WARNING_INTERVAL_S = 5.0
DEFAULT_HIDDEN_DIM = 128
DEFAULT_BACKOFF_FACTOR = 0.8

logger = logging.getLogger(__name__)


class EnvProtocol(Protocol):
    """Minimal interface for simulation environments."""

    def reset(self) -> np.ndarray:
        ...

    def step(self, option_id: int) -> Tuple[np.ndarray, float, bool, dict]:
        ...

    @property
    def state_dim(self) -> int:
        ...

    @property
    def n_joints(self) -> int:
        ...

    @property
    def goal(self) -> np.ndarray:
        ...


class ReplayBuffer:
    """Experience replay storage for transitions ``(s, a, r, s', done)``."""

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.buffer: Deque[Tuple[np.ndarray, int, float, np.ndarray, bool]] = deque(
            maxlen=capacity
        )

    def __len__(self) -> int:
        return len(self.buffer)

    def push(self, transition: Tuple[np.ndarray, int, float, np.ndarray, bool]) -> None:
        self.buffer.append(transition)

    def sample(
        self, batch_size: int
    ) -> List[Tuple[np.ndarray, int, float, np.ndarray, bool]]:
        batch_size = min(batch_size, len(self.buffer))
        return random.sample(self.buffer, batch_size)


class StateNormalizer:
    """Online mean-variance normalizer for state vectors."""

    def __init__(self, dimension: int) -> None:
        self.dimension = dimension
        self.reset()

    def reset(self) -> None:
        self.count = 0
        self.mean = np.zeros(self.dimension, dtype=np.float64)
        self.m2 = np.zeros(self.dimension, dtype=np.float64)

    def update(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float64)
        if x.shape[-1] != self.dimension:
            raise ValueError("State dimension mismatch for normalizer update.")
        self.count += 1
        delta = x - self.mean
        self.mean += delta / self.count
        delta2 = x - self.mean
        self.m2 += delta * delta2

    def variance(self) -> np.ndarray:
        if self.count < 2:
            return np.ones(self.dimension, dtype=np.float64)
        return self.m2 / max(self.count - 1, 1)

    def normalize(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if self.count < 2:
            return x.astype(np.float32)
        var = self.variance()
        return ((x - self.mean) / np.sqrt(var + EPS)).astype(np.float32)

    def state_dict(self) -> Dict[str, np.ndarray | int]:
        return {"count": self.count, "mean": self.mean.copy(), "m2": self.m2.copy()}

    def load_state_dict(self, state: Dict[str, np.ndarray | int]) -> None:
        self.count = int(state.get("count", 0))
        self.mean = np.array(state.get("mean", np.zeros(self.dimension)), dtype=np.float64)
        self.m2 = np.array(state.get("m2", np.zeros(self.dimension)), dtype=np.float64)


@dataclass
class OptionSpec:
    """Description of a discrete macro-action."""

    name: str
    goal_offset_scale: float = 1.0
    duration_scale: float = 1.0
    dmp_basis_id: int = 0
    safety_profile: str = "default"


def export_config(cfg: "HRLConfig") -> Dict[str, object]:
    """Return a serializable dictionary for ``cfg``."""

    return json.loads(json.dumps(asdict(cfg)))


def import_config(data: Dict[str, object]) -> "HRLConfig":
    """Construct :class:`HRLConfig` from a dictionary."""

    return HRLConfig(**data)


class QNetwork(nn.Module):
    """Feed-forward critic mapping states to option values."""

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = DEFAULT_HIDDEN_DIM) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - wrapper
        return self.net(x)


@dataclass
class HRLConfig:
    """Configuration for :class:`HierarchicalRLController`."""

    n_joints: int
    state_dim: int
    options: List[str]
    dmp_n_basis: int = 15
    dmp_alpha_s: float = 4.0
    dmp_alpha_z: float = 25.0
    dmp_beta_z: float = 6.25
    horizon_steps: int = 10
    dt: float = 0.05
    gamma: float = 0.99
    lr: float = 3e-4
    batch_size: int = 128
    replay_capacity: int = 100_000
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_steps: int = 50_000
    tau: float = 0.01
    cbf_alpha: float = 5.0
    qp_R_diag: float = 1.0
    max_joint_vel: float = 1.0
    use_slack: bool = True
    device: Optional[str] = None
    seed: Optional[int] = 0
    double_dqn: bool = True
    target_update: Literal["soft", "hard"] = "soft"
    target_update_interval: int = 1000
    state_norm: bool = True
    dmp_basis_overlap: float = 0.5
    qp_slack_all: bool = True
    qp_slack_weight_cbf: float = 1e3
    qp_slack_weight_bounds: float = 1e2
    goal_in_state: bool = True


class HierarchicalRLController:
    """Controller stacking DQN, DMP, and CBF-QP layers."""

    def __init__(self, cfg: HRLConfig) -> None:
        self.cfg = cfg
        self._seed_rng(cfg.seed)
        self.device = self._select_device(cfg.device)
        augmented_dim = cfg.state_dim + (2 * cfg.n_joints if cfg.goal_in_state else 0)
        self.obs_dim = augmented_dim
        self.policy_net = QNetwork(augmented_dim, len(cfg.options)).to(self.device)
        self.target_net = QNetwork(augmented_dim, len(cfg.options)).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=cfg.lr)
        self.replay_buffer = ReplayBuffer(cfg.replay_capacity)
        self.state_normalizer = StateNormalizer(augmented_dim)
        self.eps = cfg.eps_start
        self.total_env_steps = 0
        self.training_steps = 0
        self.options = self._build_option_specs(cfg.options)
        self.option_name_to_id = {spec.name: idx for idx, spec in enumerate(self.options)}
        self.dmp_centers, self.dmp_widths = self._init_dmp_basis()
        self.dmp_weights = np.zeros((cfg.n_joints, cfg.dmp_n_basis), dtype=np.float64)
        self.joint_lower = -np.pi * np.ones(cfg.n_joints, dtype=np.float64)
        self.joint_upper = np.pi * np.ones(cfg.n_joints, dtype=np.float64)
        self._goal = np.zeros(cfg.n_joints, dtype=np.float64)
        self._joint_barriers = self._build_joint_limit_barriers()
        self._custom_barriers: List[
            Tuple[str, Callable[[np.ndarray], Tuple[float, np.ndarray]]]
        ] = []
        self._last_warning_time = -math.inf

    @staticmethod
    def _select_device(device_str: Optional[str]) -> torch.device:
        if device_str == "cpu":
            return torch.device("cpu")
        if device_str == "cuda":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def _seed_rng(seed: Optional[int]) -> None:
        if seed is None:
            return
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _build_option_specs(self, names: List[str]) -> List[OptionSpec]:
        specs: List[OptionSpec] = []
        defaults: Dict[str, OptionSpec] = {
            "hold": OptionSpec(name="hold", goal_offset_scale=0.0),
            "dmp_small_step": OptionSpec(name="dmp_small_step", goal_offset_scale=0.25),
            "dmp_medium_step": OptionSpec(name="dmp_medium_step", goal_offset_scale=0.5),
            "dmp_large_step": OptionSpec(name="dmp_large_step", goal_offset_scale=1.0),
        }
        for name in names:
            specs.append(defaults.get(name, OptionSpec(name=name)))
        return specs

    def _init_dmp_basis(self) -> Tuple[np.ndarray, np.ndarray]:
        centers = np.linspace(1e-4, 1.0, self.cfg.dmp_n_basis, dtype=np.float64)[::-1]
        widths = np.zeros_like(centers)
        for idx in range(self.cfg.dmp_n_basis):
            if idx == self.cfg.dmp_n_basis - 1:
                delta = centers[idx - 1] - centers[idx] if idx > 0 else 1.0
            else:
                delta = centers[idx] - centers[idx + 1]
            delta = abs(delta) + EPS
            widths[idx] = 1.0 / (self.cfg.dmp_basis_overlap * delta**2)
        return centers, widths

    def _build_joint_limit_barriers(self) -> List[Callable[[np.ndarray], Tuple[float, np.ndarray]]]:
        barriers: List[Callable[[np.ndarray], Tuple[float, np.ndarray]]] = []
        for idx in range(self.cfg.n_joints):
            upper = self.joint_upper[idx]
            lower = self.joint_lower[idx]

            def upper_barrier(
                q: np.ndarray, i: int = idx, limit: float = upper
            ) -> Tuple[float, np.ndarray]:
                grad = np.zeros_like(q)
                grad[i] = -1.0
                return limit - q[i], grad

            def lower_barrier(
                q: np.ndarray, i: int = idx, limit: float = lower
            ) -> Tuple[float, np.ndarray]:
                grad = np.zeros_like(q)
                grad[i] = 1.0
                return q[i] - limit, grad

            barriers.append(upper_barrier)
            barriers.append(lower_barrier)
        return barriers

    def set_joint_limits(self, lower: np.ndarray, upper: np.ndarray) -> None:
        if lower.shape[0] != self.cfg.n_joints or upper.shape[0] != self.cfg.n_joints:
            raise ValueError("Joint limits must match number of joints.")
        if np.any(lower >= upper):
            raise ValueError("Lower joint limits must be strictly less than upper limits.")
        self.joint_lower = lower.astype(np.float64)
        self.joint_upper = upper.astype(np.float64)
        self._joint_barriers = self._build_joint_limit_barriers()

    def set_goal(self, g: np.ndarray) -> None:
        if g.shape[0] != self.cfg.n_joints:
            raise ValueError("Goal dimension mismatch.")
        self._goal = g.astype(np.float64)

    @property
    def goal(self) -> np.ndarray:
        return self._goal.copy()

    def reset_norm(self) -> None:
        self.state_normalizer.reset()

    def add_barrier(
        self,
        fn: Callable[[np.ndarray], Tuple[float, np.ndarray]],
        name: str = "",
    ) -> None:
        self._custom_barriers.append((name, fn))

    def clear_barriers(self) -> None:
        self._custom_barriers.clear()

    def _augment_state(self, state: np.ndarray) -> np.ndarray:
        state = np.asarray(state, dtype=np.float64)
        if self.cfg.goal_in_state:
            if state.shape[0] < self.cfg.n_joints:
                raise ValueError("State lacks joint positions required for augmentation.")
            q = state[: self.cfg.n_joints]
            error = self._goal - q
            return np.concatenate([state, error, self._goal])
        return state

    def _update_normalizer(self, state: np.ndarray) -> None:
        if self.cfg.state_norm:
            self.state_normalizer.update(state)

    def _process_state(self, state: np.ndarray, update_stats: bool = False) -> np.ndarray:
        augmented = self._augment_state(state)
        if update_stats:
            self._update_normalizer(augmented)
        if self.cfg.state_norm:
            processed = self.state_normalizer.normalize(augmented)
        else:
            processed = augmented.astype(np.float32)
        return processed

    def select_option(self, state_np: np.ndarray, explore: bool = True) -> int:
        processed = self._process_state(state_np, update_stats=False)
        state_tensor = torch.from_numpy(processed).to(self.device).unsqueeze(0)
        if explore and random.random() < self.eps:
            return random.randrange(len(self.options))
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
            return int(torch.argmax(q_values, dim=1).item())

    def push_transition(
        self, s: np.ndarray, a: int, r: float, s2: np.ndarray, done: bool
    ) -> None:
        state_aug = self._augment_state(s)
        next_state_aug = self._augment_state(s2)
        self._update_normalizer(state_aug)
        self._update_normalizer(next_state_aug)
        self.replay_buffer.push(
            (
                state_aug.astype(np.float32),
                int(a),
                float(r),
                next_state_aug.astype(np.float32),
                bool(done),
            )
        )

    def _soft_update(self, tau: float) -> None:
        for target_param, param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

    def _update_epsilon(self) -> None:
        fraction = min(1.0, self.total_env_steps / max(1, self.cfg.eps_decay_steps))
        self.eps = self.cfg.eps_start + fraction * (self.cfg.eps_end - self.cfg.eps_start)

    def train_step(self) -> Dict[str, float]:
        if len(self.replay_buffer) < self.cfg.batch_size:
            return {}
        batch = self.replay_buffer.sample(self.cfg.batch_size)
        states = np.stack([b[0] for b in batch], axis=0).astype(np.float64)
        next_states = np.stack([b[3] for b in batch], axis=0).astype(np.float64)
        if self.cfg.state_norm:
            mean = self.state_normalizer.mean
            var = self.state_normalizer.variance()
            states = ((states - mean) / np.sqrt(var + EPS)).astype(np.float32)
            next_states = ((next_states - mean) / np.sqrt(var + EPS)).astype(np.float32)
        else:
            states = states.astype(np.float32)
            next_states = next_states.astype(np.float32)
        actions = torch.tensor([b[1] for b in batch], dtype=torch.long, device=self.device)
        rewards = torch.tensor([b[2] for b in batch], dtype=torch.float32, device=self.device)
        dones = torch.tensor([float(b[4]) for b in batch], dtype=torch.float32, device=self.device)
        states_tensor = torch.from_numpy(states).to(self.device)
        next_states_tensor = torch.from_numpy(next_states).to(self.device)
        q_values = self.policy_net(states_tensor).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            if self.cfg.double_dqn:
                next_actions = self.policy_net(next_states_tensor).argmax(dim=1)
                next_q = self.target_net(next_states_tensor).gather(
                    1, next_actions.unsqueeze(1)
                ).squeeze(1)
            else:
                next_q = self.target_net(next_states_tensor).max(dim=1)[0]
            targets = rewards + self.cfg.gamma * (1.0 - dones) * next_q
        loss = nn.functional.mse_loss(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.training_steps += 1
        if self.cfg.target_update == "soft":
            self._soft_update(self.cfg.tau)
        elif self.training_steps % max(1, self.cfg.target_update_interval) == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        return {"loss": float(loss.item())}

    def save_policy(self, path: str) -> None:
        payload = {
            "policy": self.policy_net.state_dict(),
            "target": self.target_net.state_dict(),
            "config": export_config(self.cfg),
            "normalizer": self.state_normalizer.state_dict() if self.cfg.state_norm else None,
            "options": [asdict(spec) for spec in self.options],
            "dmp": {
                "weights": self.dmp_weights,
                "centers": self.dmp_centers,
                "widths": self.dmp_widths,
            },
            "goal": self._goal,
            "joint_limits": {"lower": self.joint_lower, "upper": self.joint_upper},
        }
        torch.save(payload, path)

    def load_policy(self, path: str) -> None:
        data = torch.load(path, map_location=self.device)
        if "policy" not in data:
            self.policy_net.load_state_dict(data)
            self.target_net.load_state_dict(data)
            return
        self.policy_net.load_state_dict(data["policy"])
        self.target_net.load_state_dict(data.get("target", data["policy"]))
        if self.cfg.state_norm and data.get("normalizer") is not None:
            self.state_normalizer.load_state_dict(data["normalizer"])
        if "options" in data:
            loaded_specs = [OptionSpec(**spec) for spec in data["options"]]
            if len(loaded_specs) != len(self.options):
                raise ValueError("Loaded options mismatch current controller configuration.")
            self.options = loaded_specs
            self.option_name_to_id = {spec.name: idx for idx, spec in enumerate(self.options)}
        if "dmp" in data:
            dmp_data = data["dmp"]
            self.dmp_weights = np.array(dmp_data["weights"], dtype=np.float64)
            self.dmp_centers = np.array(dmp_data["centers"], dtype=np.float64)
            self.dmp_widths = np.array(dmp_data["widths"], dtype=np.float64)
        if "goal" in data:
            self._goal = np.array(data["goal"], dtype=np.float64)
        if "joint_limits" in data:
            limits = data["joint_limits"]
            self.joint_lower = np.array(limits["lower"], dtype=np.float64)
            self.joint_upper = np.array(limits["upper"], dtype=np.float64)
            self._joint_barriers = self._build_joint_limit_barriers()
        if "config" in data:
            loaded_cfg = import_config(data["config"])
            if (
                loaded_cfg.n_joints != self.cfg.n_joints
                or loaded_cfg.state_dim != self.cfg.state_dim
                or loaded_cfg.options != self.cfg.options
            ):
                raise ValueError("Loaded configuration incompatible with existing controller.")
            self.cfg = loaded_cfg

    def export_config(self) -> Dict[str, object]:
        return export_config(self.cfg)

    def import_config(self, data: Dict[str, object]) -> None:
        new_cfg = import_config(data)
        if (
            new_cfg.n_joints != self.cfg.n_joints
            or new_cfg.state_dim != self.cfg.state_dim
            or new_cfg.options != self.cfg.options
        ):
            raise ValueError("Cannot import configuration with mismatched structural fields.")
        self.cfg = new_cfg
        for group in self.optimizer.param_groups:
            group["lr"] = self.cfg.lr

    def dmp_fit_weights(self, demo_positions: np.ndarray, T: float) -> None:
        r"""Fit DMP forcing weights from a demonstration.

        The target forcing term is obtained from
        ``v_\text{demo} = \tau \dot{x}`` and
        ``f(s) = (\tau^2 \ddot{x} - \alpha_z (\beta_z (g-x) - v_\text{demo})) /
        ((g-x_0) + \varepsilon)``. A normalized radial basis function regression
        solves for weights ``w_i`` in ``f(s) = (\sum_i \psi_i(s) w_i) s /
        (\sum_i \psi_i(s) + \varepsilon)``.
        """
        if demo_positions.shape[1] != self.cfg.n_joints:
            raise ValueError("Demonstration trajectory has incorrect joint dimension.")
        n_steps = demo_positions.shape[0]
        tau = max(T, EPS)
        dt = tau / max(n_steps - 1, 1)
        t = np.linspace(0.0, tau, n_steps, dtype=np.float64)
        s_traj = np.exp(-self.cfg.dmp_alpha_s * t / tau)
        velocities = np.gradient(demo_positions, dt, axis=0, edge_order=2)
        accelerations = np.gradient(velocities, dt, axis=0, edge_order=2)
        v_demo = tau * velocities
        g = demo_positions[-1]
        x0 = demo_positions[0]
        diff = s_traj[:, None] - self.dmp_centers[None, :]
        psi_matrix = np.exp(-self.dmp_widths[None, :] * diff**2)
        sum_psi = psi_matrix.sum(axis=1, keepdims=True) + EPS
        Phi = (psi_matrix / sum_psi) * s_traj[:, None]
        for j in range(self.cfg.n_joints):
            denom = (g[j] - x0[j]) + EPS
            feedback = self.cfg.dmp_alpha_z * (
                self.cfg.dmp_beta_z * (g[j] - demo_positions[:, j]) - v_demo[:, j]
            )
            f_target = (tau**2 * accelerations[:, j] - feedback) / denom
            weights, _, _, _ = np.linalg.lstsq(Phi, f_target, rcond=None)
            self.dmp_weights[j] = weights

    def dmp_generate(
        self,
        q0: np.ndarray,
        g: np.ndarray,
        T: float,
        n_steps: int,
        return_vel: bool = False,
    ) -> np.ndarray | Tuple[np.ndarray, np.ndarray]:
        r"""Integrate the DMP transformation system to produce a trajectory.

        The update integrates ``\tau \dot{v} = \alpha_z (\beta_z (g-x) - v)
        + (g-x_0) f(s)`` with semi-implicit Euler and ``\tau \dot{x} = v`` while
        sharing a canonical phase ``\tau \dot{s} = -\alpha_s s`` across joints.
        """
        if q0.shape[0] != self.cfg.n_joints or g.shape[0] != self.cfg.n_joints:
            raise ValueError("DMP rollout dimension mismatch.")
        tau = max(T, EPS)
        n_steps = max(int(n_steps), 2)
        dt = tau / (n_steps - 1)
        positions = np.zeros((n_steps, self.cfg.n_joints), dtype=np.float64)
        velocities = np.zeros_like(positions)
        x = q0.astype(np.float64).copy()
        v = np.zeros_like(x)
        s = 1.0
        positions[0] = x
        velocities[0] = v / tau
        for idx in range(1, n_steps):
            psi = np.exp(-self.dmp_widths * (s - self.dmp_centers) ** 2)
            psi_sum = psi.sum() + EPS
            basis = psi / psi_sum
            forcing = basis @ self.dmp_weights.T
            forcing *= s
            v_dot = (
                self.cfg.dmp_alpha_z * (self.cfg.dmp_beta_z * (g - x) - v)
                + (g - q0) * forcing
            ) / tau
            v += v_dot * dt
            x += (v / tau) * dt
            s_dot = -self.cfg.dmp_alpha_s * s / tau
            s = max(s + s_dot * dt, 0.0)
            positions[idx] = x
            velocities[idx] = v / tau
        if return_vel:
            return positions, velocities
        return positions

    def option_to_dmp(
        self,
        option_id: int,
        q: np.ndarray,
        g: np.ndarray,
        T: float,
        n_steps: int,
    ) -> np.ndarray:
        r"""Map an option to a safe joint-space trajectory.

        The method generates a DMP rollout, computes desired velocities
        ``u_{des} = \mathrm{clip}((q_{t+1} - q_t)/\Delta t)`` and filters them via
        :meth:`safety_layer_filter`, integrating ``q_{t+1} = q_t + u_{safe} \Delta t``.
        """
        if option_id < 0 or option_id >= len(self.options):
            raise ValueError("Option identifier out of range.")
        spec = self.options[option_id]
        base_goal = g.astype(np.float64)
        q = q.astype(np.float64)
        delta = base_goal - q
        if np.linalg.norm(delta) < EPS:
            delta = np.zeros_like(delta)
        goal = q + spec.goal_offset_scale * delta
        if spec.goal_offset_scale == 0.0:
            goal = q.copy()
        goal = np.clip(goal, self.joint_lower, self.joint_upper)
        duration = max(T * spec.duration_scale, self.cfg.dt)
        steps = max(int(n_steps), 2)
        raw_positions = self.dmp_generate(q, goal, duration, steps, return_vel=False)
        safe_positions = np.zeros_like(raw_positions)
        q_curr = q.copy()
        dq_curr = np.zeros_like(q_curr)
        safe_positions[0] = q_curr
        total_barriers = [fn for _, fn in self._custom_barriers]
        dt_local = duration / (steps - 1)
        R_diag = self.cfg.qp_R_diag
        alpha = self.cfg.cbf_alpha
        if spec.safety_profile == "cautious":
            R_diag *= 1.5
            alpha *= 1.2
        elif spec.safety_profile == "aggressive":
            R_diag *= 0.5
            alpha = max(alpha * 0.8, 1.0)
        for idx in range(1, steps):
            q_des = raw_positions[idx]
            u_des = np.clip((q_des - q_curr) / dt_local, -self.cfg.max_joint_vel, self.cfg.max_joint_vel)
            u_safe = self.safety_layer_filter(
                q_curr,
                dq_curr,
                u_des,
                dt_local,
                total_barriers,
                R_diag,
                alpha,
                self.cfg.use_slack,
            )
            q_curr = q_curr + u_safe * dt_local
            q_curr = np.clip(q_curr, self.joint_lower, self.joint_upper)
            dq_curr = u_safe.copy()
            safe_positions[idx] = q_curr
        return safe_positions

    def safety_layer_filter(
        self,
        q: np.ndarray,
        dq: np.ndarray,
        u_des: np.ndarray,
        dt: float,
        barriers: List[Callable[[np.ndarray], Tuple[float, np.ndarray]]],
        R_diag: float,
        alpha: float,
        use_slack: bool = True,
    ) -> np.ndarray:
        r"""Solve the CBF-QP safety filter.

        The QP minimizes ``\tfrac{1}{2}\|u-u_{des}\|_R^2`` subject to joint
        velocity/position limits and CBF inequalities
        ``\nabla h(q)^\top (dq + u) + \alpha h(q) + \delta \ge 0``. Slack penalties
        weight ``\delta`` for CBF constraints and bounds independently.
        """
        q = q.astype(np.float64)
        dq = dq.astype(np.float64)
        u_des = u_des.astype(np.float64)
        max_vel = np.full_like(u_des, self.cfg.max_joint_vel, dtype=np.float64)
        all_barriers = [*self._joint_barriers, *barriers]
        n_u = self.cfg.n_joints

        def build_qp(
            penalty_bounds: float,
            penalty_cbf: float,
        ) -> Tuple[sparse.csc_matrix, np.ndarray, sparse.csr_matrix, np.ndarray, np.ndarray]:
            use_bounds_slack = use_slack and self.cfg.qp_slack_all
            use_cbf_slack = use_slack and len(all_barriers) > 0
            idx_bounds = n_u if use_bounds_slack else None
            idx_cbf = n_u + (1 if idx_bounds is not None else 0) if use_cbf_slack else None
            total_dim = n_u
            if idx_bounds is not None:
                total_dim += 1
            if idx_cbf is not None:
                total_dim += 1
            diag = np.zeros(total_dim, dtype=np.float64)
            diag[:n_u] = R_diag
            if idx_bounds is not None:
                diag[idx_bounds] = penalty_bounds
            if idx_cbf is not None:
                diag[idx_cbf] = penalty_cbf
            P = sparse.diags(diag, offsets=0, format="csc")
            q_vec = np.zeros(total_dim, dtype=np.float64)
            q_vec[:n_u] = -R_diag * u_des
            rows: List[np.ndarray] = []
            lower: List[float] = []
            upper: List[float] = []
            for j in range(n_u):
                row = np.zeros(total_dim, dtype=np.float64)
                row[j] = 1.0
                if idx_bounds is not None:
                    row[idx_bounds] = -1.0
                rows.append(row)
                lower.append(-np.inf)
                upper.append(max_vel[j])
                row = np.zeros(total_dim, dtype=np.float64)
                row[j] = -1.0
                if idx_bounds is not None:
                    row[idx_bounds] = -1.0
                rows.append(row)
                lower.append(-np.inf)
                upper.append(max_vel[j])
                row = np.zeros(total_dim, dtype=np.float64)
                row[j] = dt
                if idx_bounds is not None:
                    row[idx_bounds] = -1.0
                rows.append(row)
                lower.append(-np.inf)
                upper.append(self.joint_upper[j] - q[j])
                row = np.zeros(total_dim, dtype=np.float64)
                row[j] = -dt
                if idx_bounds is not None:
                    row[idx_bounds] = -1.0
                rows.append(row)
                lower.append(-np.inf)
                upper.append(q[j] - self.joint_lower[j])
            if idx_bounds is not None:
                row = np.zeros(total_dim, dtype=np.float64)
                row[idx_bounds] = 1.0
                rows.append(row)
                lower.append(0.0)
                upper.append(np.inf)
            if len(all_barriers) > 0:
                for barrier_fn in all_barriers:
                    h_val, grad = barrier_fn(q)
                    if grad.shape[0] != n_u:
                        raise ValueError("Barrier gradient dimension mismatch.")
                    row = np.zeros(total_dim, dtype=np.float64)
                    row[:n_u] = grad
                    if idx_cbf is not None:
                        row[idx_cbf] = 1.0
                    rhs = -alpha * h_val - float(np.dot(grad, dq))
                    rows.append(row)
                    lower.append(rhs)
                    upper.append(np.inf)
                if idx_cbf is not None:
                    row = np.zeros(total_dim, dtype=np.float64)
                    row[idx_cbf] = 1.0
                    rows.append(row)
                    lower.append(0.0)
                    upper.append(np.inf)
            A = sparse.csr_matrix(np.vstack(rows)) if rows else sparse.csr_matrix((0, total_dim))
            l_vec = np.array(lower, dtype=np.float64) if lower else np.zeros(0, dtype=np.float64)
            u_vec = np.array(upper, dtype=np.float64) if upper else np.zeros(0, dtype=np.float64)
            return P, q_vec, A, l_vec, u_vec

        penalty_bounds = self.cfg.qp_slack_weight_bounds
        penalty_cbf = self.cfg.qp_slack_weight_cbf
        attempt = 0
        best_u = None
        while attempt < 2:
            P, q_vec, A, l_vec, u_vec = build_qp(penalty_bounds, penalty_cbf)
            solution = self._solve_qp(P, q_vec, A, l_vec, u_vec)
            if solution is not None:
                best_u = solution[:n_u]
                break
            penalty_bounds = max(penalty_bounds * 0.1, 1.0)
            penalty_cbf = max(penalty_cbf * 0.1, 1.0)
            u_des = DEFAULT_BACKOFF_FACTOR * u_des
            attempt += 1
        if best_u is None:
            self._maybe_warn("Safety QP infeasible, returning clipped command.")
            return np.clip(u_des, -self.cfg.max_joint_vel, self.cfg.max_joint_vel)
        return np.clip(best_u, -self.cfg.max_joint_vel, self.cfg.max_joint_vel)

    def _solve_qp(
        self,
        P: sparse.csc_matrix,
        q: np.ndarray,
        A: sparse.csr_matrix,
        l: np.ndarray,
        u: np.ndarray,
    ) -> Optional[np.ndarray]:
        if osqp is not None:
            prob = osqp.OSQP()
            prob.setup(P=P, q=q, A=A, l=l, u=u, verbose=False, polish=True)
            res = prob.solve()
            status = res.info.status.lower()
            if status in {"solved", "solved_inaccurate"} and res.x is not None:
                return np.array(res.x, dtype=np.float64)
        if solve_qp is None:
            self._maybe_warn("No QP solver available; safety layer falling back to clipping.")
            return None
        A_eq: List[np.ndarray] = []
        b_eq: List[float] = []
        G: List[np.ndarray] = []
        h_vec: List[float] = []
        for idx in range(A.shape[0]):
            row = A.getrow(idx).toarray().ravel()
            low = l[idx] if l.size else -np.inf
            high = u[idx] if u.size else np.inf
            if math.isclose(low, high, rel_tol=1e-9, abs_tol=1e-9):
                A_eq.append(row)
                b_eq.append(low)
            else:
                if np.isfinite(high):
                    G.append(row)
                    h_vec.append(high)
                if np.isfinite(low):
                    G.append(-row)
                    h_vec.append(-low)
        G_mat = np.vstack(G) if G else None
        h_arr = np.array(h_vec, dtype=np.float64) if G else None
        A_mat = np.vstack(A_eq) if A_eq else None
        b_arr = np.array(b_eq, dtype=np.float64) if A_eq else None
        try:
            sol = solve_qp(P.toarray(), q, G_mat, h_arr, A_mat, b_arr)
        except Exception:  # pragma: no cover - solver specific errors
            sol = None
        if sol is None:
            return None
        return np.array(sol, dtype=np.float64)

    def _maybe_warn(self, message: str) -> None:
        now = time.monotonic()
        if now - self._last_warning_time > WARNING_INTERVAL_S:
            logger.warning(message)
            self._last_warning_time = now

    def train(
        self,
        env: EnvProtocol,
        total_steps: int,
        warmup: int = 1_000,
        target_update_interval: int = 1_000,
        log_interval: int = 1_000,
    ) -> Dict[str, float]:
        """Train the controller on ``env`` for a fixed number of option steps."""
        state = env.reset()
        self.set_goal(env.goal)
        self._process_state(state, update_stats=True)
        episode_reward = 0.0
        episode = 0
        last_loss = 0.0
        for step in range(total_steps):
            if self.total_env_steps < warmup:
                option_id = random.randrange(len(self.options))
            else:
                option_id = self.select_option(state, explore=True)
            next_state, reward, done, _ = env.step(option_id)
            self.push_transition(state, option_id, reward, next_state, done)
            info = self.train_step()
            if "loss" in info:
                last_loss = info["loss"]
            self.total_env_steps += 1
            self._update_epsilon()
            episode_reward += reward
            state = next_state
            if done:
                state = env.reset()
                self.set_goal(env.goal)
                self._process_state(state, update_stats=True)
                episode += 1
                episode_reward = 0.0
            interval_cfg = max(self.cfg.target_update_interval, 1)
            if self.cfg.target_update == "hard" and (step + 1) % interval_cfg == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
            if (step + 1) % max(log_interval, 1) == 0 and logger.isEnabledFor(logging.INFO):
                logger.info("Training step %d/%d", step + 1, total_steps)
        return {
            "steps": float(total_steps),
            "episodes": float(episode),
            "last_loss": float(last_loss),
        }

    def evaluate(self, env: EnvProtocol, episodes: int = 5) -> Dict[str, float]:
        """Evaluate the greedy policy for ``episodes`` episodes."""
        rewards: List[float] = []
        for _ in range(episodes):
            state = env.reset()
            self.set_goal(env.goal)
            done = False
            episode_reward = 0.0
            while not done:
                option_id = self.select_option(state, explore=False)
                state, reward, done, _ = env.step(option_id)
                episode_reward += reward
            rewards.append(episode_reward)
        avg_reward = float(np.mean(rewards)) if rewards else 0.0
        return {"avg_reward": avg_reward}

    def run_ros2_closed_loop(
        self,
        controller_ns: str = "/joint_trajectory_controller",
        hz: float = 10.0,
        topic_joint_states: str = "/joint_states",
        topic_joint_traj: str = "/joint_trajectory",
        frame_id: str = "",
        goal: Optional[np.ndarray] = None,
    ) -> None:
        """Run a ROS 2 closed-loop executor publishing joint trajectories."""
        try:
            import rclpy  # type: ignore
            from builtin_interfaces.msg import Duration  # type: ignore
            from rclpy.node import Node  # type: ignore
            from sensor_msgs.msg import JointState  # type: ignore
            from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("ROS 2 not available") from exc

        goal_array = goal.astype(np.float64) if goal is not None else self.goal

        class HRLNode(Node):
            def __init__(self, outer: "HierarchicalRLController") -> None:
                super().__init__("hierarchical_rl_controller")
                self.outer = outer
                self.joint_state: Optional[JointState] = None
                self.goal = goal_array.copy()
                self.state_subscription = self.create_subscription(
                    JointState,
                    topic_joint_states,
                    self._joint_state_cb,
                    10,
                )
                self.goal_subscription = self.create_subscription(
                    JointTrajectoryPoint,
                    f"{controller_ns}/set_goal",
                    self._goal_cb,
                    10,
                )
                self.trajectory_publisher = self.create_publisher(
                    JointTrajectory,
                    topic_joint_traj,
                    10,
                )
                self.timer = self.create_timer(1.0 / max(hz, 1e-3), self._timer_cb)

            def _joint_state_cb(self, msg: JointState) -> None:
                self.joint_state = msg

            def _goal_cb(self, msg: JointTrajectoryPoint) -> None:
                if msg.positions:
                    self.goal = np.array(msg.positions[: self.outer.cfg.n_joints], dtype=np.float64)
                    self.outer.set_goal(self.goal)

            def _timer_cb(self) -> None:
                if self.joint_state is None or not self.joint_state.position:
                    return
                q = np.array(self.joint_state.position[: self.outer.cfg.n_joints], dtype=np.float64)
                dq = (
                    np.array(self.joint_state.velocity[: self.outer.cfg.n_joints], dtype=np.float64)
                    if self.joint_state.velocity
                    else np.zeros(self.outer.cfg.n_joints, dtype=np.float64)
                )
                state_vec = np.concatenate([q, dq])
                option = self.outer.select_option(state_vec, explore=False)
                traj = self.outer.option_to_dmp(
                    option,
                    q,
                    self.goal,
                    self.outer.cfg.dt * self.outer.cfg.horizon_steps,
                    self.outer.cfg.horizon_steps,
                )
                msg = JointTrajectory()
                msg.header.frame_id = frame_id
                msg.joint_names = list(self.joint_state.name[: self.outer.cfg.n_joints])
                msg.header.stamp = self.get_clock().now().to_msg()
                points: List[JointTrajectoryPoint] = []
                dt_local = self.outer.cfg.dt
                for idx, positions in enumerate(traj):
                    point = JointTrajectoryPoint()
                    point.positions = list(positions)
                    if idx > 0:
                        velocities = (traj[idx] - traj[idx - 1]) / dt_local
                        point.velocities = list(velocities)
                    point.time_from_start = Duration(sec=0, nanosec=int(idx * dt_local * 1e9))
                    points.append(point)
                msg.points = points
                self.trajectory_publisher.publish(msg)

        rclpy.init()
        node = HRLNode(self)
        try:
            rclpy.spin(node)
        finally:
            node.destroy_node()
            rclpy.shutdown()


class ToyJointEnv(EnvProtocol):
    """Deterministic toy environment using controller-generated DMP rollouts."""

    def __init__(self, controller: HierarchicalRLController, horizon: int = 3) -> None:
        self.controller = controller
        self._horizon = max(horizon, 2)
        self._dt = controller.cfg.dt
        self._n_joints = controller.cfg.n_joints
        self._state = np.zeros(2 * self._n_joints, dtype=np.float64)
        self._max_steps = 200
        self._step_count = 0

    @property
    def state_dim(self) -> int:
        return 2 * self._n_joints

    @property
    def n_joints(self) -> int:
        return self._n_joints

    @property
    def goal(self) -> np.ndarray:
        return self.controller.goal

    def reset(self) -> np.ndarray:
        self._state = np.zeros_like(self._state)
        self._step_count = 0
        return self._state.copy()

    def step(self, option_id: int) -> Tuple[np.ndarray, float, bool, dict]:
        """Roll out the selected option and integrate the toy system dynamics."""
        q = self._state[: self._n_joints]
        dq = self._state[self._n_joints :]
        traj = self.controller.option_to_dmp(
            option_id,
            q,
            self.goal,
            self.controller.cfg.dt * self._horizon,
            self._horizon,
        )
        q_next = traj[-1]
        dq_next = (traj[-1] - traj[-2]) / self._dt if self._horizon > 1 else np.zeros_like(q)
        self._state[: self._n_joints] = q_next
        self._state[self._n_joints :] = dq_next
        error = np.linalg.norm(self.goal - q_next, ord=1)
        reward = -float(error)
        self._step_count += 1
        done = self._step_count >= self._max_steps or error < 1e-2
        return self._state.copy(), reward, done, {}


def _smoke_test() -> None:
    """Minimal CPU smoke test for import and basic training."""
    logging.basicConfig(level=logging.INFO)
    cfg = HRLConfig(
        n_joints=1,
        state_dim=2,
        options=["hold", "dmp_small_step", "dmp_medium_step", "dmp_large_step"],
        batch_size=16,
        replay_capacity=1_000,
        eps_decay_steps=1_000,
        horizon_steps=5,
        dt=0.05,
        seed=42,
    )
    controller = HierarchicalRLController(cfg)
    controller.set_joint_limits(np.array([-1.0]), np.array([1.0]))
    controller.set_goal(np.array([0.5]))
    env = ToyJointEnv(controller, horizon=4)
    controller.train(env, total_steps=200, warmup=20, target_update_interval=100)
    metrics = controller.evaluate(env, episodes=3)
    print("Toy env average reward:", metrics["avg_reward"])


if __name__ == "__main__":
    _smoke_test()
