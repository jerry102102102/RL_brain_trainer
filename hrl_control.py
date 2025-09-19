"""Hierarchical reinforcement learning controller with DMP and CBF-QP safety.

This module implements a hierarchical reinforcement learning (HRL) stack that
combines a discrete Deep Q-Network (DQN) option selector, Dynamic Movement
Primitives (DMPs) for continuous trajectory synthesis, and a Control Barrier
Function (CBF) Quadratic Program (QP) safety filter. The design targets robotic
joint-space control while remaining lightweight enough for smoke testing.
"""
from __future__ import annotations

import logging
import math
import random
import time
from collections import deque
from dataclasses import asdict, dataclass
from typing import Callable, Deque, Dict, List, Optional, Protocol, Sequence, Tuple, Union, Literal

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
OSQP_SUCCESS_STATUSES = {"solved", "solved_inaccurate"}
DEFAULT_WARNING_INTERVAL = 5.0
LOGGER = logging.getLogger(__name__)


class EnvProtocol(Protocol):
    """Protocol describing the minimal environment interface."""

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


@dataclass
class OptionSpec:
    """Specification describing how a discrete option maps to a DMP rollout."""

    name: str
    goal_offset_scale: float = 1.0
    duration_scale: float = 1.0
    dmp_basis_id: int = 0
    safety_profile: str = "default"


_DEFAULT_OPTION_SETTINGS: Dict[str, Dict[str, float]] = {
    "hold": {"goal_offset_scale": 0.0, "duration_scale": 1.0},
    "dmp_small_step": {"goal_offset_scale": 0.25, "duration_scale": 1.0},
    "dmp_medium_step": {"goal_offset_scale": 0.5, "duration_scale": 1.0},
    "dmp_large_step": {"goal_offset_scale": 1.0, "duration_scale": 1.0},
}


def _default_option_spec(name: str, index: int = 0) -> OptionSpec:
    settings = _DEFAULT_OPTION_SETTINGS.get(name, {"goal_offset_scale": 0.5, "duration_scale": 1.0})
    return OptionSpec(
        name=name,
        goal_offset_scale=float(settings.get("goal_offset_scale", 0.5)),
        duration_scale=float(settings.get("duration_scale", 1.0)),
        dmp_basis_id=index,
        safety_profile="default",
    )


class ReplayBuffer:
    """Replay buffer storing transitions ``(s, a, r, s', done)``."""

    def __init__(self, capacity: int) -> None:
        self.capacity = int(capacity)
        self.buffer: Deque[Tuple[np.ndarray, int, float, np.ndarray, bool]] = deque(maxlen=self.capacity)

    def __len__(self) -> int:
        return len(self.buffer)

    def push(self, transition: Tuple[np.ndarray, int, float, np.ndarray, bool]) -> None:
        self.buffer.append(transition)

    def sample(self, batch_size: int) -> List[Tuple[np.ndarray, int, float, np.ndarray, bool]]:
        batch_size = min(int(batch_size), len(self.buffer))
        return random.sample(self.buffer, batch_size)


class StateNormalizer:
    """Running mean-variance normalizer for observations."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.count: int = 0
        self.mean: Optional[np.ndarray] = None
        self.m2: Optional[np.ndarray] = None

    def update(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float64)
        if self.mean is None:
            self.mean = np.zeros_like(x, dtype=np.float64)
            self.m2 = np.zeros_like(x, dtype=np.float64)
        if x.shape != self.mean.shape:
            raise ValueError("StateNormalizer received inconsistent shapes.")
        self.count += 1
        if self.count == 1:
            self.mean = x.copy()
            self.m2 = np.zeros_like(x, dtype=np.float64)
            return
        assert self.mean is not None and self.m2 is not None
        delta = x - self.mean
        self.mean += delta / self.count
        delta2 = x - self.mean
        self.m2 += delta * delta2

    def normalize(self, x: np.ndarray) -> np.ndarray:
        if self.count < 2 or self.mean is None or self.m2 is None:
            return np.asarray(x, dtype=np.float64)
        variance = self.m2 / max(self.count - 1, 1)
        std = np.sqrt(np.maximum(variance, 1e-6))
        return (np.asarray(x, dtype=np.float64) - self.mean) / std

    def state_dict(self) -> Dict[str, Union[int, np.ndarray]]:
        return {
            "count": self.count,
            "mean": None if self.mean is None else self.mean.copy(),
            "m2": None if self.m2 is None else self.m2.copy(),
        }

    def load_state_dict(self, state: Dict[str, Union[int, np.ndarray]]) -> None:
        self.count = int(state.get("count", 0))
        mean = state.get("mean")
        m2 = state.get("m2")
        self.mean = None if mean is None else np.asarray(mean, dtype=np.float64)
        self.m2 = None if m2 is None else np.asarray(m2, dtype=np.float64)


class QNetwork(nn.Module):
    """Feed-forward state-action value function approximator."""

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DMPModel:
    """Dynamic Movement Primitive model shared across joints."""

    def __init__(
        self,
        n_joints: int,
        n_basis: int,
        alpha_s: float,
        alpha_z: float,
        beta_z: float,
        basis_overlap: float,
    ) -> None:
        self.n_joints = int(n_joints)
        self.n_basis = int(n_basis)
        self.alpha_s = float(alpha_s)
        self.alpha_z = float(alpha_z)
        self.beta_z = float(beta_z)
        self.basis_overlap = float(basis_overlap)
        self.centers, self.widths = self._init_basis()
        self.weights = np.zeros((self.n_joints, self.n_basis), dtype=np.float64)

    def _init_basis(self) -> Tuple[np.ndarray, np.ndarray]:
        centers = np.linspace(1e-4, 1.0, self.n_basis, dtype=np.float64)[::-1]
        widths = np.ones_like(centers)
        if self.n_basis > 1:
            spacings = np.abs(np.diff(centers))
            spacings = np.maximum(spacings, EPS)
            widths[:-1] = 1.0 / (self.basis_overlap * spacings ** 2)
            widths[-1] = widths[-2]
        return centers, widths

    def _canonical_trajectory(self, T: float, n_steps: int) -> np.ndarray:
        tau = max(float(T), EPS)
        n_steps = max(int(n_steps), 2)
        dt = tau / (n_steps - 1)
        s = 1.0
        trajectory = np.zeros(n_steps, dtype=np.float64)
        for idx in range(n_steps):
            trajectory[idx] = s
            ds = (-self.alpha_s * s / tau) * dt
            s = max(s + ds, 0.0)
        return trajectory

    def _basis_vector(self, s_val: float) -> np.ndarray:
        diff = s_val - self.centers
        return np.exp(-self.widths * diff ** 2)

    def fit_weights(self, demo_positions: np.ndarray, T: float) -> None:
        r"""Fit forcing term weights via locally weighted regression.

        The transformation system obeys

        .. math::
            \tau \dot{v} = \alpha_z (\beta_z (g - x) - v) + (g - x_0) f(s),
            \quad \tau \dot{x} = v,

        with the canonical state :math:`s` governed by :math:`\tau \dot{s} = -\alpha_s s`.
        The target forcing term is recovered from a demonstration trajectory
        ``demo_positions`` by rearranging the dynamics with

        .. math::
            v_\text{demo} = \tau \dot{x},\quad
            f^* = \frac{\tau^2 \ddot{x} - \alpha_z (\beta_z (g - x) - v_\text{demo})}{(g - x_0) + \varepsilon}.
        """

        demo = np.asarray(demo_positions, dtype=np.float64)
        if demo.ndim != 2 or demo.shape[1] != self.n_joints:
            raise ValueError("Demo trajectory joint dimension mismatch.")
        n_steps = demo.shape[0]
        tau = max(float(T), EPS)
        dt = tau / max(n_steps - 1, 1)
        x_dot = np.gradient(demo, dt, axis=0, edge_order=2)
        x_ddot = np.gradient(x_dot, dt, axis=0, edge_order=2)
        v_demo = tau * x_dot
        s_traj = self._canonical_trajectory(T, n_steps)
        psi_matrix = np.stack([self._basis_vector(s_val) for s_val in s_traj], axis=0)
        sum_psi = psi_matrix.sum(axis=1, keepdims=True) + EPS
        Phi = (psi_matrix * s_traj[:, None]) / sum_psi
        g = demo[-1]
        x0 = demo[0]
        reg = 1e-6 * np.eye(self.n_basis, dtype=np.float64)
        Phi_T = Phi.T
        gram = Phi_T @ Phi + reg
        for joint in range(self.n_joints):
            denom = (g[joint] - x0[joint]) + EPS
            f_target = (
                (tau ** 2) * x_ddot[:, joint]
                - self.alpha_z * (self.beta_z * (g[joint] - demo[:, joint]) - v_demo[:, joint])
            ) / denom
            rhs = Phi_T @ f_target
            self.weights[joint] = np.linalg.solve(gram, rhs)

    def generate(
        self,
        q0: np.ndarray,
        g: np.ndarray,
        T: float,
        n_steps: int,
        return_vel: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        r"""Roll out the learned DMP dynamics to obtain a trajectory.

        Semi-implicit Euler integration advances the scaled velocity ``v`` via

        .. math::
            v_{k+1} = v_k + \frac{\Delta t}{\tau} \left[
                \alpha_z (\beta_z (g - x_k) - v_k) + (g - x_0) f(s_k)
            \right],

        followed by :math:`x_{k+1} = x_k + \frac{\Delta t}{\tau} v_{k+1}`.
        """

        q0 = np.asarray(q0, dtype=np.float64)
        g = np.asarray(g, dtype=np.float64)
        if q0.shape[0] != self.n_joints or g.shape[0] != self.n_joints:
            raise ValueError("Initial or goal joint dimension mismatch.")
        tau = max(float(T), EPS)
        n_steps = max(int(n_steps), 2)
        dt = tau / (n_steps - 1)
        s_traj = self._canonical_trajectory(T, n_steps)
        positions = np.zeros((n_steps, self.n_joints), dtype=np.float64)
        velocities = np.zeros_like(positions)
        x = q0.copy()
        v = np.zeros_like(q0)
        for idx, s_val in enumerate(s_traj):
            positions[idx] = x
            velocities[idx] = v / tau
            if idx == n_steps - 1:
                break
            psi = self._basis_vector(s_val)
            sum_psi = psi.sum() + EPS
            f = (self.weights @ psi) * s_val / sum_psi
            forcing = (g - q0) * f
            v_dot = (self.alpha_z * (self.beta_z * (g - x) - v) + forcing) / tau
            v = v + v_dot * dt
            x = x + (v / tau) * dt
        if return_vel:
            return positions, velocities
        return positions

    def state_dict(self) -> Dict[str, np.ndarray]:
        return {
            "weights": self.weights.copy(),
            "centers": self.centers.copy(),
            "widths": self.widths.copy(),
        }

    def load_state_dict(self, state: Dict[str, np.ndarray]) -> None:
        self.weights = np.asarray(state["weights"], dtype=np.float64)
        self.centers = np.asarray(state["centers"], dtype=np.float64)
        self.widths = np.asarray(state["widths"], dtype=np.float64)


@dataclass
class HRLConfig:
    n_joints: int
    state_dim: int
    options: Sequence[Union[str, OptionSpec]]
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
    target_update_interval: int = 1_000
    state_norm: bool = True
    dmp_basis_overlap: float = 0.5
    qp_slack_all: bool = True
    qp_slack_weight_cbf: float = 1e3
    qp_slack_weight_bounds: float = 1e2
    goal_in_state: bool = True


class HierarchicalRLController:
    """Hierarchical RL controller with DMP motion primitives and CBF-QP safety."""

    def __init__(self, cfg: HRLConfig) -> None:
        self.cfg = cfg
        self.device = self._select_device(cfg.device)
        self._seed_rngs(cfg.seed)
        self.options: List[OptionSpec] = self._build_option_registry(cfg.options)
        input_dim = int(cfg.state_dim)
        output_dim = len(self.options)
        self.policy_net = QNetwork(input_dim, output_dim).to(self.device)
        self.target_net = QNetwork(input_dim, output_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=cfg.lr)
        self.replay_buffer = ReplayBuffer(cfg.replay_capacity)
        self.loss_fn = nn.MSELoss()
        self.eps = cfg.eps_start
        self.total_env_steps = 0
        self.training_steps = 0
        self.state_normalizer = StateNormalizer() if cfg.state_norm else None
        self.dmp_model = DMPModel(
            cfg.n_joints,
            cfg.dmp_n_basis,
            cfg.dmp_alpha_s,
            cfg.dmp_alpha_z,
            cfg.dmp_beta_z,
            cfg.dmp_basis_overlap,
        )
        self.dmp_centers = self.dmp_model.centers
        self.dmp_widths = self.dmp_model.widths
        self.dmp_weights = self.dmp_model.weights
        self.joint_lower = -np.pi * np.ones(cfg.n_joints, dtype=np.float64)
        self.joint_upper = np.pi * np.ones(cfg.n_joints, dtype=np.float64)
        self._goal = np.zeros(cfg.n_joints, dtype=np.float64)
        self._joint_barriers = self._build_joint_limit_barriers()
        self._user_barriers: List[Tuple[Callable[[np.ndarray], Tuple[float, np.ndarray]], str]] = []
        self._qp_warning_last = 0.0

    @staticmethod
    def _select_device(device_str: Optional[str]) -> torch.device:
        if device_str == "cpu":
            return torch.device("cpu")
        if device_str == "cuda":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def _seed_rngs(seed: Optional[int]) -> None:
        if seed is None:
            return
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():  # pragma: no cover - optional GPU
            torch.cuda.manual_seed_all(seed)

    def _build_option_registry(self, options: Sequence[Union[str, OptionSpec]]) -> List[OptionSpec]:
        registry: List[OptionSpec] = []
        for idx, opt in enumerate(options):
            if isinstance(opt, OptionSpec):
                registry.append(opt)
            elif isinstance(opt, str):
                registry.append(_default_option_spec(opt, idx))
            else:
                raise TypeError(f"Unsupported option specification type: {type(opt)!r}")
        return registry

    def _build_joint_limit_barriers(self) -> List[Callable[[np.ndarray], Tuple[float, np.ndarray]]]:
        barriers: List[Callable[[np.ndarray], Tuple[float, np.ndarray]]] = []
        for idx in range(self.cfg.n_joints):
            upper = self.joint_upper[idx]
            lower = self.joint_lower[idx]

            def upper_barrier(q: np.ndarray, i: int = idx, limit: float = upper) -> Tuple[float, np.ndarray]:
                grad = np.zeros_like(q, dtype=np.float64)
                grad[i] = -1.0
                return float(limit - q[i]), grad

            def lower_barrier(q: np.ndarray, i: int = idx, limit: float = lower) -> Tuple[float, np.ndarray]:
                grad = np.zeros_like(q, dtype=np.float64)
                grad[i] = 1.0
                return float(q[i] - limit), grad

            barriers.append(upper_barrier)
            barriers.append(lower_barrier)
        return barriers

    def add_barrier(
        self,
        fn: Callable[[np.ndarray], Tuple[float, np.ndarray]],
        name: str = "",
    ) -> None:
        """Register an additional barrier function for the safety layer."""

        self._user_barriers.append((fn, name))

    def clear_barriers(self) -> None:
        """Remove all user-defined barrier functions while keeping joint limits."""

        self._user_barriers.clear()

    def set_joint_limits(self, lower: np.ndarray, upper: np.ndarray) -> None:
        """Update joint bounds used by the safety filter."""

        lower_arr = np.asarray(lower, dtype=np.float64)
        upper_arr = np.asarray(upper, dtype=np.float64)
        if lower_arr.shape[0] != self.cfg.n_joints or upper_arr.shape[0] != self.cfg.n_joints:
            raise ValueError("Joint limit arrays must match n_joints.")
        if np.any(lower_arr >= upper_arr):
            raise ValueError("Each joint lower bound must be strictly less than upper bound.")
        self.joint_lower = lower_arr.copy()
        self.joint_upper = upper_arr.copy()
        self._joint_barriers = self._build_joint_limit_barriers()

    def set_goal(self, g: np.ndarray) -> None:
        """Set the internal goal used for trajectory synthesis and state augmentation."""

        goal = np.asarray(g, dtype=np.float64)
        if goal.shape[0] != self.cfg.n_joints:
            raise ValueError("Goal dimension mismatch.")
        self._goal = goal.copy()

    def reset_norm(self) -> None:
        """Reset observation normalization statistics."""

        if self.state_normalizer is not None:
            self.state_normalizer.reset()

    def _collect_barriers(self) -> List[Callable[[np.ndarray], Tuple[float, np.ndarray]]]:
        barriers = list(self._joint_barriers)
        barriers.extend(fn for fn, _ in self._user_barriers)
        return barriers

    def _augment_state(self, state: np.ndarray, goal: np.ndarray) -> np.ndarray:
        base = np.asarray(state, dtype=np.float64)
        if not self.cfg.goal_in_state:
            return base
        joints = min(self.cfg.n_joints, base.shape[0])
        q = base[:joints]
        goal_segment = goal[:joints]
        error = goal_segment - q
        return np.concatenate([base, error, goal_segment], dtype=np.float64)

    def _pad_state(self, state: np.ndarray) -> np.ndarray:
        if state.shape[0] < self.cfg.state_dim:
            return np.pad(state, (0, self.cfg.state_dim - state.shape[0]))
        if state.shape[0] > self.cfg.state_dim:
            return state[: self.cfg.state_dim]
        return state

    def _process_state(self, state: np.ndarray, goal: np.ndarray, update_norm: bool) -> np.ndarray:
        augmented = self._augment_state(state, goal)
        padded = self._pad_state(augmented)
        if self.state_normalizer is not None:
            if update_norm:
                self.state_normalizer.update(padded)
            padded = self.state_normalizer.normalize(padded)
        return padded.astype(np.float32)

    def select_option(self, state_np: np.ndarray, explore: bool = True) -> int:
        """Select an option via epsilon-greedy evaluation of the policy network."""

        state_tensor = torch.as_tensor(state_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        if explore and random.random() < self.eps:
            return random.randrange(len(self.options))
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
            action = int(torch.argmax(q_values, dim=1).item())
        return action

    def push_transition(self, s: np.ndarray, a: int, r: float, s2: np.ndarray, done: bool) -> None:
        self.replay_buffer.push((np.asarray(s, dtype=np.float32), int(a), float(r), np.asarray(s2, dtype=np.float32), bool(done)))

    def _soft_update(self) -> None:
        tau = self.cfg.tau
        for target_param, param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

    def _hard_update(self) -> None:
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def _update_epsilon(self) -> None:
        progress = min(1.0, self.total_env_steps / max(self.cfg.eps_decay_steps, 1))
        self.eps = self.cfg.eps_start + progress * (self.cfg.eps_end - self.cfg.eps_start)

    def train_step(self) -> Dict[str, float]:
        if len(self.replay_buffer) < self.cfg.batch_size:
            return {}
        batch = self.replay_buffer.sample(self.cfg.batch_size)
        states = torch.as_tensor(np.stack([b[0] for b in batch]), dtype=torch.float32, device=self.device)
        actions = torch.as_tensor([b[1] for b in batch], dtype=torch.long, device=self.device)
        rewards = torch.as_tensor([b[2] for b in batch], dtype=torch.float32, device=self.device)
        next_states = torch.as_tensor(np.stack([b[3] for b in batch]), dtype=torch.float32, device=self.device)
        dones = torch.as_tensor([float(b[4]) for b in batch], dtype=torch.float32, device=self.device)
        q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            if self.cfg.double_dqn:
                next_actions = torch.argmax(self.policy_net(next_states), dim=1, keepdim=True)
                next_q = self.target_net(next_states).gather(1, next_actions).squeeze(1)
            else:
                next_q = self.target_net(next_states).max(1)[0]
            targets = rewards + self.cfg.gamma * (1.0 - dones) * next_q
        loss = self.loss_fn(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.training_steps += 1
        if self.cfg.target_update == "soft":
            self._soft_update()
        elif self.cfg.target_update == "hard":
            interval = max(self.cfg.target_update_interval, 1)
            if self.training_steps % interval == 0:
                self._hard_update()
        return {"loss": float(loss.item())}

    def _extract_goal(self, info: Optional[dict], env: EnvProtocol) -> np.ndarray:
        if info is not None and "goal" in info:
            goal = np.asarray(info["goal"], dtype=np.float64)
        else:
            goal = np.asarray(env.goal, dtype=np.float64)
        if goal.shape[0] != self.cfg.n_joints:
            raise ValueError("Environment goal dimension mismatch.")
        return goal

    def train(
        self,
        env: EnvProtocol,
        total_steps: int,
        warmup: int = 1_000,
        target_update_interval: Optional[int] = None,
        log_interval: int = 1_000,
    ) -> Dict[str, float]:
        state_raw = np.asarray(env.reset(), dtype=np.float64)
        goal = self._extract_goal({}, env)
        self.set_goal(goal)
        state = self._process_state(state_raw, self._goal, update_norm=True)
        episode_reward = 0.0
        episode = 1
        rewards: List[float] = []
        losses: List[float] = []
        original_interval = self.cfg.target_update_interval
        if target_update_interval is not None:
            self.cfg.target_update_interval = int(target_update_interval)
        try:
            for step in range(1, total_steps + 1):
                if step <= warmup:
                    action = random.randrange(len(self.options))
                else:
                    action = self.select_option(state, explore=True)
                next_state_raw, reward, done, info = env.step(action)
            next_goal = self._extract_goal(info, env)
            self.set_goal(next_goal)
            next_state = self._process_state(np.asarray(next_state_raw, dtype=np.float64), self._goal, update_norm=True)
            self.push_transition(state, action, reward, next_state, done)
            info_train = self.train_step()
            if info_train.get("loss") is not None:
                losses.append(info_train["loss"])
            episode_reward += reward
            self.total_env_steps += 1
            self._update_epsilon()
            state = next_state
            if done:
                rewards.append(episode_reward)
                state_raw = np.asarray(env.reset(), dtype=np.float64)
                goal = self._extract_goal({}, env)
                self.set_goal(goal)
                state = self._process_state(state_raw, self._goal, update_norm=True)
                episode_reward = 0.0
                episode += 1
        finally:
            self.cfg.target_update_interval = original_interval
        metrics = {
            "episodes": episode,
            "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
            "mean_loss": float(np.mean(losses)) if losses else 0.0,
        }
        if log_interval:
            metrics["epsilon"] = float(self.eps)
        return metrics

    def evaluate(self, env: EnvProtocol, episodes: int = 5) -> Dict[str, Union[float, List[float]]]:
        rewards: List[float] = []
        original_mode = self.policy_net.training
        self.policy_net.eval()
        try:
            for _ in range(episodes):
                state_raw = np.asarray(env.reset(), dtype=np.float64)
                goal = self._extract_goal({}, env)
                self.set_goal(goal)
                state = self._process_state(state_raw, self._goal, update_norm=False)
                done = False
                total_reward = 0.0
                steps = 0
                while not done and steps < 200:
                    action = self.select_option(state, explore=False)
                    next_state_raw, reward, done, info = env.step(action)
                    goal = self._extract_goal(info, env)
                    self.set_goal(goal)
                    state = self._process_state(np.asarray(next_state_raw, dtype=np.float64), self._goal, update_norm=False)
                    total_reward += reward
                    steps += 1
                rewards.append(total_reward)
        finally:
            if original_mode:
                self.policy_net.train()
        return {
            "episodes": float(episodes),
            "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
            "rewards": rewards,
        }

    def dmp_fit_weights(self, demo_positions: np.ndarray, T: float) -> None:
        self.dmp_model.fit_weights(demo_positions, T)
        self.dmp_weights = self.dmp_model.weights
        self.dmp_centers = self.dmp_model.centers
        self.dmp_widths = self.dmp_model.widths

    def dmp_generate(
        self,
        q0: np.ndarray,
        g: np.ndarray,
        T: float,
        n_steps: int,
        return_vel: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        return self.dmp_model.generate(q0, g, T, n_steps, return_vel=return_vel)

    def option_to_dmp(
        self,
        option_id: int,
        q: np.ndarray,
        g: np.ndarray,
        T: float,
        n_steps: int,
        return_vel: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        spec = self.options[int(option_id)]
        q = np.asarray(q, dtype=np.float64)
        g = np.asarray(g, dtype=np.float64)
        goal_offset = spec.goal_offset_scale * (g - q)
        g_target = np.clip(q + goal_offset, self.joint_lower, self.joint_upper)
        duration = max(float(T) * spec.duration_scale, self.cfg.dt)
        steps = max(int(round(n_steps * spec.duration_scale)), 2)
        dmp_result = self.dmp_generate(q, g_target, duration, steps, return_vel=True)
        positions, velocities = dmp_result
        dt = duration / (steps - 1)
        safe_positions: List[np.ndarray] = [q.copy()]
        safe_velocities: List[np.ndarray] = [np.zeros_like(q)]
        q_curr = q.copy()
        dq_curr = np.zeros_like(q_curr)
        barriers = self._collect_barriers()
        for idx in range(steps - 1):
            q_des_curr = positions[idx]
            q_des_next = positions[idx + 1]
            u_des = (q_des_next - q_des_curr) / max(dt, EPS)
            u_des = np.clip(u_des, -self.cfg.max_joint_vel, self.cfg.max_joint_vel)
            u_safe = self.safety_layer_filter(
                q_curr,
                dq_curr,
                u_des,
                dt,
                barriers,
                self.cfg.qp_R_diag,
                self.cfg.cbf_alpha,
                self.cfg.use_slack,
            )
            q_curr = q_curr + u_safe * dt
            dq_curr = u_safe.copy()
            safe_positions.append(q_curr.copy())
            safe_velocities.append(dq_curr.copy())
        positions_safe = np.vstack(safe_positions)
        velocities_safe = np.vstack(safe_velocities)
        if return_vel:
            return positions_safe, velocities_safe
        return positions_safe

    def _solve_safety_qp(
        self,
        q: np.ndarray,
        dq: np.ndarray,
        u_des: np.ndarray,
        dt: float,
        barriers: List[Callable[[np.ndarray], Tuple[float, np.ndarray]]],
        R_diag: float,
        alpha: float,
        use_slack: bool,
        slack_all: bool,
        bound_slack_weight: float,
        cbf_slack_weight: float,
    ) -> Optional[np.ndarray]:
        m = u_des.shape[0]
        slack_bounds = int(use_slack and slack_all)
        slack_cbf = int(use_slack)
        n_vars = m + slack_bounds + slack_cbf
        R_vec = np.full(m, R_diag, dtype=np.float64)
        P = np.zeros((n_vars, n_vars), dtype=np.float64)
        np.fill_diagonal(P[:m, :m], R_vec)
        q_vec = np.zeros(n_vars, dtype=np.float64)
        q_vec[:m] = -R_vec * u_des
        if slack_bounds:
            P[m, m] = bound_slack_weight
        if slack_cbf:
            idx = m + slack_bounds
            P[idx, idx] = cbf_slack_weight
        A_rows: List[np.ndarray] = []
        l_bounds: List[float] = []
        u_bounds: List[float] = []
        max_vel = self.cfg.max_joint_vel
        bound_slack_idx = m if slack_bounds else None
        cbf_slack_idx = m + slack_bounds if slack_cbf else None
        for i in range(m):
            row = np.zeros(n_vars, dtype=np.float64)
            row[i] = 1.0
            if bound_slack_idx is not None:
                row[bound_slack_idx] = -1.0
            A_rows.append(row)
            l_bounds.append(-np.inf)
            u_bounds.append(max_vel)
            row = np.zeros(n_vars, dtype=np.float64)
            row[i] = -1.0
            if bound_slack_idx is not None:
                row[bound_slack_idx] = -1.0
            A_rows.append(row)
            l_bounds.append(-np.inf)
            u_bounds.append(max_vel)
            upper_limit = (self.joint_upper[i] - q[i]) / max(dt, EPS)
            row = np.zeros(n_vars, dtype=np.float64)
            row[i] = 1.0
            if bound_slack_idx is not None:
                row[bound_slack_idx] = -1.0
            A_rows.append(row)
            l_bounds.append(-np.inf)
            u_bounds.append(upper_limit)
            lower_limit = (q[i] - self.joint_lower[i]) / max(dt, EPS)
            row = np.zeros(n_vars, dtype=np.float64)
            row[i] = -1.0
            if bound_slack_idx is not None:
                row[bound_slack_idx] = -1.0
            A_rows.append(row)
            l_bounds.append(-np.inf)
            u_bounds.append(lower_limit)
        for barrier in barriers:
            h_val, grad = barrier(q)
            if grad.shape[0] != m:
                continue
            row = np.zeros(n_vars, dtype=np.float64)
            row[:m] = -grad
            if cbf_slack_idx is not None:
                row[cbf_slack_idx] = -1.0
            upper = alpha * h_val + float(grad @ dq)
            A_rows.append(row)
            l_bounds.append(-np.inf)
            u_bounds.append(upper)
        if slack_bounds:
            row = np.zeros(n_vars, dtype=np.float64)
            row[bound_slack_idx] = 1.0
            A_rows.append(row)
            l_bounds.append(0.0)
            u_bounds.append(np.inf)
        if slack_cbf:
            row = np.zeros(n_vars, dtype=np.float64)
            row[cbf_slack_idx] = 1.0
            A_rows.append(row)
            l_bounds.append(0.0)
            u_bounds.append(np.inf)
        if not A_rows:
            return None
        A_mat = np.vstack(A_rows)
        l_vec = np.asarray(l_bounds, dtype=np.float64)
        u_vec = np.asarray(u_bounds, dtype=np.float64)
        solution = self._solve_qp(P, q_vec, A_mat, l_vec, u_vec)
        if solution is None:
            return None
        return solution[:m]

    def _log_qp_warning(self, message: str) -> None:
        now = time.time()
        if now - self._qp_warning_last > DEFAULT_WARNING_INTERVAL:
            LOGGER.warning(message)
            self._qp_warning_last = now

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
        q = np.asarray(q, dtype=np.float64)
        dq = np.asarray(dq, dtype=np.float64)
        u_des = np.asarray(u_des, dtype=np.float64)
        dt = float(dt)
        attempts: List[Tuple[np.ndarray, float]] = [(u_des.copy(), self.cfg.qp_slack_weight_bounds)]
        if use_slack and self.cfg.qp_slack_all:
            relaxed_weight = max(self.cfg.qp_slack_weight_bounds * 0.1, 1e-3)
            attempts.append((u_des * 0.8, relaxed_weight))
        result: Optional[np.ndarray] = None
        for idx, (u_candidate, bound_weight) in enumerate(attempts):
            solution = self._solve_safety_qp(
                q,
                dq,
                u_candidate,
                dt,
                barriers,
                R_diag,
                alpha,
                use_slack,
                self.cfg.qp_slack_all,
                bound_weight,
                self.cfg.qp_slack_weight_cbf,
            )
            if solution is not None:
                result = solution
                break
        if result is None:
            self._log_qp_warning("CBF-QP infeasible; falling back to clipped command.")
            return np.clip(u_des, -self.cfg.max_joint_vel, self.cfg.max_joint_vel)
        return np.clip(result, -self.cfg.max_joint_vel, self.cfg.max_joint_vel)

    def _solve_qp(self, P: np.ndarray, q_vec: np.ndarray, A: np.ndarray, l: np.ndarray, u: np.ndarray) -> Optional[np.ndarray]:
        P_sp = sparse.csc_matrix(P)
        A_sp = sparse.csc_matrix(A)
        if osqp is not None:
            prob = osqp.OSQP()
            prob.setup(P=P_sp, q=q_vec, A=A_sp, l=l, u=u, verbose=False, polish=True)
            res = prob.solve()
            if res.info.status in OSQP_SUCCESS_STATUSES and res.x is not None:
                return np.asarray(res.x, dtype=np.float64)
        if solve_qp is not None:
            G_list: List[np.ndarray] = []
            h_list: List[float] = []
            for idx in range(A.shape[0]):
                if np.isfinite(u[idx]):
                    G_list.append(A[idx])
                    h_list.append(u[idx])
                if np.isfinite(l[idx]):
                    G_list.append(-A[idx])
                    h_list.append(-l[idx])
            if G_list:
                G = np.vstack(G_list)
                h_vec = np.asarray(h_list, dtype=np.float64)
                try:
                    solution = solve_qp(P, q_vec, G, h_vec, None, None, None, None, solver="quadprog")
                    if solution is not None:
                        return np.asarray(solution, dtype=np.float64)
                except Exception:  # pragma: no cover - solver fallback
                    return None
        return None

    def save_policy(self, path: str) -> None:
        payload = {
            "policy_state": self.policy_net.state_dict(),
            "target_state": self.target_net.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "config": self.export_config(),
            "dmp": self.dmp_model.state_dict(),
            "state_norm": None if self.state_normalizer is None else self.state_normalizer.state_dict(),
            "options": [asdict(spec) for spec in self.options],
        }
        torch.save(payload, path)

    def load_policy(self, path: str) -> None:
        payload = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(payload["policy_state"])
        self.target_net.load_state_dict(payload.get("target_state", payload["policy_state"]))
        if "optimizer_state" in payload:
            self.optimizer.load_state_dict(payload["optimizer_state"])
        if "options" in payload:
            loaded_options = [OptionSpec(**spec) for spec in payload["options"]]
            self.options = loaded_options
        if "dmp" in payload:
            self.dmp_model.load_state_dict(payload["dmp"])
            self.dmp_weights = self.dmp_model.weights
            self.dmp_centers = self.dmp_model.centers
            self.dmp_widths = self.dmp_model.widths
        if self.state_normalizer is not None and payload.get("state_norm") is not None:
            self.state_normalizer.load_state_dict(payload["state_norm"])
        elif self.state_normalizer is not None:
            self.state_normalizer.reset()

    def export_config(self) -> Dict[str, Union[int, float, bool, Sequence[Union[str, OptionSpec]], Optional[str]]]:
        return asdict(self.cfg)

    @staticmethod
    def import_config(data: Dict[str, Union[int, float, bool, Sequence[str], Optional[str]]]) -> HRLConfig:
        return HRLConfig(**data)  # type: ignore[arg-type]

    def run_ros2_closed_loop(
        self,
        controller_ns: str = "/joint_trajectory_controller",
        hz: float = 10.0,
        topic_joint_states: str = "/joint_states",
        topic_joint_traj: str = "/joint_trajectory",
        frame_id: str = "",
        goal: Optional[np.ndarray] = None,
    ) -> None:
        try:  # pragma: no cover - optional dependency
            import rclpy
            from rclpy.node import Node
            from sensor_msgs.msg import JointState
            from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
            from builtin_interfaces.msg import Duration
            from std_msgs.msg import Float64MultiArray
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("ROS 2 not available") from exc

        cfg = self.cfg
        dt = cfg.dt
        horizon = cfg.horizon_steps
        if goal is not None:
            self.set_goal(goal)
        controller = self

        class HRLNode(Node):
            def __init__(self) -> None:
                super().__init__("hierarchical_rl_controller")
                self.state_msg: Optional[JointState] = None
                self.goal = controller._goal.copy()
                self.joint_names: Optional[Tuple[str, ...]] = None
                self.subscription = self.create_subscription(JointState, topic_joint_states, self.joint_state_callback, 10)
                self.publisher = self.create_publisher(JointTrajectory, topic_joint_traj, 10)
                self.create_subscription(Float64MultiArray, "~/set_goal", self.goal_callback, 10)
                timer_period = 1.0 / max(hz, EPS)
                self.timer = self.create_timer(timer_period, self.control_loop)

            def goal_callback(self, msg: Float64MultiArray) -> None:
                data = np.asarray(msg.data, dtype=np.float64)
                if data.size >= cfg.n_joints:
                    self.goal = data[: cfg.n_joints]
                    controller.set_goal(self.goal)
                else:
                    self.get_logger().warning("Received goal with insufficient dimension.")

            def joint_state_callback(self, msg: JointState) -> None:
                if len(msg.position) < cfg.n_joints:
                    self.get_logger().warning("JointState message has insufficient positions.")
                    return
                if self.joint_names is None:
                    self.joint_names = tuple(msg.name[: cfg.n_joints]) if msg.name else None
                elif msg.name and tuple(msg.name[: cfg.n_joints]) != self.joint_names:
                    self.get_logger().warning("Joint name ordering changed; ignoring update.")
                    return
                self.state_msg = msg

            def control_loop(self) -> None:
                if self.state_msg is None:
                    return
                msg = self.state_msg
                positions = np.array(msg.position[: cfg.n_joints], dtype=np.float64)
                if msg.velocity:
                    velocities = np.array(msg.velocity[: cfg.n_joints], dtype=np.float64)
                else:
                    velocities = np.zeros(cfg.n_joints, dtype=np.float64)
                base_state = np.concatenate([positions, velocities], dtype=np.float64)
                state_proc = controller._process_state(base_state, controller._goal, update_norm=False)
                option = controller.select_option(state_proc, explore=False)
                traj, vel = controller.option_to_dmp(option, positions, controller._goal, dt * horizon, horizon, return_vel=True)
                traj_msg = JointTrajectory()
                traj_msg.header.stamp = self.get_clock().now().to_msg()
                if frame_id:
                    traj_msg.header.frame_id = frame_id
                traj_msg.joint_names = list(self.joint_names) if self.joint_names is not None else list(msg.name[: cfg.n_joints])
                time_accum = 0.0
                for idx in range(traj.shape[0]):
                    pt = JointTrajectoryPoint()
                    pt.positions = traj[idx].tolist()
                    pt.velocities = vel[idx].tolist()
                    if idx > 0:
                        time_accum += dt
                    sec = int(time_accum)
                    nanosec = int((time_accum - sec) * 1e9)
                    pt.time_from_start = Duration(sec=sec, nanosec=nanosec)
                    traj_msg.points.append(pt)
                if traj_msg.points:
                    self.publisher.publish(traj_msg)

        rclpy.init()
        node = HRLNode()
        try:
            rclpy.spin(node)
        finally:
            node.destroy_node()
            rclpy.shutdown()


class ToyJointEnv:
    """Deterministic toy joint environment driven by short DMP rollouts."""

    def __init__(self, horizon: int = 4, dt: float = 0.1) -> None:
        self._dt = float(dt)
        self._horizon = max(int(horizon), 3)
        self._q = np.zeros(1, dtype=np.float64)
        self._dq = np.zeros(1, dtype=np.float64)
        self._goal = np.array([0.6], dtype=np.float64)
        self._step = 0
        self._max_steps = 60
        self._options = [
            OptionSpec("hold", goal_offset_scale=0.0),
            OptionSpec("dmp_small_step", goal_offset_scale=0.25),
            OptionSpec("dmp_medium_step", goal_offset_scale=0.5),
            OptionSpec("dmp_large_step", goal_offset_scale=1.0),
        ]
        self._dmp = DMPModel(1, 10, 4.0, 25.0, 6.25, 0.5)
        demo = np.linspace(0.0, self._goal[0], 40, dtype=np.float64)[:, None]
        self._dmp.fit_weights(demo, T=self._dt * (demo.shape[0] - 1))

    def reset(self) -> np.ndarray:
        self._q.fill(0.0)
        self._dq.fill(0.0)
        self._step = 0
        return np.array([self._q[0], self._dq[0]], dtype=np.float32)

    def option_to_dmp(self, option_id: int, q: np.ndarray, goal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        spec = self._options[int(option_id % len(self._options))]
        q = np.asarray(q, dtype=np.float64)
        goal = np.asarray(goal, dtype=np.float64)
        target = q + spec.goal_offset_scale * (goal - q)
        duration = self._dt * self._horizon * spec.duration_scale
        traj, vel = self._dmp.generate(q, target, duration, self._horizon, return_vel=True)
        return traj, vel

    def step(self, option_id: int) -> Tuple[np.ndarray, float, bool, dict]:
        option_id = int(option_id) % len(self._options)
        traj, vel = self.option_to_dmp(option_id, self._q, self._goal)
        self._q = traj[-1]
        self._dq = vel[-1]
        self._q = np.clip(self._q, -math.pi, math.pi)
        self._step += 1
        state = np.array([self._q[0], self._dq[0]], dtype=np.float32)
        error = float(np.linalg.norm(self._goal - self._q))
        reward = -error
        done = bool(error < 0.01 or self._step >= self._max_steps)
        info = {"goal": self.goal.copy()}
        return state, reward, done, info

    @property
    def state_dim(self) -> int:
        return 2

    @property
    def n_joints(self) -> int:
        return 1

    @property
    def goal(self) -> np.ndarray:
        return self._goal.astype(np.float32)


__all__ = ["HRLConfig", "HierarchicalRLController", "EnvProtocol", "ToyJointEnv"]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    env = ToyJointEnv()
    state_dim = env.state_dim + 2 * env.n_joints
    config = HRLConfig(
        n_joints=env.n_joints,
        state_dim=state_dim,
        options=["hold", "dmp_small_step", "dmp_medium_step", "dmp_large_step"],
        horizon_steps=6,
        dt=0.1,
        batch_size=32,
        replay_capacity=2_000,
        eps_decay_steps=1_000,
        seed=0,
    )
    controller = HierarchicalRLController(config)
    demo = np.linspace(0.0, float(env.goal[0]), 50, dtype=np.float64)[:, None]
    controller.dmp_fit_weights(demo, T=config.dt * (demo.shape[0] - 1))
    metrics = controller.train(env, total_steps=500, warmup=50, log_interval=200)
    eval_metrics = controller.evaluate(env, episodes=2)
    print("Training metrics:", metrics)
    print("Evaluation metrics:", eval_metrics)
