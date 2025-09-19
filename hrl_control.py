"""Hierarchical reinforcement learning controller module."""
from __future__ import annotations

from dataclasses import dataclass

from typing import Callable, Deque, List, Optional, Protocol, Tuple
import math
import random
from collections import deque

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover - required dependency
    raise ImportError("numpy is required for hrl_control.py") from exc

try:
    import torch
    from torch import nn
    from torch import optim
except ImportError as exc:  # pragma: no cover - required dependency
    raise ImportError("PyTorch >= 2.0 is required for hrl_control.py") from exc

try:
    from scipy import sparse
except ImportError as exc:  # pragma: no cover - required dependency
    raise ImportError("scipy is required for hrl_control.py") from exc

try:
    import osqp  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    osqp = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from qpsolvers import solve_qp  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    solve_qp = None  # type: ignore

EPS = 1e-8
DEFAULT_SLACK_PENALTY = 1e3
DEFAULT_HIDDEN_DIM = 128


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


class ReplayBuffer:
    """Simple replay buffer storing transitions (s, a, r, s', done)."""

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.buffer: Deque[Tuple[np.ndarray, int, float, np.ndarray, bool]] = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self.buffer)

    def push(self, transition: Tuple[np.ndarray, int, float, np.ndarray, bool]) -> None:
        self.buffer.append(transition)

    def sample(self, batch_size: int) -> List[Tuple[np.ndarray, int, float, np.ndarray, bool]]:
        batch_size = min(batch_size, len(self.buffer))
        return random.sample(self.buffer, batch_size)


class QNetwork(nn.Module):
    """Feed-forward network approximating the state-action value function."""

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = DEFAULT_HIDDEN_DIM) -> None:
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


@dataclass
class HRLConfig:
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


class HierarchicalRLController:
    """Modular controller stacking concept, skill, and safety layers.

    The controller maintains a discrete option policy via Deep Q-Networks,
    transforms selected options into smooth joint trajectories using Dynamic
    Movement Primitives (DMPs), and enforces hard constraints through a
    Control Barrier Function (CBF) Quadratic Program (QP) safety filter.
    """

    def __init__(self, cfg: HRLConfig) -> None:
        self.cfg = cfg
        self.device = self._select_device(cfg.device)
        self.policy_net = QNetwork(cfg.state_dim, len(cfg.options)).to(self.device)
        self.target_net = QNetwork(cfg.state_dim, len(cfg.options)).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=cfg.lr)
        self.replay_buffer = ReplayBuffer(cfg.replay_capacity)
        self.loss_fn = nn.MSELoss()
        self.eps = cfg.eps_start
        self.total_env_steps = 0
        self.training_steps = 0
        self.dmp_centers, self.dmp_widths = self._init_dmp_basis()
        self.dmp_weights = np.zeros((cfg.n_joints, cfg.dmp_n_basis), dtype=np.float64)
        self.joint_lower = -np.pi * np.ones(cfg.n_joints, dtype=np.float64)
        self.joint_upper = np.pi * np.ones(cfg.n_joints, dtype=np.float64)
        self._joint_barriers = self._build_joint_limit_barriers()

    @staticmethod
    def _select_device(device_str: Optional[str]) -> torch.device:
        if device_str == "cpu":
            return torch.device("cpu")
        if device_str == "cuda":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _init_dmp_basis(self) -> Tuple[np.ndarray, np.ndarray]:
        """Create shared radial basis functions for the canonical system."""
        grid = np.linspace(0.0, 1.0, self.cfg.dmp_n_basis)
        centers = np.exp(-self.cfg.dmp_alpha_s * grid)
        diffs = np.diff(centers)
        widths = np.empty_like(centers)
        widths[:-1] = 1.0 / (diffs ** 2 + EPS)
        widths[-1] = widths[-2] if len(widths) > 1 else 1.0
        return centers, widths

    def _canonical_trajectory(self, T: float, n_steps: int) -> np.ndarray:
        """Integrate the canonical phase variable :math:`s` for the DMP."""
        tau = max(T, EPS)
        dt = tau / max(n_steps - 1, 1)
        s = 1.0
        trajectory = np.zeros(n_steps, dtype=np.float64)
        for i in range(n_steps):
            trajectory[i] = s
            ds = (-self.cfg.dmp_alpha_s * s / tau) * dt
            s += ds
            s = max(s, 0.0)
        return trajectory

    def _build_joint_limit_barriers(self) -> List[Callable[[np.ndarray], Tuple[float, np.ndarray]]]:
        """Create CBF barrier functions for joint limits."""
        barriers: List[Callable[[np.ndarray], Tuple[float, np.ndarray]]] = []
        for idx in range(self.cfg.n_joints):
            upper = self.joint_upper[idx]
            lower = self.joint_lower[idx]

            def upper_barrier(q: np.ndarray, i: int = idx, limit: float = upper) -> Tuple[float, np.ndarray]:
                grad = np.zeros_like(q)
                grad[i] = -1.0
                return limit - q[i], grad

            def lower_barrier(q: np.ndarray, i: int = idx, limit: float = lower) -> Tuple[float, np.ndarray]:
                grad = np.zeros_like(q)
                grad[i] = 1.0
                return q[i] - limit, grad

            barriers.append(upper_barrier)
            barriers.append(lower_barrier)
        return barriers

    def select_option(self, state_np: np.ndarray, explore: bool = True) -> int:
        """Select an option via epsilon-greedy policy evaluation."""
        state_tensor = torch.from_numpy(state_np.astype(np.float32)).to(self.device).unsqueeze(0)
        if explore and random.random() < self.eps:
            return random.randrange(len(self.cfg.options))
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
            action = int(torch.argmax(q_values, dim=1).item())
        return action

    def push_transition(self, s: np.ndarray, a: int, r: float, s2: np.ndarray, done: bool) -> None:
        """Store a transition tuple in the replay buffer."""
        self.replay_buffer.push((s.astype(np.float32), a, float(r), s2.astype(np.float32), done))

    def _soft_update(self) -> None:
        """Soft-update target network parameters with coefficient :math:`\tau`."""
        tau = self.cfg.tau
        for target_param, param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

    def _update_epsilon(self) -> None:
        """Linearly decay exploration epsilon according to configuration."""
        fraction = min(1.0, self.total_env_steps / max(1, self.cfg.eps_decay_steps))
        self.eps = self.cfg.eps_start + fraction * (self.cfg.eps_end - self.cfg.eps_start)

    def train_step(self) -> dict:
        """Perform a single DQN update step using a minibatch from replay."""
        if len(self.replay_buffer) < self.cfg.batch_size:
            return {}
        batch = self.replay_buffer.sample(self.cfg.batch_size)
        states = torch.from_numpy(np.stack([b[0] for b in batch])).to(self.device)
        actions = torch.tensor([b[1] for b in batch], dtype=torch.long, device=self.device)
        rewards = torch.tensor([b[2] for b in batch], dtype=torch.float32, device=self.device)
        next_states = torch.from_numpy(np.stack([b[3] for b in batch])).to(self.device)
        dones = torch.tensor([float(b[4]) for b in batch], dtype=torch.float32, device=self.device)
        q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            target_next = self.target_net(next_states).max(1)[0]
            targets = rewards + self.cfg.gamma * (1.0 - dones) * target_next
        loss = self.loss_fn(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.training_steps += 1
        self._soft_update()
        return {"loss": float(loss.item())}

    def save_policy(self, path: str) -> None:
        """Serialize the policy network parameters to ``path``."""
        torch.save(self.policy_net.state_dict(), path)

    def load_policy(self, path: str) -> None:
        """Load policy parameters from ``path`` into policy and target networks."""
        state_dict = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(state_dict)
        self.target_net.load_state_dict(state_dict)

    def dmp_fit_weights(self, demo_positions: np.ndarray, T: float) -> None:
        r"""Fit DMP forcing term weights via locally weighted regression.

        The DMP dynamics follow :math:`\tau \dot{v} = \alpha_z (\beta_z (g-x) - v)
        + (g-x_0) f(s)` and :math:`\tau \dot{x} = v`, where the canonical
        system obeys :math:`\tau \dot{s} = -\alpha_s s` with :math:`s(0)=1`.
        The forcing term is approximated by normalized radial basis functions
        :math:`f(s) = \frac{\sum_i \psi_i(s) w_i s}{\sum_i \psi_i(s) + \varepsilon}`
        (Schaal 2006). The regression solves for weights :math:`w_i` that best
        reproduce a demonstration trajectory ``demo_positions`` over duration
        ``T``.
        """
        if demo_positions.shape[1] != self.cfg.n_joints:
            raise ValueError("Demo trajectory joint dimension mismatch.")
        n_steps = demo_positions.shape[0]
        tau = max(T, EPS)
        dt = tau / max(n_steps - 1, 1)
        velocities = np.gradient(demo_positions, dt, axis=0, edge_order=2)
        accelerations = np.gradient(velocities, dt, axis=0, edge_order=2)
        s_traj = self._canonical_trajectory(T, n_steps)
        diff = s_traj[:, None] - self.dmp_centers[None, :]
        psi_matrix = np.exp(-self.dmp_widths[None, :] * diff ** 2)
        sum_psi = psi_matrix.sum(axis=1, keepdims=True) + EPS
        Phi = (psi_matrix / sum_psi) * s_traj[:, None]
        g = demo_positions[-1]
        x0 = demo_positions[0]
        for j in range(self.cfg.n_joints):
            denom = (g[j] - x0[j]) + EPS
            feedback = self.cfg.dmp_alpha_z * (
                self.cfg.dmp_beta_z * (g[j] - demo_positions[:, j]) - tau * velocities[:, j]
            )
            f_target = (tau * accelerations[:, j] - feedback) / denom
            reg = 1e-6 * np.eye(self.cfg.dmp_n_basis)
            A = Phi.T @ Phi + reg
            b = Phi.T @ f_target
            self.dmp_weights[j] = np.linalg.solve(A, b)

    def dmp_generate(self, q0: np.ndarray, g: np.ndarray, T: float, n_steps: int) -> np.ndarray:
        r"""Roll out the learned DMP dynamics to obtain a joint trajectory.

        Parameters mirror the formulation in :meth:`dmp_fit_weights`. Given the
        learned weights :math:`w_i`, the method numerically integrates the
        transformation system for each joint with shared canonical state
        :math:`s` to produce ``n_steps`` samples over horizon ``T``.
        """
        if q0.shape[0] != self.cfg.n_joints:
            raise ValueError("Initial joint dimension mismatch.")
        if g.shape[0] != self.cfg.n_joints:
            raise ValueError("Goal joint dimension mismatch.")
        tau = max(T, EPS)
        dt = tau / max(n_steps - 1, 1)
        s_traj = self._canonical_trajectory(T, n_steps)
        positions = np.zeros((n_steps, self.cfg.n_joints), dtype=np.float64)
        x = q0.astype(np.float64).copy()
        v = np.zeros_like(x)
        for k in range(n_steps):
            positions[k] = x
            s_val = s_traj[k]
            psi = np.exp(-self.dmp_widths * (s_val - self.dmp_centers) ** 2)
            sum_psi = psi.sum() + EPS
            f = (self.dmp_weights @ psi) * s_val / sum_psi
            if k == n_steps - 1:
                break
            v_dot = (self.cfg.dmp_alpha_z * (self.cfg.dmp_beta_z * (g - x) - v) + (g - q0) * f) / tau
            v = v + v_dot * dt
            x = x + (v / tau) * dt
        return positions

    def option_to_dmp(self, option_id: int, q: np.ndarray, g: np.ndarray, T: float, n_steps: int) -> np.ndarray:
        r"""Map a discrete option to a DMP-generated joint trajectory.

        The method scales the goal displacement according to the chosen option,
        generates the nominal DMP rollout, computes discrete velocities
        :math:`u_{\text{des}} = \mathrm{clip}\left(\frac{q_{t+1}-q_t}{\Delta t}\right)`,
        and forwards each command through :meth:`safety_layer_filter` to obtain
        a safe path.
        """
        option_name = self.cfg.options[option_id]
        scaling = 0.0
        if option_name == "hold":
            scaling = 0.0
        elif "small" in option_name:
            scaling = 0.25
        elif "medium" in option_name:
            scaling = 0.5
        elif "large" in option_name:
            scaling = 1.0
        else:
            scaling = 0.5
        g_target = q + scaling * (g - q)
        g_target = np.clip(g_target, self.joint_lower, self.joint_upper)
        dmp_traj = self.dmp_generate(q, g_target, T, n_steps)
        dt = T / max(n_steps - 1, 1)
        safe_positions = [q.astype(np.float64).copy()]
        q_curr = q.astype(np.float64).copy()
        dq_curr = np.zeros_like(q_curr)
        barriers = list(self._joint_barriers)
        for k in range(n_steps - 1):
            q_des_curr = dmp_traj[k]
            q_des_next = dmp_traj[k + 1]
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
            dq_curr = u_safe
            safe_positions.append(q_curr.copy())
        return np.vstack(safe_positions)

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
        r"""Filter a desired velocity command through CBF-QP constraints.

        The CBF constraint for each barrier :math:`h(x)` enforces
        :math:`\nabla h(x)^\top (f(x) + g(x) u) + \alpha h(x) \ge 0` with
        :math:`f(x)` approximated by the current joint velocity ``dq`` and
        :math:`g(x)=I`. A slack variable :math:`\delta \ge 0` is optionally
        added to guarantee feasibility with penalty :math:`\lambda_\delta`.
        The QP minimizes

        .. math::

            \tfrac{1}{2} (u-u_{\text{des}})^\top R (u-u_{\text{des}})
            + \tfrac{1}{2} \lambda_\delta \delta^2

        subject to joint velocity/position bounds and the CBF inequalities
        (Ames 2016).
        """
        m = u_des.shape[0]
        u_clipped = np.clip(u_des, -self.cfg.max_joint_vel, self.cfg.max_joint_vel)
        n_vars = m + (1 if use_slack else 0)
        P = np.zeros((n_vars, n_vars), dtype=np.float64)
        P[np.arange(m), np.arange(m)] = R_diag
        q_vec = np.zeros(n_vars, dtype=np.float64)
        q_vec[:m] = -R_diag * u_des
        if use_slack:
            P[-1, -1] = DEFAULT_SLACK_PENALTY
        A_rows: List[np.ndarray] = []
        l_bounds: List[float] = []
        u_bounds: List[float] = []
        for j in range(m):
            row = np.zeros(n_vars, dtype=np.float64)
            row[j] = 1.0
            A_rows.append(row)
            l_bounds.append(-math.inf)
            u_bounds.append(self.cfg.max_joint_vel)
            row = np.zeros(n_vars, dtype=np.float64)
            row[j] = -1.0
            A_rows.append(row)
            l_bounds.append(-math.inf)
            u_bounds.append(self.cfg.max_joint_vel)
            upper_limit = (self.joint_upper[j] - q[j]) / max(dt, EPS)
            row = np.zeros(n_vars, dtype=np.float64)
            row[j] = 1.0
            A_rows.append(row)
            l_bounds.append(-math.inf)
            u_bounds.append(upper_limit)
            lower_limit = (q[j] - self.joint_lower[j]) / max(dt, EPS)
            row = np.zeros(n_vars, dtype=np.float64)
            row[j] = -1.0
            A_rows.append(row)
            l_bounds.append(-math.inf)
            u_bounds.append(lower_limit)
        for barrier in barriers:
            h_val, grad = barrier(q)
            if grad.shape[0] != m:
                continue
            row = np.zeros(n_vars, dtype=np.float64)
            row[:m] = -grad
            upper = alpha * h_val + float(grad @ dq)
            if use_slack:
                row[-1] = -1.0
            A_rows.append(row)
            l_bounds.append(-math.inf)
            u_bounds.append(upper)
        if use_slack:
            row = np.zeros(n_vars, dtype=np.float64)
            row[-1] = 1.0
            A_rows.append(row)
            l_bounds.append(0.0)
            u_bounds.append(math.inf)
        if not A_rows:
            return u_clipped
        A_mat = np.vstack(A_rows)
        l_vec = np.array(l_bounds, dtype=np.float64)
        u_vec = np.array(u_bounds, dtype=np.float64)
        solution = self._solve_qp(P, q_vec, A_mat, l_vec, u_vec)
        if solution is None:
            return u_clipped
        u_result = solution[:m]
        return np.clip(u_result, -self.cfg.max_joint_vel, self.cfg.max_joint_vel)

    def _solve_qp(self, P: np.ndarray, q_vec: np.ndarray, A: np.ndarray, l: np.ndarray, u: np.ndarray) -> Optional[np.ndarray]:
        """Solve a QP using OSQP or ``qpsolvers`` as fallback."""
        P_sp = sparse.csc_matrix(P)
        A_sp = sparse.csc_matrix(A)
        if osqp is not None:
            prob = osqp.OSQP()
            prob.setup(P=P_sp, q=q_vec, A=A_sp, l=l, u=u, verbose=False, polish=True)
            res = prob.solve()
            if res.info.status_val in {osqp.constant('OSQP_SOLVED'), osqp.constant('OSQP_SOLVED_INACCURATE')}:
                return np.asarray(res.x)
        if solve_qp is not None:
            G_list = []
            h_list = []
            for i in range(A.shape[0]):
                if math.isfinite(u[i]):
                    G_list.append(A[i])
                    h_list.append(u[i])
                if math.isfinite(l[i]):
                    G_list.append(-A[i])
                    h_list.append(-l[i])
            if G_list:
                G = np.vstack(G_list)
                h = np.array(h_list)
                try:
                    x = solve_qp(P, q_vec, G, h, None, None, None, None, solver="quadprog")
                    if x is not None:
                        return np.asarray(x)
                except Exception:  # pragma: no cover - solver fallback
                    return None
        return None

    def train(
        self,
        env: EnvProtocol,
        total_steps: int,
        warmup: int = 1_000,
        target_update_interval: int = 1_000,
        log_interval: int = 1_000,
    ) -> dict:
        """Train the concept layer with environment rollouts.

        The loop executes ``total_steps`` interactions, collects transitions in
        the replay buffer, performs DQN updates with mini-batches of size
        ``cfg.batch_size``, and decays epsilon linearly after an initial
        ``warmup`` period of random option exploration.
        """
        state = env.reset().astype(np.float32)
        episode_reward = 0.0
        episode = 1
        rewards: List[float] = []
        losses: List[float] = []
        for step in range(1, total_steps + 1):
            if step <= warmup:
                action = random.randrange(len(self.cfg.options))
            else:
                action = self.select_option(state, explore=True)
            next_state, reward, done, _ = env.step(action)
            next_state = next_state.astype(np.float32)
            self.push_transition(state, action, reward, next_state, done)
            info = self.train_step()
            if info.get("loss") is not None:
                losses.append(info["loss"])
            episode_reward += reward
            self.total_env_steps += 1
            self._update_epsilon()
            if step % target_update_interval == 0:
                self._soft_update()
            state = next_state
            if done:
                rewards.append(episode_reward)
                state = env.reset().astype(np.float32)
                episode_reward = 0.0
                episode += 1
            if log_interval and step % log_interval == 0:
                avg_reward = float(np.mean(rewards[-10:])) if rewards else 0.0
                avg_loss = float(np.mean(losses[-10:])) if losses else 0.0
                print(f"Step {step}: avg_reward={avg_reward:.3f}, avg_loss={avg_loss:.5f}, eps={self.eps:.3f}")
        metrics = {
            "episodes": episode,
            "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
            "mean_loss": float(np.mean(losses)) if losses else 0.0,
        }
        return metrics

    def evaluate(self, env: EnvProtocol, episodes: int = 5) -> dict:
        """Evaluate the greedy policy over ``episodes`` episodes."""
        rewards: List[float] = []
        original_mode = self.policy_net.training
        self.policy_net.eval()
        with torch.no_grad():
            for _ in range(episodes):
                state = env.reset().astype(np.float32)
                done = False
                total_reward = 0.0
                steps = 0
                while not done and steps < 200:
                    action = self.select_option(state, explore=False)
                    next_state, reward, done, _ = env.step(action)
                    state = next_state.astype(np.float32)
                    total_reward += reward
                    steps += 1
                rewards.append(total_reward)
        if original_mode:
            self.policy_net.train()
        return {
            "episodes": episodes,
            "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
            "rewards": rewards,
        }

    def run_ros2_closed_loop(
        self,
        controller_ns: str = "/joint_trajectory_controller",
        hz: float = 10.0,
        topic_joint_states: str = "/joint_states",
        topic_joint_traj: str = "/joint_trajectory",
    ) -> None:
        """Run a ROS 2 node that executes the hierarchical policy online.

        The node subscribes to ``topic_joint_states``, forms an observation,
        selects options through the trained DQN, synthesizes safe trajectories
        via DMP + CBF-QP, and publishes ``trajectory_msgs/JointTrajectory`` to
        ``topic_joint_traj`` within the specified controller namespace.
        """
        try:  # pragma: no cover - optional dependency
            import rclpy
            from rclpy.node import Node
            from sensor_msgs.msg import JointState
            from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
            from builtin_interfaces.msg import Duration
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("ROS 2 not available") from exc

        cfg = self.cfg
        dt = cfg.dt
        horizon = cfg.horizon_steps
        goal = np.zeros(cfg.n_joints, dtype=np.float64)

        class HRLNode(Node):
            def __init__(self, controller: HierarchicalRLController) -> None:
                super().__init__("hierarchical_rl_controller")
                self.controller = controller
                self.state_msg: Optional[JointState] = None
                self.goal = goal
                self.subscription = self.create_subscription(
                    JointState,
                    topic_joint_states,
                    self.joint_state_callback,
                    10,
                )
                self.publisher = self.create_publisher(JointTrajectory, topic_joint_traj, 10)
                timer_period = 1.0 / max(hz, EPS)
                self.timer = self.create_timer(timer_period, self.control_loop)

            def joint_state_callback(self, msg: JointState) -> None:
                self.state_msg = msg

            def control_loop(self) -> None:
                if self.state_msg is None:
                    return
                msg = self.state_msg
                positions = np.array(msg.position[: cfg.n_joints], dtype=np.float32)
                if msg.velocity:
                    velocities = np.array(
                        msg.velocity[: cfg.n_joints], dtype=np.float32
                    )
                else:
                    velocities = np.zeros(cfg.n_joints, dtype=np.float32)
                state = np.concatenate([positions, velocities], dtype=np.float32)
                if state.shape[0] < cfg.state_dim:
                    state = np.pad(state, (0, cfg.state_dim - state.shape[0]))
                else:
                    state = state[: cfg.state_dim]
                option = self.controller.select_option(state, explore=False)
                traj = self.controller.option_to_dmp(
                    option,
                    positions.astype(np.float64),
                    self.goal.astype(np.float64),
                    dt * horizon,
                    horizon,
                )
                traj_msg = JointTrajectory()
                traj_msg.header.stamp = self.get_clock().now().to_msg()
                traj_msg.header.frame_id = controller_ns
                traj_msg.joint_names = list(msg.name[: cfg.n_joints])
                time_accum = 0.0
                prev = traj[0]
                for idx in range(1, traj.shape[0]):
                    pt = JointTrajectoryPoint()
                    pt.positions = traj[idx].tolist()
                    vel = (traj[idx] - prev) / max(dt, EPS)
                    pt.velocities = vel.tolist()
                    time_accum += dt
                    sec = int(time_accum)
                    nanosec = int((time_accum - sec) * 1e9)
                    pt.time_from_start = Duration(sec=sec, nanosec=nanosec)
                    traj_msg.points.append(pt)
                    prev = traj[idx]
                if traj_msg.points:
                    self.publisher.publish(traj_msg)

        rclpy.init()
        node = HRLNode(self)
        try:
            rclpy.spin(node)
        finally:
            node.destroy_node()
            rclpy.shutdown()


class ToyJointEnv:
    """Deterministic toy environment for smoke testing."""

    def __init__(self) -> None:
        self._q = 0.0
        self._goal = np.array([0.5], dtype=np.float32)
        self._step = 0
        self._max_steps = 50
        self._options = ["hold", "dmp_small_step", "dmp_medium_step", "dmp_large_step"]

    def reset(self) -> np.ndarray:
        self._q = 0.0
        self._step = 0
        return np.array([self._q], dtype=np.float32)

    def step(self, option_id: int) -> Tuple[np.ndarray, float, bool, dict]:
        option_id = int(option_id)
        option_id = max(0, min(option_id, len(self._options) - 1))
        if option_id == 0:
            delta = 0.0
        elif option_id == 1:
            delta = 0.05
        elif option_id == 2:
            delta = 0.1
        else:
            delta = 0.2
        self._q += delta
        self._q = float(np.clip(self._q, -math.pi, math.pi))
        self._step += 1
        state = np.array([self._q], dtype=np.float32)
        reward = -abs(self._q - self._goal[0])
        done = abs(self._q - self._goal[0]) < 0.01 or self._step >= self._max_steps
        return state, reward, done, {}

    @property
    def state_dim(self) -> int:
        return 1

    @property
    def n_joints(self) -> int:
        return 1

    @property
    def goal(self) -> np.ndarray:
        return self._goal


__all__ = ["HRLConfig", "HierarchicalRLController", "EnvProtocol", "ToyJointEnv"]


if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    env = ToyJointEnv()
    config = HRLConfig(
        n_joints=env.n_joints,
        state_dim=env.state_dim,
        options=["hold", "dmp_small_step", "dmp_medium_step", "dmp_large_step"],
        horizon_steps=8,
        dt=0.05,
        replay_capacity=5_000,
        batch_size=32,
        eps_decay_steps=2_000,
    )
    controller = HierarchicalRLController(config)
    demo = np.linspace(0.0, env.goal[0], 40, dtype=np.float64)[:, None]
    controller.dmp_fit_weights(demo, T=1.0)
    metrics = controller.train(env, total_steps=1_000, warmup=50, target_update_interval=200, log_interval=500)
    eval_metrics = controller.evaluate(env, episodes=3)
    print("Training metrics:", metrics)
    print("Evaluation metrics:", eval_metrics)
