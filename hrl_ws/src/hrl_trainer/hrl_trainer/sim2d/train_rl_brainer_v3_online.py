from __future__ import annotations

import argparse
import json
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml

from .env import Sim2DEnv
from .planner import HighLevelHeuristicPlannerV2


def _oracle_action(obs: np.ndarray, subgoal_xy: np.ndarray, speed_hint: float = 0.7) -> np.ndarray:
    x, y, yaw, v, omega = obs[:5]
    dx, dy = float(subgoal_xy[0] - x), float(subgoal_xy[1] - y)
    desired_heading = float(np.arctan2(dy, dx))
    heading_err = (desired_heading - yaw + np.pi) % (2 * np.pi) - np.pi
    dist = float(np.hypot(dx, dy))

    v_target = np.clip(speed_hint * dist, 0.0, 1.2)
    omega_target = np.clip(1.5 * heading_err, -1.6, 1.6)
    return np.array([v_target, omega_target], dtype=np.float32)


class MemoryLSTMPolicy(nn.Module):
    def __init__(self, in_dim: int, hid: int = 128):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hid, num_layers=1, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hid, hid), nn.ReLU(), nn.Linear(hid, 2))

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        # seq: [B, T, D]
        out, _ = self.lstm(seq)
        h = out[:, -1, :]
        return self.head(h)


@dataclass
class Sample:
    seq: np.ndarray  # [T, D]
    target: np.ndarray  # [2] desired v, omega


@dataclass
class Transition:
    seq: np.ndarray  # [T, D]
    next_seq: np.ndarray  # [T, D]
    target: np.ndarray  # [2] oracle desired v, omega
    reward: float
    done: float


@dataclass
class TransitionFF:
    feat: np.ndarray  # [D]
    next_feat: np.ndarray  # [D]
    target: np.ndarray  # [2] oracle desired v, omega
    reward: float
    done: float


@dataclass
class MemorySample:
    key: np.ndarray  # [5]
    action: np.ndarray  # [2]
    quality: float
    created_step: int


class ReplayBuffer:
    """Minimal replay for recurrent transitions."""

    def __init__(self, capacity: int):
        self._buf: deque[Transition] = deque(maxlen=int(capacity))

    def add(self, tr: Transition) -> None:
        self._buf.append(tr)

    def __len__(self) -> int:
        return len(self._buf)

    def sample(self, batch_size: int) -> list[Transition]:
        idx = np.random.randint(0, len(self._buf), size=batch_size)
        return [self._buf[int(i)] for i in idx]


class OnlineRecurrentPolicy(nn.Module):
    """Recurrent tactical policy with actor+value heads for online updates."""

    def __init__(self, in_dim: int, hid: int = 128):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hid, num_layers=1, batch_first=True)
        self.actor = nn.Sequential(nn.Linear(hid, hid), nn.ReLU(), nn.Linear(hid, 2))
        self.value = nn.Sequential(nn.Linear(hid, hid), nn.ReLU(), nn.Linear(hid, 1))

    def forward(self, seq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out, _ = self.lstm(seq)
        h = out[:, -1, :]
        action = self.actor(h)
        value = self.value(h).squeeze(-1)
        return action, value


class OnlineTacticalBaseline(nn.Module):
    """Non-recurrent tactical baseline with actor+value heads."""

    def __init__(self, in_dim: int, hid: int = 128, linear_only: bool = False):
        super().__init__()
        if linear_only:
            self.backbone = nn.Identity()
            out_dim = in_dim
        else:
            self.backbone = nn.Sequential(nn.Linear(in_dim, hid), nn.ReLU(), nn.Linear(hid, hid), nn.ReLU())
            out_dim = hid
        self.actor = nn.Linear(out_dim, 2)
        self.value = nn.Linear(out_dim, 1)

    def forward(self, feat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.backbone(feat)
        action = self.actor(h)
        value = self.value(h).squeeze(-1)
        return action, value


def _rbf_controller(obs: np.ndarray, desired_vo: np.ndarray) -> np.ndarray:
    """Low-level RBF-style controller: feature basis over tracking errors."""
    v, omega = float(obs[3]), float(obs[4])
    ev = float(desired_vo[0] - v)
    ew = float(desired_vo[1] - omega)

    centers = np.array([[0.0, 0.0], [0.3, 0.0], [-0.3, 0.0], [0.0, 0.3], [0.0, -0.3], [0.3, 0.3], [-0.3, -0.3]])
    widths = 8.0
    e = np.array([ev, ew])[None, :]
    phi = np.exp(-widths * np.sum((e - centers) ** 2, axis=1))

    # Hand-tuned blend: linear term + RBF compensation
    a_lin = 1.25 * ev + 0.25 * (phi[1] - phi[2])
    a_ang = 1.35 * ew + 0.20 * (phi[3] - phi[4])
    return np.clip(np.array([a_lin, a_ang], dtype=np.float32), -1.0, 1.0)


def _build_feature(obs: np.ndarray, packet: dict, mem_action: np.ndarray | None) -> np.ndarray:
    # base obs (10 dims currently) + subgoal delta(2) + speed_hint(1) + memory action(2)
    dx = packet["subgoal_xy"][0] - obs[0]
    dy = packet["subgoal_xy"][1] - obs[1]
    sh = float(packet.get("speed_hint", 0.6))
    if mem_action is None:
        mem_action = np.zeros(2, dtype=np.float32)
    return np.concatenate([obs.astype(np.float32), np.array([dx, dy, sh], dtype=np.float32), mem_action.astype(np.float32)], dtype=np.float32)


def _retrieve_memory_action(obs: np.ndarray, memory_bank: deque[tuple[np.ndarray, np.ndarray]], memory_k: int) -> np.ndarray | None:
    if not memory_bank:
        return None
    key = obs[:5]
    keys = np.stack([k for k, _ in memory_bank], axis=0)
    vals = np.stack([v for _, v in memory_bank], axis=0)
    d2 = np.sum((keys - key[None, :]) ** 2, axis=1)
    k = min(max(int(memory_k), 1), len(memory_bank))
    nn_idx = np.argpartition(d2, kth=k - 1)[:k]
    # Inverse-distance weighting keeps retrieval stable when closest sample is noisy.
    w = 1.0 / (np.sqrt(d2[nn_idx]) + 1e-4)
    w = w / np.sum(w)
    return np.sum(vals[nn_idx] * w[:, None], axis=0).astype(np.float32)


def _insert_memory_sample(memory_bank: list[MemorySample], sample: MemorySample, capacity: int) -> int:
    evicted = 0
    if len(memory_bank) >= capacity:
        min_idx = int(np.argmin(np.array([m.quality for m in memory_bank], dtype=np.float32)))
        memory_bank.pop(min_idx)
        evicted = 1
    memory_bank.append(sample)
    return evicted


def _retrieve_memory_action_scored(
    obs: np.ndarray,
    memory_bank: list[MemorySample],
    memory_k: int,
    retrieval_scores: list[float],
) -> np.ndarray | None:
    if not memory_bank:
        return None
    key = obs[:5]
    keys = np.stack([m.key for m in memory_bank], axis=0)
    vals = np.stack([m.action for m in memory_bank], axis=0)
    quality = np.array([max(float(m.quality), 1e-6) for m in memory_bank], dtype=np.float32)

    d2 = np.sum((keys - key[None, :]) ** 2, axis=1)
    sim = 1.0 / (1.0 + np.sqrt(np.maximum(d2, 0.0)))
    score = sim * quality

    k = min(max(int(memory_k), 1), len(memory_bank))
    nn_idx = np.argpartition(-score, kth=k - 1)[:k]
    w = np.maximum(score[nn_idx], 1e-8)
    w = w / np.sum(w)
    retrieval_scores.append(float(np.max(score[nn_idx])))
    return np.sum(vals[nn_idx] * w[:, None], axis=0).astype(np.float32)


def _retrieve_memory_action_with_score(
    obs: np.ndarray,
    memory_bank: list[MemorySample],
    memory_k: int,
) -> tuple[np.ndarray | None, float | None]:
    if not memory_bank:
        return None, None
    key = obs[:5]
    keys = np.stack([m.key for m in memory_bank], axis=0)
    vals = np.stack([m.action for m in memory_bank], axis=0)
    quality = np.array([max(float(m.quality), 1e-6) for m in memory_bank], dtype=np.float32)

    d2 = np.sum((keys - key[None, :]) ** 2, axis=1)
    sim = 1.0 / (1.0 + np.sqrt(np.maximum(d2, 0.0)))
    score = sim * quality

    k = min(max(int(memory_k), 1), len(memory_bank))
    nn_idx = np.argpartition(-score, kth=k - 1)[:k]
    w = np.maximum(score[nn_idx], 1e-8)
    w = w / np.sum(w)
    act = np.sum(vals[nn_idx] * w[:, None], axis=0).astype(np.float32)
    return act, float(np.max(score[nn_idx]))


def _score_stats(scores: list[float]) -> dict:
    if not scores:
        return {"count": 0, "mean": None, "std": None, "min": None, "max": None, "p50": None, "p90": None}
    arr = np.array(scores, dtype=np.float32)
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
    }


def _clip_desired(desired: np.ndarray) -> np.ndarray:
    return np.array([np.clip(desired[0], 0.0, 1.2), np.clip(desired[1], -1.6, 1.6)], dtype=np.float32)


def _deterministic_core_mapping(obs: np.ndarray, packet: dict, heading_gain: float = 1.5) -> np.ndarray:
    """Deterministic tactical core: planner subgoal -> desired [v, omega]."""
    dx = float(packet["subgoal_xy"][0] - obs[0])
    dy = float(packet["subgoal_xy"][1] - obs[1])
    yaw = float(obs[2])
    dist = float(np.hypot(dx, dy))
    desired_heading = float(np.arctan2(dy, dx))
    heading_err = (desired_heading - yaw + np.pi) % (2 * np.pi) - np.pi
    speed_hint = float(packet.get("speed_hint", 0.7))
    return _clip_desired(np.array([speed_hint * dist, heading_gain * heading_err], dtype=np.float32))


def _bounded_delta(delta: np.ndarray, delta_clip: float) -> np.ndarray:
    lim = float(abs(delta_clip))
    return np.clip(delta.astype(np.float32), -lim, lim)


def _done_reason(info: dict) -> str:
    if bool(info.get("success", False)):
        return "success"
    if bool(info.get("collided", False)):
        return "collision"
    return "timeout"


def _timeout_bin(distance: float, near_thresh: float, mid_thresh: float) -> str:
    if distance < near_thresh:
        return "near"
    if distance < mid_thresh:
        return "mid"
    return "far"


def train_and_eval_offline_legacy(cfg: dict) -> dict:
    seed = int(cfg.get("seed", 0))
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env_cfg = cfg.get("env", {})
    planner = HighLevelHeuristicPlannerV2(waypoint_scale=float(cfg.get("waypoint_scale", 0.35)))
    seq_len = int(cfg.get("seq_len", 8))

    # ====== Stage 1: collect oracle-supervised dataset ======
    collect_eps = int(cfg.get("collect_episodes", 120))
    max_steps = int(env_cfg.get("max_steps", 250))

    memory_bank: deque[tuple[np.ndarray, np.ndarray]] = deque(maxlen=6000)  # (obs_key, desired_vo)
    dataset: list[Sample] = []

    for ep in range(collect_eps):
        env = Sim2DEnv(seed=seed + ep, max_steps=max_steps, level=str(env_cfg.get("disturbance_level", "medium")), obstacle_count=int(env_cfg.get("obstacle_count", 0)))
        obs = env.reset()
        hist: deque[np.ndarray] = deque(maxlen=seq_len)

        for _ in range(max_steps):
            packet = planner.plan(obs)
            # memory retrieval: nearest by first 5 dims
            mem_action = None
            if memory_bank:
                key = obs[:5]
                keys = np.stack([k for k, _ in memory_bank], axis=0)
                idx = int(np.argmin(np.sum((keys - key[None, :]) ** 2, axis=1)))
                mem_action = memory_bank[idx][1]

            feat = _build_feature(obs, packet, mem_action)
            hist.append(feat)
            if len(hist) < seq_len:
                desired = _oracle_action(obs, packet["subgoal_xy"], packet.get("speed_hint", 0.7))
                memory_bank.append((obs[:5].copy(), desired.copy()))
                action = _rbf_controller(obs, desired)
                obs, _, done, _ = env.step(action)
                if done:
                    break
                continue

            desired = _oracle_action(obs, packet["subgoal_xy"], packet.get("speed_hint", 0.7))
            dataset.append(Sample(seq=np.stack(list(hist), axis=0), target=desired))
            memory_bank.append((obs[:5].copy(), desired.copy()))

            action = _rbf_controller(obs, desired)
            obs, _, done, _ = env.step(action)
            if done:
                break

    if not dataset:
        raise RuntimeError("Dataset collection failed; no training samples generated.")

    in_dim = int(dataset[0].seq.shape[-1])
    model = MemoryLSTMPolicy(in_dim=in_dim, hid=int(cfg.get("hidden_dim", 128))).to(device)
    opt = optim.Adam(model.parameters(), lr=float(cfg.get("lr", 2e-4)))
    mse = nn.MSELoss()

    # ====== Stage 2: supervised train (memory+LSTM tactical) ======
    batch_size = int(cfg.get("batch_size", 256))
    epochs = int(cfg.get("epochs", 20))
    losses = []

    seqs = torch.tensor(np.stack([s.seq for s in dataset], axis=0), dtype=torch.float32, device=device)
    tgts = torch.tensor(np.stack([s.target for s in dataset], axis=0), dtype=torch.float32, device=device)

    n = seqs.shape[0]
    for _ in range(epochs):
        idx = torch.randperm(n, device=device)
        for i in range(0, n, batch_size):
            b = idx[i : i + batch_size]
            pred = model(seqs[b])
            loss = mse(pred, tgts[b])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(float(loss.item()))

    # ====== Stage 3: closed-loop eval with RBF execution ======
    eval_eps = int(cfg.get("eval_episodes", 40))
    success = 0
    conv = []
    rmse = []
    rec = []
    effort = []

    for ep in range(eval_eps):
        env = Sim2DEnv(seed=seed + 5000 + ep, max_steps=max_steps, level=str(env_cfg.get("disturbance_level_eval", env_cfg.get("disturbance_level", "medium"))), obstacle_count=int(env_cfg.get("obstacle_count", 0)))
        obs = env.reset()
        hist: deque[np.ndarray] = deque(maxlen=seq_len)
        dist_hist = []
        e_sum = 0.0
        disturbed_step = None
        recover = None

        for t in range(max_steps):
            packet = planner.plan(obs)
            mem_action = None
            if memory_bank:
                key = obs[:5]
                keys = np.stack([k for k, _ in memory_bank], axis=0)
                idx = int(np.argmin(np.sum((keys - key[None, :]) ** 2, axis=1)))
                mem_action = memory_bank[idx][1]

            feat = _build_feature(obs, packet, mem_action)
            hist.append(feat)
            if len(hist) < seq_len:
                desired = _oracle_action(obs, packet["subgoal_xy"], packet.get("speed_hint", 0.7))
            else:
                seq = torch.tensor(np.stack(list(hist), axis=0)[None, ...], dtype=torch.float32, device=device)
                with torch.no_grad():
                    desired = model(seq).squeeze(0).cpu().numpy().astype(np.float32)

            action = _rbf_controller(obs, desired)
            obs, _, done, info = env.step(action)
            dcur = float(info["distance"])
            dist_hist.append(dcur)
            e_sum += float(np.linalg.norm(action))

            if dcur > 1.0 and disturbed_step is None:
                disturbed_step = t
            if disturbed_step is not None and recover is None and dcur < 0.3:
                recover = t - disturbed_step

            if done:
                if info.get("success", False):
                    success += 1
                    conv.append(t + 1)
                break

        rmse.append(float(np.sqrt(np.mean(np.square(dist_hist)))) if dist_hist else 0.0)
        effort.append(e_sum)
        if recover is not None:
            rec.append(recover)

    return {
        "device": str(device),
        "dataset_size": len(dataset),
        "mean_loss": float(np.mean(losses)) if losses else None,
        "success_rate": success / max(eval_eps, 1),
        "time_to_convergence": float(np.mean(conv)) if conv else None,
        "tracking_rmse": float(np.mean(rmse)) if rmse else None,
        "recovery_time": float(np.mean(rec)) if rec else None,
        "control_effort": float(np.mean(effort)) if effort else None,
    }


def train_and_eval_online_v3(cfg: dict) -> dict:
    """V3 online loop: act -> step -> store -> periodic update."""
    seed = int(cfg.get("seed", 0))
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env_cfg = cfg.get("env", {})
    planner = HighLevelHeuristicPlannerV2(waypoint_scale=float(cfg.get("waypoint_scale", 0.35)))

    seq_len = int(cfg.get("seq_len", 8))
    max_steps = int(env_cfg.get("max_steps", 250))
    train_eps = int(cfg.get("train_episodes", cfg.get("collect_episodes", 120)))
    eval_eps = int(cfg.get("eval_episodes", 40))

    replay = ReplayBuffer(capacity=int(cfg.get("replay_capacity", 20000)))
    memory_bank: deque[tuple[np.ndarray, np.ndarray]] = deque(maxlen=int(cfg.get("memory_bucket_quota", 6000)))
    memory_k = int(cfg.get("memory_k", 5))

    batch_size = int(cfg.get("batch_size", 256))
    lr = float(cfg.get("lr", 2e-4))
    gamma = float(cfg.get("gamma", 0.98))
    update_every = int(cfg.get("update_every_steps", 20))
    grad_updates = int(cfg.get("grad_updates_per_step", 2))
    warmup_steps = int(cfg.get("warmup_steps", max(seq_len * 6, batch_size)))
    noise_std = float(cfg.get("explore_noise_std", 0.05))
    actor_coef = float(cfg.get("actor_loss_coef", 1.0))
    critic_coef = float(cfg.get("critic_loss_coef", 0.25))

    in_dim = 10 + 2 + 1 + 2
    model = OnlineRecurrentPolicy(in_dim=in_dim, hid=int(cfg.get("hidden_dim", 128))).to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    mse = nn.MSELoss()

    global_steps = 0
    update_count = 0
    actor_losses: list[float] = []
    critic_losses: list[float] = []
    episode_returns: list[float] = []
    train_done_reasons: Counter[str] = Counter()

    for ep in range(train_eps):
        env = Sim2DEnv(
            seed=seed + ep,
            max_steps=max_steps,
            level=str(env_cfg.get("disturbance_level", "medium")),
            obstacle_count=int(env_cfg.get("obstacle_count", 0)),
        )
        obs = env.reset()
        hist: deque[np.ndarray] = deque(maxlen=seq_len)
        ep_return = 0.0

        for _ in range(max_steps):
            packet = planner.plan(obs)
            mem_action = _retrieve_memory_action(obs, memory_bank, memory_k)
            feat = _build_feature(obs, packet, mem_action)
            hist.append(feat)

            oracle_desired = _oracle_action(obs, packet["subgoal_xy"], packet.get("speed_hint", 0.7))
            desired = oracle_desired
            seq_arr = None
            if len(hist) == seq_len:
                seq_arr = np.stack(list(hist), axis=0).astype(np.float32)
                seq = torch.tensor(seq_arr[None, ...], dtype=torch.float32, device=device)
                with torch.no_grad():
                    pred_action, _ = model(seq)
                desired = pred_action.squeeze(0).cpu().numpy().astype(np.float32)
                desired += np.random.normal(0.0, noise_std, size=2).astype(np.float32)
                desired = _clip_desired(desired)

            # Keep high-level planner and low-level RBF fixed; only tactical recurrent policy is learned.
            action = _rbf_controller(obs, desired)
            next_obs, reward, done, info = env.step(action)
            ep_return += float(reward)
            global_steps += 1

            memory_bank.append((obs[:5].copy(), oracle_desired.copy()))

            if seq_arr is not None:
                next_packet = planner.plan(next_obs)
                next_mem_action = _retrieve_memory_action(next_obs, memory_bank, memory_k)
                next_feat = _build_feature(next_obs, next_packet, next_mem_action)
                next_hist = deque(hist, maxlen=seq_len)
                next_hist.append(next_feat)
                next_seq_arr = np.stack(list(next_hist), axis=0).astype(np.float32)
                replay.add(
                    Transition(
                        seq=seq_arr,
                        next_seq=next_seq_arr,
                        target=oracle_desired.copy(),
                        reward=float(reward),
                        done=float(done),
                    )
                )

            if global_steps >= warmup_steps and global_steps % update_every == 0 and len(replay) >= batch_size:
                model.train()
                for _ in range(grad_updates):
                    batch = replay.sample(batch_size)
                    b_seq = torch.tensor(np.stack([b.seq for b in batch], axis=0), dtype=torch.float32, device=device)
                    b_next_seq = torch.tensor(np.stack([b.next_seq for b in batch], axis=0), dtype=torch.float32, device=device)
                    b_tgt = torch.tensor(np.stack([b.target for b in batch], axis=0), dtype=torch.float32, device=device)
                    b_rew = torch.tensor([b.reward for b in batch], dtype=torch.float32, device=device)
                    b_done = torch.tensor([b.done for b in batch], dtype=torch.float32, device=device)

                    pred_action, pred_value = model(b_seq)
                    with torch.no_grad():
                        _, next_value = model(b_next_seq)
                        td_target = b_rew + gamma * (1.0 - b_done) * next_value

                    actor_loss = mse(pred_action, b_tgt)
                    critic_loss = mse(pred_value, td_target)
                    loss = actor_coef * actor_loss + critic_coef * critic_loss

                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    opt.step()
                    update_count += 1

                    actor_losses.append(float(actor_loss.item()))
                    critic_losses.append(float(critic_loss.item()))
                model.eval()

            obs = next_obs
            if done:
                train_done_reasons[_done_reason(info)] += 1
                break

        episode_returns.append(ep_return)

    timeout_bins_cfg = cfg.get("timeout_bins", {})
    near_thresh = float(timeout_bins_cfg.get("near", 0.25))
    mid_thresh = float(timeout_bins_cfg.get("mid", 0.75))
    if mid_thresh <= near_thresh:
        mid_thresh = near_thresh + 0.25

    success = 0
    conv = []
    rmse = []
    rec = []
    effort = []
    eval_done_reasons: Counter[str] = Counter()
    timeout_distance_bins: Counter[str] = Counter({"near": 0, "mid": 0, "far": 0})

    model.eval()
    for ep in range(eval_eps):
        env = Sim2DEnv(
            seed=seed + 5000 + ep,
            max_steps=max_steps,
            level=str(env_cfg.get("disturbance_level_eval", env_cfg.get("disturbance_level", "medium"))),
            obstacle_count=int(env_cfg.get("obstacle_count", 0)),
        )
        obs = env.reset()
        hist: deque[np.ndarray] = deque(maxlen=seq_len)
        dist_hist = []
        e_sum = 0.0
        disturbed_step = None
        recover = None

        for t in range(max_steps):
            packet = planner.plan(obs)
            mem_action = _retrieve_memory_action(obs, memory_bank, memory_k)
            feat = _build_feature(obs, packet, mem_action)
            hist.append(feat)

            if len(hist) < seq_len:
                desired = _oracle_action(obs, packet["subgoal_xy"], packet.get("speed_hint", 0.7))
            else:
                seq = torch.tensor(np.stack(list(hist), axis=0)[None, ...], dtype=torch.float32, device=device)
                with torch.no_grad():
                    pred_action, _ = model(seq)
                desired = _clip_desired(pred_action.squeeze(0).cpu().numpy().astype(np.float32))

            action = _rbf_controller(obs, desired)
            obs, _, done, info = env.step(action)
            dcur = float(info["distance"])
            dist_hist.append(dcur)
            e_sum += float(np.linalg.norm(action))

            if dcur > 1.0 and disturbed_step is None:
                disturbed_step = t
            if disturbed_step is not None and recover is None and dcur < 0.3:
                recover = t - disturbed_step

            if done:
                reason = _done_reason(info)
                eval_done_reasons[reason] += 1
                if reason == "timeout":
                    timeout_distance_bins[_timeout_bin(dcur, near_thresh, mid_thresh)] += 1
                if info.get("success", False):
                    success += 1
                    conv.append(t + 1)
                break

        rmse.append(float(np.sqrt(np.mean(np.square(dist_hist)))) if dist_hist else 0.0)
        effort.append(e_sum)
        if recover is not None:
            rec.append(recover)

    return {
        "device": str(device),
        "train_episodes": train_eps,
        "eval_episodes": eval_eps,
        "global_steps": global_steps,
        "memory_size": len(memory_bank),
        "replay_size": len(replay),
        "online_updates": update_count,
        "mean_actor_loss": float(np.mean(actor_losses)) if actor_losses else None,
        "mean_critic_loss": float(np.mean(critic_losses)) if critic_losses else None,
        "episode_return_mean": float(np.mean(episode_returns)) if episode_returns else None,
        "success_rate": success / max(eval_eps, 1),
        "done_reasons": {k: int(v) for k, v in eval_done_reasons.items()},
        "timeout_distance_bins": {k: int(timeout_distance_bins.get(k, 0)) for k in ("near", "mid", "far")},
        "train_done_reasons": {k: int(v) for k, v in train_done_reasons.items()},
        "time_to_convergence": float(np.mean(conv)) if conv else None,
        "tracking_rmse": float(np.mean(rmse)) if rmse else None,
        "recovery_time": float(np.mean(rec)) if rec else None,
        "control_effort": float(np.mean(effort)) if effort else None,
    }


def train_and_eval_online_v3_ff(cfg: dict, memory_mode: str = "memory_on") -> dict:
    """L2 isolation run: fixed planner/RBF, non-recurrent tactical baseline."""
    seed = int(cfg.get("seed", 0))
    np.random.seed(seed)
    torch.manual_seed(seed)

    memory_enabled = str(memory_mode).lower() == "memory_on"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env_cfg = cfg.get("env", {})
    planner = HighLevelHeuristicPlannerV2(waypoint_scale=float(cfg.get("waypoint_scale", 0.35)))

    # Isolation guard: keep obstacles disabled as requested.
    max_steps = int(env_cfg.get("max_steps", 250))
    train_eps = int(cfg.get("train_episodes", cfg.get("collect_episodes", 120)))
    eval_eps = int(cfg.get("eval_episodes", 40))

    replay = ReplayBuffer(capacity=int(cfg.get("replay_capacity", 20000)))
    memory_capacity = int(cfg.get("memory_bucket_quota", 6000))
    memory_bank: list[MemorySample] = []
    memory_k = int(cfg.get("memory_k", 5))
    memory_progress_eps = float(cfg.get("memory_progress_eps", 1e-4))
    memory_quality_progress_scale = float(cfg.get("memory_quality_progress_scale", 1.0))
    memory_quality_success_bonus = float(cfg.get("memory_quality_success_bonus", 0.5))
    memory_quality_min = float(cfg.get("memory_quality_min", 0.05))
    memory_success_segment_len = int(cfg.get("memory_success_segment_len", 8))
    memory_write_attempts = 0
    memory_write_accepted = 0
    memory_eviction_count = 0
    retrieval_scores: list[float] = []

    batch_size = int(cfg.get("batch_size", 256))
    lr = float(cfg.get("lr", 2e-4))
    gamma = float(cfg.get("gamma", 0.98))
    update_every = int(cfg.get("update_every_steps", 20))
    grad_updates = int(cfg.get("grad_updates_per_step", 2))
    warmup_steps = int(cfg.get("warmup_steps", batch_size))
    noise_std = float(cfg.get("explore_noise_std", 0.05))
    actor_coef = float(cfg.get("actor_loss_coef", 1.0))
    critic_coef = float(cfg.get("critic_loss_coef", 0.25))

    in_dim = 10 + 2 + 1 + 2
    model = OnlineTacticalBaseline(
        in_dim=in_dim,
        hid=int(cfg.get("hidden_dim", 128)),
        linear_only=bool(cfg.get("linear_policy", False)),
    ).to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    mse = nn.MSELoss()

    global_steps = 0
    update_count = 0
    actor_losses: list[float] = []
    critic_losses: list[float] = []
    episode_returns: list[float] = []
    train_done_reasons: Counter[str] = Counter()

    model.eval()
    for ep in range(train_eps):
        env = Sim2DEnv(
            seed=seed + ep,
            max_steps=max_steps,
            level=str(env_cfg.get("disturbance_level", "medium")),
            obstacle_count=int(env_cfg.get("obstacle_count", 0)),
        )
        obs = env.reset()
        ep_return = 0.0
        prev_distance = None
        episode_candidates: list[tuple[np.ndarray, np.ndarray, float]] = []

        for _ in range(max_steps):
            packet = planner.plan(obs)
            mem_action = _retrieve_memory_action_scored(obs, memory_bank, memory_k, retrieval_scores) if memory_enabled else None
            feat = _build_feature(obs, packet, mem_action)

            oracle_desired = _oracle_action(obs, packet["subgoal_xy"], packet.get("speed_hint", 0.7))
            if global_steps < warmup_steps:
                desired = oracle_desired
            else:
                feat_t = torch.tensor(feat[None, ...], dtype=torch.float32, device=device)
                with torch.no_grad():
                    pred_action, _ = model(feat_t)
                desired = pred_action.squeeze(0).cpu().numpy().astype(np.float32)
                desired += np.random.normal(0.0, noise_std, size=2).astype(np.float32)
                desired = _clip_desired(desired)

            action = _rbf_controller(obs, desired)
            next_obs, reward, done, info = env.step(action)
            ep_return += float(reward)
            global_steps += 1

            if memory_enabled:
                dcur = float(info.get("distance", np.nan))
                progress_delta = 0.0
                if prev_distance is not None and np.isfinite(dcur):
                    progress_delta = prev_distance - dcur
                if np.isfinite(dcur):
                    prev_distance = dcur
                base_quality = max(memory_quality_min, memory_quality_progress_scale * max(progress_delta, 0.0))
                success_now = bool(done and info.get("success", False))

                memory_write_attempts += 1
                if progress_delta > memory_progress_eps or success_now:
                    quality = base_quality + (memory_quality_success_bonus if success_now else 0.0)
                    memory_eviction_count += _insert_memory_sample(
                        memory_bank,
                        MemorySample(
                            key=obs[:5].copy(),
                            action=oracle_desired.copy(),
                            quality=float(max(quality, memory_quality_min)),
                            created_step=global_steps,
                        ),
                        memory_capacity,
                    )
                    memory_write_accepted += 1

                episode_candidates.append((obs[:5].copy(), oracle_desired.copy(), float(base_quality)))

            next_packet = planner.plan(next_obs)
            next_mem_action = (
                _retrieve_memory_action_scored(next_obs, memory_bank, memory_k, retrieval_scores) if memory_enabled else None
            )
            next_feat = _build_feature(next_obs, next_packet, next_mem_action)
            replay.add(
                TransitionFF(
                    feat=feat.copy(),
                    next_feat=next_feat.copy(),
                    target=oracle_desired.copy(),
                    reward=float(reward),
                    done=float(done),
                )
            )

            if global_steps >= warmup_steps and global_steps % update_every == 0 and len(replay) >= batch_size:
                model.train()
                for _ in range(grad_updates):
                    batch = replay.sample(batch_size)
                    b_feat = torch.tensor(np.stack([b.feat for b in batch], axis=0), dtype=torch.float32, device=device)
                    b_next_feat = torch.tensor(np.stack([b.next_feat for b in batch], axis=0), dtype=torch.float32, device=device)
                    b_tgt = torch.tensor(np.stack([b.target for b in batch], axis=0), dtype=torch.float32, device=device)
                    b_rew = torch.tensor([b.reward for b in batch], dtype=torch.float32, device=device)
                    b_done = torch.tensor([b.done for b in batch], dtype=torch.float32, device=device)

                    pred_action, pred_value = model(b_feat)
                    with torch.no_grad():
                        _, next_value = model(b_next_feat)
                        td_target = b_rew + gamma * (1.0 - b_done) * next_value

                    actor_loss = mse(pred_action, b_tgt)
                    critic_loss = mse(pred_value, td_target)
                    loss = actor_coef * actor_loss + critic_coef * critic_loss

                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    opt.step()
                    update_count += 1

                    actor_losses.append(float(actor_loss.item()))
                    critic_losses.append(float(critic_loss.item()))
                model.eval()

            obs = next_obs
            if done:
                if memory_enabled and bool(info.get("success", False)) and memory_success_segment_len > 0:
                    for key, act, base_quality in episode_candidates[-memory_success_segment_len:]:
                        memory_write_attempts += 1
                        memory_write_accepted += 1
                        memory_eviction_count += _insert_memory_sample(
                            memory_bank,
                            MemorySample(
                                key=key.copy(),
                                action=act.copy(),
                                quality=float(max(base_quality + memory_quality_success_bonus, memory_quality_min)),
                                created_step=global_steps,
                            ),
                            memory_capacity,
                        )
                train_done_reasons[_done_reason(info)] += 1
                break

        episode_returns.append(ep_return)

    timeout_bins_cfg = cfg.get("timeout_bins", {})
    near_thresh = float(timeout_bins_cfg.get("near", 0.25))
    mid_thresh = float(timeout_bins_cfg.get("mid", 0.75))
    if mid_thresh <= near_thresh:
        mid_thresh = near_thresh + 0.25

    success = 0
    conv = []
    rmse = []
    rec = []
    effort = []
    eval_done_reasons: Counter[str] = Counter()
    timeout_distance_bins: Counter[str] = Counter({"near": 0, "mid": 0, "far": 0})

    model.eval()
    for ep in range(eval_eps):
        env = Sim2DEnv(
            seed=seed + 5000 + ep,
            max_steps=max_steps,
            level=str(env_cfg.get("disturbance_level_eval", env_cfg.get("disturbance_level", "medium"))),
            obstacle_count=int(env_cfg.get("obstacle_count", 0)),
        )
        obs = env.reset()
        dist_hist = []
        e_sum = 0.0
        disturbed_step = None
        recover = None

        for t in range(max_steps):
            packet = planner.plan(obs)
            mem_action = _retrieve_memory_action_scored(obs, memory_bank, memory_k, retrieval_scores) if memory_enabled else None
            feat = _build_feature(obs, packet, mem_action)

            feat_t = torch.tensor(feat[None, ...], dtype=torch.float32, device=device)
            with torch.no_grad():
                pred_action, _ = model(feat_t)
            desired = _clip_desired(pred_action.squeeze(0).cpu().numpy().astype(np.float32))

            action = _rbf_controller(obs, desired)
            obs, _, done, info = env.step(action)
            dcur = float(info["distance"])
            dist_hist.append(dcur)
            e_sum += float(np.linalg.norm(action))

            if dcur > 1.0 and disturbed_step is None:
                disturbed_step = t
            if disturbed_step is not None and recover is None and dcur < 0.3:
                recover = t - disturbed_step

            if done:
                reason = _done_reason(info)
                eval_done_reasons[reason] += 1
                if reason == "timeout":
                    timeout_distance_bins[_timeout_bin(dcur, near_thresh, mid_thresh)] += 1
                if info.get("success", False):
                    success += 1
                    conv.append(t + 1)
                break

        rmse.append(float(np.sqrt(np.mean(np.square(dist_hist)))) if dist_hist else 0.0)
        effort.append(e_sum)
        if recover is not None:
            rec.append(recover)

    return {
        "memory_mode": "memory_on" if memory_enabled else "memory_off",
        "device": str(device),
        "train_episodes": train_eps,
        "eval_episodes": eval_eps,
        "global_steps": global_steps,
        "memory_size": len(memory_bank),
        "memory_write_attempts": int(memory_write_attempts),
        "memory_write_accepted": int(memory_write_accepted),
        "memory_write_accept_rate": float(memory_write_accepted / max(memory_write_attempts, 1)) if memory_enabled else 0.0,
        "memory_retrieval_score_stats": _score_stats(retrieval_scores),
        "memory_eviction_count": int(memory_eviction_count),
        "replay_size": len(replay),
        "online_updates": update_count,
        "mean_actor_loss": float(np.mean(actor_losses)) if actor_losses else None,
        "mean_critic_loss": float(np.mean(critic_losses)) if critic_losses else None,
        "episode_return_mean": float(np.mean(episode_returns)) if episode_returns else None,
        "success_rate": success / max(eval_eps, 1),
        "done_reasons": {k: int(v) for k, v in eval_done_reasons.items()},
        "timeout_distance_bins": {k: int(timeout_distance_bins.get(k, 0)) for k in ("near", "mid", "far")},
        "train_done_reasons": {k: int(v) for k, v in train_done_reasons.items()},
        "time_to_convergence": float(np.mean(conv)) if conv else None,
        "tracking_rmse": float(np.mean(rmse)) if rmse else None,
        "recovery_time": float(np.mean(rec)) if rec else None,
        "control_effort": float(np.mean(effort)) if effort else None,
    }


def run_l2_memory_ablation(cfg: dict) -> dict:
    modes = ["memory_off", "memory_on"]
    results = {mode: train_and_eval_online_v3_ff(cfg, memory_mode=mode) for mode in modes}
    off = results["memory_off"]
    on = results["memory_on"]
    return {
        "train_mode": "l2_memory_ablation",
        "seed": int(cfg.get("seed", 0)),
        "obstacle_count_forced": 0,
        "fixed_layers": {"planner": "HighLevelHeuristicPlannerV2", "controller": "rbf_controller"},
        "policy": {
            "type": "non_recurrent_tactical_baseline",
            "linear_only": bool(cfg.get("linear_policy", False)),
        },
        "results": results,
        "comparison": {
            "success_rate_delta_on_minus_off": float(on["success_rate"] - off["success_rate"]),
            "timeout_near_delta_on_minus_off": int(on["timeout_distance_bins"]["near"] - off["timeout_distance_bins"]["near"]),
            "timeout_mid_delta_on_minus_off": int(on["timeout_distance_bins"]["mid"] - off["timeout_distance_bins"]["mid"]),
            "timeout_far_delta_on_minus_off": int(on["timeout_distance_bins"]["far"] - off["timeout_distance_bins"]["far"]),
            "done_reasons": {"memory_off": off["done_reasons"], "memory_on": on["done_reasons"]},
        },
    }


def _train_memory_residual_bank(cfg: dict) -> tuple[list[MemorySample], dict]:
    seed = int(cfg.get("seed", 0))
    np.random.seed(seed)
    env_cfg = cfg.get("env", {})
    planner = HighLevelHeuristicPlannerV2(waypoint_scale=float(cfg.get("waypoint_scale", 0.35)))

    max_steps = int(env_cfg.get("max_steps", 250))
    train_eps = int(cfg.get("train_episodes", cfg.get("collect_episodes", 120)))
    memory_capacity = int(cfg.get("memory_bucket_quota", 6000))
    memory_k = int(cfg.get("memory_k", 5))
    residual_clip = float(cfg.get("memory_residual_clip", 0.15))
    residual_gain = float(cfg.get("memory_residual_gain", 0.8))

    memory_progress_eps = float(cfg.get("memory_progress_eps", 1e-4))
    memory_quality_progress_scale = float(cfg.get("memory_quality_progress_scale", 1.0))
    memory_quality_success_bonus = float(cfg.get("memory_quality_success_bonus", 0.5))
    memory_quality_min = float(cfg.get("memory_quality_min", 0.05))
    memory_success_segment_len = int(cfg.get("memory_success_segment_len", 8))

    memory_bank: list[MemorySample] = []
    retrieval_scores: list[float] = []
    memory_write_attempts = 0
    memory_write_accepted = 0
    memory_eviction_count = 0
    train_done_reasons: Counter[str] = Counter()

    for ep in range(train_eps):
        env = Sim2DEnv(
            seed=seed + ep,
            max_steps=max_steps,
            level=str(env_cfg.get("disturbance_level", "medium")),
            obstacle_count=0,
        )
        obs = env.reset()
        prev_distance = float(np.hypot(float(obs[5] - obs[0]), float(obs[6] - obs[1])))
        episode_candidates: list[tuple[np.ndarray, np.ndarray, float]] = []

        for _ in range(max_steps):
            packet = planner.plan(obs)
            core_desired = _deterministic_core_mapping(obs, packet, heading_gain=float(cfg.get("core_heading_gain", 1.5)))
            mem_delta = _retrieve_memory_action_scored(obs, memory_bank, memory_k, retrieval_scores)
            if mem_delta is None:
                mem_delta = np.zeros(2, dtype=np.float32)
            desired = _clip_desired(core_desired + _bounded_delta(mem_delta, residual_clip))
            action = _rbf_controller(obs, desired)

            next_obs, _, done, info = env.step(action)
            dcur = float(info.get("distance", prev_distance))
            progress_delta = prev_distance - dcur
            prev_distance = dcur

            # Build residual target from one-step dynamics mismatch in (v, omega).
            next_vo = np.array([next_obs[3], next_obs[4]], dtype=np.float32)
            raw_residual = residual_gain * (desired - next_vo)
            residual_target = _bounded_delta(raw_residual, residual_clip)

            base_quality = max(memory_quality_min, memory_quality_progress_scale * max(progress_delta, 0.0))
            success_now = bool(done and info.get("success", False))

            memory_write_attempts += 1
            if progress_delta > memory_progress_eps or success_now:
                quality = base_quality + (memory_quality_success_bonus if success_now else 0.0)
                memory_eviction_count += _insert_memory_sample(
                    memory_bank,
                    MemorySample(
                        key=obs[:5].copy(),
                        action=residual_target.copy(),
                        quality=float(max(quality, memory_quality_min)),
                        created_step=int(ep * max_steps),
                    ),
                    memory_capacity,
                )
                memory_write_accepted += 1

            episode_candidates.append((obs[:5].copy(), residual_target.copy(), float(base_quality)))
            obs = next_obs
            if done:
                if bool(info.get("success", False)) and memory_success_segment_len > 0:
                    for key, residual, base_q in episode_candidates[-memory_success_segment_len:]:
                        memory_write_attempts += 1
                        memory_write_accepted += 1
                        memory_eviction_count += _insert_memory_sample(
                            memory_bank,
                            MemorySample(
                                key=key.copy(),
                                action=residual.copy(),
                                quality=float(max(base_q + memory_quality_success_bonus, memory_quality_min)),
                                created_step=int(ep * max_steps),
                            ),
                            memory_capacity,
                        )
                train_done_reasons[_done_reason(info)] += 1
                break

    train_diag = {
        "memory_size": len(memory_bank),
        "memory_write_attempts": int(memory_write_attempts),
        "memory_write_accepted": int(memory_write_accepted),
        "memory_write_accept_rate": float(memory_write_accepted / max(memory_write_attempts, 1)),
        "memory_retrieval_score_stats": _score_stats(retrieval_scores),
        "memory_eviction_count": int(memory_eviction_count),
        "train_done_reasons": {k: int(v) for k, v in train_done_reasons.items()},
    }
    return memory_bank, train_diag


def _eval_deterministic_core(cfg: dict, memory_bank: list[MemorySample] | None, mode: str) -> dict:
    seed = int(cfg.get("seed", 0))
    np.random.seed(seed)
    env_cfg = cfg.get("env", {})
    planner = HighLevelHeuristicPlannerV2(waypoint_scale=float(cfg.get("waypoint_scale", 0.35)))
    eval_eps = int(cfg.get("eval_episodes", 40))
    max_steps = int(env_cfg.get("max_steps", 250))
    memory_k = int(cfg.get("memory_k", 5))
    residual_clip = float(cfg.get("memory_residual_clip", 0.15))
    heading_gain = float(cfg.get("core_heading_gain", 1.5))

    timeout_bins_cfg = cfg.get("timeout_bins", {})
    near_thresh = float(timeout_bins_cfg.get("near", 0.25))
    mid_thresh = float(timeout_bins_cfg.get("mid", 0.75))
    if mid_thresh <= near_thresh:
        mid_thresh = near_thresh + 0.25

    success = 0
    done_reasons: Counter[str] = Counter()
    timeout_distance_bins: Counter[str] = Counter({"near": 0, "mid": 0, "far": 0})
    episode_returns: list[float] = []
    retrieval_scores: list[float] = []

    for ep in range(eval_eps):
        env = Sim2DEnv(
            seed=seed + 5000 + ep,
            max_steps=max_steps,
            level=str(env_cfg.get("disturbance_level_eval", env_cfg.get("disturbance_level", "medium"))),
            obstacle_count=0,
        )
        obs = env.reset()
        ep_return = 0.0

        for _ in range(max_steps):
            packet = planner.plan(obs)
            core_desired = _deterministic_core_mapping(obs, packet, heading_gain=heading_gain)

            residual = np.zeros(2, dtype=np.float32)
            if mode == "core_plus_memory_residual" and memory_bank:
                retrieved = _retrieve_memory_action_scored(obs, memory_bank, memory_k, retrieval_scores)
                if retrieved is not None:
                    residual = _bounded_delta(retrieved, residual_clip)

            desired = _clip_desired(core_desired + residual)
            action = _rbf_controller(obs, desired)
            obs, reward, done, info = env.step(action)
            ep_return += float(reward)

            if done:
                reason = _done_reason(info)
                done_reasons[reason] += 1
                if reason == "timeout":
                    timeout_distance_bins[_timeout_bin(float(info["distance"]), near_thresh, mid_thresh)] += 1
                if bool(info.get("success", False)):
                    success += 1
                break

        episode_returns.append(ep_return)

    out = {
        "mode": mode,
        "success_rate": float(success / max(eval_eps, 1)),
        "done_reasons": {k: int(v) for k, v in done_reasons.items()},
        "timeout_distance_bins": {k: int(timeout_distance_bins.get(k, 0)) for k in ("near", "mid", "far")},
        "episode_return_mean": float(np.mean(episode_returns)) if episode_returns else None,
    }
    if mode == "core_plus_memory_residual":
        out["memory_retrieval_score_stats"] = _score_stats(retrieval_scores)
    return out


def _write_l2_deterministic_report(report: dict, report_path: Path) -> None:
    core = report["results"]["core_only"]
    mem = report["results"]["core_plus_memory_residual"]
    comp = report["comparison"]
    lines = [
        "# L2 Deterministic Core + Memory Residual (Fixed L1/L3, No LSTM)",
        "",
        "## Setup",
        f"- Branch: `v3-online-memory`",
        f"- Script: `hrl_ws/src/hrl_trainer/hrl_trainer/sim2d/train_rl_brainer_v3_online.py`",
        f"- Config mode: `l2_deterministic_plus_memory`",
        f"- Modes: `core_only` vs `core_plus_memory_residual`",
        f"- Deterministic core mapping: heading+distance to desired `[v, omega]`",
        f"- Memory residual clip: `+/-{report['residual_clip']}`",
        f"- Obstacles forced to `0` for train/eval",
        "",
        "## Results",
        "",
        "| Metric | core_only | core_plus_memory_residual | Delta (plus-core) |",
        "|---|---:|---:|---:|",
        f"| success_rate | {core['success_rate']:.3f} | {mem['success_rate']:.3f} | {comp['success_rate_delta_plus_minus_core']:+.3f} |",
        f"| avg_return | {core['episode_return_mean']:.3f} | {mem['episode_return_mean']:.3f} | {comp['avg_return_delta_plus_minus_core']:+.3f} |",
        f"| done_reasons.success | {core['done_reasons'].get('success', 0)} | {mem['done_reasons'].get('success', 0)} | {mem['done_reasons'].get('success', 0) - core['done_reasons'].get('success', 0):+d} |",
        f"| done_reasons.timeout | {core['done_reasons'].get('timeout', 0)} | {mem['done_reasons'].get('timeout', 0)} | {mem['done_reasons'].get('timeout', 0) - core['done_reasons'].get('timeout', 0):+d} |",
        f"| timeout_distance_bins.near | {core['timeout_distance_bins']['near']} | {mem['timeout_distance_bins']['near']} | {comp['timeout_near_delta_plus_minus_core']:+d} |",
        f"| timeout_distance_bins.mid | {core['timeout_distance_bins']['mid']} | {mem['timeout_distance_bins']['mid']} | {comp['timeout_mid_delta_plus_minus_core']:+d} |",
        f"| timeout_distance_bins.far | {core['timeout_distance_bins']['far']} | {mem['timeout_distance_bins']['far']} | {comp['timeout_far_delta_plus_minus_core']:+d} |",
        "",
        "## Done Reasons",
        f"- core_only: `{json.dumps(core['done_reasons'])}`",
        f"- core_plus_memory_residual: `{json.dumps(mem['done_reasons'])}`",
    ]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_l2_deterministic_plus_memory(cfg: dict, report_path: Path | None = None) -> dict:
    memory_bank, train_diag = _train_memory_residual_bank(cfg)
    core_only = _eval_deterministic_core(cfg, memory_bank=None, mode="core_only")
    core_plus = _eval_deterministic_core(cfg, memory_bank=memory_bank, mode="core_plus_memory_residual")

    report = {
        "train_mode": "l2_deterministic_plus_memory",
        "seed": int(cfg.get("seed", 0)),
        "obstacle_count_forced": 0,
        "fixed_layers": {"planner": "HighLevelHeuristicPlannerV2", "controller": "rbf_controller"},
        "policy": {
            "type": "deterministic_core_plus_optional_memory_residual",
            "core": "heading_distance_mapping",
            "memory_residual_clip": float(cfg.get("memory_residual_clip", 0.15)),
        },
        "residual_clip": float(cfg.get("memory_residual_clip", 0.15)),
        "memory_training_diagnostics": train_diag,
        "results": {"core_only": core_only, "core_plus_memory_residual": core_plus},
        "comparison": {
            "success_rate_delta_plus_minus_core": float(core_plus["success_rate"] - core_only["success_rate"]),
            "avg_return_delta_plus_minus_core": float(
                (core_plus["episode_return_mean"] or 0.0) - (core_only["episode_return_mean"] or 0.0)
            ),
            "timeout_near_delta_plus_minus_core": int(
                core_plus["timeout_distance_bins"]["near"] - core_only["timeout_distance_bins"]["near"]
            ),
            "timeout_mid_delta_plus_minus_core": int(
                core_plus["timeout_distance_bins"]["mid"] - core_only["timeout_distance_bins"]["mid"]
            ),
            "timeout_far_delta_plus_minus_core": int(
                core_plus["timeout_distance_bins"]["far"] - core_only["timeout_distance_bins"]["far"]
            ),
            "done_reasons": {
                "core_only": core_only["done_reasons"],
                "core_plus_memory_residual": core_plus["done_reasons"],
            },
        },
    }

    print(
        "L2 deterministic summary: "
        f"core_only(sr={core_only['success_rate']:.3f}, ret={core_only['episode_return_mean']:.3f}) | "
        f"core_plus_memory_residual(sr={core_plus['success_rate']:.3f}, ret={core_plus['episode_return_mean']:.3f}) | "
        f"delta(sr={report['comparison']['success_rate_delta_plus_minus_core']:+.3f}, "
        f"ret={report['comparison']['avg_return_delta_plus_minus_core']:+.3f})"
    )

    if report_path is not None:
        _write_l2_deterministic_report(report, report_path)
    return report


def _seq_array_from_window(window: deque[np.ndarray], seq_len: int, feat_dim: int) -> np.ndarray:
    if not window:
        return np.zeros((seq_len, feat_dim), dtype=np.float32)
    items = list(window)[-seq_len:]
    if len(items) < seq_len:
        pad = [items[0]] * (seq_len - len(items))
        items = pad + items
    return np.stack(items, axis=0).astype(np.float32)


def _train_lstm_residual_policy(cfg: dict, memory_bank: list[MemorySample], seed: int) -> tuple[MemoryLSTMPolicy, dict]:
    np.random.seed(seed)
    torch.manual_seed(seed)

    env_cfg = cfg.get("env", {})
    planner = HighLevelHeuristicPlannerV2(waypoint_scale=float(cfg.get("waypoint_scale", 0.35)))

    seq_len = int(cfg.get("seq_len", 10))
    hidden_dim = int(cfg.get("hidden_dim", 128))
    train_eps = int(cfg.get("lstm_train_episodes", cfg.get("train_episodes", 120)))
    max_steps = int(env_cfg.get("max_steps", 250))
    memory_k = int(cfg.get("memory_k", 5))
    residual_clip = float(cfg.get("memory_residual_clip", 0.15))
    residual_gain = float(cfg.get("memory_residual_gain", 0.8))
    batch_size = int(cfg.get("lstm_batch_size", cfg.get("batch_size", 256)))
    epochs = int(cfg.get("lstm_epochs", cfg.get("epochs", 8)))
    lr = float(cfg.get("lstm_lr", cfg.get("lr", 1.5e-4)))
    feat_dim = 15

    samples: list[Sample] = []
    for ep in range(train_eps):
        env = Sim2DEnv(
            seed=seed + 2000 + ep,
            max_steps=max_steps,
            level=str(env_cfg.get("disturbance_level", "medium")),
            obstacle_count=int(env_cfg.get("obstacle_count", 0)),
        )
        obs = env.reset()
        window: deque[np.ndarray] = deque(maxlen=seq_len)

        for _ in range(max_steps):
            packet = planner.plan(obs)
            core_desired = _deterministic_core_mapping(obs, packet, heading_gain=float(cfg.get("core_heading_gain", 1.5)))
            mem_action, _ = _retrieve_memory_action_with_score(obs, memory_bank, memory_k)
            mem_residual = (
                _bounded_delta(mem_action, residual_clip) if mem_action is not None else np.zeros(2, dtype=np.float32)
            )
            feat = _build_feature(obs, packet, mem_action)
            window.append(feat)

            desired = _clip_desired(core_desired + mem_residual)
            action = _rbf_controller(obs, desired)
            next_obs, _, done, _ = env.step(action)
            next_vo = np.array([next_obs[3], next_obs[4]], dtype=np.float32)
            target = _bounded_delta(residual_gain * (core_desired - next_vo), residual_clip)
            samples.append(Sample(seq=_seq_array_from_window(window, seq_len, feat_dim), target=target))
            obs = next_obs
            if done:
                break

    policy = MemoryLSTMPolicy(in_dim=feat_dim, hid=hidden_dim)
    optimizer = optim.Adam(policy.parameters(), lr=lr)
    mse = nn.MSELoss()
    losses: list[float] = []

    if samples:
        for _ in range(epochs):
            n = len(samples)
            for _ in range(max(1, n // max(batch_size, 1))):
                idx = np.random.randint(0, n, size=min(batch_size, n))
                seq = torch.from_numpy(np.stack([samples[int(i)].seq for i in idx], axis=0)).float()
                tgt = torch.from_numpy(np.stack([samples[int(i)].target for i in idx], axis=0)).float()
                pred = policy(seq)
                loss = mse(pred, tgt)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                losses.append(float(loss.item()))

    policy.eval()
    return policy, {
        "sample_count": int(len(samples)),
        "seq_len": seq_len,
        "hidden_dim": hidden_dim,
        "epochs": epochs,
        "batch_size": batch_size,
        "mean_lstm_loss": float(np.mean(losses)) if losses else None,
    }


def _eval_three_layer_mode(
    cfg: dict,
    seed: int,
    mode: str,
    memory_bank: list[MemorySample] | None,
    lstm_policy: MemoryLSTMPolicy | None,
) -> dict:
    np.random.seed(seed)
    env_cfg = cfg.get("env", {})
    planner = HighLevelHeuristicPlannerV2(waypoint_scale=float(cfg.get("waypoint_scale", 0.35)))

    eval_eps = int(cfg.get("eval_episodes", 40))
    max_steps = int(env_cfg.get("max_steps", 250))
    memory_k = int(cfg.get("memory_k", 5))
    residual_clip = float(cfg.get("memory_residual_clip", 0.15))
    lstm_residual_clip = float(cfg.get("lstm_residual_clip", residual_clip))
    heading_gain = float(cfg.get("core_heading_gain", 1.5))
    seq_len = int(cfg.get("seq_len", 10))
    feat_dim = 15

    timeout_bins_cfg = cfg.get("timeout_bins", {})
    near_thresh = float(timeout_bins_cfg.get("near", 0.25))
    mid_thresh = float(timeout_bins_cfg.get("mid", 0.75))
    if mid_thresh <= near_thresh:
        mid_thresh = near_thresh + 0.25

    gate_cfg = cfg.get("lstm_gate", {})
    gate_enabled = bool(gate_cfg.get("enabled", False))
    gate_near_goal = float(gate_cfg.get("near_goal_threshold", 0.25))
    gate_uncertainty = float(gate_cfg.get("uncertainty_threshold", 0.05))

    success = 0
    done_reasons: Counter[str] = Counter()
    timeout_distance_bins: Counter[str] = Counter({"near": 0, "mid": 0, "far": 0})
    episode_returns: list[float] = []
    tracking_rmse: list[float] = []
    control_effort: list[float] = []
    retrieval_scores: list[float] = []
    gate_triggered = 0
    gate_total = 0

    for ep in range(eval_eps):
        env = Sim2DEnv(
            seed=seed + 7000 + ep,
            max_steps=max_steps,
            level=str(env_cfg.get("disturbance_level_eval", env_cfg.get("disturbance_level", "medium"))),
            obstacle_count=int(env_cfg.get("obstacle_count", 0)),
        )
        obs = env.reset()
        ep_return = 0.0
        dist_hist: list[float] = []
        effort_sum = 0.0
        window: deque[np.ndarray] = deque(maxlen=seq_len)

        for _ in range(max_steps):
            packet = planner.plan(obs)
            core_desired = _deterministic_core_mapping(obs, packet, heading_gain=heading_gain)
            mem_action, mem_score = (None, None)
            if memory_bank:
                mem_action, mem_score = _retrieve_memory_action_with_score(obs, memory_bank, memory_k)
                if mem_score is not None:
                    retrieval_scores.append(mem_score)
            mem_residual = np.zeros(2, dtype=np.float32)
            if mode in ("core_plus_memory_residual", "core_plus_lstm_memory_residual") and mem_action is not None:
                mem_residual = _bounded_delta(mem_action, residual_clip)

            lstm_residual = np.zeros(2, dtype=np.float32)
            if mode == "core_plus_lstm_memory_residual" and lstm_policy is not None:
                feat = _build_feature(obs, packet, mem_action)
                window.append(feat)
                seq_arr = _seq_array_from_window(window, seq_len, feat_dim)
                seq_t = torch.from_numpy(seq_arr[None, :, :]).float()
                with torch.no_grad():
                    raw_lstm = lstm_policy(seq_t).squeeze(0).cpu().numpy().astype(np.float32)
                gated = True
                if gate_enabled:
                    distance = float(np.hypot(float(obs[5] - obs[0]), float(obs[6] - obs[1])))
                    high_uncertainty = bool(mem_score is None or mem_score < gate_uncertainty)
                    gated = bool(distance < gate_near_goal or high_uncertainty)
                gate_total += 1
                if gated:
                    gate_triggered += 1
                    lstm_residual = _bounded_delta(raw_lstm, lstm_residual_clip)

            desired = _clip_desired(core_desired + mem_residual + lstm_residual)
            action = _rbf_controller(obs, desired)
            obs, reward, done, info = env.step(action)
            ep_return += float(reward)
            dist_hist.append(float(info["distance"]))
            effort_sum += float(info.get("control_effort", np.linalg.norm(action)))

            if done:
                reason = _done_reason(info)
                done_reasons[reason] += 1
                if reason == "timeout":
                    timeout_distance_bins[_timeout_bin(float(info["distance"]), near_thresh, mid_thresh)] += 1
                if bool(info.get("success", False)):
                    success += 1
                break

        episode_returns.append(ep_return)
        tracking_rmse.append(float(np.sqrt(np.mean(np.square(dist_hist)))) if dist_hist else 0.0)
        control_effort.append(effort_sum)

    out = {
        "mode": mode,
        "seed": int(seed),
        "success_rate": float(success / max(eval_eps, 1)),
        "done_reasons": {k: int(v) for k, v in done_reasons.items()},
        "timeout_distance_bins": {k: int(timeout_distance_bins.get(k, 0)) for k in ("near", "mid", "far")},
        "episode_return_mean": float(np.mean(episode_returns)) if episode_returns else None,
        "tracking_rmse": float(np.mean(tracking_rmse)) if tracking_rmse else None,
        "control_effort": float(np.mean(control_effort)) if control_effort else None,
    }
    if retrieval_scores:
        out["memory_retrieval_score_stats"] = _score_stats(retrieval_scores)
    if mode == "core_plus_lstm_memory_residual":
        out["gate_enabled"] = gate_enabled
        out["gate_activation_rate"] = float(gate_triggered / max(gate_total, 1))
    return out


def _mean_std(values: list[float]) -> dict:
    arr = np.array(values, dtype=np.float32)
    return {"mean": float(np.mean(arr)), "std": float(np.std(arr))}


def _aggregate_three_layer_by_seed(seed_results: dict[str, list[dict]]) -> dict:
    metrics = ["success_rate", "episode_return_mean", "tracking_rmse", "control_effort"]
    modes: dict[str, dict] = {}
    for mode, rows in seed_results.items():
        agg: dict = {"n_seeds": int(len(rows))}
        for metric in metrics:
            vals = [float(r[metric]) for r in rows if r.get(metric) is not None]
            if vals:
                agg[metric] = _mean_std(vals)

        done_keys = set()
        for r in rows:
            done_keys.update(r.get("done_reasons", {}).keys())
        done_agg = {}
        for k in sorted(done_keys):
            done_agg[k] = _mean_std([float(r.get("done_reasons", {}).get(k, 0)) for r in rows])
        agg["done_reasons"] = done_agg

        timeout_agg = {}
        for k in ("near", "mid", "far"):
            timeout_agg[k] = _mean_std([float(r.get("timeout_distance_bins", {}).get(k, 0)) for r in rows])
        agg["timeout_distance_bins"] = timeout_agg

        gate_vals = [float(r.get("gate_activation_rate")) for r in rows if r.get("gate_activation_rate") is not None]
        if gate_vals:
            agg["gate_activation_rate"] = _mean_std(gate_vals)

        modes[mode] = agg
    return modes


def _print_three_layer_key_table(aggregated: dict) -> None:
    print("V3 three-layer ablation (seed mean±std)")
    print("| mode | success_rate | avg_return | tracking_rmse | control_effort | timeout near/mid/far |")
    print("|---|---:|---:|---:|---:|---:|")
    for mode in ("core_only", "core_plus_memory_residual", "core_plus_lstm_memory_residual"):
        row = aggregated[mode]
        sr = row["success_rate"]
        ret = row["episode_return_mean"]
        rmse = row["tracking_rmse"]
        effort = row["control_effort"]
        tbin = row["timeout_distance_bins"]
        print(
            f"| {mode} | {sr['mean']:.3f}±{sr['std']:.3f} | {ret['mean']:.3f}±{ret['std']:.3f} | "
            f"{rmse['mean']:.3f}±{rmse['std']:.3f} | {effort['mean']:.3f}±{effort['std']:.3f} | "
            f"{tbin['near']['mean']:.1f}/{tbin['mid']['mean']:.1f}/{tbin['far']['mean']:.1f} |"
        )


def _write_three_layer_lstm_report(report: dict, report_path: Path) -> None:
    modes = report["aggregated"]
    seed_count = int(report["seed_count"])
    cfg = report["config_snapshot"]
    c = modes["core_only"]
    b = modes["core_plus_memory_residual"]
    l = modes["core_plus_lstm_memory_residual"]
    lines = [
        "# V3 Three-Layer LSTM Ablation",
        "",
        "## Setup",
        "- Branch: `v3-online-memory`",
        "- L1 planner fixed: `HighLevelHeuristicPlannerV2`",
        "- L3 controller fixed: `rbf_controller`",
        "- L2 deterministic core fixed: heading+distance mapping to desired `[v, omega]`",
        "- Ablations:",
        "  - A: `core_only`",
        "  - B: `core_plus_memory_residual`",
        "  - C: `core_plus_lstm_memory_residual` (LSTM residual over deterministic core, bounded and gated)",
        f"- Seeds: `{report['seeds']}` (count={seed_count})",
        f"- Fairness controls: same seeds/env/eval episodes per mode, obstacle_count forced to `0`, identical L1/L3 and core mapping",
        "",
        "## Key Hyperparameters",
        f"- train_episodes: `{cfg.get('train_episodes', cfg.get('collect_episodes', 120))}`",
        f"- eval_episodes: `{cfg.get('eval_episodes', 40)}`",
        f"- seq_len: `{cfg.get('seq_len', 10)}`",
        f"- hidden_dim: `{cfg.get('hidden_dim', 128)}`",
        f"- memory_residual_clip: `{cfg.get('memory_residual_clip', 0.15)}`",
        f"- lstm_residual_clip: `{cfg.get('lstm_residual_clip', cfg.get('memory_residual_clip', 0.15))}`",
        f"- gate: `{json.dumps(cfg.get('lstm_gate', {}), sort_keys=True)}`",
        "",
        "## Aggregate Results (mean +/- std over seeds)",
        "",
        "| mode | success_rate | avg_return | tracking_rmse | control_effort | timeout near/mid/far |",
        "|---|---:|---:|---:|---:|---:|",
        f"| A core_only | {c['success_rate']['mean']:.3f} +/- {c['success_rate']['std']:.3f} | "
        f"{c['episode_return_mean']['mean']:.3f} +/- {c['episode_return_mean']['std']:.3f} | "
        f"{c['tracking_rmse']['mean']:.3f} +/- {c['tracking_rmse']['std']:.3f} | "
        f"{c['control_effort']['mean']:.3f} +/- {c['control_effort']['std']:.3f} | "
        f"{c['timeout_distance_bins']['near']['mean']:.1f}/{c['timeout_distance_bins']['mid']['mean']:.1f}/{c['timeout_distance_bins']['far']['mean']:.1f} |",
        f"| B core+memory | {b['success_rate']['mean']:.3f} +/- {b['success_rate']['std']:.3f} | "
        f"{b['episode_return_mean']['mean']:.3f} +/- {b['episode_return_mean']['std']:.3f} | "
        f"{b['tracking_rmse']['mean']:.3f} +/- {b['tracking_rmse']['std']:.3f} | "
        f"{b['control_effort']['mean']:.3f} +/- {b['control_effort']['std']:.3f} | "
        f"{b['timeout_distance_bins']['near']['mean']:.1f}/{b['timeout_distance_bins']['mid']['mean']:.1f}/{b['timeout_distance_bins']['far']['mean']:.1f} |",
        f"| C core+LSTM+memory | {l['success_rate']['mean']:.3f} +/- {l['success_rate']['std']:.3f} | "
        f"{l['episode_return_mean']['mean']:.3f} +/- {l['episode_return_mean']['std']:.3f} | "
        f"{l['tracking_rmse']['mean']:.3f} +/- {l['tracking_rmse']['std']:.3f} | "
        f"{l['control_effort']['mean']:.3f} +/- {l['control_effort']['std']:.3f} | "
        f"{l['timeout_distance_bins']['near']['mean']:.1f}/{l['timeout_distance_bins']['mid']['mean']:.1f}/{l['timeout_distance_bins']['far']['mean']:.1f} |",
        "",
        "## Conclusions",
        f"- Delta B-A success_rate: {report['comparison']['success_rate_delta_b_minus_a_mean']:+.3f}",
        f"- Delta C-B success_rate: {report['comparison']['success_rate_delta_c_minus_b_mean']:+.3f}",
        f"- Delta C-A success_rate: {report['comparison']['success_rate_delta_c_minus_a_mean']:+.3f}",
        f"- Delta C-A avg_return: {report['comparison']['avg_return_delta_c_minus_a_mean']:+.3f}",
        (
            f"- LSTM gate activation (C): {l['gate_activation_rate']['mean']:.3f} +/- {l['gate_activation_rate']['std']:.3f}"
            if "gate_activation_rate" in l
            else "- LSTM gate activation (C): not enabled"
        ),
    ]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_v3_three_layer_lstm_ablation(cfg: dict, report_path: Path | None = None) -> dict:
    seeds = cfg.get("seeds")
    if seeds is None:
        base = int(cfg.get("seed", 11))
        seeds = [base, base + 101, base + 202]
    seeds = [int(s) for s in seeds]
    if len(seeds) < 3:
        raise ValueError("three_layer_lstm_ablation requires at least 3 seeds for mean/std reporting.")

    per_seed: dict[str, list[dict]] = {
        "core_only": [],
        "core_plus_memory_residual": [],
        "core_plus_lstm_memory_residual": [],
    }
    train_diagnostics: dict[int, dict] = {}

    for seed in seeds:
        seed_cfg = dict(cfg)
        seed_cfg["seed"] = int(seed)
        memory_bank, mem_diag = _train_memory_residual_bank(seed_cfg)
        lstm_policy, lstm_diag = _train_lstm_residual_policy(seed_cfg, memory_bank, seed)
        train_diagnostics[int(seed)] = {"memory": mem_diag, "lstm": lstm_diag}

        per_seed["core_only"].append(
            _eval_three_layer_mode(seed_cfg, seed, mode="core_only", memory_bank=None, lstm_policy=None)
        )
        per_seed["core_plus_memory_residual"].append(
            _eval_three_layer_mode(
                seed_cfg,
                seed,
                mode="core_plus_memory_residual",
                memory_bank=memory_bank,
                lstm_policy=None,
            )
        )
        per_seed["core_plus_lstm_memory_residual"].append(
            _eval_three_layer_mode(
                seed_cfg,
                seed,
                mode="core_plus_lstm_memory_residual",
                memory_bank=memory_bank,
                lstm_policy=lstm_policy,
            )
        )

    aggregated = _aggregate_three_layer_by_seed(per_seed)
    comp = {
        "success_rate_delta_b_minus_a_mean": float(
            aggregated["core_plus_memory_residual"]["success_rate"]["mean"]
            - aggregated["core_only"]["success_rate"]["mean"]
        ),
        "success_rate_delta_c_minus_b_mean": float(
            aggregated["core_plus_lstm_memory_residual"]["success_rate"]["mean"]
            - aggregated["core_plus_memory_residual"]["success_rate"]["mean"]
        ),
        "success_rate_delta_c_minus_a_mean": float(
            aggregated["core_plus_lstm_memory_residual"]["success_rate"]["mean"]
            - aggregated["core_only"]["success_rate"]["mean"]
        ),
        "avg_return_delta_c_minus_a_mean": float(
            aggregated["core_plus_lstm_memory_residual"]["episode_return_mean"]["mean"]
            - aggregated["core_only"]["episode_return_mean"]["mean"]
        ),
    }

    report = {
        "train_mode": "three_layer_lstm_ablation",
        "seed_count": int(len(seeds)),
        "seeds": seeds,
        "fixed_layers": {
            "l1_planner": "HighLevelHeuristicPlannerV2",
            "l2_core": "deterministic_heading_distance_mapping",
            "l3_controller": "rbf_controller",
        },
        "l2_residual_model": {
            "memory_residual_clip": float(cfg.get("memory_residual_clip", 0.15)),
            "lstm_residual_clip": float(cfg.get("lstm_residual_clip", cfg.get("memory_residual_clip", 0.15))),
            "gate": cfg.get("lstm_gate", {}),
        },
        "per_seed_results": per_seed,
        "aggregated": aggregated,
        "comparison": comp,
        "train_diagnostics": train_diagnostics,
        "config_snapshot": cfg,
    }

    _print_three_layer_key_table(aggregated)
    if report_path is not None:
        _write_three_layer_lstm_report(report, report_path)
    return report


def _build_semantic_plan_packet(obs: np.ndarray, route_complexity: str) -> dict:
    """L1 semantic packet: obstacle-aware regional route intent (no direct control)."""
    x, y, gx, gy = float(obs[0]), float(obs[1]), float(obs[5]), float(obs[6])
    obs_dx, obs_dy, obs_dist = float(obs[7]), float(obs[8]), float(obs[9])
    start = np.array([x, y], dtype=np.float32)
    goal = np.array([gx, gy], dtype=np.float32)
    delta = goal - start
    norm = float(np.linalg.norm(delta))
    if norm < 1e-6:
        delta = np.array([1.0, 0.0], dtype=np.float32)
        norm = 1.0
    unit = delta / norm
    perp = np.array([-unit[1], unit[0]], dtype=np.float32)

    complexity = str(route_complexity).lower()
    alias = {
        "minimal": "minimal",
        "very_low": "very_low",
        "low": "low",
        "medium": "medium",
        "high": "high",
        "very_high": "very_high",
        "extreme": "extreme",
    }
    level = alias.get(complexity, "medium")

    near_obs = bool(np.isfinite(obs_dist) and obs_dist < 0.60)
    side = 1.0
    if near_obs:
        side = -1.0 if obs_dy >= 0.0 else 1.0
    elif float(obs[4]) < 0.0:
        side = -1.0

    profiles = {
        "minimal": {"alphas": [1.00], "curve": [0.00], "handoff": 0.16, "speed": 0.86},
        "very_low": {"alphas": [0.55, 1.00], "curve": [0.12, 0.00], "handoff": 0.17, "speed": 0.82},
        "low": {"alphas": [0.32, 0.72, 1.00], "curve": [0.22, -0.10, 0.00], "handoff": 0.19, "speed": 0.76},
        "medium": {"alphas": [0.25, 0.54, 0.82, 1.00], "curve": [0.32, -0.20, 0.14, 0.00], "handoff": 0.21, "speed": 0.70},
        "high": {"alphas": [0.18, 0.42, 0.66, 0.86, 1.00], "curve": [0.40, -0.28, 0.24, -0.12, 0.00], "handoff": 0.23, "speed": 0.64},
        "very_high": {"alphas": [0.15, 0.34, 0.55, 0.75, 0.90, 1.00], "curve": [0.46, -0.34, 0.28, -0.22, 0.12, 0.00], "handoff": 0.24, "speed": 0.60},
        "extreme": {"alphas": [0.12, 0.28, 0.45, 0.62, 0.78, 0.92, 1.00], "curve": [0.52, -0.40, 0.34, -0.30, 0.22, -0.12, 0.00], "handoff": 0.25, "speed": 0.56},
    }
    prof = profiles[level]
    obs_urgency = float(np.clip((0.60 - obs_dist) / 0.60, 0.0, 1.0)) if np.isfinite(obs_dist) else 0.0
    curvature_scale = 1.0 + 0.8 * obs_urgency
    anchors = []
    for a, c in zip(prof["alphas"], prof["curve"]):
        lateral = side * curvature_scale * c
        anchor = start + float(a) * delta + float(lateral) * norm * perp
        anchors.append(anchor.astype(np.float32))
    anchors[-1] = goal.astype(np.float32)

    return {
        "semantic_intent": "reach_goal_via_regional_corridors",
        "route_complexity": level,
        "route_tags": {
            "obstacle_aware": True,
            "near_obstacle": bool(near_obs),
            "corridor_side": "left" if side > 0 else "right",
        },
        "region_waypoints": anchors,
        "active_region_idx": 0,
        "handoff_radius": float(prof["handoff"]),
        "speed_hint": float(prof["speed"]),
    }


def _active_region_waypoint(obs: np.ndarray, packet: dict) -> np.ndarray:
    pts: list[np.ndarray] = packet["region_waypoints"]
    idx = int(packet.get("active_region_idx", 0))
    idx = min(max(idx, 0), len(pts) - 1)
    handoff = float(packet.get("handoff_radius", 0.18))
    pos = np.array([float(obs[0]), float(obs[1])], dtype=np.float32)
    while idx < len(pts) - 1:
        if float(np.linalg.norm(pts[idx] - pos)) > handoff:
            break
        idx += 1
    packet["active_region_idx"] = idx
    return pts[idx]


def _l2_plan_local_trajectory(obs: np.ndarray, packet: dict, horizon: int = 3) -> np.ndarray:
    """L2 local planner: converts semantic route waypoint into short local trajectory."""
    target = _active_region_waypoint(obs, packet)
    pos = np.array([float(obs[0]), float(obs[1])], dtype=np.float32)
    to_target = target - pos
    dist = float(np.linalg.norm(to_target))
    if dist < 1e-6:
        return np.stack([target], axis=0)
    forward = to_target / dist
    lateral = np.array([-forward[1], forward[0]], dtype=np.float32)

    obs_dx, obs_dy, obs_dist = float(obs[7]), float(obs[8]), float(obs[9])
    avoid = np.zeros(2, dtype=np.float32)
    if np.isfinite(obs_dist) and obs_dist < 0.45:
        # Push trajectory away from close obstacle, with a lateral bias for smoother bypassing.
        obs_vec = np.array([obs_dx, obs_dy], dtype=np.float32)
        away = -obs_vec / max(float(np.linalg.norm(obs_vec)), 1e-6)
        urgency = float(np.clip((0.45 - obs_dist) / 0.45, 0.0, 1.0))
        lat_sign = 1.0 if float(np.dot(lateral, away)) >= 0.0 else -1.0
        avoid = urgency * (0.20 * away + 0.12 * lat_sign * lateral)

    step = float(np.clip(0.22 + 0.25 * dist, 0.12, 0.42))
    local = []
    for k in range(max(int(horizon), 1)):
        alpha = float((k + 1) / max(int(horizon), 1))
        raw = pos + alpha * step * forward + alpha * avoid
        toward = pos + min(alpha * step, dist) * forward
        blend = 0.70 * toward + 0.30 * raw
        local.append(blend.astype(np.float32))
    return np.stack(local, axis=0)


def _desired_from_waypoint(obs: np.ndarray, waypoint_xy: np.ndarray, speed_hint: float) -> np.ndarray:
    dx = float(waypoint_xy[0] - obs[0])
    dy = float(waypoint_xy[1] - obs[1])
    yaw = float(obs[2])
    dist = float(np.hypot(dx, dy))
    desired_heading = float(np.arctan2(dy, dx))
    heading_err = (desired_heading - yaw + np.pi) % (2 * np.pi) - np.pi
    return _clip_desired(np.array([speed_hint * dist, 1.4 * heading_err], dtype=np.float32))


def _l3_follow_trajectory(obs: np.ndarray, trajectory_xy: np.ndarray, speed_hint: float) -> tuple[np.ndarray, np.ndarray]:
    """L3 fixed deterministic follower: trajectory -> desired [v, omega] -> low-level action."""
    if trajectory_xy.ndim != 2 or trajectory_xy.shape[0] == 0:
        waypoint = np.array([float(obs[5]), float(obs[6])], dtype=np.float32)
    else:
        look_idx = min(1, trajectory_xy.shape[0] - 1)
        waypoint = trajectory_xy[look_idx]
    desired = _desired_from_waypoint(obs, waypoint, speed_hint=speed_hint)
    return _rbf_controller(obs, desired), desired


def _sample_disturbance(levels: list[str], rng: np.random.Generator) -> str:
    if not levels:
        return "medium"
    idx = int(rng.integers(0, len(levels)))
    return str(levels[idx])


def _tier_episodes(tier_cfg: dict, key: str, fallback: int) -> int:
    return int(tier_cfg.get(key, fallback))


def _run_hierarchy_train_c(
    cfg: dict,
    tier_cfg: dict,
    seed: int,
) -> tuple[list[MemorySample], MemoryLSTMPolicy, dict]:
    """Train C-mode correction memory and LSTM residual for L2 local trajectory."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed + 1903)

    env_cfg = cfg.get("env", {})
    max_steps = int(env_cfg.get("max_steps", 250))
    seq_len = int(cfg.get("seq_len", 10))
    hidden_dim = int(cfg.get("hidden_dim", 128))
    horizon = int(cfg.get("local_trajectory_horizon", 3))
    residual_clip = float(cfg.get("memory_residual_clip", 0.10))
    memory_k = int(cfg.get("memory_k", 5))

    memory_capacity = int(cfg.get("memory_bucket_quota", 2400))
    memory_progress_eps = float(cfg.get("memory_progress_eps", 1e-4))
    memory_quality_min = float(cfg.get("memory_quality_min", 0.05))
    memory_quality_progress_scale = float(cfg.get("memory_quality_progress_scale", 1.0))
    memory_quality_success_bonus = float(cfg.get("memory_quality_success_bonus", 0.5))
    memory_success_segment_len = int(cfg.get("memory_success_segment_len", 8))

    train_eps = _tier_episodes(tier_cfg, "train_episodes", int(cfg.get("train_episodes", 70)))
    obstacle_count = int(tier_cfg.get("obstacle_count", 0))
    disturbance_levels = [str(x) for x in tier_cfg.get("disturbance_levels", ["medium"])]

    memory_bank: list[MemorySample] = []
    retrieval_scores: list[float] = []
    memory_write_attempts = 0
    memory_write_accepted = 0
    memory_evictions = 0
    samples: list[Sample] = []

    feat_dim = 15
    target_gain = float(cfg.get("l2_residual_target_gain", 0.65))
    train_done_reasons: Counter[str] = Counter()

    for ep in range(train_eps):
        level = _sample_disturbance(disturbance_levels, rng)
        env = Sim2DEnv(seed=seed + ep, max_steps=max_steps, level=level, obstacle_count=obstacle_count)
        obs = env.reset()
        packet = _build_semantic_plan_packet(obs, route_complexity=str(tier_cfg.get("route_complexity", "low")))
        window: deque[np.ndarray] = deque(maxlen=seq_len)
        prev_distance = float(np.hypot(float(obs[5] - obs[0]), float(obs[6] - obs[1])))
        success_candidates: list[tuple[np.ndarray, np.ndarray, float]] = []

        for _ in range(max_steps):
            base_traj = _l2_plan_local_trajectory(obs, packet, horizon=horizon)
            subgoal = base_traj[0]
            pseudo_packet = {"subgoal_xy": subgoal, "speed_hint": packet["speed_hint"]}
            mem_action, mem_score = _retrieve_memory_action_with_score(obs, memory_bank, memory_k)
            if mem_score is not None:
                retrieval_scores.append(mem_score)
            feat = _build_feature(obs, pseudo_packet, mem_action)
            window.append(feat)

            action, _ = _l3_follow_trajectory(obs, base_traj, speed_hint=float(packet["speed_hint"]))
            next_obs, _, done, info = env.step(action)
            pos = np.array([float(obs[0]), float(obs[1])], dtype=np.float32)
            next_pos = np.array([float(next_obs[0]), float(next_obs[1])], dtype=np.float32)
            moved = next_pos - pos
            desired_step = subgoal - pos
            residual_target = _bounded_delta(target_gain * (desired_step - moved), residual_clip)
            samples.append(Sample(seq=_seq_array_from_window(window, seq_len, feat_dim), target=residual_target))

            dcur = float(info.get("distance", prev_distance))
            progress_delta = prev_distance - dcur
            prev_distance = dcur
            base_quality = max(memory_quality_min, memory_quality_progress_scale * max(progress_delta, 0.0))
            success_now = bool(done and info.get("success", False))

            memory_write_attempts += 1
            if progress_delta > memory_progress_eps or success_now:
                quality = base_quality + (memory_quality_success_bonus if success_now else 0.0)
                memory_evictions += _insert_memory_sample(
                    memory_bank,
                    MemorySample(
                        key=obs[:5].copy(),
                        action=residual_target.copy(),
                        quality=float(max(quality, memory_quality_min)),
                        created_step=int(ep * max_steps),
                    ),
                    memory_capacity,
                )
                memory_write_accepted += 1

            success_candidates.append((obs[:5].copy(), residual_target.copy(), float(base_quality)))
            obs = next_obs
            if done:
                if bool(info.get("success", False)) and memory_success_segment_len > 0:
                    for key, residual, base_q in success_candidates[-memory_success_segment_len:]:
                        memory_write_attempts += 1
                        memory_write_accepted += 1
                        memory_evictions += _insert_memory_sample(
                            memory_bank,
                            MemorySample(
                                key=key.copy(),
                                action=residual.copy(),
                                quality=float(max(base_q + memory_quality_success_bonus, memory_quality_min)),
                                created_step=int(ep * max_steps),
                            ),
                            memory_capacity,
                        )
                train_done_reasons[_done_reason(info)] += 1
                break

    lstm = MemoryLSTMPolicy(in_dim=feat_dim, hid=hidden_dim)
    epochs = int(cfg.get("lstm_epochs", 8))
    batch_size = int(cfg.get("lstm_batch_size", 192))
    lr = float(cfg.get("lstm_lr", 2e-4))
    losses: list[float] = []
    if samples:
        optimizer = optim.Adam(lstm.parameters(), lr=lr)
        mse = nn.MSELoss()
        for _ in range(epochs):
            n = len(samples)
            for _ in range(max(1, n // max(batch_size, 1))):
                idx = np.random.randint(0, n, size=min(batch_size, n))
                seq = torch.from_numpy(np.stack([samples[int(i)].seq for i in idx], axis=0)).float()
                tgt = torch.from_numpy(np.stack([samples[int(i)].target for i in idx], axis=0)).float()
                pred = lstm(seq)
                loss = mse(pred, tgt)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(lstm.parameters(), 5.0)
                optimizer.step()
                losses.append(float(loss.item()))
    lstm.eval()

    diag = {
        "train_episodes": int(train_eps),
        "sample_count": int(len(samples)),
        "memory_size": int(len(memory_bank)),
        "memory_write_attempts": int(memory_write_attempts),
        "memory_write_accepted": int(memory_write_accepted),
        "memory_write_accept_rate": float(memory_write_accepted / max(memory_write_attempts, 1)),
        "memory_eviction_count": int(memory_evictions),
        "memory_retrieval_score_stats": _score_stats(retrieval_scores),
        "train_done_reasons": {k: int(v) for k, v in train_done_reasons.items()},
        "mean_lstm_loss": float(np.mean(losses)) if losses else None,
    }
    return memory_bank, lstm, diag


def _eval_hierarchy_mode(
    cfg: dict,
    tier_cfg: dict,
    seed: int,
    mode: str,
    memory_bank: list[MemorySample] | None,
    lstm_policy: MemoryLSTMPolicy | None,
) -> dict:
    np.random.seed(seed)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed + 911)

    env_cfg = cfg.get("env", {})
    max_steps = int(env_cfg.get("max_steps", 250))
    eval_eps = _tier_episodes(tier_cfg, "eval_episodes", int(cfg.get("eval_episodes", 30)))
    horizon = int(cfg.get("local_trajectory_horizon", 3))
    memory_k = int(cfg.get("memory_k", 5))
    memory_clip = float(cfg.get("memory_residual_clip", 0.10))
    lstm_clip = float(cfg.get("lstm_residual_clip", memory_clip))
    seq_len = int(cfg.get("seq_len", 10))
    feat_dim = 15
    obstacle_count = int(tier_cfg.get("obstacle_count", 0))
    disturbance_levels = [str(x) for x in tier_cfg.get("disturbance_levels", ["medium"])]

    timeout_bins_cfg = cfg.get("timeout_bins", {})
    near_thresh = float(timeout_bins_cfg.get("near", 0.25))
    mid_thresh = float(timeout_bins_cfg.get("mid", 0.75))
    if mid_thresh <= near_thresh:
        mid_thresh = near_thresh + 0.25

    done_reasons: Counter[str] = Counter()
    timeout_distance_bins: Counter[str] = Counter({"near": 0, "mid": 0, "far": 0})
    success = 0
    episode_returns: list[float] = []
    control_effort: list[float] = []
    tracking_rmse: list[float] = []
    path_efficiency: list[float] = []
    progress_ratio: list[float] = []
    waypoint_tracking_rmse: list[float] = []
    min_clearance: list[float] = []
    retrieval_scores: list[float] = []
    gate_total = 0
    gate_triggered = 0

    gate_cfg = cfg.get("lstm_gate", {})
    gate_enabled = bool(gate_cfg.get("enabled", True))
    gate_near_goal = float(gate_cfg.get("near_goal_threshold", 0.35))
    gate_uncertainty = float(gate_cfg.get("uncertainty_threshold", 0.05))

    for ep in range(eval_eps):
        level = _sample_disturbance(disturbance_levels, rng)
        env = Sim2DEnv(seed=seed + 7000 + ep, max_steps=max_steps, level=level, obstacle_count=obstacle_count)
        obs = env.reset()
        packet = _build_semantic_plan_packet(obs, route_complexity=str(tier_cfg.get("route_complexity", "low")))

        start = np.array([float(obs[0]), float(obs[1])], dtype=np.float32)
        goal = np.array([float(obs[5]), float(obs[6])], dtype=np.float32)
        direct_dist = float(np.linalg.norm(goal - start))
        start_dist = direct_dist
        ep_return = 0.0
        effort_sum = 0.0
        dist_hist: list[float] = []
        wp_err_hist: list[float] = []
        path_len = 0.0
        clear_min = 1e9
        last_pos = start.copy()
        window: deque[np.ndarray] = deque(maxlen=seq_len)

        for _ in range(max_steps):
            target = _active_region_waypoint(obs, packet)
            if mode == "no_l2_shortcut":
                traj = np.stack([target], axis=0)
            else:
                traj = _l2_plan_local_trajectory(obs, packet, horizon=horizon)

            mem_action = None
            mem_score = None
            if mode == "l2_memory_lstm" and memory_bank:
                mem_action, mem_score = _retrieve_memory_action_with_score(obs, memory_bank, memory_k)
                if mem_score is not None:
                    retrieval_scores.append(mem_score)

            if mode == "l2_memory_lstm":
                pseudo_packet = {"subgoal_xy": traj[0], "speed_hint": packet["speed_hint"]}
                feat = _build_feature(obs, pseudo_packet, mem_action)
                window.append(feat)
                residual = np.zeros(2, dtype=np.float32)
                if mem_action is not None:
                    residual += _bounded_delta(mem_action, memory_clip)
                if lstm_policy is not None:
                    seq_arr = _seq_array_from_window(window, seq_len, feat_dim)
                    seq_t = torch.from_numpy(seq_arr[None, :, :]).float()
                    with torch.no_grad():
                        lstm_res = lstm_policy(seq_t).squeeze(0).cpu().numpy().astype(np.float32)
                    allow = True
                    if gate_enabled:
                        dist_goal = float(np.linalg.norm(goal - np.array([float(obs[0]), float(obs[1])], dtype=np.float32)))
                        high_unc = bool(mem_score is None or mem_score < gate_uncertainty)
                        allow = bool(dist_goal < gate_near_goal or high_unc)
                    gate_total += 1
                    if allow:
                        gate_triggered += 1
                        residual += _bounded_delta(lstm_res, lstm_clip)
                traj = traj + residual[None, :]

            wp_err_hist.append(float(np.linalg.norm(traj[0] - np.array([float(obs[0]), float(obs[1])], dtype=np.float32))))
            clear_min = min(clear_min, float(obs[9]))
            action, _ = _l3_follow_trajectory(obs, traj, speed_hint=float(packet["speed_hint"]))
            next_obs, reward, done, info = env.step(action)
            ep_return += float(reward)
            dist_hist.append(float(info.get("distance", 0.0)))
            effort_sum += float(info.get("control_effort", np.linalg.norm(action)))
            next_pos = np.array([float(next_obs[0]), float(next_obs[1])], dtype=np.float32)
            path_len += float(np.linalg.norm(next_pos - last_pos))
            last_pos = next_pos
            obs = next_obs

            if done:
                reason = _done_reason(info)
                done_reasons[reason] += 1
                if reason == "timeout":
                    timeout_distance_bins[_timeout_bin(float(info["distance"]), near_thresh, mid_thresh)] += 1
                if bool(info.get("success", False)):
                    success += 1
                break

        final_dist = float(dist_hist[-1]) if dist_hist else start_dist
        episode_returns.append(ep_return)
        control_effort.append(effort_sum)
        tracking_rmse.append(float(np.sqrt(np.mean(np.square(dist_hist)))) if dist_hist else 0.0)
        path_efficiency.append(float(np.clip(direct_dist / max(path_len, direct_dist, 1e-6), 0.0, 1.0)))
        progress_ratio.append(float(np.clip((start_dist - final_dist) / max(start_dist, 1e-6), -1.0, 1.0)))
        waypoint_tracking_rmse.append(float(np.sqrt(np.mean(np.square(wp_err_hist)))) if wp_err_hist else 0.0)
        min_clearance.append(float(clear_min if clear_min < 1e8 else 0.0))

    out = {
        "mode": mode,
        "seed": int(seed),
        "tier": str(tier_cfg.get("name", "unknown")),
        "eval_episodes": int(eval_eps),
        "success_rate": float(success / max(eval_eps, 1)),
        "episode_return_mean": float(np.mean(episode_returns)) if episode_returns else None,
        "tracking_rmse": float(np.mean(tracking_rmse)) if tracking_rmse else None,
        "control_effort": float(np.mean(control_effort)) if control_effort else None,
        "done_reasons": {k: int(v) for k, v in done_reasons.items()},
        "timeout_distance_bins": {k: int(timeout_distance_bins.get(k, 0)) for k in ("near", "mid", "far")},
        "trajectory_quality": {
            "path_efficiency": float(np.mean(path_efficiency)) if path_efficiency else None,
            "progress_ratio": float(np.mean(progress_ratio)) if progress_ratio else None,
            "waypoint_tracking_rmse": float(np.mean(waypoint_tracking_rmse)) if waypoint_tracking_rmse else None,
            "min_obstacle_clearance": float(np.mean(min_clearance)) if min_clearance else None,
        },
    }
    if retrieval_scores:
        out["memory_retrieval_score_stats"] = _score_stats(retrieval_scores)
    if mode == "l2_memory_lstm":
        out["gate_activation_rate"] = float(gate_triggered / max(gate_total, 1))
    return out


def _aggregate_hierarchy_seed_rows(rows: list[dict]) -> dict:
    agg = {
        "n_seeds": int(len(rows)),
        "success_rate": _mean_std([float(r["success_rate"]) for r in rows]),
        "episode_return_mean": _mean_std([float(r["episode_return_mean"]) for r in rows if r.get("episode_return_mean") is not None]),
        "tracking_rmse": _mean_std([float(r["tracking_rmse"]) for r in rows if r.get("tracking_rmse") is not None]),
        "control_effort": _mean_std([float(r["control_effort"]) for r in rows if r.get("control_effort") is not None]),
    }

    done_keys = set()
    for r in rows:
        done_keys.update(r.get("done_reasons", {}).keys())
    agg["done_reasons"] = {
        k: _mean_std([float(r.get("done_reasons", {}).get(k, 0)) for r in rows]) for k in sorted(done_keys)
    }
    agg["timeout_distance_bins"] = {
        k: _mean_std([float(r.get("timeout_distance_bins", {}).get(k, 0)) for r in rows]) for k in ("near", "mid", "far")
    }

    q_metrics = ("path_efficiency", "progress_ratio", "waypoint_tracking_rmse", "min_obstacle_clearance")
    agg["trajectory_quality"] = {
        q: _mean_std(
            [float(r.get("trajectory_quality", {}).get(q)) for r in rows if r.get("trajectory_quality", {}).get(q) is not None]
        )
        for q in q_metrics
    }

    gate_vals = [float(r.get("gate_activation_rate")) for r in rows if r.get("gate_activation_rate") is not None]
    if gate_vals:
        agg["gate_activation_rate"] = _mean_std(gate_vals)
    return agg


def _print_hierarchy_matrix(aggregated: dict, tier_order: list[str]) -> None:
    print("V3 hierarchy-meaning ablation (seed mean±std)")
    print("| tier | mode | success_rate | timeout near/mid/far | path_eff | wp_rmse | control_effort |")
    print("|---|---|---:|---:|---:|---:|---:|")
    mode_order = ("no_l2_shortcut", "l2_no_memory", "l2_memory_lstm")
    for tier in tier_order:
        tier_rows = aggregated[tier]
        for mode in mode_order:
            row = tier_rows[mode]
            sr = row["success_rate"]
            tbin = row["timeout_distance_bins"]
            tq = row["trajectory_quality"]
            ce = row["control_effort"]
            print(
                f"| {tier} | {mode} | {sr['mean']:.3f}±{sr['std']:.3f} | "
                f"{tbin['near']['mean']:.1f}/{tbin['mid']['mean']:.1f}/{tbin['far']['mean']:.1f} | "
                f"{tq['path_efficiency']['mean']:.3f} | {tq['waypoint_tracking_rmse']['mean']:.3f} | "
                f"{ce['mean']:.3f}±{ce['std']:.3f} |"
            )


def run_v3_hierarchy_meaning_ablation(cfg: dict) -> dict:
    """Validate L2 meaning under strict hierarchy contract with tiered benchmark."""
    min_seed_count = int(cfg.get("min_seed_count", 3))
    seeds = [int(s) for s in cfg.get("seeds", [11, 29, 47])]
    if len(seeds) < min_seed_count:
        raise ValueError(f"hierarchy_meaning_ablation requires at least {min_seed_count} seeds.")

    tiers_cfg = cfg.get("tiers", {})
    if not isinstance(tiers_cfg, dict) or not tiers_cfg:
        raise ValueError("Missing tiers config for hierarchy_meaning_ablation.")
    tier_order = [str(t) for t in cfg.get("tier_order", list(tiers_cfg.keys()))]
    missing = [t for t in tier_order if t not in tiers_cfg]
    if missing:
        raise ValueError(f"Missing tier config(s) in tiers: {missing}")
    min_tier_count = int(cfg.get("min_tier_count", 3))
    if len(tier_order) < min_tier_count:
        raise ValueError(f"hierarchy_meaning_ablation requires at least {min_tier_count} tiers.")

    per_seed: dict[str, dict[str, dict[str, list[dict]]]] = {}
    aggregated: dict[str, dict[str, dict]] = {}
    train_diagnostics: dict[str, dict[int, dict]] = {t: {} for t in tier_order}

    for tier in tier_order:
        tier_cfg = dict(tiers_cfg[tier])
        tier_cfg["name"] = tier
        per_seed[tier] = {"no_l2_shortcut": [], "l2_no_memory": [], "l2_memory_lstm": []}
        for seed in seeds:
            memory_bank, lstm_policy, diag = _run_hierarchy_train_c(cfg, tier_cfg, seed)
            train_diagnostics[tier][int(seed)] = diag

            per_seed[tier]["no_l2_shortcut"].append(
                _eval_hierarchy_mode(cfg, tier_cfg, seed, mode="no_l2_shortcut", memory_bank=None, lstm_policy=None)
            )
            per_seed[tier]["l2_no_memory"].append(
                _eval_hierarchy_mode(cfg, tier_cfg, seed, mode="l2_no_memory", memory_bank=None, lstm_policy=None)
            )
            per_seed[tier]["l2_memory_lstm"].append(
                _eval_hierarchy_mode(
                    cfg,
                    tier_cfg,
                    seed,
                    mode="l2_memory_lstm",
                    memory_bank=memory_bank,
                    lstm_policy=lstm_policy,
                )
            )

        aggregated[tier] = {
            mode: _aggregate_hierarchy_seed_rows(per_seed[tier][mode])
            for mode in ("no_l2_shortcut", "l2_no_memory", "l2_memory_lstm")
        }

    comparison = {}
    for tier in tier_order:
        row = aggregated[tier]
        comparison[tier] = {
            "success_rate_delta_b_minus_a": float(row["l2_no_memory"]["success_rate"]["mean"] - row["no_l2_shortcut"]["success_rate"]["mean"]),
            "success_rate_delta_c_minus_b": float(row["l2_memory_lstm"]["success_rate"]["mean"] - row["l2_no_memory"]["success_rate"]["mean"]),
            "success_rate_delta_c_minus_a": float(
                row["l2_memory_lstm"]["success_rate"]["mean"] - row["no_l2_shortcut"]["success_rate"]["mean"]
            ),
        }

    report = {
        "train_mode": "hierarchy_meaning_ablation",
        "seed_count": int(len(seeds)),
        "seeds": seeds,
        "contract": {
            "l1": "semantic/regional route representation only (no direct control)",
            "l2": "local route/trajectory planner",
            "l3": "fixed deterministic trajectory follower",
        },
        "modes": {
            "A": "no_l2_shortcut",
            "B": "l2_no_memory",
            "C": "l2_memory_lstm",
        },
        "tier_order": tier_order,
        "tiers": tiers_cfg,
        "per_seed_results": per_seed,
        "aggregated": aggregated,
        "comparison": comparison,
        "train_diagnostics": train_diagnostics,
        "config_snapshot": cfg,
    }
    _print_hierarchy_matrix(aggregated, tier_order=tier_order)
    return report


def _resolve_report_path() -> Path:
    here = Path(__file__).resolve()
    for p in [here] + list(here.parents):
        if (p / "docs").exists():
            return p / "docs" / "L2_DETERMINISTIC_PLUS_MEMORY.md"
    return Path("docs/L2_DETERMINISTIC_PLUS_MEMORY.md")


def _resolve_three_layer_report_path() -> Path:
    here = Path(__file__).resolve()
    for p in [here] + list(here.parents):
        if (p / "docs").exists():
            return p / "docs" / "V3_THREE_LAYER_LSTM_ABLATION.md"
    return Path("docs/V3_THREE_LAYER_LSTM_ABLATION.md")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train RL-brainer v3 (online-memory, WIP)")
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    mode = str(cfg.get("train_mode", "online_v3"))
    if mode == "offline_legacy":
        out = train_and_eval_offline_legacy(cfg)
    elif mode == "online_v3_ff":
        out = train_and_eval_online_v3_ff(cfg, memory_mode=str(cfg.get("memory_mode", "memory_on")))
    elif mode == "l2_memory_ablation":
        out = run_l2_memory_ablation(cfg)
    elif mode == "l2_deterministic_plus_memory":
        out = run_l2_deterministic_plus_memory(cfg, report_path=_resolve_report_path())
    elif mode == "three_layer_lstm_ablation":
        out = run_v3_three_layer_lstm_ablation(cfg, report_path=_resolve_three_layer_report_path())
    elif mode == "hierarchy_meaning_ablation":
        out = run_v3_hierarchy_meaning_ablation(cfg)
    else:
        out = train_and_eval_online_v3(cfg)
    print(json.dumps(out, indent=2))
    if args.out:
        p = Path(args.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
