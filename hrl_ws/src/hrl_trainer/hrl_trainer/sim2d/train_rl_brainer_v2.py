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


def _oracle_action(obs: np.ndarray, subgoal_xy: np.ndarray, speed_hint: float = 0.7, option_id: str = "APPROACH") -> np.ndarray:
    x, y, yaw, v, omega = obs[:5]
    dx, dy = float(subgoal_xy[0] - x), float(subgoal_xy[1] - y)
    desired_heading = float(np.arctan2(dy, dx))
    heading_err = (desired_heading - yaw + np.pi) % (2 * np.pi) - np.pi
    dist = float(np.hypot(dx, dy))

    v_target = np.clip(speed_hint * dist, 0.0, 1.2)
    omega_target = np.clip(1.5 * heading_err, -1.8, 1.8)

    # Docking specialization
    if option_id == "DOCK_ALIGN":
        v_target = 0.02 * dist
        omega_target = np.clip(2.0 * heading_err, -1.8, 1.8)
    elif option_id == "DOCK_APPROACH":
        if abs(heading_err) > 0.2:
            v_target *= 0.25
        else:
            v_target *= 0.6
        omega_target = np.clip(1.2 * heading_err, -1.2, 1.2)

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


def _retrieve_memory_action(memory_bank: deque[tuple[np.ndarray, np.ndarray]], obs: np.ndarray, k: int = 5) -> tuple[np.ndarray | None, float | None]:
    if not memory_bank:
        return None, None
    key = obs[:5]
    keys = np.stack([m[0] for m in memory_bank], axis=0)
    d2 = np.sum((keys - key[None, :]) ** 2, axis=1)
    k = max(1, min(k, len(memory_bank)))
    idxs = np.argpartition(d2, k - 1)[:k]
    weights = 1.0 / (np.sqrt(d2[idxs]) + 1e-6)
    weights = weights / np.sum(weights)
    acts = np.stack([memory_bank[i][1] for i in idxs], axis=0)
    act = np.sum(acts * weights[:, None], axis=0).astype(np.float32)
    return act, float(np.mean(np.sqrt(d2[idxs])))


def _build_feature(obs: np.ndarray, packet: dict, mem_action: np.ndarray | None) -> np.ndarray:
    # base obs (10 dims currently) + subgoal delta(2) + speed_hint(1) + memory action(2)
    dx = packet["subgoal_xy"][0] - obs[0]
    dy = packet["subgoal_xy"][1] - obs[1]
    sh = float(packet.get("speed_hint", 0.6))
    if mem_action is None:
        mem_action = np.zeros(2, dtype=np.float32)
    return np.concatenate([obs.astype(np.float32), np.array([dx, dy, sh], dtype=np.float32), mem_action.astype(np.float32)], dtype=np.float32)


def train_and_eval(cfg: dict) -> dict:
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
    memory_k = int(cfg.get("memory_k", 5))

    memory_bank: deque[tuple[np.ndarray, np.ndarray]] = deque(maxlen=6000)  # (obs_key, desired_vo)
    dataset: list[Sample] = []
    option_counter = Counter()
    retrieval_dists = []

    for ep in range(collect_eps):
        env = Sim2DEnv(seed=seed + ep, max_steps=max_steps, level=str(env_cfg.get("disturbance_level", "medium")), obstacle_count=int(env_cfg.get("obstacle_count", 0)))
        obs = env.reset()
        hist: deque[np.ndarray] = deque(maxlen=seq_len)

        for _ in range(max_steps):
            packet = planner.plan(obs)
            option_counter[packet["option_id"]] += 1
            mem_action, md = _retrieve_memory_action(memory_bank, obs, k=memory_k)
            if md is not None:
                retrieval_dists.append(md)

            feat = _build_feature(obs, packet, mem_action)
            hist.append(feat)
            if len(hist) < seq_len:
                desired = _oracle_action(obs, packet["subgoal_xy"], packet.get("speed_hint", 0.7), packet.get("option_id", "APPROACH"))
                memory_bank.append((obs[:5].copy(), desired.copy()))
                action = _rbf_controller(obs, desired)
                obs, _, done, _ = env.step(action)
                if done:
                    break
                continue

            desired = _oracle_action(obs, packet["subgoal_xy"], packet.get("speed_hint", 0.7), packet.get("option_id", "APPROACH"))
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
    done_reasons = Counter()

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
            mem_action, md = _retrieve_memory_action(memory_bank, obs, k=memory_k)
            if md is not None:
                retrieval_dists.append(md)

            feat = _build_feature(obs, packet, mem_action)
            hist.append(feat)
            if len(hist) < seq_len:
                desired = _oracle_action(obs, packet["subgoal_xy"], packet.get("speed_hint", 0.7), packet.get("option_id", "APPROACH"))
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
                    done_reasons["success"] += 1
                elif info.get("collided", False):
                    done_reasons["collision"] += 1
                else:
                    done_reasons["timeout"] += 1
                break

        rmse.append(float(np.sqrt(np.mean(np.square(dist_hist)))) if dist_hist else 0.0)
        effort.append(e_sum)
        if recover is not None:
            rec.append(recover)

    return {
        "device": str(device),
        "dataset_size": len(dataset),
        "memory_bank_size": len(memory_bank),
        "mean_memory_retrieval_distance": float(np.mean(retrieval_dists)) if retrieval_dists else None,
        "option_counts": dict(option_counter),
        "mean_loss": float(np.mean(losses)) if losses else None,
        "success_rate": success / max(eval_eps, 1),
        "done_reasons": dict(done_reasons),
        "time_to_convergence": float(np.mean(conv)) if conv else None,
        "tracking_rmse": float(np.mean(rmse)) if rmse else None,
        "recovery_time": float(np.mean(rec)) if rec else None,
        "control_effort": float(np.mean(effort)) if effort else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train RL-brainer v2 (Memory + LSTM + RBF)")
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    out = train_and_eval(cfg)
    print(json.dumps(out, indent=2))
    if args.out:
        p = Path(args.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
