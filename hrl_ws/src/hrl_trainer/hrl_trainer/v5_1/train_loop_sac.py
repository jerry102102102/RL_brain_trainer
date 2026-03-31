from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from .artifact_schema import validate_train_row
from .reward_v1 import compute_reward_v1
from .sac_agent import SACAgent, SACConfig


class ToyReachEnv:
    def __init__(
        self,
        seed: int = 0,
        max_steps: int = 160,
        success_dwell_steps: int = 10,
        near_goal_tol: float = 0.05,
        goal_tol_pos: float = 0.02,
        goal_tol_yaw: float = 0.08,
    ):
        self.rng = np.random.default_rng(seed)
        self.max_steps = max_steps
        self.success_dwell_steps = int(success_dwell_steps)
        self.near_goal_tol = float(near_goal_tol)
        self.goal_tol_pos = float(goal_tol_pos)
        self.goal_tol_yaw = float(goal_tol_yaw)
        self.step_idx = 0
        self.near_goal_streak = 0
        self.goal = np.zeros(3, dtype=np.float32)
        self.state = np.zeros(3, dtype=np.float32)
        self.prev_action = np.zeros(3, dtype=np.float32)

    def reset(self, episode_seed: int) -> np.ndarray:
        self.rng = np.random.default_rng(episode_seed)
        self.step_idx = 0
        self.near_goal_streak = 0
        self.state = self.rng.uniform(-0.06, 0.06, size=3).astype(np.float32)
        self.prev_action = np.zeros(3, dtype=np.float32)
        return self.state.copy()

    def step(self, action: np.ndarray) -> tuple[np.ndarray, dict[str, float | bool], bool, bool]:
        self.step_idx += 1
        clamp = np.clip(action, -0.08, 0.08)
        clamp_ratio = float(np.mean(np.abs(action - clamp) > 1e-6))
        noise = self.rng.normal(0.0, 0.002, size=3).astype(np.float32)
        self.state = self.state - 0.35 * clamp.astype(np.float32) + noise
        err = float(np.linalg.norm(self.state - self.goal))
        yaw_err = float(abs(self.state[-1]))
        safety = bool(np.any(np.abs(clamp) > 0.075))
        intervention = False
        timeout = self.step_idx >= self.max_steps
        goal_hit = bool(err <= self.goal_tol_pos and yaw_err <= self.goal_tol_yaw)
        near_goal = bool(err <= self.near_goal_tol)
        near_goal_streak_prev = int(self.near_goal_streak)
        self.near_goal_streak = self.near_goal_streak + 1 if near_goal else 0
        success = bool(self.near_goal_streak >= self.success_dwell_steps)
        done = timeout or safety or intervention or success
        truncated = timeout and not success
        return self.state.copy(), {
            "error": err,
            "yaw_error": yaw_err,
            "clamp_ratio": clamp_ratio,
            "safety": safety,
            "intervention": intervention,
            "success": success,
            "goal_hit": goal_hit,
            "near_goal": near_goal,
            "near_goal_streak_prev": near_goal_streak_prev,
            "near_goal_streak_curr": int(self.near_goal_streak),
        }, done, truncated


def run_train(episodes: int, seed: int, artifact_root: Path) -> Path:
    cfg = SACConfig(obs_dim=3, action_dim=3)
    agent = SACAgent(cfg, seed=seed)
    env = ToyReachEnv(seed=seed)
    artifact_root.mkdir(parents=True, exist_ok=True)
    train_metrics_path = artifact_root / "train_metrics.jsonl"

    global_step = 0
    with train_metrics_path.open("w", encoding="utf-8") as fp:
        for ep in range(episodes):
            obs = env.reset(seed + ep)
            err_prev = float(np.linalg.norm(obs))
            act_prev = None
            act_prev2 = None
            for _ in range(env.max_steps):
                action = agent.select_action(obs, deterministic=False)
                nxt, info, done, truncated = env.step(action)
                terminal_reason = "ongoing"
                if done:
                    if info["success"]:
                        terminal_reason = "success"
                    elif info["safety"]:
                        terminal_reason = "safety_abort"
                    elif info["intervention"]:
                        terminal_reason = "intervention_abort"
                    else:
                        terminal_reason = "timeout"

                rb = compute_reward_v1(
                    error_prev=err_prev,
                    error_curr=float(info["error"]),
                    yaw_error_curr=float(info["yaw_error"]),
                    action_curr=action,
                    action_prev=act_prev,
                    action_prev2=act_prev2,
                    clamp_ratio=float(info["clamp_ratio"]),
                    safety_violation_event=bool(info["safety"]),
                    intervention_event=bool(info["intervention"]),
                    terminal_reason=terminal_reason,
                    near_goal_streak_prev=int(info.get("near_goal_streak_prev", 0)),
                    near_goal_streak_curr=int(info.get("near_goal_streak_curr", 0)),
                )
                agent.remember(obs, action, rb.reward_total, nxt, bool(done), bool(truncated), info={"terminal_reason": terminal_reason})
                global_step += 1

                if global_step >= cfg.warmup_steps:
                    for row in agent.update():
                        row["global_step"] = global_step
                        assert validate_train_row(row).ok, "invalid train row schema"
                        fp.write(json.dumps(row, ensure_ascii=True) + "\n")
                obs = nxt
                err_prev = float(info["error"])
                act_prev2 = act_prev
                act_prev = action
                if done:
                    break

    ckpt = artifact_root / "checkpoint_latest.pt"
    agent.save(ckpt)
    return ckpt


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Train v5_1 SAC loop")
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--seed", type=int, default=20260331)
    p.add_argument("--artifact-root", type=Path, default=Path("artifacts/v5_1/train/smoke"))
    args = p.parse_args(argv)
    ckpt = run_train(args.episodes, args.seed, args.artifact_root)
    print(f"checkpoint={ckpt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
