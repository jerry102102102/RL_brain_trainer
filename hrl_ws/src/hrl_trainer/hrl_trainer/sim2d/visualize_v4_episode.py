from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches

from .env import Sim2DEnv
from .planner import HighLevelHeuristicPlannerV2
from .train_rl_brainer_v4 import (
    OnlineRecurrentPolicy,
    _bounded_delta,
    _build_feature,
    _clip_desired,
    _deterministic_core_mapping,
    _oracle_action,
    _rbf_controller,
    _retrieve_memory_action,
)


def run_episode(
    seed: int,
    level: str,
    obstacle_count: int,
    max_steps: int,
    waypoint_scale: float,
    model_path: str | None,
    mode: str = "l2",
    min_start_goal_dist: float = 1.1,
    control_mode: str = "velocity",
):
    env = Sim2DEnv(
        seed=seed,
        max_steps=max_steps,
        level=level,
        obstacle_count=obstacle_count,
        control_mode=control_mode,
        min_start_goal_dist=min_start_goal_dist,
    )
    planner = HighLevelHeuristicPlannerV2(waypoint_scale=waypoint_scale)

    obs = env.reset()
    states = [env.state.copy()]
    packets = []
    actions = []
    infos = []

    memory_bank = []
    seq_len = 10
    hist = []

    model = None
    if model_path:
        model = OnlineRecurrentPolicy(in_dim=15, hid=128)
        ckpt = np.load(model_path, allow_pickle=True) if model_path.endswith('.npz') else None
        if ckpt is not None:
            raise RuntimeError("npz checkpoint not supported for this visualizer; pass a PyTorch .pt state_dict")
        import torch

        sd = torch.load(model_path, map_location="cpu")
        model.load_state_dict(sd)
        model.eval()

    for _ in range(max_steps):
        packet = planner.plan(obs)
        if mode == "l3_only":
            # bypass local-L2 style subgoaling: hand goal directly to deterministic core
            packet = {
                "subgoal_xy": np.array([obs[5], obs[6]], dtype=np.float32),
                "speed_hint": 0.7,
            }

        mem_action = _retrieve_memory_action(obs, memory_bank, memory_k=5)
        feat = _build_feature(obs, packet, mem_action)
        hist.append(feat)
        if len(hist) > seq_len:
            hist.pop(0)

        core_desired = _deterministic_core_mapping(obs, packet)
        desired = _oracle_action(obs, packet["subgoal_xy"], packet.get("speed_hint", 0.7))

        if mode == "l3_only":
            desired = core_desired
        elif model is not None and len(hist) == seq_len:
            import torch

            seq = torch.tensor(np.stack(hist, axis=0)[None, ...], dtype=torch.float32)
            with torch.no_grad():
                pred_action, _ = model(seq)
            pred = pred_action.squeeze(0).numpy().astype(np.float32)
            desired = _clip_desired(core_desired + _bounded_delta(pred, 0.45))

        action = _rbf_controller(obs, desired)
        next_obs, _, done, info = env.step(action)

        memory_bank.append((obs[:5].copy(), desired.copy()))
        if len(memory_bank) > 6000:
            memory_bank.pop(0)

        packets.append(packet)
        actions.append(action.copy())
        infos.append(info)
        states.append(env.state.copy())

        obs = next_obs
        if done:
            break

    return env, states, packets, actions, infos


def draw_and_save(env, states, packets, infos, out_path: Path, fps: int = 12):
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 6), dpi=120)
    world = env.world_half_extent
    goal_xy = states[0][5:7]

    # static items
    ax.set_xlim(-world, world)
    ax.set_ylim(-world, world)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.2)
    ax.set_title("V4 Sim2D rollout")

    for ox, oy, rr in env.obstacles:
        ax.add_patch(patches.Circle((ox, oy), rr, color="tomato", alpha=0.5))

    ax.add_patch(patches.Circle((goal_xy[0], goal_xy[1]), 0.06, color="limegreen", alpha=0.9, label="goal"))

    robot_patch = patches.RegularPolygon(
        (states[0][0], states[0][1]),
        numVertices=5,
        radius=env.robot_circ_radius,
        orientation=float(states[0][2]),
        color="royalblue",
        alpha=0.9,
    )
    ax.add_patch(robot_patch)
    path_line, = ax.plot([], [], color="navy", lw=2, alpha=0.7, label="trajectory")
    subgoal_dot, = ax.plot([], [], "o", color="orange", ms=5, label="subgoal")
    status_txt = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top", ha="left", fontsize=9)
    ax.legend(loc="lower right")

    xs, ys = [], []

    frames = []
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

    canvas = FigureCanvas(fig)

    for i in range(len(states) - 1):
        s = states[i + 1]
        x, y, yaw = float(s[0]), float(s[1]), float(s[2])
        xs.append(x)
        ys.append(y)
        robot_patch.xy = (x, y)
        robot_patch.orientation = yaw
        path_line.set_data(xs, ys)

        if i < len(packets):
            sg = packets[i]["subgoal_xy"]
            subgoal_dot.set_data([sg[0]], [sg[1]])

        info = infos[i] if i < len(infos) else {"distance": 0.0, "success": False, "collided": False}
        status_txt.set_text(
            f"step={i+1}\n"
            f"dist={info.get('distance', 0.0):.3f}\n"
            f"success={info.get('success', False)} collided={info.get('collided', False)}"
        )

        canvas.draw()
        w, h = fig.canvas.get_width_height()
        img = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)[..., :3]
        frames.append(img.copy())

    plt.close(fig)

    try:
        import imageio.v2 as imageio

        imageio.mimsave(out_path, frames, fps=fps)
    except Exception as e:
        fallback = out_path.with_suffix(".npz")
        np.savez_compressed(fallback, frames=np.stack(frames, axis=0))
        raise RuntimeError(f"Failed to save gif/mp4 ({e}). Saved raw frames to {fallback}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--level", type=str, default="hard", choices=["easy", "medium", "hard"])
    ap.add_argument("--obstacle-count", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=220)
    ap.add_argument("--waypoint-scale", type=float, default=0.35)
    ap.add_argument("--model", type=str, default=None, help="Optional .pt state_dict for OnlineRecurrentPolicy")
    ap.add_argument("--mode", type=str, default="l2", choices=["l2", "l3_only"], help="l2=planner/local rollout, l3_only=direct goal handoff")
    ap.add_argument("--min-start-goal-dist", type=float, default=1.1)
    ap.add_argument("--control-mode", type=str, default="velocity", choices=["velocity", "acceleration"])
    ap.add_argument("--out", type=str, default="/tmp/v4_episode.gif")
    ap.add_argument("--fps", type=int, default=12)
    args = ap.parse_args()

    env, states, packets, actions, infos = run_episode(
        seed=args.seed,
        level=args.level,
        obstacle_count=args.obstacle_count,
        max_steps=args.max_steps,
        waypoint_scale=args.waypoint_scale,
        model_path=args.model,
        mode=args.mode,
        min_start_goal_dist=args.min_start_goal_dist,
        control_mode=args.control_mode,
    )
    draw_and_save(env, states, packets, infos, Path(args.out), fps=args.fps)

    final = infos[-1] if infos else {"success": False, "collided": False, "distance": None}
    print({
        "out": args.out,
        "steps": len(infos),
        "success": final.get("success", False),
        "collided": final.get("collided", False),
        "final_distance": final.get("distance", None),
    })


if __name__ == "__main__":
    main()
