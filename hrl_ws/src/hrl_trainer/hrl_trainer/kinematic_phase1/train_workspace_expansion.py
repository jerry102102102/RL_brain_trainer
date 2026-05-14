"""Train Approach policy with Workspace Expansion Curriculum gates."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from .eval.eval_workspace_expansion import evaluate_workspace_expansion_checkpoint
from .training.callbacks import PeriodicCheckpointCallback, PointCurriculumCallback
from .training.policy_config import (
    approach_default_config_path,
    config_dir,
    deep_merge,
    load_yaml_file,
    ppo_default_config_path,
    repo_root,
    to_algorithm_kwargs,
    to_env_config,
    write_json,
)


def _require_training_dependencies() -> None:
    try:
        import gymnasium  # noqa: F401
        import stable_baselines3  # noqa: F401
    except Exception as exc:
        raise RuntimeError("Workspace expansion training requires gymnasium and stable-baselines3.") from exc


def _load_overlay_with_bases(path: Path) -> dict[str, Any]:
    overlay = load_yaml_file(path)
    base_config = overlay.pop("base_config", None)
    if not base_config:
        return overlay
    base_path = Path(str(base_config))
    if not base_path.is_absolute():
        candidate = path.parent / base_path
        cfg_candidate = config_dir() / base_path
        base_path = candidate if candidate.exists() else cfg_candidate if cfg_candidate.exists() else repo_root() / base_path
    return deep_merge(_load_overlay_with_bases(base_path), overlay)


def _load_config(explicit_path: str | None) -> dict[str, Any]:
    cfg = deep_merge(load_yaml_file(approach_default_config_path()), load_yaml_file(ppo_default_config_path()))
    if explicit_path:
        cfg = deep_merge(cfg, _load_overlay_with_bases(Path(explicit_path)))
    return cfg


class WorkspaceEvalGateCallback:  # pragma: no cover - exercised in long SB3 runs
    def __init__(
        self,
        *,
        artifact_root: Path,
        approach_config_path: Path,
        finisher_checkpoint: Path,
        finisher_config_path: Path,
        eval_interval: int,
        episodes: int,
        seed: int,
        stage_indices: list[int],
        current_stage_index: int,
        gate_config: dict[str, Any],
    ) -> None:
        from stable_baselines3.common.callbacks import BaseCallback

        class _Callback(BaseCallback):
            def __init__(self, outer: WorkspaceEvalGateCallback) -> None:
                super().__init__()
                self.outer = outer
                self.best_score = float("-inf")
                self.next_eval_timesteps = int(outer.eval_interval)

            def _on_step(self) -> bool:
                if self.n_calls == 1 or self.num_timesteps < self.next_eval_timesteps:
                    return True
                while self.next_eval_timesteps <= self.num_timesteps:
                    self.next_eval_timesteps += int(self.outer.eval_interval)
                candidate = self.outer.candidates_dir / f"candidate_step_{self.num_timesteps}"
                self.model.save(str(candidate))
                summary = evaluate_workspace_expansion_checkpoint(
                    approach_checkpoint=candidate,
                    approach_config_path=self.outer.approach_config_path,
                    artifact_root=self.outer.eval_dir / f"eval_step_{self.num_timesteps}",
                    finisher_checkpoint=self.outer.finisher_checkpoint,
                    finisher_config_path=self.outer.finisher_config_path,
                    episodes=self.outer.episodes,
                    seed=self.outer.seed,
                    stage_indices=self.outer.stage_indices,
                    gate_config=self.outer.gate_config,
                )
                selection = summary["best_model_selection"]
                record = {
                    "timesteps": int(self.num_timesteps),
                    "candidate": str(candidate) + ".zip",
                    **selection,
                }
                with self.outer.eval_history_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record) + "\n")
                score = float(selection["score"])
                if bool(selection["retention_ok"]) and score > self.best_score:
                    self.best_score = score
                    best_target = self.outer.best_dir / "model_best_by_gate"
                    self.model.save(str(best_target))
                    write_json(self.outer.artifact_root / "best_model_selection_summary.json", record)
                return True

        self.artifact_root = artifact_root
        self.approach_config_path = approach_config_path
        self.finisher_checkpoint = finisher_checkpoint
        self.finisher_config_path = finisher_config_path
        self.eval_interval = max(int(eval_interval), 1)
        self.episodes = max(int(episodes), 1)
        self.seed = int(seed)
        self.stage_indices = list(stage_indices)
        self.current_stage_index = int(current_stage_index)
        self.gate_config = dict(gate_config)
        self.candidates_dir = artifact_root / "gate_candidates"
        self.eval_dir = artifact_root / "gate_evals"
        self.best_dir = artifact_root / "best_checkpoint"
        self.eval_history_path = artifact_root / "eval_history.jsonl"
        self.candidates_dir.mkdir(parents=True, exist_ok=True)
        self.eval_dir.mkdir(parents=True, exist_ok=True)
        self.best_dir.mkdir(parents=True, exist_ok=True)
        self.callback = _Callback(self)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Workspace Expansion Curriculum PPO training.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-id", default="workspace_expand_stage6to9_ppo_big_001")
    parser.add_argument("--artifact-root")
    parser.add_argument("--total-timesteps", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--resume-from")
    parser.add_argument("--no-gate-callback", action="store_true")
    return parser


def main() -> None:  # pragma: no cover - CLI integration
    _require_training_dependencies()
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CallbackList
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.utils import FloatSchedule
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

    args = build_arg_parser().parse_args()
    config_path = Path(args.config)
    cfg = _load_config(args.config)
    env_cfg = to_env_config(cfg)
    algo_kwargs = to_algorithm_kwargs(cfg, "ppo")
    runtime_cfg = cfg.get("training", {})
    workspace_cfg = cfg.get("workspace_expansion", {})
    gate_cfg = workspace_cfg.get("gate", {})
    if args.total_timesteps is not None:
        algo_kwargs["total_timesteps"] = args.total_timesteps
    if args.seed is not None:
        algo_kwargs["seed"] = args.seed

    artifact_root = Path(args.artifact_root) if args.artifact_root else repo_root() / "artifacts/kinematic_phase1/workspace_expansion" / args.run_id
    artifact_root.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = artifact_root / "checkpoints"
    latest_dir = artifact_root / "latest_checkpoint"
    latest_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(config_path, artifact_root / "config_resolved.yaml")
    write_json(artifact_root / "training_launch_summary.json", {"run_id": args.run_id, "config": cfg})

    from .envs.arm_kinematic_env import ArmKinematicEnv

    def _make_env() -> ArmKinematicEnv:
        return ArmKinematicEnv(config=env_cfg)

    n_envs = int(runtime_cfg.get("n_envs", 1))
    device = str(runtime_cfg.get("device", "auto"))
    vec_env_kind = str(runtime_cfg.get("vec_env", "dummy" if n_envs == 1 else "subproc")).lower()
    vec_env_cls = DummyVecEnv if vec_env_kind == "dummy" else SubprocVecEnv
    vec_env_kwargs = {} if vec_env_cls is DummyVecEnv else {"start_method": str(runtime_cfg.get("start_method", "fork"))}
    vec_env = make_vec_env(_make_env, n_envs=n_envs, seed=algo_kwargs.get("seed"), vec_env_cls=vec_env_cls, vec_env_kwargs=vec_env_kwargs)

    total_timesteps = int(algo_kwargs.get("total_timesteps", 100_000))
    model_kwargs = {k: v for k, v in algo_kwargs.items() if k != "total_timesteps"}
    resume_from = Path(args.resume_from or workspace_cfg.get("init_approach_checkpoint", ""))
    if resume_from and str(resume_from) and resume_from.exists():
        model = PPO.load(str(resume_from), env=vec_env, device=device)
        model.tensorboard_log = str(artifact_root / "tb")
        if "learning_rate" in model_kwargs:
            learning_rate = float(model_kwargs["learning_rate"])
            model.learning_rate = learning_rate
            model.lr_schedule = FloatSchedule(learning_rate)
            for param_group in model.policy.optimizer.param_groups:
                param_group["lr"] = learning_rate
        print(f"Resuming workspace expansion from {resume_from}")
    else:
        model = PPO("MultiInputPolicy", vec_env, verbose=1, tensorboard_log=str(artifact_root / "tb"), device=device, **model_kwargs)

    callbacks = [PeriodicCheckpointCallback(checkpoints_dir, save_freq=int(runtime_cfg.get("checkpoint_freq", 100_000)))]
    curriculum_cb = None
    if env_cfg.curriculum_config.enabled and env_cfg.curriculum_config.stages:
        curriculum_cb = PointCurriculumCallback(
            success_rate_threshold=env_cfg.curriculum_config.success_rate_threshold,
            window_episodes=env_cfg.curriculum_config.window_episodes,
            min_episodes_per_stage=env_cfg.curriculum_config.min_episodes_per_stage,
            max_stage_index=len(env_cfg.curriculum_config.stages) - 1,
            initial_stage_index=int(workspace_cfg.get("start_stage_index", 0)),
        )
        callbacks.append(curriculum_cb)

    if not args.no_gate_callback and workspace_cfg.get("finisher_checkpoint"):
        stages = list(range(len(env_cfg.curriculum_config.stages)))
        gate_callback = WorkspaceEvalGateCallback(
            artifact_root=artifact_root,
            approach_config_path=config_path,
            finisher_checkpoint=Path(workspace_cfg["finisher_checkpoint"]),
            finisher_config_path=Path(workspace_cfg["finisher_config"]),
            eval_interval=int(workspace_cfg.get("eval_interval", 200_000)),
            episodes=int(workspace_cfg.get("gate_eval_episodes", 24)),
            seed=int(workspace_cfg.get("eval_seed", 700001)),
            stage_indices=stages,
            current_stage_index=len(stages) - 1,
            gate_config=gate_cfg,
        )
        callbacks.append(gate_callback.callback)

    # Treat workspace expansion as a fine-tuning run with a fresh local clock.
    # SB3 checkpoints store prior timesteps; if we keep that clock, asking for
    # 5M steps from a 5.5M-step checkpoint exits almost immediately.
    model.learn(total_timesteps=total_timesteps, callback=CallbackList(callbacks), reset_num_timesteps=True)
    latest_model = latest_dir / "model_latest"
    model.save(str(latest_model))
    model.save(str(artifact_root / "model_latest"))
    curriculum_summary = curriculum_cb.summary() if curriculum_cb is not None else None

    final_eval = None
    if workspace_cfg.get("finisher_checkpoint"):
        final_eval = evaluate_workspace_expansion_checkpoint(
            approach_checkpoint=latest_model,
            approach_config_path=config_path,
            artifact_root=artifact_root / "final_eval",
            finisher_checkpoint=Path(workspace_cfg["finisher_checkpoint"]),
            finisher_config_path=Path(workspace_cfg["finisher_config"]),
            episodes=int(workspace_cfg.get("final_eval_episodes", 80)),
            seed=int(workspace_cfg.get("eval_seed", 700001)),
            stage_indices=list(range(len(env_cfg.curriculum_config.stages))),
            gate_config=gate_cfg,
        )
        for name in ("stage_metrics.json", "workspace_failure_report.json", "best_model_selection_summary.json"):
            src = artifact_root / "final_eval" / name
            if src.exists():
                shutil.copyfile(src, artifact_root / name)

    write_json(
        artifact_root / "training_summary.json",
        {
            "policy_type": "approach",
            "algorithm": "ppo",
            "run_id": args.run_id,
            "model_path": str(latest_model) + ".zip",
            "resume_from": str(resume_from) if resume_from else None,
            "n_envs": n_envs,
            "device": device,
            "curriculum_summary": curriculum_summary,
            "final_workspace_eval": final_eval,
        },
    )
    print(json.dumps({"run_id": args.run_id, "artifact_root": str(artifact_root), "model_latest": str(latest_model) + ".zip"}, indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()
