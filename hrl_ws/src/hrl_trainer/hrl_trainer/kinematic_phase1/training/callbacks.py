"""Training callbacks for Phase 1 SB3 runs."""

from __future__ import annotations

from collections import deque
from pathlib import Path

try:  # pragma: no cover - runtime dependency
    from stable_baselines3.common.callbacks import BaseCallback

    SB3_CALLBACKS_AVAILABLE = True
except Exception:  # pragma: no cover - fallback
    BaseCallback = object
    SB3_CALLBACKS_AVAILABLE = False


class PeriodicCheckpointCallback(BaseCallback):  # pragma: no cover - exercised with SB3 installed
    def __init__(self, save_dir: Path, save_freq: int) -> None:
        if not SB3_CALLBACKS_AVAILABLE:
            raise RuntimeError("Stable-Baselines3 callbacks are unavailable")
        super().__init__()
        self.save_dir = save_dir
        self.save_freq = max(int(save_freq), 1)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            self.model.save(str(self.save_dir / f"checkpoint_step_{self.num_timesteps}"))
        return True


class PointCurriculumCallback(BaseCallback):  # pragma: no cover - exercised with SB3 installed
    def __init__(
        self,
        *,
        success_rate_threshold: float,
        window_episodes: int,
        min_episodes_per_stage: int,
        max_stage_index: int,
    ) -> None:
        if not SB3_CALLBACKS_AVAILABLE:
            raise RuntimeError("Stable-Baselines3 callbacks are unavailable")
        super().__init__()
        self.success_rate_threshold = float(success_rate_threshold)
        self.window_episodes = max(int(window_episodes), 1)
        self.min_episodes_per_stage = max(int(min_episodes_per_stage), 1)
        self.max_stage_index = max(int(max_stage_index), 0)
        self.current_stage_index = 0
        self.stage_episode_count = 0
        self.recent_successes: deque[int] = deque(maxlen=self.window_episodes)
        self.history: list[dict[str, float | int | str]] = []

    def _promote(self, next_stage: int, trigger_success_rate: float) -> None:
        self.training_env.env_method("set_curriculum_stage", next_stage)
        self.history.append(
            {
                "from_stage_index": self.current_stage_index,
                "to_stage_index": next_stage,
                "trigger_success_rate": float(trigger_success_rate),
                "total_timesteps": int(self.num_timesteps),
            }
        )
        self.current_stage_index = next_stage
        self.stage_episode_count = 0
        self.recent_successes.clear()

    def _on_training_start(self) -> None:
        self.training_env.env_method("set_curriculum_stage", self.current_stage_index)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos")
        dones = self.locals.get("dones")
        if infos is None:
            infos = []
        if dones is None:
            dones = []
        for done, info in zip(dones, infos):
            if not done:
                continue
            self.stage_episode_count += 1
            self.recent_successes.append(1 if bool(info.get("success", False)) else 0)
            if self.current_stage_index >= self.max_stage_index:
                continue
            if self.stage_episode_count < self.min_episodes_per_stage:
                continue
            if len(self.recent_successes) < self.window_episodes:
                continue
            success_rate = float(sum(self.recent_successes)) / float(len(self.recent_successes))
            if success_rate >= self.success_rate_threshold:
                self._promote(self.current_stage_index + 1, success_rate)
        return True

    def summary(self) -> dict[str, object]:
        success_rate = float(sum(self.recent_successes)) / float(len(self.recent_successes)) if self.recent_successes else 0.0
        return {
            "stage_index": self.current_stage_index,
            "stage_episode_count": self.stage_episode_count,
            "recent_success_rate": success_rate,
            "history": list(self.history),
        }


class DockReverseCurriculumCallback(BaseCallback):  # pragma: no cover - exercised with SB3 installed
    def __init__(
        self,
        *,
        stages: list[dict[str, object]],
        window_episodes: int,
    ) -> None:
        if not SB3_CALLBACKS_AVAILABLE:
            raise RuntimeError("Stable-Baselines3 callbacks are unavailable")
        super().__init__()
        if not stages:
            raise ValueError("DockReverseCurriculumCallback requires at least one stage")
        self.stages = list(stages)
        self.window_episodes = max(int(window_episodes), 1)
        self.current_stage_index = 0
        self.stage_episode_count = 0
        self.recent_successes: deque[int] = deque(maxlen=self.window_episodes)
        self.history: list[dict[str, object]] = []

    def _stage_payload(self, stage: dict[str, object]) -> dict[str, object]:
        payload: dict[str, object] = {"dock_reset": {}}
        for key in (
            "action_delta_scale",
            "dock_residual_action_limit",
            "dock_delta_q_change_limit_scale",
        ):
            if key in stage:
                payload[key] = stage[key]
        for key in (
            "close_bucket_probability",
            "close_bucket_min_pos_error_m",
            "close_bucket_max_pos_error_m",
            "close_bucket_max_ori_error_rad",
            "close_init_q_noise",
            "init_q_noise",
        ):
            if key in stage:
                payload["dock_reset"][key] = stage[key]
        return payload

    def _apply_stage(self, stage_index: int) -> None:
        stage = self.stages[stage_index]
        self.training_env.env_method("apply_dock_training_stage", self._stage_payload(stage))

    def _promote(self, next_stage_index: int, trigger_success_rate: float) -> None:
        prev_stage = self.stages[self.current_stage_index]
        next_stage = self.stages[next_stage_index]
        self._apply_stage(next_stage_index)
        self.history.append(
            {
                "from_stage_index": self.current_stage_index,
                "from_stage_name": prev_stage.get("name", f"stage_{self.current_stage_index}"),
                "to_stage_index": next_stage_index,
                "to_stage_name": next_stage.get("name", f"stage_{next_stage_index}"),
                "trigger_success_rate": float(trigger_success_rate),
                "stage_episode_count": int(self.stage_episode_count),
                "total_timesteps": int(self.num_timesteps),
            }
        )
        self.current_stage_index = next_stage_index
        self.stage_episode_count = 0
        self.recent_successes.clear()

    def _on_training_start(self) -> None:
        self._apply_stage(self.current_stage_index)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos")
        dones = self.locals.get("dones")
        if infos is None:
            infos = []
        if dones is None:
            dones = []
        for done, info in zip(dones, infos):
            if not done:
                continue
            self.stage_episode_count += 1
            self.recent_successes.append(1 if bool(info.get("success", False)) else 0)
            if self.current_stage_index >= len(self.stages) - 1:
                continue

            stage = self.stages[self.current_stage_index]
            min_episodes = max(int(stage.get("min_episodes", self.window_episodes)), 1)
            success_rate_threshold = float(stage.get("success_rate_threshold", 1.0))
            stage_window = max(int(stage.get("window_episodes", self.window_episodes)), 1)
            if self.stage_episode_count < min_episodes:
                continue
            if len(self.recent_successes) < min(stage_window, self.window_episodes):
                continue
            recent = list(self.recent_successes)[-min(stage_window, len(self.recent_successes)) :]
            success_rate = float(sum(recent)) / float(len(recent))
            if success_rate >= success_rate_threshold:
                self._promote(self.current_stage_index + 1, success_rate)
        return True

    def summary(self) -> dict[str, object]:
        success_rate = float(sum(self.recent_successes)) / float(len(self.recent_successes)) if self.recent_successes else 0.0
        return {
            "stage_index": self.current_stage_index,
            "stage_name": self.stages[self.current_stage_index].get("name", f"stage_{self.current_stage_index}"),
            "stage_episode_count": self.stage_episode_count,
            "recent_success_rate": success_rate,
            "history": list(self.history),
        }
