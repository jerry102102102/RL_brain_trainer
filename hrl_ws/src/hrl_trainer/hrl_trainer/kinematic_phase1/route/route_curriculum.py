"""Route prefix curriculum callback."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

try:  # pragma: no cover - runtime dependency
    from stable_baselines3.common.callbacks import BaseCallback

    SB3_CALLBACKS_AVAILABLE = True
except Exception:  # pragma: no cover
    BaseCallback = object
    SB3_CALLBACKS_AVAILABLE = False


@dataclass(frozen=True)
class RouteCurriculumStage:
    name: str
    prefix_end_index: int


class RoutePrefixCurriculumCallback(BaseCallback):  # pragma: no cover - exercised in SB3 runs
    def __init__(
        self,
        *,
        stages: list[RouteCurriculumStage],
        promotion_success_rate: float,
        promotion_route_ready_hit_rate: float,
        promotion_orientation_hit_rate: float,
        promotion_max_regression_rate: float,
        window_episodes: int,
        min_episodes_per_stage: int = 128,
    ) -> None:
        if not SB3_CALLBACKS_AVAILABLE:
            raise RuntimeError("Stable-Baselines3 callbacks are unavailable")
        super().__init__()
        if not stages:
            raise ValueError("RoutePrefixCurriculumCallback requires at least one stage")
        self.stages = list(stages)
        self.promotion_success_rate = float(promotion_success_rate)
        self.promotion_route_ready_hit_rate = float(promotion_route_ready_hit_rate)
        self.promotion_orientation_hit_rate = float(promotion_orientation_hit_rate)
        self.promotion_max_regression_rate = float(promotion_max_regression_rate)
        self.window_episodes = max(int(window_episodes), 1)
        self.min_episodes_per_stage = max(int(min_episodes_per_stage), 1)
        self.current_stage_index = 0
        self.stage_episode_count = 0
        self.successes: deque[int] = deque(maxlen=self.window_episodes)
        self.ready_hits: deque[int] = deque(maxlen=self.window_episodes)
        self.orientation_hits: deque[int] = deque(maxlen=self.window_episodes)
        self.regressions: deque[int] = deque(maxlen=self.window_episodes)
        self.history: list[dict[str, object]] = []

    def _apply_stage(self) -> None:
        stage = self.stages[self.current_stage_index]
        self.training_env.env_method("set_route_window", max_route_index=int(stage.prefix_end_index), min_route_index=1)

    def _on_training_start(self) -> None:
        self._apply_stage()

    def _promote(self, metrics: dict[str, float]) -> None:
        if self.current_stage_index >= len(self.stages) - 1:
            return
        prev = self.stages[self.current_stage_index]
        self.current_stage_index += 1
        nxt = self.stages[self.current_stage_index]
        self.history.append(
            {
                "from_stage": prev.name,
                "to_stage": nxt.name,
                "from_prefix_end_index": int(prev.prefix_end_index),
                "to_prefix_end_index": int(nxt.prefix_end_index),
                "total_timesteps": int(self.num_timesteps),
                **metrics,
            }
        )
        self.stage_episode_count = 0
        self.successes.clear()
        self.ready_hits.clear()
        self.orientation_hits.clear()
        self.regressions.clear()
        self._apply_stage()

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
            self.successes.append(1 if bool(info.get("success", False)) else 0)
            self.ready_hits.append(1 if bool(info.get("route_ready", False)) else 0)
            self.orientation_hits.append(1 if bool(info.get("route_orientation_hit", False)) else 0)
            self.regressions.append(1 if bool(info.get("route_regression", False)) else 0)
            if self.stage_episode_count < self.min_episodes_per_stage or len(self.successes) < self.window_episodes:
                continue
            metrics = self._metrics()
            if (
                metrics["recent_success_rate"] >= self.promotion_success_rate
                and metrics["recent_route_ready_hit_rate"] >= self.promotion_route_ready_hit_rate
                and metrics["recent_orientation_hit_rate"] >= self.promotion_orientation_hit_rate
                and metrics["recent_regression_rate"] <= self.promotion_max_regression_rate
            ):
                self._promote(metrics)
        return True

    def _metrics(self) -> dict[str, float]:
        def mean(xs: deque[int]) -> float:
            return float(sum(xs)) / float(len(xs)) if xs else 0.0

        return {
            "recent_success_rate": mean(self.successes),
            "recent_route_ready_hit_rate": mean(self.ready_hits),
            "recent_orientation_hit_rate": mean(self.orientation_hits),
            "recent_regression_rate": mean(self.regressions),
        }

    def summary(self) -> dict[str, object]:
        stage = self.stages[self.current_stage_index]
        return {
            "stage_index": int(self.current_stage_index),
            "stage_name": stage.name,
            "prefix_end_index": int(stage.prefix_end_index),
            "stage_episode_count": int(self.stage_episode_count),
            **self._metrics(),
            "history": list(self.history),
        }


def build_prefix_stages(prefixes: list[int]) -> list[RouteCurriculumStage]:
    return [RouteCurriculumStage(name=f"prefix_{int(p)}", prefix_end_index=int(p)) for p in prefixes]
