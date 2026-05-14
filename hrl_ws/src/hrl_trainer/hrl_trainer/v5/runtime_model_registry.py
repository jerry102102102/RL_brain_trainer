"""Single source of truth for Phase 3A runtime model assets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml


DEFAULT_PHASE3A_MODEL_REGISTRY = Path(__file__).resolve().parent / "configs" / "phase3a_runtime_models.yaml"


@dataclass(frozen=True)
class RuntimeModelAsset:
    role: str
    checkpoint: str
    config: str
    note: str = ""

    def resolve_checkpoint(self, repo_root: str | Path) -> Path:
        path = Path(self.checkpoint)
        return path if path.is_absolute() else Path(repo_root) / path

    def resolve_config(self, repo_root: str | Path) -> Path:
        path = Path(self.config)
        return path if path.is_absolute() else Path(repo_root) / path

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "checkpoint": self.checkpoint,
            "config": self.config,
            "note": self.note,
        }


@dataclass(frozen=True)
class Phase3ARuntimeRegistry:
    schema_version: str
    pipeline: str
    approach: RuntimeModelAsset
    finisher: RuntimeModelAsset
    handoff: dict[str, Any]
    ros_topics: dict[str, str]

    def validate_paths(self, repo_root: str | Path) -> dict[str, Any]:
        repo = Path(repo_root)
        checks = {}
        for name, asset in (("approach", self.approach), ("finisher", self.finisher)):
            checkpoint = asset.resolve_checkpoint(repo)
            config = asset.resolve_config(repo)
            checks[name] = {
                "checkpoint": str(checkpoint),
                "checkpoint_exists": checkpoint.exists(),
                "config": str(config),
                "config_exists": config.exists(),
            }
        return checks

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "pipeline": self.pipeline,
            "approach": self.approach.to_dict(),
            "finisher": self.finisher.to_dict(),
            "handoff": dict(self.handoff),
            "ros_topics": dict(self.ros_topics),
        }


def _asset(payload: Mapping[str, Any], *, default_role: str) -> RuntimeModelAsset:
    return RuntimeModelAsset(
        role=str(payload.get("role", default_role)),
        checkpoint=str(payload["checkpoint"]),
        config=str(payload["config"]),
        note=str(payload.get("note", "")),
    )


def load_phase3a_model_registry(path: str | Path | None = None) -> Phase3ARuntimeRegistry:
    registry_path = Path(path) if path is not None else DEFAULT_PHASE3A_MODEL_REGISTRY
    payload = yaml.safe_load(registry_path.read_text(encoding="utf-8")) or {}
    return Phase3ARuntimeRegistry(
        schema_version=str(payload.get("schema_version", "v5.phase3a.runtime_models.v1")),
        pipeline=str(payload.get("pipeline", "APPROACH_FINISHER")),
        approach=_asset(payload["approach"], default_role="APPROACH"),
        finisher=_asset(payload["finisher"], default_role="FINISHER"),
        handoff=dict(payload.get("handoff", {})),
        ros_topics={str(k): str(v) for k, v in dict(payload.get("ros_topics", {})).items()},
    )

