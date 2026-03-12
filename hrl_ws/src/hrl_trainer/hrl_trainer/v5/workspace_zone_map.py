"""V5 WP1.5 workspace zone map and canonical hover anchor loader."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


DEFAULT_WORKSPACE_ZONE_MAP_PATH = Path(__file__).resolve().parents[2] / "config" / "v5_workspace_zone_map.yaml"


class WorkspaceZoneMapError(ValueError):
    """Raised when workspace zone map payload is invalid."""


@dataclass(frozen=True)
class ZoneRegion:
    center_xyz: tuple[float, float, float]
    size_xyz: tuple[float, float, float]
    yaw: float


@dataclass(frozen=True)
class WorkspaceZone:
    zone_id: str
    region_world: ZoneRegion
    hover_anchor_ids: tuple[str, ...]


@dataclass(frozen=True)
class HoverAnchor:
    anchor_id: str
    zone_id: str
    xyz: tuple[float, float, float]
    rpy: tuple[float, float, float]


class WorkspaceZoneMap:
    def __init__(self, zones: Sequence[WorkspaceZone], anchors: Sequence[HoverAnchor]):
        if not zones:
            raise WorkspaceZoneMapError("WorkspaceZoneMap requires at least one zone")
        if not anchors:
            raise WorkspaceZoneMapError("WorkspaceZoneMap requires at least one anchor")

        self._zones = tuple(zones)
        self._anchors = tuple(anchors)
        self._by_zone_id = {zone.zone_id: zone for zone in self._zones}
        self._by_anchor_id = {anchor.anchor_id: anchor for anchor in self._anchors}

        if len(self._by_zone_id) != len(self._zones):
            raise WorkspaceZoneMapError("zone_id values must be unique")
        if len(self._by_anchor_id) != len(self._anchors):
            raise WorkspaceZoneMapError("anchor_id values must be unique")

        for anchor in self._anchors:
            if anchor.zone_id not in self._by_zone_id:
                raise WorkspaceZoneMapError(f"anchor {anchor.anchor_id!r} references unknown zone {anchor.zone_id!r}")

        for zone in self._zones:
            if not zone.hover_anchor_ids:
                raise WorkspaceZoneMapError(f"zone {zone.zone_id!r} must include at least one hover anchor")
            for anchor_id in zone.hover_anchor_ids:
                anchor = self._by_anchor_id.get(anchor_id)
                if anchor is None:
                    raise WorkspaceZoneMapError(f"zone {zone.zone_id!r} references unknown anchor {anchor_id!r}")
                if anchor.zone_id != zone.zone_id:
                    raise WorkspaceZoneMapError(
                        f"anchor {anchor.anchor_id!r} zone mismatch: {anchor.zone_id!r} != {zone.zone_id!r}"
                    )

    @property
    def zones(self) -> tuple[WorkspaceZone, ...]:
        return self._zones

    @property
    def anchors(self) -> tuple[HoverAnchor, ...]:
        return self._anchors

    def get_zone(self, zone_id: str) -> WorkspaceZone:
        return self._by_zone_id[zone_id]

    def get_anchor(self, anchor_id: str) -> HoverAnchor:
        return self._by_anchor_id[anchor_id]

    def zone_for_point(self, xyz: Sequence[float]) -> str | None:
        x, y, z = float(xyz[0]), float(xyz[1]), float(xyz[2])
        for zone in self._zones:
            cx, cy, cz = zone.region_world.center_xyz
            sx, sy, sz = zone.region_world.size_xyz
            if abs(x - cx) <= (sx / 2.0) and abs(y - cy) <= (sy / 2.0) and abs(z - cz) <= (sz / 2.0):
                return zone.zone_id
        return None

    @classmethod
    def from_yaml(cls, path: str | Path) -> "WorkspaceZoneMap":
        with Path(path).open("r", encoding="utf-8") as fp:
            return cls.from_dict(yaml.safe_load(fp) or {})

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "WorkspaceZoneMap":
        zones_payload = payload.get("zones")
        anchors_payload = payload.get("anchors")
        if not isinstance(zones_payload, list):
            raise WorkspaceZoneMapError("zones must be a list")
        if not isinstance(anchors_payload, list):
            raise WorkspaceZoneMapError("anchors must be a list")

        zones: list[WorkspaceZone] = []
        for zone_entry in zones_payload:
            if not isinstance(zone_entry, Mapping):
                raise WorkspaceZoneMapError("zone entries must be mappings")
            region = zone_entry.get("region_world")
            if not isinstance(region, Mapping):
                raise WorkspaceZoneMapError("zone.region_world must be mapping")
            zone = WorkspaceZone(
                zone_id=str(zone_entry["zone_id"]),
                region_world=ZoneRegion(
                    center_xyz=_as_float3(region.get("center_xyz", []), field_name="region_world.center_xyz"),
                    size_xyz=_as_float3(region.get("size_xyz", []), field_name="region_world.size_xyz"),
                    yaw=float(region.get("yaw", 0.0)),
                ),
                hover_anchor_ids=tuple(str(anchor_id) for anchor_id in zone_entry.get("hover_anchor_ids", [])),
            )
            zones.append(zone)

        anchors: list[HoverAnchor] = []
        for anchor_entry in anchors_payload:
            if not isinstance(anchor_entry, Mapping):
                raise WorkspaceZoneMapError("anchor entries must be mappings")
            pose = anchor_entry.get("pose")
            if not isinstance(pose, Mapping):
                raise WorkspaceZoneMapError("anchor.pose must be mapping")
            anchor = HoverAnchor(
                anchor_id=str(anchor_entry["anchor_id"]),
                zone_id=str(anchor_entry["zone_id"]),
                xyz=_as_float3(pose.get("xyz", []), field_name="anchor.pose.xyz"),
                rpy=_as_float3(pose.get("rpy", []), field_name="anchor.pose.rpy"),
            )
            anchors.append(anchor)

        return cls(zones, anchors)


def _as_float3(values: Sequence[Any], *, field_name: str) -> tuple[float, float, float]:
    if len(values) != 3:
        raise WorkspaceZoneMapError(f"{field_name} must contain exactly 3 values")
    return (float(values[0]), float(values[1]), float(values[2]))


def load_runtime_workspace_zone_map(path: str | Path | None = None) -> WorkspaceZoneMap:
    zone_map_path = Path(path) if path is not None else DEFAULT_WORKSPACE_ZONE_MAP_PATH
    if not zone_map_path.exists():
        raise FileNotFoundError(f"Workspace zone map config not found: {zone_map_path}")
    return WorkspaceZoneMap.from_yaml(zone_map_path)
