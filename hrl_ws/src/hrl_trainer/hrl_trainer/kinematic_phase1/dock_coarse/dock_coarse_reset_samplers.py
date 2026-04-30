"""Dock-Coarse reset helpers.

The first implementation intentionally reuses the existing Dock close-bucket
sampler through ``DockResetConfig``. That keeps the coarse branch compatible
with the strict finisher while giving it a wider reset band.
"""

from __future__ import annotations

from ..envs.reset_samplers import DockResetConfig, ResetSample, sample_dock_reset

__all__ = ["DockResetConfig", "ResetSample", "sample_dock_reset"]
