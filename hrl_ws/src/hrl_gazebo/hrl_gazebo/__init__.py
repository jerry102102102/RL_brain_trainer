"""Utility package for HRL Gazebo resources."""

from importlib import resources as _resources
from pathlib import Path as _Path

__all__ = ["get_share_path"]


def get_share_path() -> _Path:
    """Return the path to the package share directory."""

    return _Path(_resources.files(__name__))
