"""Inference-only ROS 2 node for executing trained HRL policies."""

from __future__ import annotations

from typing import List, Optional

import rclpy
from rclpy.parameter import Parameter

from .hrl_trainer import HRLTrainerNode


class HRLPolicyNode(HRLTrainerNode):
    """Specialised node that disables training and prefers final checkpoints."""

    def _declare_default_parameters(self) -> None:
        super()._declare_default_parameters()
        self.set_parameters(
            [
                Parameter("training_mode", Parameter.Type.STRING, "eval"),
                Parameter("use_tensorboard", Parameter.Type.BOOL, False),
            ]
        )

    def _maybe_load_checkpoint(self) -> None:  # type: ignore[override]
        self._checkpoint_candidates = ["final.pt", "latest.pt"]
        super()._maybe_load_checkpoint()


def main(args: Optional[List[str]] = None) -> None:
    rclpy.init(args=args)
    node = HRLPolicyNode()
    try:
        rclpy.spin(node)
    finally:
        if node.writer is not None:
            node.writer.close()
        node.destroy_node()
        rclpy.shutdown()


__all__ = ["HRLPolicyNode", "main"]
