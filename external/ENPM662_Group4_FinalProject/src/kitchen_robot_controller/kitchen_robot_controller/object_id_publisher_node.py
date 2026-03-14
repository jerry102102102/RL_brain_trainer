from __future__ import annotations

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import String


class ObjectIdPublisherNode(Node):
    def __init__(self) -> None:
        super().__init__("object_id_publisher_node")

        self.declare_parameter("input_topic", "/tray1/pose")
        self.declare_parameter("output_topic", "/v5/perception/object_pose_est")
        self.declare_parameter("id_value", "tray1")
        self.declare_parameter("publish_rate_hz", 10.0)
        self.declare_parameter("enable_stats_log", False)
        self.declare_parameter("stats_log_every_n", 300)
        if not self.has_parameter("use_sim_time"):
            self.declare_parameter("use_sim_time", True)

        input_topic = str(self.get_parameter("input_topic").value)
        output_topic = str(self.get_parameter("output_topic").value)
        self.id_value = str(self.get_parameter("id_value").value) or "tray1"
        publish_rate_hz = float(self.get_parameter("publish_rate_hz").value)
        self.publish_period = 1.0 / max(0.5, publish_rate_hz)
        self.enable_stats_log = bool(self.get_parameter("enable_stats_log").value)
        self.stats_log_every_n = int(self.get_parameter("stats_log_every_n").value)

        self.pub = self.create_publisher(String, output_topic, 20)
        self.sub = self.create_subscription(PoseStamped, input_topic, self._on_pose, 20)
        self.timer = self.create_timer(self.publish_period, self._on_timer)

        self.have_seen_pose = False
        self.published = 0

    def _publish(self) -> None:
        msg = String()
        msg.data = self.id_value
        self.pub.publish(msg)
        self.published += 1
        if self.enable_stats_log and self.published % max(1, self.stats_log_every_n) == 0:
            self.get_logger().info(f"object id publisher stats: published={self.published} id={self.id_value}")

    def _on_pose(self, _msg: PoseStamped) -> None:
        self.have_seen_pose = True
        self._publish()

    def _on_timer(self) -> None:
        if not self.have_seen_pose:
            return
        self._publish()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ObjectIdPublisherNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
