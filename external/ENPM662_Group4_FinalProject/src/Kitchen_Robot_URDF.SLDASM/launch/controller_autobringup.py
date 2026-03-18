#!/usr/bin/env python3

import argparse
import sys
import time
from typing import Dict

import rclpy
from rclpy.node import Node

from controller_manager_msgs.srv import (
    ConfigureController,
    ListControllers,
    LoadController,
    SwitchController,
)


class ControllerAutoBringup(Node):
    def __init__(self, controller_manager: str) -> None:
        super().__init__('controller_autobringup')
        cm = controller_manager.rstrip('/')
        self._list_client = self.create_client(ListControllers, f'{cm}/list_controllers')
        self._load_client = self.create_client(LoadController, f'{cm}/load_controller')
        self._configure_client = self.create_client(ConfigureController, f'{cm}/configure_controller')
        self._switch_client = self.create_client(SwitchController, f'{cm}/switch_controller')

    def wait_ready(self, timeout_sec: float) -> bool:
        deadline = time.monotonic() + timeout_sec
        while time.monotonic() < deadline and rclpy.ok():
            if (
                self._list_client.wait_for_service(timeout_sec=0.5)
                and self._load_client.wait_for_service(timeout_sec=0.5)
                and self._configure_client.wait_for_service(timeout_sec=0.5)
                and self._switch_client.wait_for_service(timeout_sec=0.5)
            ):
                self.get_logger().info('READY: /controller_manager services available')
                return True
        return False

    def _call(self, client, request):
        future = client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
        if not future.done() or future.result() is None:
            return None
        return future.result()

    def list_states(self) -> Dict[str, str]:
        res = self._call(self._list_client, ListControllers.Request())
        if res is None:
            return {}
        return {c.name: c.state.lower() for c in res.controller}

    def load(self, name: str) -> bool:
        req = LoadController.Request()
        req.name = name
        res = self._call(self._load_client, req)
        if res is None:
            self.get_logger().warn(f'LOAD: timeout calling load_controller for {name}')
            return False
        ok = bool(res.ok)
        if ok:
            self.get_logger().info(f'LOAD: loaded controller {name}')
        else:
            self.get_logger().warn(f'LOAD: load_controller returned not ok for {name}')
        return ok

    def configure(self, name: str) -> bool:
        req = ConfigureController.Request()
        req.name = name
        res = self._call(self._configure_client, req)
        if res is None:
            self.get_logger().warn(f'CONFIGURE: timeout calling configure_controller for {name}')
            return False
        ok = bool(res.ok)
        if ok:
            self.get_logger().info(f'CONFIGURE: configured controller {name}')
        else:
            self.get_logger().warn(f'CONFIGURE: configure_controller returned not ok for {name}')
        return ok

    def activate(self, name: str) -> bool:
        req = SwitchController.Request()
        req.activate_controllers = [name]
        req.deactivate_controllers = []
        req.strictness = getattr(SwitchController.Request, 'BEST_EFFORT', 1)
        req.activate_asap = True
        req.timeout.sec = 5
        req.timeout.nanosec = 0
        res = self._call(self._switch_client, req)
        if res is None:
            self.get_logger().warn(f'ACTIVATE: timeout calling switch_controller for {name}')
            return False
        ok = bool(res.ok)
        if ok:
            self.get_logger().info(f'ACTIVATE: activated controller {name}')
        else:
            self.get_logger().warn(f'ACTIVATE: switch_controller returned not ok for {name}; will retry')
        return ok


def parse_args():
    parser = argparse.ArgumentParser(description='Robust ros2_control controller auto bring-up helper')
    parser.add_argument('--controller-manager', default='/controller_manager')
    parser.add_argument('--timeout', type=float, default=30.0)
    parser.add_argument('--period', type=float, default=1.0)
    parser.add_argument(
        '--controllers',
        nargs='+',
        default=['joint_state_broadcaster', 'arm_controller'],
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rclpy.init(args=None)
    node = ControllerAutoBringup(args.controller_manager)

    try:
        if not node.wait_ready(timeout_sec=args.timeout):
            node.get_logger().warn('DEGRADED: /controller_manager services not ready before timeout; continuing without hard failure')
            return 0

        deadline = time.monotonic() + args.timeout
        while time.monotonic() < deadline and rclpy.ok():
            states = node.list_states()
            if not states:
                node.get_logger().warn('EMPTY_LIST_DETECTED: list_controllers returned empty; forcing load/configure/activate sequence')
                for controller in args.controllers:
                    node.get_logger().info(f'LOAD: empty list recovery load request for {controller}')
                    node.load(controller)
                    node.get_logger().info(f'CONFIGURE: empty list recovery configure request for {controller}')
                    node.configure(controller)
                    node.get_logger().info(f'ACTIVATE: empty list recovery activate request for {controller}')
                    node.activate(controller)
                time.sleep(args.period)
                continue

            for controller in args.controllers:
                state = states.get(controller)

                if state is None:
                    node.load(controller)
                    continue

                if state == 'active':
                    node.get_logger().info(f'ACTIVE: {controller} already active')
                    continue

                if state == 'unconfigured':
                    node.configure(controller)
                    continue

                if state in ('inactive', 'configured'):
                    # ACTIVATE may fail transiently; keep retrying until global timeout.
                    node.activate(controller)
                    continue

                if state == 'finalized':
                    # allow recovery path for finalized -> configure in next loop.
                    node.get_logger().warn(f'CONFIGURE: {controller} is finalized; attempting configure')
                    node.configure(controller)
                    continue

                node.get_logger().warn(f'DEGRADED: {controller} in unexpected state={state}; retrying')

            states = node.list_states()
            if states and all(states.get(c, '') == 'active' for c in args.controllers):
                node.get_logger().info(
                    'ACTIVE: all target controllers are active: '
                    + ', '.join(f'{c}={states.get(c)}' for c in args.controllers)
                )
                return 0

            time.sleep(args.period)

        final_states = node.list_states()
        node.get_logger().warn(
            'DEGRADED: timeout waiting for active controllers: '
            + ', '.join(f'{c}={final_states.get(c, "missing")}' for c in args.controllers)
        )
        return 0
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    sys.exit(main())
