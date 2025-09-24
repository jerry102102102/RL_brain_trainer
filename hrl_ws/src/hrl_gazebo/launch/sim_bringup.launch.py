"""Launch Gazebo with the HRL manipulator and ros2_control controllers."""

from __future__ import annotations

import os
from typing import List

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node


def _declare_arguments(pkg_share: str) -> List[DeclareLaunchArgument]:
    default_world = os.path.join(pkg_share, "worlds", "empty.world")
    default_model = os.path.join(pkg_share, "urdf", "manipulator.urdf.xacro")
    default_ros2_control = os.path.join(pkg_share, "urdf", "ros2_control.yaml")
    return [
        DeclareLaunchArgument("use_gui", default_value="true", description="Launch Gazebo with the classic GUI."),
        DeclareLaunchArgument("world", default_value=default_world, description="Gazebo world file."),
        DeclareLaunchArgument("model", default_value=default_model, description="Manipulator xacro description."),
        DeclareLaunchArgument(
            "ros2_control_config",
            default_value=default_ros2_control,
            description="ros2_control YAML configuration file.",
        ),
        DeclareLaunchArgument(
            "command_controller",
            default_value="forward_position_controller",
            description="Name of the command controller to spawn.",
        ),
    ]


def _launch_setup(context, *args, **kwargs):  # type: ignore[override]
    pkg_share = get_package_share_directory("hrl_gazebo")
    robot_description = Command(["xacro ", LaunchConfiguration("model")])
    world = LaunchConfiguration("world")
    ros2_control_config = LaunchConfiguration("ros2_control_config")
    command_controller = LaunchConfiguration("command_controller")

    gazebo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory("gazebo_ros"), "launch", "gazebo.launch.py")
        ),
        launch_arguments={
            "world": world,
            "gui": LaunchConfiguration("use_gui"),
        }.items(),
    )

    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        parameters=[{"robot_description": robot_description}],
        output="screen",
    )

    ros2_control_node = Node(
        package="controller_manager",
        executable="ros2_control_node",
        parameters=[{"robot_description": robot_description}, ros2_control_config],
        output="screen",
    )

    spawn_entity = Node(
        package="gazebo_ros",
        executable="spawn_entity.py",
        arguments=["-topic", "robot_description", "-entity", "hrl_manipulator"],
        output="screen",
    )

    joint_state_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["joint_state_broadcaster"],
        output="screen",
    )

    forward_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[command_controller, "-c", "/controller_manager"],
        output="screen",
    )

    return [
        gazebo_launch,
        robot_state_publisher,
        ros2_control_node,
        spawn_entity,
        joint_state_spawner,
        forward_controller_spawner,
    ]


def generate_launch_description() -> LaunchDescription:
    pkg_share = get_package_share_directory("hrl_gazebo")
    declared_arguments = _declare_arguments(pkg_share)
    return LaunchDescription(declared_arguments + [OpaqueFunction(function=_launch_setup)])
