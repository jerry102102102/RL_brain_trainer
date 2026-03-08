#!/usr/bin/env python3
import os

from launch import LaunchDescription
from launch.actions import (
    IncludeLaunchDescription,
    SetEnvironmentVariable,
    DeclareLaunchArgument,
    RegisterEventHandler,
    ExecuteProcess,
    TimerAction,
)
from launch.conditions import IfCondition
from launch.conditions import UnlessCondition
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration

from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():

    # === package paths ===
    pkg_name = 'kitchen_robot_description'
    pkg_share = get_package_share_directory(pkg_name)
    res_root = os.path.dirname(pkg_share)

    urdf_file = os.path.join(pkg_share, 'urdf', 'Kitchen_Robot_UR7e.SLDASM.urdf')
    burner_urdf = os.path.join(pkg_share, 'urdf', 'burner.urdf')
    tray_urdf = os.path.join(pkg_share, 'urdf', 'tray.urdf')
    cam_overhead_sdf = os.path.join(pkg_share, 'models', 'cam_overhead.sdf')
    cam_side_sdf = os.path.join(pkg_share, 'models', 'cam_side.sdf')

    controllers = os.path.join(pkg_share, 'config', 'controller.yaml')
    tmp_params = '/tmp/kitchen_robot_controller.yaml'

    # === Launch args ===
    declare_headless = DeclareLaunchArgument(
        'headless',
        default_value='false',
        description='Run gz sim without GUI (server-only).'
    )
    declare_use_software_renderer = DeclareLaunchArgument(
        'use_software_renderer',
        default_value='false',
        description='Force software GL renderer. Keep false if GPU is available.'
    )
    declare_gz_args = DeclareLaunchArgument(
        'gz_args',
        default_value='-r -v 2 worlds/v5_kitchen_empty.sdf',
        description='Arguments passed to gz sim (GUI path).'
    )
    declare_gz_headless_args = DeclareLaunchArgument(
        'gz_headless_args',
        default_value='-r -s --headless-rendering -v 2 worlds/v5_kitchen_empty.sdf',
        description='Arguments passed to gz sim (headless path).'
    )

    # === Environment variables ===
    set_res_gz = SetEnvironmentVariable('GZ_SIM_RESOURCE_PATH', os.pathsep.join([pkg_share, res_root]))
    set_res_ign = SetEnvironmentVariable('IGN_GAZEBO_RESOURCE_PATH', os.pathsep.join([pkg_share, res_root]))
    soft_gl = SetEnvironmentVariable('LIBGL_ALWAYS_SOFTWARE', '1', condition=IfCondition(LaunchConfiguration('use_software_renderer')))
    no_audio = SetEnvironmentVariable('GZ_AUDIO', '0')

    # Copy controller.yaml into /tmp so the <parameters> tag can read it
    prep_params = ExecuteProcess(cmd=['/bin/bash', '-c', f'cp "{controllers}" "{tmp_params}"'], output='screen')

    gz_share = get_package_share_directory('ros_gz_sim')

    gz_gui = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(gz_share, 'launch', 'gz_sim.launch.py')),
        launch_arguments={'gz_args': LaunchConfiguration('gz_args')}.items(),
        condition=UnlessCondition(LaunchConfiguration('headless'))
    )

    gz_headless = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(gz_share, 'launch', 'gz_sim.launch.py')),
        launch_arguments={'gz_args': LaunchConfiguration('gz_headless_args')}.items(),
        condition=IfCondition(LaunchConfiguration('headless'))
    )

    # === /robot_description ===
    with open(urdf_file, 'r') as f:
        robot_xml = f.read()

    rsp = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        parameters=[{'use_sim_time': True, 'robot_description': robot_xml}],
        output='screen'
    )

    # === Spawn main robot into Gazebo (table top ~ z = 1.04) ===
    TABLE_Z = 1.04
    spawn = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=[
            '-name', 'kitchen_robot',
            '-topic', 'robot_description',
            '-x', '0',
            '-y', '0',
            '-z', f'{TABLE_Z:.3f}',
        ],
        output='screen'
    )

    # === Holders (from URDF) ===
    holders_xy = [
        (-0.91994, -1.18100),
        (-0.31034, -1.18100),
        (0.29926, -1.18100),
        (0.90886, -1.18100),
        (0.90886, 1.18100),
        (0.29926, 1.18100),
        (-0.31034, 1.18100),
        (-0.91994, 1.18100),
    ]

    BURNER_H = 0.1778
    TRAY_H = 0.4667
    burner_world_z = TABLE_Z + BURNER_H
    tray_world_z = TABLE_Z + TRAY_H
    BURNER_DY = 0.02
    TRAY_DY = 0.02

    prop_spawns = []
    for i, (hx, hy) in enumerate(holders_xy, start=1):
        if i != 1:
            continue
        sign_to_center = 1.0 if hy < 0.0 else -1.0
        y_burner = hy + sign_to_center * BURNER_DY
        y_tray = hy + sign_to_center * TRAY_DY

        burner_node = Node(
            package='ros_gz_sim',
            executable='create',
            arguments=[
                '-name', f'burner{i}',
                '-file', burner_urdf,
                '-x', f'{hx:.6f}',
                '-y', f'{y_burner:.6f}',
                '-z', f'{burner_world_z:.4f}',
            ],
            output='screen'
        )
        tray_node = Node(
            package='ros_gz_sim',
            executable='create',
            arguments=[
                '-name', f'tray{i}',
                '-file', tray_urdf,
                '-x', f'{hx:.6f}',
                '-y', f'{y_tray:.6f}',
                '-z', f'{tray_world_z:.4f}',
            ],
            output='screen'
        )
        prop_spawns.extend([burner_node, tray_node])

    # === Static cameras (for L1/VLM perception) ===
    spawn_cam_overhead = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=[
            '-name', 'cam_overhead',
            '-file', cam_overhead_sdf,
            '-x', '0.0',
            '-y', '0.0',
            '-z', '2.4',
            '-R', '3.14159',
            '-P', '1.35',
            '-Y', '0.0',
        ],
        output='screen'
    )

    spawn_cam_side = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=[
            '-name', 'cam_side',
            '-file', cam_side_sdf,
            '-x', '1.8',
            '-y', '-0.2',
            '-z', '1.6',
            '-R', '3.14159',
            '-P', '0.0',
            '-Y', '2.9',
        ],
        output='screen'
    )

    # === ros2_control controllers ===
    jsp = Node(
        package='controller_manager',
        executable='spawner',
        arguments=[
            'joint_state_broadcaster',
            '--controller-manager', '/controller_manager',
            '--param-file', controllers,
            '--switch-timeout', '30',
        ],
        output='screen'
    )

    arm_ctrl = Node(
        package='controller_manager',
        executable='spawner',
        arguments=[
            'arm_controller',
            '--controller-manager', '/controller_manager',
            '--param-file', controllers,
            '--switch-timeout', '30',
        ],
        output='screen'
    )

    spawn_then_jsp = RegisterEventHandler(
        OnProcessExit(target_action=spawn, on_exit=[TimerAction(period=2.0, actions=[jsp])])
    )
    jsp_then_arm = RegisterEventHandler(
        OnProcessExit(target_action=jsp, on_exit=[TimerAction(period=2.0, actions=[arm_ctrl])])
    )

    # === Bridges ===
    bridge_clock = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=['/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock'],
        output='screen'
    )

    # Plate tracking stream: bridge world pose info into ROS for downstream filtering.
    # NOTE: /model/tray1/pose currently has no publisher in this scene, so we bridge
    # /world/empty/pose/info (gz.msgs.Pose_V) instead.
    bridge_world_pose_info = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=['/world/empty/pose/info@tf2_msgs/msg/TFMessage[gz.msgs.Pose_V'],
        remappings=[('/world/empty/pose/info', '/tray_tracking/pose_stream')],
        output='screen'
    )

    # Tray pose adapter: robustly emit /tray1/pose even when source child/frame ids are empty.
    tray_pose_adapter = Node(
        package='kitchen_robot_controller',
        executable='tray_pose_adapter_node',
        name='tray_pose_adapter',
        parameters=[{
            'use_sim_time': True,
            'input_topic': '/tray_tracking/pose_stream',
            'output_topic': '/tray1/pose',
            'default_frame_id': 'world',
            'target_child_frame_patterns': ['tray1', 'tray1*', '*tray1*'],
            'expected_xyz': [holders_xy[0][0], holders_xy[0][1] + TRAY_DY, tray_world_z],
            'publish_rate_hz': 10.0,
        }],
        output='screen'
    )

    object_id_publisher = Node(
        package='kitchen_robot_controller',
        executable='object_id_publisher_node',
        name='object_id_publisher',
        parameters=[{
            'use_sim_time': True,
            'input_topic': '/tray1/pose',
            'output_topic': '/v5/perception/object_pose_est',
            'id_value': 'tray1',
            'publish_rate_hz': 10.0,
        }],
        output='screen'
    )

    static_tf_world_cam_overhead = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_tf_world_cam_overhead',
        arguments=[
            '--x', '0.0', '--y', '0.0', '--z', '2.4',
            '--roll', '3.14159', '--pitch', '1.35', '--yaw', '0.0',
            '--frame-id', 'world', '--child-frame-id', 'cam_overhead',
        ],
        output='screen'
    )

    static_tf_cam_overhead_optical = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_tf_cam_overhead_optical',
        arguments=[
            '--x', '0.0', '--y', '0.0', '--z', '0.0',
            '--roll', '-1.57079632679', '--pitch', '0.0', '--yaw', '-1.57079632679',
            '--frame-id', 'cam_overhead', '--child-frame-id', 'cam_overhead_optical_frame',
        ],
        output='screen'
    )

    static_tf_world_cam_side = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_tf_world_cam_side',
        arguments=[
            '--x', '1.8', '--y', '-0.2', '--z', '1.6',
            '--roll', '3.14159', '--pitch', '0.0', '--yaw', '2.9',
            '--frame-id', 'world', '--child-frame-id', 'cam_side',
        ],
        output='screen'
    )

    static_tf_cam_side_optical = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_tf_cam_side_optical',
        arguments=[
            '--x', '0.0', '--y', '0.0', '--z', '0.0',
            '--roll', '-1.57079632679', '--pitch', '0.0', '--yaw', '-1.57079632679',
            '--frame-id', 'cam_side', '--child-frame-id', 'cam_side_optical_frame',
        ],
        output='screen'
    )

    bridge_cam_overhead = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=['/cam_overhead/image@sensor_msgs/msg/Image[gz.msgs.Image'],
        remappings=[('/cam_overhead/image', '/v5/cam/overhead/rgb')],
        output='screen'
    )

    bridge_cam_side = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=['/cam_side/image@sensor_msgs/msg/Image[gz.msgs.Image'],
        remappings=[('/cam_side/image', '/v5/cam/side/rgb')],
        output='screen'
    )

    bridge_cam_overhead_info = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=['/cam_overhead/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo'],
        remappings=[('/cam_overhead/camera_info', '/v5/cam/overhead/camera_info')],
        output='screen'
    )

    bridge_cam_side_info = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=['/cam_side/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo'],
        remappings=[('/cam_side/camera_info', '/v5/cam/side/camera_info')],
        output='screen'
    )

    return LaunchDescription([
        declare_headless,
        declare_use_software_renderer,
        declare_gz_args,
        declare_gz_headless_args,
        set_res_gz, set_res_ign, soft_gl, no_audio,
        prep_params,
        gz_gui,
        gz_headless,
        rsp,
        spawn,
        *prop_spawns,
        spawn_cam_overhead,
        spawn_cam_side,
        spawn_then_jsp,
        jsp_then_arm,
        bridge_clock,
        static_tf_world_cam_overhead,
        static_tf_cam_overhead_optical,
        static_tf_world_cam_side,
        static_tf_cam_side_optical,
        bridge_world_pose_info,
        tray_pose_adapter,
        object_id_publisher,
        bridge_cam_overhead,
        bridge_cam_side,
        bridge_cam_overhead_info,
        bridge_cam_side_info,
    ])
