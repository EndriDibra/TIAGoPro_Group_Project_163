# Copyright (c) 2024 PAL Robotics S.L. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Major parts of this file are based on the original work by PAL Robotics S.L.
# Modifications for WSL compatibility have been added.


import os
import yaml
import tempfile
from os import environ, pathsep
from ament_index_python.packages import get_package_prefix, get_package_share_directory

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    SetEnvironmentVariable,
    SetLaunchConfiguration,
    GroupAction,
    OpaqueFunction,
)
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration

from launch_pal.include_utils import (
    include_scoped_launch_py_description,
    include_launch_py_description,
)
from launch_pal.actions import CheckPublicSim

from launch_pal.arg_utils import LaunchArgumentsBase, read_launch_argument
from launch_pal.robot_arguments import CommonArgs
from tiago_pro_description.launch_arguments import TiagoProArgs
from dataclasses import dataclass
from launch_ros.actions import Node

from launch.actions import ExecuteProcess, TimerAction


@dataclass(frozen=True)
class LaunchArguments(LaunchArgumentsBase):
    base_type: DeclareLaunchArgument = TiagoProArgs.base_type
    arm_type_right: DeclareLaunchArgument = TiagoProArgs.arm_type_right
    arm_type_left: DeclareLaunchArgument = TiagoProArgs.arm_type_left
    end_effector_right: DeclareLaunchArgument = TiagoProArgs.end_effector_right
    end_effector_left: DeclareLaunchArgument = TiagoProArgs.end_effector_left
    ft_sensor_right: DeclareLaunchArgument = TiagoProArgs.ft_sensor_right
    ft_sensor_left: DeclareLaunchArgument = TiagoProArgs.ft_sensor_left
    tool_changer_right: DeclareLaunchArgument = TiagoProArgs.tool_changer_right
    tool_changer_left: DeclareLaunchArgument = TiagoProArgs.tool_changer_left
    wrist_model_right: DeclareLaunchArgument = TiagoProArgs.wrist_model_right
    wrist_model_left: DeclareLaunchArgument = TiagoProArgs.wrist_model_left
    camera_model: DeclareLaunchArgument = TiagoProArgs.camera_model
    laser_model: DeclareLaunchArgument = TiagoProArgs.laser_model

    navigation: DeclareLaunchArgument = CommonArgs.navigation
    advanced_navigation: DeclareLaunchArgument = CommonArgs.advanced_navigation
    slam: DeclareLaunchArgument = CommonArgs.slam
    docking: DeclareLaunchArgument = CommonArgs.docking
    moveit: DeclareLaunchArgument = CommonArgs.moveit
    world_name: DeclareLaunchArgument = CommonArgs.world_name
    tuck_arm: DeclareLaunchArgument = CommonArgs.tuck_arm
    is_public_sim: DeclareLaunchArgument = CommonArgs.is_public_sim
    
    headless: DeclareLaunchArgument = DeclareLaunchArgument(
        "headless", default_value="False",
        description="Run Gazebo without GUI (headless mode)")


def private_navigation(context, *args, **kwargs):
    actions = []
    base_type = read_launch_argument('base_type', context)
    camera_model = read_launch_argument('camera_model', context)
    docking = read_launch_argument('docking', context)
    advanced_navigation = read_launch_argument('advanced_navigation', context)
    use_sim_time = read_launch_argument('use_sim_time', context)
    rviz_cfg_pkg = base_type + '_2dnav'
    if advanced_navigation == 'True':
        rviz_cfg_pkg = base_type + '_advanced_2dnav'
    
    # Robot info
    robot_info = {
        "robot_info_publisher": {
            "ros__parameters": {
                "robot_type": "tiago_pro",
                "base_type": base_type,
                "laser_model": "sick-571",
                "camera_model": camera_model,
                "advanced_navigation": (advanced_navigation == 'True'),
                "has_dock": (docking == 'True'),
                "use_sim_time": (use_sim_time == 'True'),
            }
        }
    }
    
    temp_yaml = tempfile.mkdtemp()
    temp_robot_info = os.path.join(temp_yaml, '99_robot_info.yaml')
    with open(temp_robot_info, 'w') as f:
        yaml.safe_dump(robot_info, f)
    
    robot_info_env = SetEnvironmentVariable('ROBOT_INFO_PATH', temp_yaml)
    actions.append(robot_info_env)
    
    robot_info_publisher = Node(
        package='robot_info_publisher',
        executable='robot_info_publisher',
        name='robot_info_publisher',
        output='screen',
    )
    actions.append(robot_info_publisher)
    
    # Laser sensors
    laser_bringup_launch = include_launch_py_description(
        pkg_name=base_type + '_laser_sensors',
        paths=['launch', 'laser_sim.launch.py'],
    )
    actions.append(laser_bringup_launch)
    
    # ===== WSL2 FIX: EKF for ODOM =====
    ekf_node = Node(
        package='robot_localization',
        executable='ekf_node',
        name='ekf_odom',
        output='screen',
        parameters=[{
            'use_sim_time': True,
            'frequency': 10.0,
            'two_d_mode': True,
            'publish_tf': True,
            'map_frame': 'map',
            'odom_frame': 'odom',
            'base_link_frame': 'base_footprint',
            'world_frame': 'odom',
            'odom0': '/mobile_base_controller/odom',
            'odom0_config': [True, True, False, False, False, True,
                            True, True, False, False, False, True,
                            False, False, False],
        }]
    )
    actions.append(ekf_node)
    # ===== END WSL2 FIX =====
    
    # Navigation (using PAL's default config with user overrides from /home/user/.pal/config/)
    nav_bringup_launch = include_launch_py_description(
        pkg_name=base_type + '_2dnav',
        paths=['launch', 'navigation.launch.py'],
    )
    actions.append(nav_bringup_launch)
    
    # Localization
    loc_bringup_launch = include_launch_py_description(
        pkg_name=base_type + '_2dnav',
        paths=['launch', 'localization.launch.py'],
        condition=UnlessCondition(LaunchConfiguration('slam'))
    )
    actions.append(loc_bringup_launch)

    # SLAM
    slam_bringup_launch = include_launch_py_description(
        pkg_name=base_type + '_2dnav',
        paths=['launch', 'slam.launch.py'],
        condition=IfCondition(LaunchConfiguration('slam'))
    )
    actions.append(slam_bringup_launch)

    # Docking
    docking_bringup_launch = include_launch_py_description(
        pkg_name=base_type + '_docking',
        paths=['launch', 'docking_sim.launch.py'],
        condition=IfCondition(LaunchConfiguration('docking'))
    )
    actions.append(docking_bringup_launch)

    # Stores Server
    db_bringup_launch = Node(
        package='pal_stores_server',
        executable='pal_stores_server',
        arguments=[os.path.join(
            os.environ['HOME'], '.pal', 'stores.db'
        )],
        output='screen',
        condition=IfCondition(LaunchConfiguration('advanced_navigation'))
    )
    actions.append(db_bringup_launch)

    # Advanced Navigation
    advanced_nav_bringup_launch = include_launch_py_description(
        pkg_name=base_type + '_advanced_2dnav',
        paths=['launch', 'advanced_navigation.launch.py'],
        condition=IfCondition(LaunchConfiguration('advanced_navigation'))
    )
    actions.append(advanced_nav_bringup_launch)

    # RViz
    rviz_bringup_launch = Node(
        package='rviz2',
        executable='rviz2',
        arguments=['-d', os.path.join(
            get_package_share_directory(rviz_cfg_pkg),
            'config',
            'rviz',
            'navigation.rviz',
        )],
        parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}],
        output='screen',
    )
    actions.append(rviz_bringup_launch)
    return actions


def declare_actions(launch_description: LaunchDescription, launch_args: LaunchArguments):

    # Set use_sim_time to True
    set_sim_time = SetLaunchConfiguration("use_sim_time", "True")
    launch_description.add_action(set_sim_time)

    # ===== WSL2 FIX: CLEANUP =====
    cleanup_wsl = ExecuteProcess(
        cmd=['bash', '-c', 
             'rm -f /tmp/ros2-control-controller-spawner.lock; '
             'pkill -9 gzserver gzclient 2>/dev/null || true; '
             'sleep 2'],
        name='wsl_cleanup',
        output='screen'
    )
    launch_description.add_action(cleanup_wsl)
    # ===== END WSL2 FIX =====

    # ===== WSL2 FIX: REMAP    # WSL2 FIX
    # relay_node = Node(
    #     package='topic_tools',
    #     executable='relay',
    #     name='scan_topic_relay',
    #     parameters=[{
    #         'input_topic': '/scan_front_raw',
    #         'output_topic': '/scan',
    #     }],
    # )
    # launch_description.add_action(relay_node)

    # static_tf_publisher_node = Node(
    #     package='tf2_ros',
    #     executable='static_transform_publisher',
    #     name='static_transform_publisher',
    #     arguments=['0', '0', '0', '0', '0', '0', 'odom', 'base_footprint'],
    # )
    # launch_description.add_action(static_tf_publisher_node)
    # ===== END WSL2 FIX =====

    # Shows error if is_public_sim is not set to True when using public simulation
    public_sim_check = CheckPublicSim()
    launch_description.add_action(public_sim_check)

    robot_name = "tiago_pro"
    packages = ["tiago_pro_description", "pal_sea_arm_description",
                "omni_base_description", "pal_pro_gripper_description",
                "tiago_pro_head_description", "allegro_hand_description",
                "pal_urdf_utils"]

    model_path = get_model_paths(packages)

    gazebo_model_path_env_var = SetEnvironmentVariable(
        "GAZEBO_MODEL_PATH", model_path)

    gazebo = include_scoped_launch_py_description(
        pkg_name="pal_gazebo_worlds",
        paths=["launch", "pal_gazebo.launch.py"],
        env_vars=[gazebo_model_path_env_var],
        launch_arguments={
            "world_name":  launch_args.world_name,
            "model_paths": packages,
            "resource_paths": packages,
        })

    launch_description.add_action(gazebo)

    navigation = GroupAction(
        condition=IfCondition(LaunchConfiguration('navigation')),
        actions=[
            # Private Navigation
            OpaqueFunction(
                function=private_navigation,
                condition=UnlessCondition(LaunchConfiguration('is_public_sim'))
            ),
        ]
    )

    launch_description.add_action(navigation)

    move_group = include_scoped_launch_py_description(
        pkg_name="tiago_pro_moveit_config",
        paths=["launch", "move_group.launch.py"],
        launch_arguments={
            "robot_name": robot_name,
            "use_sim_time": LaunchConfiguration("use_sim_time"),
            "base_type": launch_args.base_type,
            "arm_type_right": launch_args.arm_type_right,
            "arm_type_left": launch_args.arm_type_left,
            "end_effector_right": launch_args.end_effector_right,
            "end_effector_left": launch_args.end_effector_left,
            "ft_sensor_right": launch_args.ft_sensor_right,
            "ft_sensor_left": launch_args.ft_sensor_left
        },
        condition=IfCondition(LaunchConfiguration("moveit")))

    launch_description.add_action(move_group)

    robot_spawn = include_scoped_launch_py_description(
        pkg_name="tiago_pro_gazebo",
        paths=["launch", "robot_spawn.launch.py"])

    launch_description.add_action(robot_spawn)

    tiago_bringup = include_scoped_launch_py_description(
        pkg_name="tiago_pro_bringup", paths=["launch", "tiago_pro_bringup.launch.py"],
        launch_arguments={
            "use_sim_time": LaunchConfiguration("use_sim_time"),
            "arm_type_right": launch_args.arm_type_right,
            "arm_type_left": launch_args.arm_type_left,
            "end_effector_right": launch_args.end_effector_right,
            "end_effector_left": launch_args.end_effector_left,
            "ft_sensor_right": launch_args.ft_sensor_right,
            "ft_sensor_left": launch_args.ft_sensor_left,
            "tool_changer_right": launch_args.tool_changer_right,
            "tool_changer_left": launch_args.tool_changer_left,
            "wrist_model_right": launch_args.wrist_model_right,
            "wrist_model_left": launch_args.wrist_model_left,
            "laser_model": launch_args.laser_model,
            "camera_model": launch_args.camera_model,
            "base_type": launch_args.base_type,
            "is_public_sim": launch_args.is_public_sim}
    )

    # ===== WSL2 FIX: DELAY BEFORE BRINGUP =====    
    # Wrap with delay to ensure Gazebo + controller_manager are ready
    tiago_bringup_delayed = TimerAction(
        period=5.0,  # 5 second delay for WSL2
        actions=[tiago_bringup]
    )
    # (The name of the added action would have to be changed if reverting)
    # ===== END WSL2 FIX =====

    launch_description.add_action(tiago_bringup_delayed)
    # launch_description.add_action(tiago_bringup)  

    tuck_arm = Node(
        package="tiago_pro_gazebo",
        executable="tuck_arm.py",
        emulate_tty=True,
        output="both",
        condition=IfCondition(LaunchConfiguration('tuck_arm'))
    )

    launch_description.add_action(tuck_arm)

    # Social Costmap Node
    social_costmap_node = Node(
        package='tiago_social_nav',
        executable='social_costmap_node',
        name='social_costmap_node',
        output='screen',
        parameters=[{'use_sim_time': True}],
    )
    launch_description.add_action(social_costmap_node)

    # VLM Navigator Node
    vlm_navigator_node = Node(
        package='tiago_social_vlm',
        executable='vlm_navigator',
        name='vlm_navigator',
        output='screen',
        parameters=[
            {'vlm_backend': os.environ.get('VLM_BACKEND', 'mistral')},  # Default to mistral, override with VLM_BACKEND env var
            {'mistral_api_key': os.environ.get('MISTRAL_API_KEY', '')},  # Read from environment
            {'controller_server_name': 'controller_server'},
            {'controller_name': 'FollowPath'},
            {'default_max_speed': 1.0},
            {'use_sim_time': True}
        ],
    )
    launch_description.add_action(vlm_navigator_node)

    # Human Spawner Node (delayed to ensure Gazebo is ready)
    human_spawner_node = Node(
        package='human_spawner',
        executable='human_controller',
        name='human_controller',
        output='screen',
        parameters=[{'use_sim_time': True}],
    )
    human_spawner_delayed = TimerAction(
        period=10.0,  # Wait for Gazebo to be fully ready
        actions=[human_spawner_node]
    )
    launch_description.add_action(human_spawner_delayed)

    # Social Command Node
    social_cmd_node = Node(
        package='tiago_social_cmd',
        executable='social_cmd_node',
        name='social_cmd_node',
        output='screen',
        parameters=[
            os.path.join(get_package_share_directory('tiago_social_cmd'), 'config', 'scenarios.yaml')
        ]
    )
    social_cmd_delayed = TimerAction(
        period=15.0, # Wait for Nav2 and Spawner
        actions=[social_cmd_node]
    )
    launch_description.add_action(social_cmd_delayed)

    return


def get_model_paths(packages_names):
    model_paths = ""
    for package_name in packages_names:
        if model_paths != "":
            model_paths += pathsep

        package_path = get_package_prefix(package_name)
        model_path = os.path.join(package_path, "share")

        model_paths += model_path

    if "GAZEBO_MODEL_PATH" in environ:
        model_paths += pathsep + environ["GAZEBO_MODEL_PATH"]

    return model_paths


def get_resource_paths(packages_names):
    resource_paths = ""
    for package_name in packages_names:
        if resource_paths != "":
            resource_paths += pathsep

        package_path = get_package_prefix(package_name)
        resource_paths += package_path

    if "GAZEBO_RESOURCE_PATH" in environ:
        resource_paths += pathsep + environ["GAZEBO_RESOURCE_PATH"]

    return resource_paths


def generate_launch_description():

    # Create the launch description
    ld = LaunchDescription()

    launch_arguments = LaunchArguments()

    launch_arguments.add_to_launch_description(ld)

    declare_actions(ld, launch_arguments)

    return ld
