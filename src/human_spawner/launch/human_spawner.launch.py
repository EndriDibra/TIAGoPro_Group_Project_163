"""
Launch file for the human spawner node.
"""

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='human_spawner',
            executable='human_controller',
            name='human_controller',
            output='screen',
            parameters=[{'use_sim_time': True}],
        ),
    ])
