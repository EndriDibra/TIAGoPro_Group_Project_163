from launch import LaunchDescription
from launch_ros.actions import Node
import os
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    pkg_share = get_package_share_directory('tiago_social_cmd')
    config_file = os.path.join(pkg_share, 'config', 'scenarios.yaml')

    return LaunchDescription([
        Node(
            package='tiago_social_cmd',
            executable='social_cmd_node',
            name='social_cmd_node',
            output='screen',
            parameters=[config_file]
        )
    ])
