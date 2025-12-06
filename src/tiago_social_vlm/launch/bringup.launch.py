from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    vlm_backend_arg = DeclareLaunchArgument(
        'vlm_backend',
        default_value='mock',
        description="VLM backend to use: 'mock' (testing), 'smol' (local GPU), or 'mistral' (cloud API)"
    )

    mistral_key_arg = DeclareLaunchArgument(
        'mistral_api_key',
        default_value='',
        description='API Key for Mistral VLM (required if vlm_backend=mistral)'
    )

    vlm_node = Node(
        package='tiago_social_vlm',
        executable='vlm_navigator',
        name='vlm_navigator',
        output='screen',
        parameters=[
            {'vlm_backend': LaunchConfiguration('vlm_backend')},
            {'mistral_api_key': LaunchConfiguration('mistral_api_key')},
            {'controller_server_name': 'controller_server'},
            {'controller_name': 'FollowPath'}, # Change this if your controller plugin has a different name
            {'default_max_speed': 1.0},
            {'use_sim_time': True}
        ],
        # Note: This node listens to /vlm/goal_pose (dedicated topic) to avoid
        # conflict with bt_navigator which also subscribes to /goal_pose.
        # To send goals to this node, publish to /vlm/goal_pose:
        #   ros2 topic pub --once /vlm/goal_pose geometry_msgs/msg/PoseStamped \
        #     "{header: {frame_id: 'map'}, pose: {position: {x: 1.0, y: 0.5}, orientation: {w: 1.0}}}"
    )

    return LaunchDescription([
        vlm_backend_arg,
        mistral_key_arg,
        vlm_node
    ])
