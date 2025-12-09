#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose
import difflib
import time
import threading

class SocialCmdNode(Node):
    def __init__(self):
        super().__init__('social_cmd_node', 
                         allow_undeclared_parameters=True,
                         automatically_declare_parameters_from_overrides=True)

        # Parameters
        # Note: 'automatically_declare_parameters_from_overrides=True' will declare 
        # parameters found in the YAML file (like 'scenario_sequence' and 'loop_sequence').
        # Manually declaring them again would verify defaults but causes a "ParameterAlreadyDeclaredException"
        # if they exist in YAML. We will safely retrieve them instead.

        # Publishers
        self.task_pub = self.create_publisher(String, '/social_task', 10)

        # Subscribers
        self.create_subscription(String, '/command_input', self.command_callback, 10)

        # Action Client
        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        # State
        self.current_goal_handle = None
        self.executor_thread = None
        self.stop_sequence = False
        self.nav_complete_event = threading.Event()

        # Load initial config
        self.scenarios = self.get_scenarios_from_params()
        
        # Get parameters safely (handling both declared-via-yaml and missing cases)
        self.sequence = self._get_param_or_default('scenario_sequence', [])
        self.loop = self._get_param_or_default('loop_sequence', False)

        self.get_logger().info(f'Social Cmd Node Ready. Scenarios: {list(self.scenarios.keys())}')

        if self.sequence:
            self.start_sequence_execution()

    def _get_param_or_default(self, name, default):
        try:
            param = self.get_parameter(name)
            return param.value
        except rclpy.exceptions.ParameterNotDeclaredException:
            # If not in YAML, it wasn't automatically declared. 
            # We can declare it now with default or just return default.
            # Declaring it allows runtime interaction later.
            return self.declare_parameter(name, default).value

    def get_scenarios_from_params(self):
        scenarios = {}
        # get_parameters_by_prefix returns map relative to prefix
        # e.g. 'kitchen_task.keywords': Parameter(...)
        # This allows us to handle the nested structure defined in YAML
        params = self.get_parameters_by_prefix('scenarios')
        
        for name, param in params.items():
            parts = name.split('.')
            if len(parts) >= 2: # scenario_name.field (and potentially subfields)
                scenario_name = parts[0]
                field = parts[1]
                
                if scenario_name not in scenarios:
                    scenarios[scenario_name] = {}
                
                # We only support 1 level of nesting (scenario -> fields) for now
                scenarios[scenario_name][field] = param.value
                
        return scenarios

    def command_callback(self, msg):
        self.get_logger().info(f"Received command: {msg.data}")
        self.process_command(msg.data)

    def process_command(self, cmd_text):
        # Fuzzy match
        scenario_name = self.match_command(cmd_text)
        if scenario_name:
            return self.execute_scenario(scenario_name)
        else:
            self.get_logger().warn(f"No matching scenario found for: {cmd_text}")
            return False

    def match_command(self, cmd_text):
        # 1. Check exact keys
        if cmd_text in self.scenarios:
            return cmd_text
        
        # 2. Check keywords in values
        # We can implement fuzzy logic here.
        # Simple approach: Check if input text is "close" to a key
        matches = difflib.get_close_matches(cmd_text, self.scenarios.keys(), n=1, cutoff=0.6)
        if matches:
            return matches[0]
        
        # 3. Check specific keywords defined in scenarios
        # (Assuming scenarios have a 'keywords' list)
        for name, data in self.scenarios.items():
            keywords = data.get('keywords', [])
            for k in keywords:
                if k in cmd_text:
                    return name
        
        return None

    def execute_scenario(self, name):
        data = self.scenarios.get(name)
        if not data:
            return False

        self.get_logger().info(f"Executing scenario: {name}")

        # 1. Publish Task
        task_msg = data.get('task_msg', '')
        if task_msg:
            msg = String()
            msg.data = task_msg
            self.task_pub.publish(msg)
            self.get_logger().info(f"Published task: {task_msg}")

        # 2. Send Nav Goal
        nav_goal = data.get('nav_goal', None)
        if nav_goal:
            # Check if nav_goal is a list [x, y, yaw] or string (alias)
            if isinstance(nav_goal, str):
                # Handle alias lookup if we had a locations dict. 
                # For now assume explicit coords in list.
                pass
            elif isinstance(nav_goal, list) and len(nav_goal) >= 3:
                self.send_goal(nav_goal)
                return True # Navigation started
        
        return False

    def send_goal(self, pose_data):
        if not self.nav_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("Nav2 action server not available!")
            return

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = 'map'
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()
        
        goal_msg.pose.pose.position.x = float(pose_data[0])
        goal_msg.pose.pose.position.y = float(pose_data[1])
        # Convert yaw to quaternion (assuming z is yaw)
        import math
        yaw = float(pose_data[2])
        goal_msg.pose.pose.orientation.z = math.sin(yaw / 2.0)
        goal_msg.pose.pose.orientation.w = math.cos(yaw / 2.0)

        self.get_logger().info(f"Sending goal: {pose_data}")
        
        self.nav_complete_event.clear() # Reset event before sending
        self.future = self.nav_client.send_goal_async(goal_msg)
        self.future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Goal rejected')
            self.nav_complete_event.set() # Unblock if rejected
            return

        self.get_logger().info('Goal accepted')
        self.current_goal_handle = goal_handle
        get_result_future = goal_handle.get_result_async()
        get_result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        result = future.result().result
        status = future.result().status
        self.get_logger().info(f'Nav Result Status: {status}')
        # Signal completion for sequencer
        self.nav_complete_event.set()

    def start_sequence_execution(self):
        self.executor_thread = threading.Thread(target=self.run_sequence)
        self.executor_thread.start()

    def run_sequence(self):
        # Give some time for initial setup
        time.sleep(2.0)
        
        while rclpy.ok() and not self.stop_sequence:
            for step_cmd in self.sequence:
                if not rclpy.ok(): break
                
                # Check for "wait" command
                if step_cmd.startswith("wait "):
                    try:
                        dur = float(step_cmd.split()[1])
                        self.get_logger().info(f"Waiting {dur} seconds...")
                        time.sleep(dur)
                    except:
                        pass
                    continue
                
                self.get_logger().info(f"Auto-running: {step_cmd}")
                nav_started = self.process_command(step_cmd)
                
                if nav_started:
                    self.get_logger().info(f"Waiting for navigation to complete...")
                    self.nav_complete_event.wait()
                    self.get_logger().info(f"Navigation completed.")
                else:
                    # If no nav, just a small pause to avoid spamming if loop is tight
                    time.sleep(1.0) 

            if not self.loop:
                break

def main(args=None):
    rclpy.init(args=args)
    node = SocialCmdNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
