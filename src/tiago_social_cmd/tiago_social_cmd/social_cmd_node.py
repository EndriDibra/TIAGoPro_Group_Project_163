#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from action_msgs.msg import GoalStatus
from std_msgs.msg import String
from std_srvs.srv import Trigger
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose
import difflib
import time
import threading
import math

class SocialCmdNode(Node):
    def __init__(self):
        super().__init__('social_cmd_node', 
                         allow_undeclared_parameters=True,
                         automatically_declare_parameters_from_overrides=True)

        # Publishers
        self.task_pub = self.create_publisher(String, '/social_task', 10)

        # Subscribers
        self.create_subscription(String, '/command_input', self.command_callback, 10)

        # Action Client
        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        
        # Service Client for cleanup
        self.remove_humans_client = self.create_client(Trigger, '/human_spawner/remove_all')

        # State
        self.current_goal_handle = None
        self.executor_thread = None
        self.stop_sequence = False
        self.nav_complete_event = threading.Event()

        # Execution State
        self.current_scenario_name = None
        self.execution_state = "IDLE"  # IDLE, SETUP, RECOVERY, ACTION
        
        # Retry logic for start position
        self.recovery_pose = [0.0, 0.0, 0.0]
        self.max_start_retries = 3
        self.current_start_attempt = 0
        self.current_start_pose = None

        # Load initial config
        self.scenarios = self.get_scenarios_from_params()
        
        # Get parameters safely
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
            return self.declare_parameter(name, default).value

    def get_scenarios_from_params(self):
        scenarios = {}
        params = self.get_parameters_by_prefix('scenarios')
        
        for name, param in params.items():
            parts = name.split('.')
            if len(parts) >= 2:
                scenario_name = parts[0]
                field = parts[1]
                
                if scenario_name not in scenarios:
                    scenarios[scenario_name] = {}
                
                scenarios[scenario_name][field] = param.value
                
        return scenarios

    def command_callback(self, msg):
        self.get_logger().info(f"Received command: {msg.data}")
        self.process_command(msg.data)

    def process_command(self, cmd_text):
        # Fuzzy match
        scenario_name = self.match_command(cmd_text)
        if scenario_name:
            return self.start_scenario_execution(scenario_name)
        else:
            self.get_logger().warn(f"No matching scenario found for: {cmd_text}")
            return False

    def match_command(self, cmd_text):
        if cmd_text in self.scenarios:
            return cmd_text
        
        matches = difflib.get_close_matches(cmd_text, self.scenarios.keys(), n=1, cutoff=0.6)
        if matches:
            return matches[0]
        
        for name, data in self.scenarios.items():
            keywords = data.get('keywords', [])
            for k in keywords:
                if k in cmd_text:
                    return name
        return None

    def start_scenario_execution(self, name):
        data = self.scenarios.get(name)
        if not data: return False

        self.get_logger().info(f"Starting scenario sequence: {name}")
        self.current_scenario_name = name
        
        # Reset retry state
        self.current_start_attempt = 0
        self.current_start_pose = data.get('start_pose', None)
        
        # CLEANUP: Remove old humans
        if self.remove_humans_client.wait_for_service(timeout_sec=1.0):
            req = Trigger.Request()
            self.remove_humans_client.call_async(req)
            self.get_logger().info(f"Triggered human cleanup.")
        else:
            self.get_logger().warn(f"Human cleanup service not available.")

        # Check for start_pose (Setup Phase)
        if self.current_start_pose:
            self.execution_state = "SETUP"
            self.current_start_attempt += 1
            self.get_logger().info(f"Phase: SETUP -> Navigate to start ({self.current_start_pose}) [Attempt {self.current_start_attempt}/{self.max_start_retries}]")
            return self.send_goal(self.current_start_pose)
        else:
            # Skip directly to ACTION phase
            self.execution_state = "ACTION"
            return self.execute_action_phase(name)

    def execute_action_phase(self, name):
        self.execution_state = "ACTION"
        data = self.scenarios.get(name)
        
        self.get_logger().info(f"Phase: ACTION -> Trigger & Navigate")
        
        # 1. Publish Task (Trigger Human)
        task_msg = data.get('task_msg', '')
        if task_msg:
            msg = String()
            msg.data = task_msg
            self.task_pub.publish(msg)
            self.get_logger().info(f"Published task trigger: {task_msg}")

        # 2. Send Nav Goal (Robot Action)
        nav_goal = data.get('nav_goal', None)
        if nav_goal:
             return self.send_goal(nav_goal)
        
        # If no nav goal, we are done
        self.nav_complete_event.set()
        self.execution_state = "IDLE"
        return True

    def send_goal(self, pose_data):
        if not self.nav_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("Nav2 action server not available!")
            return False

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = 'map'
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()
        
        goal_msg.pose.pose.position.x = float(pose_data[0])
        goal_msg.pose.pose.position.y = float(pose_data[1])
        
        yaw = float(pose_data[2])
        goal_msg.pose.pose.orientation.z = math.sin(yaw / 2.0)
        goal_msg.pose.pose.orientation.w = math.cos(yaw / 2.0)

        self.get_logger().info(f"Sending goal: {pose_data}")
        
        self.nav_complete_event.clear()
        self.future = self.nav_client.send_goal_async(goal_msg)
        self.future.add_done_callback(self.goal_response_callback)
        return True

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Goal rejected')
            self.nav_complete_event.set()
            return

        self.get_logger().info('Goal accepted')
        self.current_goal_handle = goal_handle
        get_result_future = goal_handle.get_result_async()
        get_result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        status = future.result().status
        status_name = self._status_to_string(status)
        self.get_logger().info(f'Nav Result Status: {status_name}')
        
        if self.execution_state == "SETUP":
            if status == GoalStatus.STATUS_SUCCEEDED:
                # Successfully reached start position -> proceed to ACTION
                self.get_logger().info("Reached start position successfully.")
                self.execute_action_phase(self.current_scenario_name)
            else:
                # Failed to reach start position -> try recovery
                self.get_logger().warn(f"Failed to reach start position (status: {status_name})")
                self.attempt_recovery()
                
        elif self.execution_state == "RECOVERY":
            if status == GoalStatus.STATUS_SUCCEEDED:
                # Reached recovery pose -> retry start position
                self.get_logger().info("Reached recovery pose, retrying start position...")
                self.current_start_attempt += 1
                if self.current_start_attempt <= self.max_start_retries:
                    self.execution_state = "SETUP"
                    self.get_logger().info(f"Phase: SETUP -> Navigate to start ({self.current_start_pose}) [Attempt {self.current_start_attempt}/{self.max_start_retries}]")
                    self.send_goal(self.current_start_pose)
                else:
                    self.skip_scenario("Max retries exceeded after recovery")
            else:
                # Even recovery failed -> skip scenario
                self.skip_scenario(f"Recovery navigation also failed (status: {status_name})")
                
        elif self.execution_state == "ACTION":
            # ACTION phase complete (success or not, we proceed)
            if status != GoalStatus.STATUS_SUCCEEDED:
                self.get_logger().warn(f"Did not reach end goal (status: {status_name}), but proceeding anyway.")
            else:
                self.get_logger().info("Reached end goal successfully.")
            self.execution_state = "IDLE"
            self.nav_complete_event.set()
        else:
            # IDLE or unknown state
            self.execution_state = "IDLE"
            self.nav_complete_event.set()
    
    def attempt_recovery(self):
        """Navigate to recovery pose before retrying start position."""
        if self.current_start_attempt >= self.max_start_retries:
            self.skip_scenario("Max retries exceeded for start position")
            return
            
        self.get_logger().info(f"Attempting recovery: navigating to {self.recovery_pose}")
        self.execution_state = "RECOVERY"
        self.send_goal(self.recovery_pose)
    
    def skip_scenario(self, reason):
        """Skip current scenario and signal completion."""
        self.get_logger().error(f"SKIPPING scenario '{self.current_scenario_name}': {reason}")
        self.execution_state = "IDLE"
        self.nav_complete_event.set()
    
    def _status_to_string(self, status):
        """Convert GoalStatus to human-readable string."""
        status_map = {
            GoalStatus.STATUS_UNKNOWN: "UNKNOWN",
            GoalStatus.STATUS_ACCEPTED: "ACCEPTED",
            GoalStatus.STATUS_EXECUTING: "EXECUTING",
            GoalStatus.STATUS_CANCELING: "CANCELING",
            GoalStatus.STATUS_SUCCEEDED: "SUCCEEDED",
            GoalStatus.STATUS_CANCELED: "CANCELED",
            GoalStatus.STATUS_ABORTED: "ABORTED",
        }
        return status_map.get(status, f"STATUS_{status}")

    def start_sequence_execution(self):
        self.executor_thread = threading.Thread(target=self.run_sequence)
        self.executor_thread.start()

    def run_sequence(self):
        time.sleep(2.0)
        while rclpy.ok() and not self.stop_sequence:
            for step_cmd in self.sequence:
                if not rclpy.ok(): break
                
                if step_cmd.startswith("wait "):
                    try:
                        dur = float(step_cmd.split()[1])
                        self.get_logger().info(f"Waiting {dur} seconds...")
                        time.sleep(dur)
                    except Exception:
                        pass
                    continue
                
                self.get_logger().info(f"Auto-running: {step_cmd}")
                nav_started = self.process_command(step_cmd)
                
                if nav_started:
                    self.get_logger().info(f"Waiting for scenario completion...")
                    self.nav_complete_event.wait()
                    self.get_logger().info(f"Scenario completed.")
                else:
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
