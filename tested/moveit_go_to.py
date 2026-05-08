import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from pymoveit2 import MoveIt2
from pymoveit2.robots import kinova


class MoveItGoTo(Node):

    def __init__(self):
        super().__init__("moveit_go_to")

        self._joint_state_received = False

        self.create_subscription(
            JointState,
            "/joint_states",
            self._joint_state_callback,
            10
        )

        self.moveit2 = MoveIt2(
            node=self,
            joint_names=kinova.joint_names(),
            base_link_name=kinova.base_link_name(),
            end_effector_name=kinova.end_effector_name(),
            group_name=kinova.MOVE_GROUP_ARM,
        )

    def _joint_state_callback(self, msg):
        self._joint_state_received = True

    def wait_for_joint_states(self):
        self.get_logger().info("Waiting for /joint_states...")
        while not self._joint_state_received:
            rclpy.spin_once(self, timeout_sec=0.1)

        self.get_logger().info("Joint states received ✅")

    def go_to_joint(self, joint_positions_rad: list):

        # 🔥 IMPORTANT FIX
        self.wait_for_joint_states()

        self.get_logger().info("Planning to joint target...")
        self.moveit2.move_to_configuration(joint_positions_rad)

        self.get_logger().info("Executing trajectory...")
        self.moveit2.wait_until_executed()

        self.get_logger().info("Done.")     