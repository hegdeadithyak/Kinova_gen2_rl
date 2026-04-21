#!/usr/bin/env python3
import rclpy
import math
from moveit.planning import MoveItPy

def main(args=None):
    rclpy.init(args=args)

    # 1. Initialize MoveIt 2 Python API
    # Note: This node MUST be launched with MoveIt parameters (URDF, SRDF, Kinematics)
    kinova_moveit = MoveItPy(node_name="prescoop_planner")

    # The group name must match the one defined in your Kinova MoveIt config (usually "arm")
    planning_group_name = "arm"
    arm = kinova_moveit.get_planning_component(planning_group_name)

    # 2. Your target angles in DEGREES
    PRE_SCOOP_DEG = [372.75, 217.16, 57.26, 295.66, 192.63, 251.18]
    
    # 3. Convert degrees to RADIANS
    PRE_SCOOP_RAD = [math.radians(deg) for deg in PRE_SCOOP_DEG]

    kinova_moveit.get_logger().info(f"Target in Radians: {PRE_SCOOP_RAD}")

    # 4. Create a target robot state
    robot_model = kinova_moveit.get_robot_model()
    target_state = robot_model.create_robot_state()
    target_state.update() # Sync with current state before modifying
    
    # Apply our target radians to the planning group
    target_state.set_joint_group_positions(planning_group_name, PRE_SCOOP_RAD)

    # Set this state as our planning goal
    arm.set_goal_state(robot_state=target_state)

    # 5. Plan the trajectory from current position to PRE_SCOOP
    kinova_moveit.get_logger().info("Planning path to PRE_SCOOP...")
    plan_result = arm.plan()

    # 6. Execute the trajectory
    if plan_result:
        kinova_moveit.get_logger().info("Plan found! Sending to trajectory bridge...")
        # This triggers the FollowJointTrajectory action server (your trajectory bridge)
        kinova_moveit.execute(plan_result.trajectory, controllers=[])
    else:
        kinova_moveit.get_logger().error("Planning failed. Check URDF joint limits and collision margins.")

    rclpy.shutdown()

if __name__ == '__main__':
    main()