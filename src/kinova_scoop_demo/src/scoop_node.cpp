#include <rclcpp/rclcpp.hpp>
#include <moveit/move_group_interface/move_group_interface.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    
    // MoveIt 2 requires node options to automatically declare parameters
    rclcpp::NodeOptions node_options;
    node_options.automatically_declare_parameters_from_overrides(true);
    auto node = rclcpp::Node::make_shared("scoop_node", node_options);

    // Spin up a thread for the executor so MoveIt can communicate with ROS
    rclcpp::executors::MultiThreadedExecutor executor;
    executor.add_node(node);
    std::thread spinner([&executor]() { executor.spin(); });

    // Initialize the MoveGroupInterface (Make sure "arm" matches your SRDF group name)
    auto move_group = moveit::planning_interface::MoveGroupInterface(node, "arm");
    auto logger = node->get_logger();

    std::vector<geometry_msgs::msg::Pose> waypoints;
    geometry_msgs::msg::Pose target_pose = move_group.getCurrentPose().pose;

    RCLCPP_INFO(logger, "Starting scoop sequence...");

    // 1. APPROACH
    waypoints.push_back(target_pose);

    // 2. DIP
    target_pose.position.z -= 0.10; 
    waypoints.push_back(target_pose);

    // 3. SWEEP & PITCH
    target_pose.position.x += 0.08; 
    tf2::Quaternion q_orig, q_rot, q_new;
    tf2::fromMsg(target_pose.orientation, q_orig);
    q_rot.setRPY(0.0, -M_PI/4.0, 0.0); // Pitch up by 45 degrees
    q_new = q_rot * q_orig;
    q_new.normalize();
    target_pose.orientation = tf2::toMsg(q_new);
    waypoints.push_back(target_pose);

    // 4. LIFT
    target_pose.position.z += 0.15;
    waypoints.push_back(target_pose);

    // Plan the Cartesian path
    moveit_msgs::msg::RobotTrajectory trajectory;
    const double jump_threshold = 0.0;
    const double eef_step = 0.01;

    double fraction = move_group.computeCartesianPath(waypoints, eef_step, jump_threshold, trajectory);
    RCLCPP_INFO(logger, "Visualizing Cartesian path (%.2f%% achieved)", fraction * 100.0);

    if (fraction > 0.9) {
        RCLCPP_INFO(logger, "Executing trajectory...");
        move_group.execute(trajectory);
    } else {
        RCLCPP_WARN(logger, "Could not compute the full Cartesian path.");
    }

    // Cleanup
    rclcpp::shutdown();
    spinner.join();
    return 0;
}
