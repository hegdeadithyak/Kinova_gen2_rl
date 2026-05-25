/**
 * feeding_safe_mover.cpp
 *
 * Safety-focused MoveIt 2 node for the Kinova Gen2 assistive feeding robot.
 *
 * Subscribes to /goal_point (geometry_msgs/PointStamped) and moves the arm
 * end-effector (spoon) to the requested XYZ position while holding the
 * current end-effector orientation locked via an OMPL path constraint that
 * is active on EVERY waypoint of the planned trajectory, not just the goal.
 *
 * Safety guarantees (by design, not by runtime flag):
 *   - Position-only planning is not implemented and cannot be triggered.
 *   - The path constraint is set before planning and cleared only after.
 *   - Every trajectory is validated for wrist-joint deviation before execution.
 *   - If planning or validation fails the arm does not move.
 */

#include <atomic>
#include <cmath>
#include <memory>
#include <string>
#include <thread>
#include <chrono>

#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/point_stamped.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>

#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit_msgs/msg/constraints.hpp>
#include <moveit_msgs/msg/joint_constraint.hpp>
#include <moveit_msgs/msg/orientation_constraint.hpp>

using MoveGroupInterface = moveit::planning_interface::MoveGroupInterface;
using PointStamped       = geometry_msgs::msg::PointStamped;


class FeedingSafeMover : public rclcpp::Node
{
public:
    explicit FeedingSafeMover()
        : Node("feeding_safe_mover"), busy_(false)
    {
        declare_parameter("base_frame",               "root");
        declare_parameter("ee_link",                  "j2s6s200_end_effector");
        declare_parameter("joint_5",                  "j2s6s200_joint_5");
        declare_parameter("joint_6",                  "j2s6s200_joint_6");
        declare_parameter("offset_x",                 0.0);
        declare_parameter("offset_y",                 0.0);
        declare_parameter("offset_z",                 0.0);
        declare_parameter("ori_tol_x",                0.05);
        declare_parameter("ori_tol_y",                0.05);
        declare_parameter("ori_tol_z",                0.20);
        declare_parameter("wrist_tol",                0.08);
        declare_parameter("planning_time",            10.0);
        declare_parameter("velocity_scale",           0.15);
        declare_parameter("acceleration_scale",       0.15);
        declare_parameter("enable_joint_constraints", false);

        base_frame_               = get_parameter("base_frame").as_string();
        ee_link_                  = get_parameter("ee_link").as_string();
        joint_5_                  = get_parameter("joint_5").as_string();
        joint_6_                  = get_parameter("joint_6").as_string();
        ori_tol_x_                = get_parameter("ori_tol_x").as_double();
        ori_tol_y_                = get_parameter("ori_tol_y").as_double();
        ori_tol_z_                = get_parameter("ori_tol_z").as_double();
        wrist_tol_                = get_parameter("wrist_tol").as_double();
        planning_time_            = get_parameter("planning_time").as_double();
        velocity_scale_           = get_parameter("velocity_scale").as_double();
        acceleration_scale_       = get_parameter("acceleration_scale").as_double();
        enable_joint_constraints_ = get_parameter("enable_joint_constraints").as_bool();

        // MutuallyExclusive ensures no two goal callbacks interleave.
        auto cb_group = create_callback_group(
            rclcpp::CallbackGroupType::MutuallyExclusive);
        rclcpp::SubscriptionOptions sub_opts;
        sub_opts.callback_group = cb_group;

        goal_sub_ = create_subscription<PointStamped>(
            "/goal_point", 10,
            [this](PointStamped::SharedPtr msg) { goal_cb(std::move(msg)); },
            sub_opts);

        RCLCPP_INFO(get_logger(),
            "FeedingSafeMover waiting for /goal_point  "
            "ee=%s  base=%s  planning_time=%.1fs  vel=%.2f  acc=%.2f",
            ee_link_.c_str(), base_frame_.c_str(),
            planning_time_, velocity_scale_, acceleration_scale_);
    }

    // Must be called after the executor is already spinning.
    void init_move_group()
    {
        move_group_ = std::make_unique<MoveGroupInterface>(
            shared_from_this(), "arm");

        move_group_->setPlanningTime(planning_time_);
        move_group_->setMaxVelocityScalingFactor(velocity_scale_);
        move_group_->setMaxAccelerationScalingFactor(acceleration_scale_);
        move_group_->setPlannerId("RRTConnectkConfigDefault");
        move_group_->setEndEffectorLink(ee_link_);
        move_group_->setPoseReferenceFrame(base_frame_);

        RCLCPP_INFO(get_logger(),
            "MoveGroupInterface ready  group=arm  planner=RRTConnectkConfigDefault");
    }

private:
    // ------------------------------------------------------------------
    // Subscription callback
    // ------------------------------------------------------------------
    void goal_cb(PointStamped::SharedPtr msg)
    {
        if (busy_.exchange(true)) {
            RCLCPP_WARN(get_logger(),
                "Previous goal still executing — new /goal_point ignored");
            return;
        }
        execute_goal(std::move(msg));
        busy_ = false;
    }

    // ------------------------------------------------------------------
    // Planning and execution
    // ------------------------------------------------------------------
    void execute_goal(PointStamped::SharedPtr msg)
    {
        if (!move_group_) {
            RCLCPP_ERROR(get_logger(),
                "MoveGroupInterface not yet initialized — ignoring goal");
            return;
        }

        // Offsets re-read at runtime so they can be patched via ros2 param set.
        const double ox = get_parameter("offset_x").as_double();
        const double oy = get_parameter("offset_y").as_double();
        const double oz = get_parameter("offset_z").as_double();
        const double gx = msg->point.x + ox;
        const double gy = msg->point.y + oy;
        const double gz = msg->point.z + oz;

        RCLCPP_INFO(get_logger(),
            "Goal received: raw=(%.3f, %.3f, %.3f) + offset -> (%.3f, %.3f, %.3f)",
            msg->point.x, msg->point.y, msg->point.z, gx, gy, gz);

        // --- Step 1: current robot state ---
        move_group_->setStartStateToCurrentState();
        auto current_state = move_group_->getCurrentState(/*timeout_s=*/2.0);
        if (!current_state) {
            RCLCPP_ERROR(get_logger(),
                "getCurrentState() timed out — robot state unavailable, aborting");
            return;
        }

        // --- Step 2: lock the current EE orientation (the spoon must stay level) ---
        geometry_msgs::msg::PoseStamped ee_stamped;
        try {
            ee_stamped = move_group_->getCurrentPose(ee_link_);
        } catch (const std::exception& e) {
            RCLCPP_ERROR(get_logger(),
                "getCurrentPose failed: %s — aborting", e.what());
            return;
        }

        const auto& locked_q = ee_stamped.pose.orientation;
        const double qnorm = std::sqrt(
            locked_q.x * locked_q.x + locked_q.y * locked_q.y +
            locked_q.z * locked_q.z + locked_q.w * locked_q.w);
        if (qnorm < 0.9 || qnorm > 1.1) {
            RCLCPP_ERROR(get_logger(),
                "EE orientation quaternion is invalid (norm=%.4f) — aborting. "
                "Check that TF is publishing and the robot is in a valid state.",
                qnorm);
            return;
        }

        RCLCPP_INFO(get_logger(),
            "Spoon orientation locked: q=(%.4f, %.4f, %.4f, %.4f)",
            locked_q.x, locked_q.y, locked_q.z, locked_q.w);

        // --- Step 3: snapshot initial wrist positions for post-plan safety check ---
        double init_j5 = 0.0, init_j6 = 0.0;
        bool   j5_ok   = false, j6_ok   = false;
        try {
            init_j5 = current_state->getVariablePosition(joint_5_);
            j5_ok   = true;
        } catch (...) {
            RCLCPP_WARN(get_logger(),
                "Joint '%s' not found in robot state; wrist-j5 check skipped",
                joint_5_.c_str());
        }
        try {
            init_j6 = current_state->getVariablePosition(joint_6_);
            j6_ok   = true;
        } catch (...) {
            RCLCPP_WARN(get_logger(),
                "Joint '%s' not found in robot state; wrist-j6 check skipped",
                joint_6_.c_str());
        }

        // --- Step 4: target pose (goal XYZ + LOCKED orientation) ---
        geometry_msgs::msg::Pose target;
        target.position.x  = gx;
        target.position.y  = gy;
        target.position.z  = gz;
        target.orientation = locked_q;   // orientation is not free — it is constrained
        move_group_->setPoseTarget(target, ee_link_);

        // --- Step 5: orientation PATH constraint (enforced on every waypoint) ---
        moveit_msgs::msg::OrientationConstraint oc;
        oc.header.frame_id           = base_frame_;
        oc.link_name                 = ee_link_;
        oc.orientation               = locked_q;
        oc.absolute_x_axis_tolerance = ori_tol_x_;
        oc.absolute_y_axis_tolerance = ori_tol_y_;
        oc.absolute_z_axis_tolerance = ori_tol_z_;
        oc.weight                    = 1.0;

        moveit_msgs::msg::Constraints path_constraints;
        path_constraints.name = "spoon_orientation_lock";
        path_constraints.orientation_constraints.push_back(oc);

        RCLCPP_INFO(get_logger(),
            "Orientation path constraint active: "
            "tol_x=%.3f  tol_y=%.3f  tol_z=%.3f rad  (on every waypoint)",
            ori_tol_x_, ori_tol_y_, ori_tol_z_);

        // --- Step 6: optional wrist joint constraints (disabled by default) ---
        if (enable_joint_constraints_ && j5_ok && j6_ok) {
            moveit_msgs::msg::JointConstraint jc5, jc6;

            jc5.joint_name      = joint_5_;
            jc5.position        = init_j5;
            jc5.tolerance_above = wrist_tol_;
            jc5.tolerance_below = wrist_tol_;
            jc5.weight          = 1.0;

            jc6.joint_name      = joint_6_;
            jc6.position        = init_j6;
            jc6.tolerance_above = wrist_tol_;
            jc6.tolerance_below = wrist_tol_;
            jc6.weight          = 1.0;

            path_constraints.joint_constraints.push_back(jc5);
            path_constraints.joint_constraints.push_back(jc6);

            RCLCPP_INFO(get_logger(),
                "Joint constraints: j5=%.3f±%.3f  j6=%.3f±%.3f rad",
                init_j5, wrist_tol_, init_j6, wrist_tol_);
        }

        move_group_->setPathConstraints(path_constraints);

        // --- Step 7: plan ---
        RCLCPP_INFO(get_logger(), "Planning with RRTConnectkConfigDefault ...");
        MoveGroupInterface::Plan plan;
        const bool plan_ok = static_cast<bool>(move_group_->plan(plan));

        // Clear constraints and targets regardless of outcome.
        move_group_->clearPathConstraints();
        move_group_->clearPoseTargets();

        if (!plan_ok) {
            RCLCPP_ERROR(get_logger(),
                "Planning FAILED with active orientation constraint. "
                "The goal may be unreachable while preserving spoon orientation. "
                "NOT executing. (Position-only fallback is intentionally absent.)");
            return;
        }

        RCLCPP_INFO(get_logger(),
            "Planning succeeded (%zu waypoints). Running safety validation ...",
            plan.trajectory_.joint_trajectory.points.size());

        // --- Step 8: validate wrist deviation in every trajectory waypoint ---
        if (!validate_trajectory(plan, init_j5, init_j6, j5_ok, j6_ok)) {
            RCLCPP_ERROR(get_logger(),
                "Trajectory REJECTED by wrist safety validator — NOT executing");
            return;
        }

        // --- Step 9: execute ---
        RCLCPP_INFO(get_logger(),
            "All safety checks passed. Executing at vel=%.0f%%  acc=%.0f%% ...",
            velocity_scale_ * 100.0, acceleration_scale_ * 100.0);

        const bool exec_ok = static_cast<bool>(move_group_->execute(plan));
        if (!exec_ok) {
            RCLCPP_ERROR(get_logger(), "Execution failed");
        } else {
            RCLCPP_INFO(get_logger(),
                "Motion complete — spoon at (%.3f, %.3f, %.3f)", gx, gy, gz);
        }
    }

    // ------------------------------------------------------------------
    // Wrist safety validator
    // Rejects the plan if joint_5 or joint_6 deviate more than
    // wrist_tol * 1.5 from their initial positions at ANY waypoint.
    // ------------------------------------------------------------------
    bool validate_trajectory(const MoveGroupInterface::Plan& plan,
                              double init_j5, double init_j6,
                              bool j5_ok, bool j6_ok) const
    {
        const auto& traj = plan.trajectory_.joint_trajectory;

        if (traj.points.empty()) {
            RCLCPP_ERROR(get_logger(), "[Validate] Trajectory contains no waypoints");
            return false;
        }

        int idx5 = -1, idx6 = -1;
        for (std::size_t i = 0; i < traj.joint_names.size(); ++i) {
            if (traj.joint_names[i] == joint_5_) idx5 = static_cast<int>(i);
            if (traj.joint_names[i] == joint_6_) idx6 = static_cast<int>(i);
        }

        if (j5_ok && idx5 < 0) {
            RCLCPP_ERROR(get_logger(),
                "[Validate] Joint '%s' missing from trajectory joint list — rejecting",
                joint_5_.c_str());
            return false;
        }
        if (j6_ok && idx6 < 0) {
            RCLCPP_ERROR(get_logger(),
                "[Validate] Joint '%s' missing from trajectory joint list — rejecting",
                joint_6_.c_str());
            return false;
        }

        const double max_dev = wrist_tol_ * 1.5;
        double worst5 = 0.0, worst6 = 0.0;

        for (std::size_t pt = 0; pt < traj.points.size(); ++pt) {
            const auto& pos = traj.points[pt].positions;
            if (j5_ok && idx5 >= 0) {
                const double dev = std::abs(pos[static_cast<std::size_t>(idx5)] - init_j5);
                if (dev > worst5) worst5 = dev;
                if (dev > max_dev) {
                    RCLCPP_ERROR(get_logger(),
                        "[Validate] j5 deviation %.4f rad exceeds limit %.4f rad "
                        "at waypoint %zu — REJECTED",
                        dev, max_dev, pt);
                    return false;
                }
            }
            if (j6_ok && idx6 >= 0) {
                const double dev = std::abs(pos[static_cast<std::size_t>(idx6)] - init_j6);
                if (dev > worst6) worst6 = dev;
                if (dev > max_dev) {
                    RCLCPP_ERROR(get_logger(),
                        "[Validate] j6 deviation %.4f rad exceeds limit %.4f rad "
                        "at waypoint %zu — REJECTED",
                        dev, max_dev, pt);
                    return false;
                }
            }
        }

        RCLCPP_INFO(get_logger(),
            "[Validate] PASSED — %zu waypoints  "
            "worst_j5=%.4f rad  worst_j6=%.4f rad  (limit=%.4f rad)",
            traj.points.size(), worst5, worst6, max_dev);
        return true;
    }

    // ---- members ----
    std::atomic<bool>                             busy_;
    std::unique_ptr<MoveGroupInterface>           move_group_;
    rclcpp::Subscription<PointStamped>::SharedPtr goal_sub_;

    std::string base_frame_;
    std::string ee_link_;
    std::string joint_5_;
    std::string joint_6_;
    double      ori_tol_x_;
    double      ori_tol_y_;
    double      ori_tol_z_;
    double      wrist_tol_;
    double      planning_time_;
    double      velocity_scale_;
    double      acceleration_scale_;
    bool        enable_joint_constraints_;
};


int main(int argc, char* argv[])
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<FeedingSafeMover>();

    // MultiThreadedExecutor is required: the goal callback blocks the thread
    // during move_group_->plan() / execute().  Other threads in the pool handle
    // the incoming action and service responses that planning depends on.
    rclcpp::executors::MultiThreadedExecutor exec(rclcpp::ExecutorOptions(), 4);
    exec.add_node(node);

    // Spin in a background thread first so that MoveGroupInterface can
    // connect to the move_group action server during init_move_group().
    std::thread spin_thread([&exec]() { exec.spin(); });

    node->init_move_group();

    spin_thread.join();
    rclcpp::shutdown();
    return 0;
}
