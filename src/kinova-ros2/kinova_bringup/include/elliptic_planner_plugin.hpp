#pragma once
// elliptic_planner_plugin.hpp

#include <moveit/planning_interface/planning_interface.h>
#include <moveit/planning_scene/planning_scene.h>
#include <moveit/robot_state/robot_state.h>
#include <moveit/robot_state/conversions.h>
#include <moveit/robot_trajectory/robot_trajectory.h>
#include <rclcpp/rclcpp.hpp>
#include <Eigen/Geometry>

namespace elliptic_planner
{

// ── Context: one instance per plan request ────────────────────────────────────

class EllipticPlanningContext : public planning_interface::PlanningContext
{
public:
  EllipticPlanningContext(
      const std::string & group_name,
      const planning_scene::PlanningSceneConstPtr & scene,
      const rclcpp::Logger & logger,
      double lift_height,
      int    n_waypoints,
      double total_time);

  bool solve(planning_interface::MotionPlanResponse & res) override;
  bool solve(planning_interface::MotionPlanDetailedResponse & res) override;
  bool terminate() override { return true; }
  void clear() override {}

private:
  std::vector<Eigen::Vector3d> buildArc(
      const Eigen::Vector3d & A,
      const Eigen::Vector3d & B) const;

  std::vector<double> minJerkTimes() const;

  bool ikPosition(
      const moveit::core::RobotState & seed,
      const Eigen::Vector3d & target,
      const std::string & ee_link,
      const moveit::core::JointModelGroup * jmg,
      moveit::core::RobotState & result_state,
      int    max_iter = 500,
      double tol      = 3e-3,
      double damping  = 1e-3) const;

  planning_scene::PlanningSceneConstPtr scene_;
  rclcpp::Logger logger_;
  double lift_height_;
  int    n_waypoints_;
  double total_time_;
};

// ── Manager: registered with pluginlib ───────────────────────────────────────

class EllipticPlannerManager : public planning_interface::PlannerManager
{
public:
  EllipticPlannerManager() = default;

  bool initialize(
      const moveit::core::RobotModelConstPtr & model,
      const rclcpp::Node::SharedPtr & node,
      const std::string & parameter_namespace) override;

  std::string getDescription() const override
  { return "EllipticPlanner — minimum-jerk arc for assistive feeding"; }

  void getPlanningAlgorithms(std::vector<std::string> & algs) const override
  { algs = { "EllipticArc" }; }

  // Pure virtual in PlannerManager — must implement or won't compile
  void setPlannerConfigurations(
      const planning_interface::PlannerConfigurationMap & pcs) override
  { config_map_ = pcs; }

  planning_interface::PlanningContextPtr getPlanningContext(
      const planning_scene::PlanningSceneConstPtr & scene,
      const planning_interface::MotionPlanRequest & req,
      moveit_msgs::msg::MoveItErrorCodes & error_code) const override;

  bool canServiceRequest(
      const planning_interface::MotionPlanRequest & /*req*/) const override
  { return true; }

private:
  moveit::core::RobotModelConstPtr robot_model_;
  rclcpp::Node::SharedPtr node_;
  planning_interface::PlannerConfigurationMap config_map_;
  double lift_height_ { 0.10 };
  int    n_waypoints_ { 40   };
  double total_time_  { 4.0  };
};

}  // namespace elliptic_planner