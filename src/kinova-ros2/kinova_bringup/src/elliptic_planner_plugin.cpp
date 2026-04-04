// elliptic_planner_plugin.cpp

#include "elliptic_planner_plugin.hpp"
#include <moveit/robot_state/conversions.h>
#include <moveit/robot_model/robot_model.h>
#include <pluginlib/class_list_macros.hpp>
#include <cmath>

PLUGINLIB_EXPORT_CLASS(
    elliptic_planner::EllipticPlannerManager,
    planning_interface::PlannerManager)

namespace elliptic_planner
{

EllipticPlanningContext::EllipticPlanningContext(
    const std::string & group_name,
    const planning_scene::PlanningSceneConstPtr & scene,
    const rclcpp::Logger & logger,
    double lift_height,
    int    n_waypoints,
    double total_time)
: planning_interface::PlanningContext(group_name, group_name)
, scene_(scene)
, logger_(logger)
, lift_height_(lift_height)
, n_waypoints_(n_waypoints)
, total_time_(total_time)
{}

// ── Arc ───────────────────────────────────────────────────────────────────────

std::vector<Eigen::Vector3d>
EllipticPlanningContext::buildArc(
    const Eigen::Vector3d & A,
    const Eigen::Vector3d & B) const
{
  Eigen::Vector3d AB   = B - A;
  double dist          = AB.norm();
  if (dist < 0.02)
    RCLCPP_WARN(logger_, "[EllipticPlanner] A-B only %.1fcm", dist*100.0);

  Eigen::Vector3d u_AB = AB.normalized();
  Eigen::Vector3d wz(0, 0, 1);
  Eigen::Vector3d v_raw = wz - wz.dot(u_AB) * u_AB;
  if (v_raw.norm() < 1e-6) {
    Eigen::Vector3d wx(1, 0, 0);
    v_raw = wx - wx.dot(u_AB) * u_AB;
  }
  Eigen::Vector3d u_lift = v_raw.normalized();
  Eigen::Vector3d center = (A + B) * 0.5;

  std::vector<Eigen::Vector3d> arc;
  arc.reserve(n_waypoints_);
  for (int i = 0; i < n_waypoints_; ++i) {
    double s = M_PI * (1.0 - static_cast<double>(i) / (n_waypoints_ - 1));
    arc.push_back(center + (dist*0.5)*std::cos(s)*u_AB
                         + lift_height_*std::sin(s)*u_lift);
  }
  return arc;
}

// ── Timing ────────────────────────────────────────────────────────────────────

std::vector<double> EllipticPlanningContext::minJerkTimes() const
{
  std::vector<double> t(n_waypoints_);
  for (int i = 0; i < n_waypoints_; ++i) {
    double tau = static_cast<double>(i) / (n_waypoints_ - 1);
    t[i] = (10*std::pow(tau,3) - 15*std::pow(tau,4) + 6*std::pow(tau,5)) * total_time_;
  }
  return t;
}

// ── IK ────────────────────────────────────────────────────────────────────────

bool EllipticPlanningContext::ikPosition(
    const moveit::core::RobotState & seed,
    const Eigen::Vector3d & target,
    const std::string & ee_link,
    const moveit::core::JointModelGroup * jmg,
    moveit::core::RobotState & result_state,
    int max_iter, double tol, double damping) const
{
  result_state = seed;
  for (int iter = 0; iter < max_iter; ++iter) {
    result_state.updateLinkTransforms();
    Eigen::Vector3d err =
        target - result_state.getGlobalLinkTransform(ee_link).translation();
    if (err.norm() < tol) return true;

    Eigen::MatrixXd J;
    result_state.getJacobian(jmg,
        result_state.getLinkModel(ee_link),
        Eigen::Vector3d::Zero(), J);
    Eigen::MatrixXd Jp  = J.topRows(3);
    Eigen::Matrix3d A   = Jp*Jp.transpose() + damping*Eigen::Matrix3d::Identity();
    Eigen::VectorXd dq  = Jp.transpose() * A.ldlt().solve(err);
    dq = dq.cwiseMax(-0.5).cwiseMin(0.5);

    std::vector<double> pos;
    result_state.copyJointGroupPositions(jmg, pos);
    for (size_t j = 0; j < pos.size() && j < (size_t)dq.size(); ++j)
      pos[j] += dq[j];
    result_state.setJointGroupPositions(jmg, pos);
    result_state.enforceBounds(jmg);
  }
  return false;
}

// ── solve ─────────────────────────────────────────────────────────────────────

bool EllipticPlanningContext::solve(
    planning_interface::MotionPlanResponse & res)
{
  const auto & req         = getMotionPlanRequest();
  const auto & robot_model = scene_->getRobotModel();
  const std::string & group = getGroupName();

  const moveit::core::JointModelGroup * jmg =
      robot_model->getJointModelGroup(group);
  if (!jmg) {
    RCLCPP_ERROR(logger_, "[EllipticPlanner] Group '%s' not found", group.c_str());
    res.error_code_.val = moveit_msgs::msg::MoveItErrorCodes::INVALID_GROUP_NAME;
    return false;
  }

  // ── Debug: log exactly what goal_constraints contains ────────────────────
  RCLCPP_INFO(logger_, "[EllipticPlanner] goal_constraints size: %zu",
              req.goal_constraints.size());
  for (size_t gi = 0; gi < req.goal_constraints.size(); ++gi) {
    const auto & gc = req.goal_constraints[gi];
    RCLCPP_INFO(logger_,
        "[EllipticPlanner]   gc[%zu]: position=%zu  orientation=%zu  joint=%zu",
        gi,
        gc.position_constraints.size(),
        gc.orientation_constraints.size(),
        gc.joint_constraints.size());
    for (size_t ji = 0; ji < gc.joint_constraints.size(); ++ji)
      RCLCPP_INFO(logger_, "[EllipticPlanner]     joint[%zu]: %s = %.4f",
                  ji,
                  gc.joint_constraints[ji].joint_name.c_str(),
                  gc.joint_constraints[ji].position);
    for (size_t pi = 0; pi < gc.position_constraints.size(); ++pi) {
      const auto & pc = gc.position_constraints[pi];
      RCLCPP_INFO(logger_,
          "[EllipticPlanner]     pos_constraint[%zu]: link=%s  "
          "primitive_poses=%zu",
          pi, pc.link_name.c_str(),
          pc.constraint_region.primitive_poses.size());
      if (!pc.constraint_region.primitive_poses.empty()) {
        const auto & p = pc.constraint_region.primitive_poses[0].position;
        RCLCPP_INFO(logger_,
            "[EllipticPlanner]       target=(%.3f, %.3f, %.3f)", p.x, p.y, p.z);
      }
    }
  }

  // ── Start state ──────────────────────────────────────────────────────────
  // Use scene's current state if request start_state is empty
  moveit::core::RobotState start_state = scene_->getCurrentState();
  if (!req.start_state.joint_state.name.empty()) {
    moveit::core::robotStateMsgToRobotState(req.start_state, start_state);
  } else {
    RCLCPP_WARN(logger_,
        "[EllipticPlanner] Empty start_state in request — "
        "using scene current state");
  }
  start_state.updateLinkTransforms();

  // ── EE link ──────────────────────────────────────────────────────────────
  const std::string & ee_link = jmg->getLinkModelNames().back();
  RCLCPP_INFO(logger_, "[EllipticPlanner] group=%s  ee=%s",
              group.c_str(), ee_link.c_str());

  // ── Fork position A (start EE position) ──────────────────────────────────
  Eigen::Vector3d A =
      start_state.getGlobalLinkTransform(ee_link).translation();

  // ── Mouth position B — handle ALL goal formats RViz may send ─────────────
  //
  // RViz sends goals in one of three formats depending on how the goal
  // was specified in the MotionPlanning panel:
  //
  //   Format 1: position_constraints  — "Goal State" set via pose marker
  //             target = constraint_region.primitive_poses[0].position
  //
  //   Format 2: joint_constraints     — "Goal State" set via joint sliders
  //             or after RViz runs IK on a pose goal
  //             run FK on these to get EE position B
  //
  //   Format 3: orientation_constraints only (rare, ignore for position IK)
  //             fall through to error

  Eigen::Vector3d B = Eigen::Vector3d::Zero();
  bool goal_found   = false;

  if (!req.goal_constraints.empty()) {
    const auto & gc = req.goal_constraints[0];

    // Format 1: position constraint
    if (!gc.position_constraints.empty()) {
      const auto & pc = gc.position_constraints[0];
      if (!pc.constraint_region.primitive_poses.empty()) {
        const auto & p = pc.constraint_region.primitive_poses[0].position;
        B           = Eigen::Vector3d(p.x, p.y, p.z);
        goal_found  = true;
        RCLCPP_INFO(logger_,
            "[EllipticPlanner] Goal from position_constraint: "
            "(%.3f, %.3f, %.3f)", B.x(), B.y(), B.z());
      }
    }

    // Format 2: joint constraints → FK
    if (!goal_found && !gc.joint_constraints.empty()) {
      moveit::core::RobotState goal_state(robot_model);
      goal_state = start_state;   // inherit current state, override goal joints
      for (const auto & jc : gc.joint_constraints)
        goal_state.setJointPositions(jc.joint_name, &jc.position);
      goal_state.updateLinkTransforms();
      B          = goal_state.getGlobalLinkTransform(ee_link).translation();
      goal_found = true;
      RCLCPP_INFO(logger_,
          "[EllipticPlanner] Goal from joint_constraints FK: "
          "(%.3f, %.3f, %.3f)", B.x(), B.y(), B.z());
    }
  }

  if (!goal_found) {
    RCLCPP_ERROR(logger_,
        "[EllipticPlanner] No usable goal in request.\n"
        "  In RViz: drag the interactive marker to the mouth position,\n"
        "  then click Plan. The goal must produce either position_constraints\n"
        "  or joint_constraints in the MotionPlanRequest.");
    res.error_code_.val =
        moveit_msgs::msg::MoveItErrorCodes::INVALID_GOAL_CONSTRAINTS;
    return false;
  }

  RCLCPP_INFO(logger_,
      "[EllipticPlanner] Fork(%.3f,%.3f,%.3f) → Mouth(%.3f,%.3f,%.3f)  "
      "dist=%.1fcm",
      A.x(), A.y(), A.z(), B.x(), B.y(), B.z(), (B-A).norm()*100.0);

  // ── Arc + IK ──────────────────────────────────────────────────────────────
  auto arc        = buildArc(A, B);
  auto timestamps = minJerkTimes();

  auto trajectory = std::make_shared<robot_trajectory::RobotTrajectory>(
      robot_model, group);

  moveit::core::RobotState q = start_state;
  int n_ok = 0;

  for (int i = 0; i < n_waypoints_; ++i) {
    moveit::core::RobotState q_sol = q;
    if (ikPosition(q, arc[i], ee_link, jmg, q_sol))
      { q = q_sol; ++n_ok; }
    else
      { q = q_sol; }
    trajectory->addSuffixWayPoint(q, 0.0);
  }

  for (int i = 1; i < n_waypoints_; ++i)
    trajectory->setWayPointDurationFromPrevious(
        i, std::max(timestamps[i] - timestamps[i-1], 0.001));

  double apex_z = 0.0;
  for (auto & p : arc) apex_z = std::max(apex_z, p.z());

  RCLCPP_INFO(logger_,
      "[EllipticPlanner] IK %d/%d  apex_z=%.3fm  duration=%.1fs",
      n_ok, n_waypoints_, apex_z, timestamps.back());

  res.trajectory_     = trajectory;
  res.planning_time_  = 0.0;
  res.error_code_.val = moveit_msgs::msg::MoveItErrorCodes::SUCCESS;
  return true;
}

bool EllipticPlanningContext::solve(
    planning_interface::MotionPlanDetailedResponse & res)
{
  planning_interface::MotionPlanResponse r;
  bool ok = solve(r);
  if (ok) {
    res.trajectory_.push_back(r.trajectory_);
    res.description_.push_back("elliptic_arc");
    res.processing_time_.push_back(r.planning_time_);
  }
  res.error_code_ = r.error_code_;
  return ok;
}

// ── EllipticPlannerManager ────────────────────────────────────────────────────

bool EllipticPlannerManager::initialize(
    const moveit::core::RobotModelConstPtr & model,
    const rclcpp::Node::SharedPtr & node,
    const std::string & parameter_namespace)
{
  robot_model_ = model;
  node_        = node;

  auto ns = [&](const std::string & p) {
    return parameter_namespace.empty() ? p : parameter_namespace + "." + p;
  };

  if (!node_->has_parameter(ns("lift_height")))
    node_->declare_parameter(ns("lift_height"), 0.10);
  if (!node_->has_parameter(ns("n_waypoints")))
    node_->declare_parameter(ns("n_waypoints"), 40);
  if (!node_->has_parameter(ns("total_time")))
    node_->declare_parameter(ns("total_time"), 4.0);

  lift_height_ = node_->get_parameter(ns("lift_height")).as_double();
  n_waypoints_ = node_->get_parameter(ns("n_waypoints")).as_int();
  total_time_  = node_->get_parameter(ns("total_time")).as_double();

  RCLCPP_INFO(node_->get_logger(),
      "[EllipticPlanner] Initialized  lift=%.0fcm  n=%d  T=%.1fs",
      lift_height_*100.0, n_waypoints_, total_time_);
  return true;
}

planning_interface::PlanningContextPtr
EllipticPlannerManager::getPlanningContext(
    const planning_scene::PlanningSceneConstPtr & scene,
    const planning_interface::MotionPlanRequest & req,
    moveit_msgs::msg::MoveItErrorCodes & error_code) const
{
  error_code.val = moveit_msgs::msg::MoveItErrorCodes::SUCCESS;
  return std::make_shared<EllipticPlanningContext>(
      req.group_name, scene, node_->get_logger(),
      lift_height_, n_waypoints_, total_time_);
}

}  // namespace elliptic_planner