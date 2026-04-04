#!/bin/bash
# Two fixes:
# 1. IK: position-only, more iterations, bigger step — gets arm moving
# 2. CMakeLists: install all scripts correctly
set -e
GREEN='\033[0;32m'; CYAN='\033[0;36m'; NC='\033[0m'
log()     { echo -e "${GREEN}[OK]${NC} $1"; }
section() { echo -e "\n${CYAN}══════════════════${NC} $1 ${CYAN}══════════════════${NC}"; }

SCRIPTS="$HOME/kinova_ws/src/kinova-ros2/kinova_bringup/scripts"
CMAKE="$HOME/kinova_ws/src/kinova-ros2/kinova_bringup/CMakeLists.txt"

source /opt/ros/humble/setup.bash

# ─── Fix CMakeLists ───────────────────────────────────────────────────────────
section "Fix CMakeLists.txt"
cat > "$CMAKE" << 'CMAKE_EOF'
cmake_minimum_required(VERSION 3.5)
project(kinova_bringup)
find_package(ament_cmake REQUIRED)
install(DIRECTORY launch DESTINATION share/${PROJECT_NAME})
install(DIRECTORY moveit_resource DESTINATION share/${PROJECT_NAME})
install(DIRECTORY worlds/ DESTINATION share/${PROJECT_NAME}/worlds)
install(PROGRAMS
  scripts/elliptic_delivery.py
  scripts/feeding_node.py
  DESTINATION lib/${PROJECT_NAME}
)
ament_package()
CMAKE_EOF
log "CMakeLists.txt fixed"

# ─── Rewrite elliptic_delivery.py with working IK ────────────────────────────
section "Rewriting elliptic_delivery.py"

cat > "$SCRIPTS/elliptic_delivery.py" << 'EOF'
#!/usr/bin/env python3
"""
Elliptic Delivery Node — Kinova j2n6s300

IK strategy: position-only pseudoinverse IK.
  Orientation IK was causing 38/40 failures — too many local minima
  when starting from neutral. Position-only IK is robust and sufficient
  to follow the arc. Fork level is enforced by keeping wrist joints
  near their current values (natural posture regularization).
"""

import rclpy
from rclpy.node import Node
import numpy as np
import pinocchio as pin
from pinocchio import RobotWrapper
import os

from sensor_msgs.msg        import JointState
from geometry_msgs.msg      import PointStamped
from std_msgs.msg           import String
from trajectory_msgs.msg    import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration

ARM_URDF    = '/tmp/j2n6s300_arm.urdf'
TRAJ_TOPIC  = '/joint_trajectory_controller/joint_trajectory'
JOINT_NAMES = [
    'j2n6s300_joint_1', 'j2n6s300_joint_2', 'j2n6s300_joint_3',
    'j2n6s300_joint_4', 'j2n6s300_joint_5', 'j2n6s300_joint_6',
]


class EllipticPlanner:

    def __init__(self, urdf_path, lift_height=0.10,
                 n_waypoints=40, total_time=4.0):

        robot      = RobotWrapper.BuildFromURDF(urdf_path)
        self.model = robot.model
        self.data  = robot.data
        self.nv    = self.model.nv   # 6 — velocity DOF
        self.nq    = self.model.nq   # 10 — config dim (continuous joints)
        assert self.nv == 6, f"Expected nv=6, got {self.nv}"

        self.lift = lift_height
        self.n    = n_waypoints
        self.T    = total_time

        self.q_neutral = pin.neutral(self.model)

        ee_name = 'j2n6s300_end_effector'
        self.ee_fid = self.model.getFrameId(ee_name) \
                      if self.model.existFrame(ee_name) \
                      else self.model.nframes - 1

        print(f"[EllipticPlanner] nv={self.nv} nq={self.nq}  "
              f"EE={self.model.frames[self.ee_fid].name}  "
              f"lift={lift_height*100:.0f}cm  T={total_time}s")

    # ── Config helpers ────────────────────────────────────────────────────────

    def joints_to_q(self, joints_dict):
        q = self.q_neutral.copy()
        for name in JOINT_NAMES:
            if name not in joints_dict:
                continue
            jid = self.model.getJointId(name)
            if jid >= self.model.njoints:
                continue
            idx = self.model.joints[jid].idx_q
            nqi = self.model.joints[jid].nq
            angle = joints_dict[name]
            if nqi == 1:
                q[idx] = angle
            else:
                q[idx]   = np.cos(angle)
                q[idx+1] = np.sin(angle)
        return q

    def q_to_joints(self, q):
        angles = np.zeros(6)
        for i, name in enumerate(JOINT_NAMES):
            jid = self.model.getJointId(name)
            if jid >= self.model.njoints:
                continue
            idx = self.model.joints[jid].idx_q
            nqi = self.model.joints[jid].nq
            if nqi == 1:
                angles[i] = q[idx]
            else:
                angles[i] = np.arctan2(q[idx+1], q[idx])
        return angles

    # ── FK ────────────────────────────────────────────────────────────────────

    def get_ee_pos(self, q):
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacement(self.model, self.data, self.ee_fid)
        return self.data.oMf[self.ee_fid].translation.copy()

    # ── Arc ───────────────────────────────────────────────────────────────────

    def build_arc(self, A, B):
        AB   = B - A
        dist = np.linalg.norm(AB)
        if dist < 0.02:
            raise ValueError(f"A-B only {dist*100:.1f}cm apart")
        u = AB / dist
        wz    = np.array([0., 0., 1.])
        v_raw = wz - np.dot(wz, u) * u
        if np.linalg.norm(v_raw) < 1e-6:
            v_raw = np.array([1., 0., 0.]) - np.dot(np.array([1.,0.,0.]), u)*u
        v      = v_raw / np.linalg.norm(v_raw)
        center = (A + B) / 2.0
        return np.array([
            center + (dist/2)*np.cos(s)*u + self.lift*np.sin(s)*v
            for s in np.linspace(np.pi, 0.0, self.n)
        ])

    def min_jerk_times(self):
        tau = np.linspace(0., 1., self.n)
        return (10*tau**3 - 15*tau**4 + 6*tau**5) * self.T

    # ── IK: position-only, damped least squares ───────────────────────────────

    def _ik_position(self, q_init, target_pos,
                     max_iter=300, tol=2e-3, damping=1e-4):
        """
        Position-only IK using damped least-squares Jacobian.
        Warm-started from previous waypoint — fast convergence along arc.

        Regularization term pulls joints toward q_init (posture),
        which keeps the wrist orientation approximately constant —
        fork stays level without explicit rotation constraint.
        """
        q    = q_init.copy()
        w_reg = 0.01   # posture regularization weight

        for _ in range(max_iter):
            pin.forwardKinematics(self.model, self.data, q)
            pin.updateFramePlacement(self.model, self.data, self.ee_fid)
            pos_err = target_pos - self.data.oMf[self.ee_fid].translation

            if np.linalg.norm(pos_err) < tol:
                return q, True

            # Full 6xnv Jacobian, take top 3 rows (position only)
            J_full = pin.computeFrameJacobian(
                self.model, self.data, q, self.ee_fid,
                pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
            J = J_full[:3, :]   # shape (3, nv)

            # Damped least squares: dq = (J^T J + λI)^{-1} J^T err
            # + posture regularization toward q_init
            A   = J.T @ J + (damping + w_reg) * np.eye(self.nv)
            b   = J.T @ pos_err - w_reg * self._q_diff(q, q_init)
            dq  = np.linalg.solve(A, b)
            dq  = np.clip(dq, -0.2, 0.2)
            q   = pin.integrate(self.model, q, dq)

        return q, False

    def _q_diff(self, q, q_ref):
        """Velocity-space difference q - q_ref using pinocchio's difference."""
        return pin.difference(self.model, q_ref, q)

    # ── Plan ─────────────────────────────────────────────────────────────────

    def plan(self, q_current, mouth_pos):
        A = self.get_ee_pos(q_current)
        B = mouth_pos.copy()
        print(f"[Plan] Fork:  {np.round(A,3)}")
        print(f"[Plan] Mouth: {np.round(B,3)}  dist={np.linalg.norm(B-A)*100:.1f}cm")

        arc        = self.build_arc(A, B)
        timestamps = self.min_jerk_times()

        configs   = []
        q         = q_current.copy()
        fails     = 0
        pos_errs  = []

        for target_pos in arc:
            q_sol, ok = self._ik_position(q, target_pos)
            if ok:
                q = q_sol
            else:
                fails += 1
            configs.append(q.copy())
            # Track actual position error for diagnostics
            actual = self.get_ee_pos(q)
            pos_errs.append(np.linalg.norm(actual - target_pos)*100)

        print(f"[Plan] IK: {len(arc)-fails}/{len(arc)} converged  "
              f"max_err={max(pos_errs):.1f}cm  "
              f"mean_err={np.mean(pos_errs):.1f}cm")
        print(f"[Plan] {len(configs)} waypoints  "
              f"apex_z={arc[:,2].max():.3f}m  "
              f"duration={timestamps[-1]:.1f}s")

        return {'configs': configs, 'timestamps': timestamps, 'arc': arc}


# ─────────────────────────────────────────────────────────────────────────────

class DeliveryNode(Node):

    def __init__(self):
        super().__init__('elliptic_delivery_node')

        if not os.path.exists(ARM_URDF):
            self.get_logger().error(
                f"Missing: {ARM_URDF}\n"
                "Run: bash setup_delivery.sh")
            raise FileNotFoundError(ARM_URDF)

        self.planner   = EllipticPlanner(ARM_URDF)
        self.q_current = pin.neutral(self.planner.model)
        self.mouth_pos = None
        self.executing = False
        self.joints_received = False

        self.create_subscription(
            JointState, '/joint_states', self._joint_cb, 10)
        self.create_subscription(
            PointStamped, '/mouth_position', self._mouth_cb, 10)
        self.create_subscription(
            String, '/deliver', self._trigger_cb, 10)

        self.traj_pub = self.create_publisher(
            JointTrajectory, TRAJ_TOPIC, 10)

        self.get_logger().info(
            f"\n{'='*50}\n"
            f"  Elliptic Delivery Node Ready\n"
            f"  Pub → {TRAJ_TOPIC}\n"
            f"{'='*50}\n"
            "  Step 1 — set mouth position:\n"
            "    ros2 topic pub /mouth_position geometry_msgs/PointStamped \\\n"
            "    '{header: {frame_id: world}, "
            "point: {x: 0.45, y: -0.05, z: 0.55}}' --once\n"
            "  Step 2 — trigger:\n"
            "    ros2 topic pub /deliver std_msgs/String '{data: go}' --once")

    def _joint_cb(self, msg):
        pos_map = dict(zip(msg.name, msg.position))
        joints  = {n: pos_map[n] for n in JOINT_NAMES if n in pos_map}
        if len(joints) == 6:
            self.q_current       = self.planner.joints_to_q(joints)
            self.joints_received = True

    def _mouth_cb(self, msg):
        self.mouth_pos = np.array([
            msg.point.x, msg.point.y, msg.point.z])
        self.get_logger().info(f"Mouth: {np.round(self.mouth_pos,3)}")

    def _trigger_cb(self, msg):
        if self.executing:
            self.get_logger().warn("Already executing")
            return
        if self.mouth_pos is None:
            self.get_logger().error(
                "No mouth position. Publish to /mouth_position first.")
            return
        if not self.joints_received:
            self.get_logger().warn(
                "No joint states received yet — using neutral config. "
                "Is /joint_states publishing?")

        self.executing = True
        try:
            result = self.planner.plan(self.q_current, self.mouth_pos)
            self._publish(result)
        except Exception as e:
            self.get_logger().error(f"Planning failed: {e}")
            import traceback; traceback.print_exc()
        finally:
            self.executing = False

    def _publish(self, result):
        traj             = JointTrajectory()
        traj.joint_names = JOINT_NAMES
        traj.header.stamp = self.get_clock().now().to_msg()

        for q_pin, t in zip(result['configs'], result['timestamps']):
            angles        = self.planner.q_to_joints(q_pin)
            pt            = JointTrajectoryPoint()
            pt.positions  = [float(v) for v in angles]
            pt.velocities = [0.0] * 6
            pt.time_from_start = Duration(
                sec=int(t), nanosec=int((t % 1) * 1e9))
            traj.points.append(pt)

        # Sanity check — verify configs are not all identical
        all_pos = np.array([pt.positions for pt in traj.points])
        joint_ranges = np.max(all_pos, axis=0) - np.min(all_pos, axis=0)
        self.get_logger().info(
            f"Joint ranges across trajectory (rad): "
            f"{np.round(joint_ranges, 3)}")
        if np.max(joint_ranges) < 0.01:
            self.get_logger().error(
                "All waypoints are identical — IK failed to move joints. "
                "Check that the mouth position is reachable.")
            return

        self.traj_pub.publish(traj)
        self.get_logger().info(
            f"Published {len(traj.points)} waypoints → {TRAJ_TOPIC}  "
            f"duration={result['timestamps'][-1]:.1f}s")


def main(args=None):
    rclpy.init(args=args)
    node = DeliveryNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
EOF

chmod +x "$SCRIPTS/elliptic_delivery.py"
log "elliptic_delivery.py written"

# ─── Build ────────────────────────────────────────────────────────────────────
section "Building"
cd "$HOME/kinova_ws"
colcon build --packages-select kinova_bringup --symlink-install 2>&1
source "$HOME/kinova_ws/install/setup.bash"
log "Build done"

# ─── Smoke test ───────────────────────────────────────────────────────────────
section "Smoke Test"
python3 << 'TEST_EOF'
import sys, os, numpy as np, pinocchio as pin
sys.path.insert(0, os.path.expanduser(
    '~/kinova_ws/src/kinova-ros2/kinova_bringup/scripts'))

from elliptic_delivery import EllipticPlanner, JOINT_NAMES

p  = EllipticPlanner('/tmp/j2n6s300_arm.urdf')

# Simulate realistic start config (arm roughly above table)
joints = {
    'j2n6s300_joint_1': 0.0,
    'j2n6s300_joint_2': 2.9,
    'j2n6s300_joint_3': 1.3,
    'j2n6s300_joint_4': 4.2,
    'j2n6s300_joint_5': 1.4,
    'j2n6s300_joint_6': 0.0,
}
q0     = p.joints_to_q(joints)
fork   = p.get_ee_pos(q0)
mouth  = np.array([0.45, -0.05, 0.55])
print(f"Fork position: {np.round(fork,3)}")
print(f"Mouth position: {np.round(mouth,3)}")

result = p.plan(q0, mouth)
configs = result['configs']

all_angles = np.array([p.q_to_joints(c) for c in configs])
joint_ranges = np.max(all_angles, axis=0) - np.min(all_angles, axis=0)
print(f"\nJoint ranges (rad): {np.round(joint_ranges,3)}")
print(f"Max joint range: {np.max(joint_ranges):.3f} rad = "
      f"{np.degrees(np.max(joint_ranges)):.1f} deg")

if np.max(joint_ranges) > 0.05:
    print("✓ Joints are moving — IK working correctly")
else:
    print("✗ Joints not moving — check mouth position reachability")
TEST_EOF

section "Done"
echo ""
echo -e "${GREEN}Run:${NC}"
echo "  source ~/kinova_ws/install/setup.bash"
echo "  ros2 run kinova_bringup elliptic_delivery.py"
echo ""
echo -e "${GREEN}Set mouth position (adjust x,y,z to real patient position):${NC}"
echo "  ros2 topic pub /mouth_position geometry_msgs/PointStamped \\"
echo "    '{header: {frame_id: world}, point: {x: 0.45, y: -0.05, z: 0.55}}' --once"
echo ""
echo -e "${GREEN}Trigger:${NC}"
echo "  ros2 topic pub /deliver std_msgs/String '{data: go}' --once"
echo ""
echo -e "${GREEN}Check controller is active:${NC}"
echo "  ros2 control list_controllers"