#!/usr/bin/env python3
"""
Elliptic Delivery Node — Kinova j2n6s300

IK fix: pure damped least-squares, no regularization, with random restarts.
The previous regularization term was fighting the Jacobian and preventing
convergence from large initial errors (~78cm).

Initial joint config from launch file (kinova_launch.py _patch_urdf INIT):
  joint_1: 4.71  joint_2: 2.71  joint_3: 1.57
  joint_4: 4.71  joint_5: 0.0   joint_6: 3.14
These are used as IK seed when joint_states not yet received.
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

# Initial config from kinova_launch.py INIT dict
# Used as IK seed before /joint_states is received
INIT_ANGLES = {
    'j2n6s300_joint_1': 4.71,
    'j2n6s300_joint_2': 2.71,
    'j2n6s300_joint_3': 1.57,
    'j2n6s300_joint_4': 4.71,
    'j2n6s300_joint_5': 0.0,
    'j2n6s300_joint_6': 3.14,
}


class EllipticPlanner:

    def __init__(self, urdf_path, lift_height=0.10,
                 n_waypoints=40, total_time=4.0):

        robot      = RobotWrapper.BuildFromURDF(urdf_path)
        self.model = robot.model
        self.data  = robot.data
        self.nv    = self.model.nv   # 6
        self.nq    = self.model.nq   # 10 (continuous joints stored as cos/sin)
        assert self.nv == 6, f"Expected nv=6, got {self.nv}"

        self.lift = lift_height
        self.n    = n_waypoints
        self.T    = total_time

        self.q_neutral = pin.neutral(self.model)

        ee_name = 'j2n6s300_end_effector'
        self.ee_fid = (self.model.getFrameId(ee_name)
                       if self.model.existFrame(ee_name)
                       else self.model.nframes - 1)

        print(f"[EllipticPlanner] nv={self.nv} nq={self.nq}  "
              f"EE={self.model.frames[self.ee_fid].name}  "
              f"lift={lift_height*100:.0f}cm  T={total_time}s")

    # ── Config helpers ────────────────────────────────────────────────────────

    def joints_to_q(self, joints_dict):
        """Dict {joint_name: angle_rad} → pinocchio config vector (nq,)"""
        q = self.q_neutral.copy()
        for name in JOINT_NAMES:
            if name not in joints_dict:
                continue
            jid = self.model.getJointId(name)
            if jid >= self.model.njoints:
                continue
            idx   = self.model.joints[jid].idx_q
            nqi   = self.model.joints[jid].nq
            angle = joints_dict[name]
            if nqi == 1:
                q[idx] = angle
            else:
                q[idx]   = np.cos(angle)
                q[idx+1] = np.sin(angle)
        return q

    def q_to_joints(self, q):
        """Pinocchio config vector → np.array(6,) joint angles"""
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

    # ── IK ────────────────────────────────────────────────────────────────────

    def _ik_one_attempt(self, q_init, target_pos,
                        max_iter=500, tol=3e-3, damping=1e-3):
        """
        Pure damped least-squares position IK.
        Uses J^T (JJ^T + λI)^{-1} form — numerically stable for
        near-singular Jacobians (arm near singularity).
        No regularization — it was fighting convergence.
        Step clipped to 0.5 rad max per iteration.
        """
        q = q_init.copy()
        for _ in range(max_iter):
            pin.forwardKinematics(self.model, self.data, q)
            pin.updateFramePlacement(self.model, self.data, self.ee_fid)
            err = target_pos - self.data.oMf[self.ee_fid].translation

            if np.linalg.norm(err) < tol:
                return q, True

            J   = pin.computeFrameJacobian(
                      self.model, self.data, q, self.ee_fid,
                      pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)[:3, :]
            JJT = J @ J.T
            dq  = J.T @ np.linalg.solve(JJT + damping * np.eye(3), err)
            dq  = np.clip(dq, -0.5, 0.5)
            q   = pin.integrate(self.model, q, dq)

        return q, False

    def ik_position(self, q_seed, target_pos, n_restarts=5):
        """
        IK with random restarts on failure.
        Attempt 0: warm start from q_seed (previous waypoint — fast along arc).
        Attempts 1-N: random configs within joint limits.
        Returns best q found even if not fully converged.
        """
        best_q   = q_seed.copy()
        best_err = np.linalg.norm(target_pos - self.get_ee_pos(q_seed))

        q_sol, ok = self._ik_one_attempt(q_seed, target_pos)
        err = np.linalg.norm(target_pos - self.get_ee_pos(q_sol))
        if ok:
            return q_sol, True
        if err < best_err:
            best_err, best_q = err, q_sol

        for _ in range(n_restarts):
            q_rand    = pin.randomConfiguration(self.model)
            q_sol, ok = self._ik_one_attempt(q_rand, target_pos)
            err       = np.linalg.norm(target_pos - self.get_ee_pos(q_sol))
            if ok:
                return q_sol, True
            if err < best_err:
                best_err, best_q = err, q_sol

        return best_q, False

    # ── Arc ───────────────────────────────────────────────────────────────────

    def build_arc(self, A, B):
        """
        Elliptical arc from A (fork) to B (mouth).
        Lives in the plane of (A→B vector, world-Z).
        s: π→0  so P(π)=A, P(0)=B, apex at s=π/2 (highest point).
        """
        AB   = B - A
        dist = np.linalg.norm(AB)
        if dist < 0.02:
            raise ValueError(f"A-B only {dist*100:.1f}cm apart")

        u_AB  = AB / dist
        wz    = np.array([0., 0., 1.])
        v_raw = wz - np.dot(wz, u_AB) * u_AB
        if np.linalg.norm(v_raw) < 1e-6:
            v_raw = np.array([1., 0., 0.]) - \
                    np.dot(np.array([1., 0., 0.]), u_AB) * u_AB
        u_lift = v_raw / np.linalg.norm(v_raw)
        center = (A + B) / 2.0

        return np.array([
            center + (dist/2)*np.cos(s)*u_AB + self.lift*np.sin(s)*u_lift
            for s in np.linspace(np.pi, 0.0, self.n)
        ])

    def min_jerk_times(self):
        """s(τ)=10τ³−15τ⁴+6τ⁵: zero vel+acc at both endpoints."""
        tau = np.linspace(0., 1., self.n)
        return (10*tau**3 - 15*tau**4 + 6*tau**5) * self.T

    # ── Plan ─────────────────────────────────────────────────────────────────

    def plan(self, q_current, mouth_pos):
        A = self.get_ee_pos(q_current)
        B = mouth_pos.copy()
        print(f"[Plan] Fork:  {np.round(A, 3)}")
        print(f"[Plan] Mouth: {np.round(B, 3)}  "
              f"dist={np.linalg.norm(B-A)*100:.1f}cm")

        arc        = self.build_arc(A, B)
        timestamps = self.min_jerk_times()
        configs    = []
        q          = q_current.copy()
        n_ok       = 0
        pos_errs   = []

        for target_pos in arc:
            q_sol, ok = self.ik_position(q, target_pos)
            q = q_sol   # always advance — best attempt even if not converged
            if ok:
                n_ok += 1
            configs.append(q.copy())
            pos_errs.append(
                np.linalg.norm(self.get_ee_pos(q) - target_pos) * 100)

        print(f"[Plan] IK: {n_ok}/{len(arc)} converged  "
              f"max_err={max(pos_errs):.1f}cm  "
              f"mean_err={np.mean(pos_errs):.1f}cm")
        print(f"[Plan] apex_z={arc[:, 2].max():.3f}m  "
              f"duration={timestamps[-1]:.1f}s")

        return {'configs': configs, 'timestamps': timestamps, 'arc': arc}


# ─────────────────────────────────────────────────────────────────────────────

class DeliveryNode(Node):

    def __init__(self):
        super().__init__('elliptic_delivery_node')

        if not os.path.exists(ARM_URDF):
            self.get_logger().error(
                f"Missing {ARM_URDF}. Run setup_delivery.sh first.")
            raise FileNotFoundError(ARM_URDF)

        self.planner = EllipticPlanner(ARM_URDF)

        # Seed from launch file — better than pin.neutral() for this robot
        self.q_current       = self.planner.joints_to_q(INIT_ANGLES)
        self.joints_received = False
        self.mouth_pos       = None
        self.executing       = False

        self.create_subscription(
            JointState, '/joint_states', self._joint_cb, 10)
        self.create_subscription(
            PointStamped, '/mouth_position', self._mouth_cb, 10)
        self.create_subscription(
            String, '/deliver', self._trigger_cb, 10)

        self.traj_pub = self.create_publisher(
            JointTrajectory, TRAJ_TOPIC, 10)

        # From kinova_launch.py FIX B:
        # Patient mouth world position: (0.660, 0, 0.990)
        self.get_logger().info(
            f"\n{'='*52}\n"
            f"  Elliptic Delivery Node Ready\n"
            f"  Pub → {TRAJ_TOPIC}\n"
            f"{'='*52}\n"
            "  Mouth position from your SDF (FIX B):\n"
            "    ros2 topic pub /mouth_position "
            "geometry_msgs/PointStamped \\\n"
            "    '{header: {frame_id: world}, "
            "point: {x: 0.66, y: 0.0, z: 0.99}}' --once\n\n"
            "  Trigger:\n"
            "    ros2 topic pub /deliver "
            "std_msgs/String '{data: go}' --once\n"
            f"{'='*52}")

    def _joint_cb(self, msg):
        pos_map = dict(zip(msg.name, msg.position))
        joints  = {n: pos_map[n] for n in JOINT_NAMES if n in pos_map}
        if len(joints) == 6:
            self.q_current       = self.planner.joints_to_q(joints)
            self.joints_received = True

    def _mouth_cb(self, msg):
        self.mouth_pos = np.array([
            msg.point.x, msg.point.y, msg.point.z])
        self.get_logger().info(
            f"Mouth position: {np.round(self.mouth_pos, 3)}")

    def _trigger_cb(self, msg):
        if self.executing:
            self.get_logger().warn("Already executing — ignoring")
            return
        if self.mouth_pos is None:
            self.get_logger().error(
                "No mouth position. Publish /mouth_position first.")
            return
        if not self.joints_received:
            self.get_logger().warn(
                "No /joint_states yet — using launch-file config as IK seed.")

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
        all_angles   = np.array([
            self.planner.q_to_joints(c) for c in result['configs']])
        joint_ranges = np.max(all_angles, axis=0) - np.min(all_angles, axis=0)

        self.get_logger().info(
            f"Joint ranges (rad): {np.round(joint_ranges, 3)}")

        if np.max(joint_ranges) < 0.05:
            self.get_logger().error(
                "Joints not moving — IK failed.\n"
                f"  Fork:  {np.round(self.planner.get_ee_pos(self.q_current),3)}\n"
                f"  Mouth: {np.round(self.mouth_pos, 3)}\n"
                "  Verify mouth is within reach (~0.9m from robot base).")
            return

        traj              = JointTrajectory()
        traj.joint_names  = JOINT_NAMES
        traj.header.stamp = self.get_clock().now().to_msg()

        # Prepend current robot state as t=0 waypoint.
        # Without this the controller rejects the trajectory with
        # "start state tolerance violated" if the first IK waypoint
        # differs from the actual joint positions.
        current_angles = self.planner.q_to_joints(self.q_current)
        pt0             = JointTrajectoryPoint()
        pt0.positions   = [float(v) for v in current_angles]
        pt0.velocities  = [0.0] * 6
        pt0.time_from_start = Duration(sec=0, nanosec=0)
        traj.points.append(pt0)

        for angles, t in zip(all_angles, result['timestamps']):
            pt              = JointTrajectoryPoint()
            pt.positions    = [float(v) for v in angles]
            pt.velocities   = [0.0] * 6
            # Offset by 0.5s to give controller time to accept — 
            # total duration becomes T + 0.5s
            t_offset = t + 0.5
            pt.time_from_start = Duration(
                sec=int(t_offset), nanosec=int((t_offset % 1) * 1e9))
            traj.points.append(pt)

        self.traj_pub.publish(traj)
        self.get_logger().info(
            f"Published {len(traj.points)} waypoints → {TRAJ_TOPIC}  "
            f"duration={result['timestamps'][-1]+0.5:.1f}s")


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