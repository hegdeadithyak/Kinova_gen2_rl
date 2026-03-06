"""
jaco_reach_env.py
─────────────────
Gymnasium environment for the Kinova j2n6s300 reach task running inside
Ignition Gazebo (Fortress) with ROS 2 Humble.

Observation  (18-dim float32):
  [0:6]   current joint positions  (radians)
  [6:12]  current joint velocities (rad/s)
  [12:15] end-effector XYZ in base frame (metres)
  [15:18] target XYZ in base frame (metres)

Action  (6-dim continuous float32, clipped to [-1, 1]):
  Delta joint position per step, scaled by ACTION_SCALE.
  Applied to joints 1-6 (fingers not controlled by RL).

Reward:
  r = -dist(ee, target)                         dense distance penalty
    + REACH_BONUS  if dist < REACH_THRESHOLD     sparse success bonus
    - COLLISION_PENALTY if joint limit breached  safety penalty
    - TIME_PENALTY                               encourage speed

Episode ends when:
  - dist(ee, target) < REACH_THRESHOLD  (success)
  - Any joint exceeds soft limits        (failure)
  - MAX_STEPS reached                    (timeout)
"""

import math
import time
import threading
from typing import Optional, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor

from sensor_msgs.msg import JointState
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point

import tf2_ros
import tf2_geometry_msgs  # noqa: F401 — registers transform support


# ─────────────────────────────── CONSTANTS ────────────────────────────
ROBOT_NAME      = 'j2n6s300'
ARM_JOINTS      = [f'{ROBOT_NAME}_joint_{i}' for i in range(1, 7)]
BASE_LINK       = f'{ROBOT_NAME}_link_base'
EE_LINK         = f'{ROBOT_NAME}_end_effector'

# Joint soft limits (radians) — slightly inside hardware limits
JOINT_LIMITS_LOW  = np.array([0.0,  0.8,  0.0,  0.0,  0.0,  0.0])
JOINT_LIMITS_HIGH = np.array([6.28, 5.48, 6.28, 6.28, 6.28, 6.28])

# RL hyper-parameters
MAX_STEPS         = 200
ACTION_SCALE      = 0.05          # radians per step
STEP_DURATION     = 0.5           # seconds per action step (sim time)
REACH_THRESHOLD   = 0.05          # metres — success radius
REACH_BONUS       = 100.0
COLLISION_PENALTY = 50.0
TIME_PENALTY      = 0.01

# Target sampling workspace (metres, in base frame)
TARGET_X_RANGE = (0.2,  0.6)
TARGET_Y_RANGE = (-0.4, 0.4)
TARGET_Z_RANGE = (0.0,  0.6)

# Home joint positions (radians) matching launch file initial values
HOME_POSITIONS = np.array([4.71, 2.71, 1.57, 4.71, 0.0, 3.14])


class _ROSBridge(Node):
    """
    Internal ROS2 node that manages all topic I/O for the environment.
    Runs in its own thread so the Gymnasium step() call can block
    cleanly without stalling the ROS executor.
    """

    def __init__(self):
        super().__init__('kinova_rl_env_bridge')

        # ── State cache ────────────────────────────────────────────────
        self._joint_lock   = threading.Lock()
        self._joint_pos    = HOME_POSITIONS.copy()
        self._joint_vel    = np.zeros(6)
        self._joint_ready  = threading.Event()

        # ── TF buffer ──────────────────────────────────────────────────
        self._tf_buffer    = tf2_ros.Buffer()
        self._tf_listener  = tf2_ros.TransformListener(self._tf_buffer, self)

        # ── Subscriptions ──────────────────────────────────────────────
        self.create_subscription(
            JointState, '/joint_states',
            self._joint_cb, 10,
        )

        # ── Action client ──────────────────────────────────────────────
        self._traj_client = ActionClient(
            self,
            FollowJointTrajectory,
            '/joint_trajectory_controller/follow_joint_trajectory',
        )

        # ── Target marker publisher ────────────────────────────────────
        self._marker_pub = self.create_publisher(
            Marker, '/rl/target_marker', 10,
        )

        # ── Agent thoughts publisher ───────────────────────────────────
        # Published as a flat Float64MultiArray so the thoughts_node
        # can subscribe and render a live terminal dashboard.
        from std_msgs.msg import Float64MultiArray
        self._thoughts_pub = self.create_publisher(
            Float64MultiArray, '/rl/agent_thoughts', 10,
        )

        self.get_logger().info('[RL Bridge] Ready.')

    # ── Callbacks ─────────────────────────────────────────────────────

    def _joint_cb(self, msg: JointState):
        """Cache the 6 arm joint positions + velocities."""
        name_to_idx = {n: i for i, n in enumerate(msg.name)}
        pos = np.zeros(6)
        vel = np.zeros(6)
        all_found = True
        for k, joint in enumerate(ARM_JOINTS):
            if joint in name_to_idx:
                i = name_to_idx[joint]
                pos[k] = msg.position[i] if msg.position else 0.0
                vel[k] = msg.velocity[i] if msg.velocity else 0.0
            else:
                all_found = False
        if all_found:
            with self._joint_lock:
                self._joint_pos = pos
                self._joint_vel = vel
            self._joint_ready.set()

    # ── Public API ─────────────────────────────────────────────────────

    def get_joint_state(self) -> Tuple[np.ndarray, np.ndarray]:
        with self._joint_lock:
            return self._joint_pos.copy(), self._joint_vel.copy()

    def get_ee_position(self) -> Optional[np.ndarray]:
        """Return end-effector XYZ in base frame, or None if TF unavailable."""
        try:
            t = self._tf_buffer.lookup_transform(
                BASE_LINK, EE_LINK,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=1.0),
            )
            tr = t.transform.translation
            return np.array([tr.x, tr.y, tr.z])
        except Exception:
            return None

    def send_joint_goal(self, positions: np.ndarray, duration_sec: float = 0.5):
        """
        Send a FollowJointTrajectory goal and wait for it to complete.
        Returns True if succeeded, False otherwise.
        """
        if not self._traj_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('FollowJointTrajectory action server not available!')
            return False

        traj = JointTrajectory()
        traj.joint_names = ARM_JOINTS

        pt = JointTrajectoryPoint()
        pt.positions  = positions.tolist()
        pt.velocities = [0.0] * 6
        secs  = int(duration_sec)
        nsecs = int((duration_sec - secs) * 1e9)
        pt.time_from_start = Duration(sec=secs, nanosec=nsecs)
        traj.points = [pt]

        goal = FollowJointTrajectory.Goal()
        goal.trajectory = traj

        future = self._traj_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future, timeout_sec=10.0)

        if not future.result() or not future.result().accepted:
            return False

        result_future = future.result().get_result_async()
        rclpy.spin_until_future_complete(self, result_future, timeout_sec=10.0)
        return True

    def publish_target_marker(self, target: np.ndarray):
        """Publish a red sphere marker at the target position in RViz."""
        m = Marker()
        m.header.frame_id = BASE_LINK
        m.header.stamp    = self.get_clock().now().to_msg()
        m.ns              = 'rl_target'
        m.id              = 0
        m.type            = Marker.SPHERE
        m.action          = Marker.ADD
        m.pose.position.x = float(target[0])
        m.pose.position.y = float(target[1])
        m.pose.position.z = float(target[2])
        m.pose.orientation.w = 1.0
        m.scale.x = m.scale.y = m.scale.z = REACH_THRESHOLD * 2
        m.color.r = 1.0
        m.color.g = 0.2
        m.color.b = 0.2
        m.color.a = 0.8
        self._marker_pub.publish(m)

    def publish_thoughts(self, thoughts: dict):
        """
        Pack agent thoughts into a Float64MultiArray and publish.
        Layout:
          [0]  episode reward so far
          [1]  distance to target
          [2]  step count
          [3]  value estimate (if available, else 0)
          [4]  entropy (if available, else 0)
          [5:11] chosen action (6 joints)
          [11:17] joint positions
        """
        from std_msgs.msg import Float64MultiArray
        msg = Float64MultiArray()
        msg.data = [
            float(thoughts.get('ep_reward',   0.0)),
            float(thoughts.get('distance',    0.0)),
            float(thoughts.get('step',        0.0)),
            float(thoughts.get('value',       0.0)),
            float(thoughts.get('entropy',     0.0)),
        ]
        action = thoughts.get('action', np.zeros(6))
        joints = thoughts.get('joint_pos', np.zeros(6))
        msg.data.extend([float(a) for a in action])
        msg.data.extend([float(j) for j in joints])
        self._thoughts_pub.publish(msg)


class JacoReachEnv(gym.Env):
    """
    Gymnasium reach task for Kinova j2n6s300 in Ignition Gazebo.

    Usage:
        env = JacoReachEnv(headless=True)   # for training
        env = JacoReachEnv(headless=False)  # for visualization + thoughts

    headless=False:
        - Publishes target marker to /rl/target_marker (visible in RViz)
        - Publishes agent thoughts to /rl/agent_thoughts (read by thoughts_node)
    """

    metadata = {'render_modes': ['human', 'none']}

    def __init__(self, headless: bool = True, render_mode: str = 'none'):
        super().__init__()

        self.headless    = headless
        self.render_mode = render_mode
        self._step_count = 0
        self._ep_reward  = 0.0
        self._target     = np.zeros(3)
        self._last_dist  = 0.0

        # ── Init ROS2 ──────────────────────────────────────────────────
        if not rclpy.ok():
            rclpy.init()
        self._ros = _ROSBridge()
        self._executor = MultiThreadedExecutor()
        self._executor.add_node(self._ros)
        self._spin_thread = threading.Thread(
            target=self._executor.spin, daemon=True,
        )
        self._spin_thread.start()

        # Wait for first joint state to arrive
        self._ros.get_logger().info('[RL Env] Waiting for first joint state...')
        self._ros._joint_ready.wait(timeout=30.0)
        self._ros.get_logger().info('[RL Env] Joint states received. Environment ready.')

        # ── Spaces ────────────────────────────────────────────────────
        # obs: 6 joint pos + 6 joint vel + 3 ee pos + 3 target pos = 18
        obs_low  = np.concatenate([
            JOINT_LIMITS_LOW,
            np.full(6,  -10.0),
            np.full(3,  -2.0),
            np.full(3,  -2.0),
        ]).astype(np.float32)
        obs_high = np.concatenate([
            JOINT_LIMITS_HIGH,
            np.full(6,   10.0),
            np.full(3,   2.0),
            np.full(3,   2.0),
        ]).astype(np.float32)

        self.observation_space = spaces.Box(
            low=obs_low, high=obs_high, dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(6,), dtype=np.float32,
        )

    # ── Gymnasium API ─────────────────────────────────────────────────

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._step_count = 0
        self._ep_reward  = 0.0

        # Move arm to home
        self._ros.send_joint_goal(HOME_POSITIONS, duration_sec=2.0)
        time.sleep(0.5)

        # Sample a new random target
        rng = self.np_random
        self._target = np.array([
            rng.uniform(*TARGET_X_RANGE),
            rng.uniform(*TARGET_Y_RANGE),
            rng.uniform(*TARGET_Z_RANGE),
        ])

        if not self.headless:
            self._ros.publish_target_marker(self._target)

        obs = self._get_obs()
        self._last_dist = self._dist_to_target(obs[12:15])
        return obs, {}

    def step(self, action: np.ndarray):
        self._step_count += 1

        # Clip and scale action
        action = np.clip(action, -1.0, 1.0)
        delta  = action * ACTION_SCALE

        # Compute new joint targets (clamped to soft limits)
        pos, _ = self._ros.get_joint_state()
        new_pos = np.clip(pos + delta, JOINT_LIMITS_LOW, JOINT_LIMITS_HIGH)

        # Send goal to controller
        self._ros.send_joint_goal(new_pos, duration_sec=STEP_DURATION)

        # Get new observation
        obs     = self._get_obs()
        ee_pos  = obs[12:15]
        dist    = self._dist_to_target(ee_pos)

        # ── Reward ────────────────────────────────────────────────────
        reward = -dist                              # dense distance penalty

        if dist < REACH_THRESHOLD:
            reward += REACH_BONUS

        # Joint limit violation penalty
        jp = obs[:6]
        limit_violation = np.any(jp <= JOINT_LIMITS_LOW + 0.01) or \
                          np.any(jp >= JOINT_LIMITS_HIGH - 0.01)
        if limit_violation:
            reward -= COLLISION_PENALTY

        reward -= TIME_PENALTY                      # time penalty
        reward += (self._last_dist - dist) * 10.0  # progress reward

        self._last_dist  = dist
        self._ep_reward += reward

        # ── Termination ───────────────────────────────────────────────
        terminated = bool(dist < REACH_THRESHOLD)
        truncated  = bool(
            self._step_count >= MAX_STEPS or limit_violation
        )

        info = {
            'distance':       dist,
            'success':        terminated,
            'ep_reward':      self._ep_reward,
            'step':           self._step_count,
            'joint_pos':      obs[:6],
            'ee_pos':         ee_pos,
            'target':         self._target,
        }

        # ── Publish thoughts (viz mode) ────────────────────────────────
        if not self.headless:
            self._ros.publish_thoughts({
                'ep_reward': self._ep_reward,
                'distance':  dist,
                'step':      self._step_count,
                'action':    action,
                'joint_pos': obs[:6],
            })
            self._ros.publish_target_marker(self._target)

        return obs, float(reward), terminated, truncated, info

    def render(self):
        pass  # Visualization handled by RViz + thoughts_node

    def close(self):
        self._executor.shutdown()
        self._ros.destroy_node()

    # ── Helpers ───────────────────────────────────────────────────────

    def _get_obs(self) -> np.ndarray:
        pos, vel = self._ros.get_joint_state()
        ee_pos   = self._ros.get_ee_position()
        if ee_pos is None:
            ee_pos = np.zeros(3)
        return np.concatenate([
            pos, vel, ee_pos, self._target,
        ]).astype(np.float32)

    def _dist_to_target(self, ee_pos: np.ndarray) -> float:
        return float(np.linalg.norm(ee_pos - self._target))
