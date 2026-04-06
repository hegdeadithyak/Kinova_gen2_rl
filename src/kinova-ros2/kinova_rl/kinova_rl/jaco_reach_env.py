"""
jaco_reach_env.py  (v2 — HER + reward shaping)
───────────────────────────────────────────────
Gymnasium GoalEnv for the Kinova j2n6s300 reach task.

KEY UPGRADES over v1
────────────────────
1. GoalEnv dict observation — observation/achieved_goal/desired_goal keys
   required for stable-baselines3 HerReplayBuffer to work.

2. compute_reward() method — HER calls this to relabel every failed
   transition as a success by substituting the actual end-effector
   position as the goal. Multiplies useful training signal by ~10x.

3. Potential-based reward shaping (Ng et al. 1999):
       r = Φ(s') − Φ(s)    where Φ(s) = −dist * POTENTIAL_SCALE
         + REACH_BONUS      if dist < threshold
         − smoothness penalty (discourages jerky large actions)
         − joint limit penalty
         − tiny time penalty
   Moving closer always gives +reward. Moving away gives −reward.
   This gives a dense gradient at every single step.

4. Curriculum learning — threshold starts at 0.10 m (easy) and
   automatically tightens to 0.05 m after 200 successes.

5. info['is_success'] key — required by SB3 EvalCallback + HER.
"""

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
from std_msgs.msg import Float64MultiArray

import tf2_ros
import tf2_geometry_msgs  # noqa: F401

# ─────────────────────────────── CONSTANTS ────────────────────────────
ROBOT_NAME   = 'j2n6s300'
ARM_JOINTS   = [f'{ROBOT_NAME}_joint_{i}' for i in range(1, 7)]
BASE_LINK    = f'{ROBOT_NAME}_link_base'
EE_LINK      = f'{ROBOT_NAME}_end_effector'

JOINT_LIMITS_LOW  = np.array([0.0,  0.8,  0.0,  0.0,  0.0,  0.0], dtype=np.float32)
JOINT_LIMITS_HIGH = np.array([6.28, 5.48, 6.28, 6.28, 6.28, 6.28], dtype=np.float32)

MAX_STEPS              = 200
ACTION_SCALE           = 0.05      # rad per step

# Reward tuning
POTENTIAL_SCALE        = 10.0      # Φ(s) = -dist * scale  → dense shaping
REACH_BONUS            = 50.0      # sparse success bonus
JOINT_LIMIT_PENALTY    = 20.0
ACTION_SMOOTH_PENALTY  = 0.1       # discourages jerk
TIME_PENALTY           = 0.005

# Curriculum
REACH_THRESHOLD_EASY   = 0.10      # m — first N successes
REACH_THRESHOLD_HARD   = 0.05      # m — after curriculum kicks in
CURRICULUM_SUCCESSES   = 200

# Target workspace (m, base frame)
TARGET_X_RANGE = (0.20, 0.55)
TARGET_Y_RANGE = (-0.35, 0.35)
TARGET_Z_RANGE = (0.05, 0.55)

HOME_POSITIONS = np.array([4.71, 2.71, 1.57, 4.71, 0.0, 3.14], dtype=np.float32)


class _ROSBridge(Node):
    def __init__(self):
        super().__init__('kinova_rl_env_bridge')
        self._joint_lock  = threading.Lock()
        self._joint_pos   = HOME_POSITIONS.copy()
        self._joint_vel   = np.zeros(6, dtype=np.float32)
        self._joint_ready = threading.Event()

        self._tf_buffer   = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

        self.create_subscription(JointState, '/joint_states', self._joint_cb, 10)
        self._traj_client  = ActionClient(
            self, FollowJointTrajectory,
            '/joint_trajectory_controller/follow_joint_trajectory',
        )
        self._marker_pub   = self.create_publisher(Marker, '/rl/target_marker', 10)
        self._thoughts_pub = self.create_publisher(Float64MultiArray, '/rl/agent_thoughts', 10)
        self.get_logger().info('[RL Bridge] Ready.')

    def _joint_cb(self, msg: JointState):
        n2i = {n: i for i, n in enumerate(msg.name)}
        pos = np.zeros(6, dtype=np.float32)
        vel = np.zeros(6, dtype=np.float32)
        ok  = True
        for k, j in enumerate(ARM_JOINTS):
            if j in n2i:
                i = n2i[j]
                pos[k] = msg.position[i] if msg.position else 0.0
                vel[k] = msg.velocity[i] if msg.velocity else 0.0
            else:
                ok = False
        if ok:
            with self._joint_lock:
                self._joint_pos = pos
                self._joint_vel = vel
            self._joint_ready.set()

    def get_joint_state(self):
        with self._joint_lock:
            return self._joint_pos.copy(), self._joint_vel.copy()

    def get_ee_position(self) -> Optional[np.ndarray]:
        try:
            t = self._tf_buffer.lookup_transform(
                BASE_LINK, EE_LINK, rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=1.0),
            )
            tr = t.transform.translation
            return np.array([tr.x, tr.y, tr.z], dtype=np.float32)
        except Exception:
            return None

    def send_joint_goal(self, positions: np.ndarray, duration_sec: float = 0.5) -> bool:
        if not self._traj_client.wait_for_server(timeout_sec=5.0):
            return False
        traj = JointTrajectory()
        traj.joint_names = ARM_JOINTS
        pt = JointTrajectoryPoint()
        pt.positions  = positions.tolist()
        pt.velocities = [0.0] * 6
        s = int(duration_sec)
        pt.time_from_start = Duration(sec=s, nanosec=int((duration_sec - s) * 1e9))
        traj.points = [pt]
        goal = FollowJointTrajectory.Goal()
        goal.trajectory = traj
        future = self._traj_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future, timeout_sec=10.0)
        if not future.result() or not future.result().accepted:
            return False
        rclpy.spin_until_future_complete(self, future.result().get_result_async(), timeout_sec=10.0)
        return True

    def publish_target_marker(self, target: np.ndarray, threshold: float = 0.05):
        m = Marker()
        m.header.frame_id    = BASE_LINK
        m.header.stamp       = self.get_clock().now().to_msg()
        m.ns, m.id           = 'rl_target', 0
        m.type, m.action     = Marker.SPHERE, Marker.ADD
        m.pose.position.x    = float(target[0])
        m.pose.position.y    = float(target[1])
        m.pose.position.z    = float(target[2])
        m.pose.orientation.w = 1.0
        m.scale.x = m.scale.y = m.scale.z = threshold * 2
        m.color.r, m.color.g, m.color.b, m.color.a = 1.0, 0.2, 0.2, 0.8
        self._marker_pub.publish(m)

    def publish_thoughts(self, thoughts: dict):
        msg = Float64MultiArray()
        # Build as plain Python list first — msg.data is array.array, not list
        data = [
            float(thoughts.get('ep_reward', 0.0)),
            float(thoughts.get('distance',  0.0)),
            float(thoughts.get('step',      0.0)),
            float(thoughts.get('value',     0.0)),
            float(thoughts.get('entropy',   0.0)),
        ]
        data += [float(a) for a in thoughts.get('action',    np.zeros(6))]
        data += [float(j) for j in thoughts.get('joint_pos', np.zeros(6))]
        msg.data = data
        self._thoughts_pub.publish(msg)


class JacoReachEnv(gym.Env):
    """
    GoalEnv-compatible reach task. Use with SAC + HerReplayBuffer.

    headless=True  → fast training, no marker/thoughts publishing
    headless=False → Gazebo/RViz visualization + thoughts dashboard
    """

    metadata = {'render_modes': ['human', 'none']}

    def __init__(self, headless: bool = True, render_mode: str = 'none'):
        super().__init__()
        self.headless    = headless
        self.render_mode = render_mode

        self._step_count      = 0
        self._ep_reward       = 0.0
        self._target          = np.zeros(3, dtype=np.float32)
        self._prev_ee_pos     = np.zeros(3, dtype=np.float32)
        self._prev_action     = np.zeros(6, dtype=np.float32)
        self._total_successes = 0
        self._reach_threshold = REACH_THRESHOLD_EASY

        if not rclpy.ok():
            rclpy.init()
        self._ros = _ROSBridge()
        self._executor = MultiThreadedExecutor()
        self._executor.add_node(self._ros)
        self._spin_thread = threading.Thread(target=self._executor.spin, daemon=True)
        self._spin_thread.start()

        self._ros.get_logger().info('[RL Env] Waiting for first joint state...')
        self._ros._joint_ready.wait(timeout=30.0)
        self._ros.get_logger().info('[RL Env] Ready.')

        # GoalEnv observation space
        obs_low  = np.concatenate([JOINT_LIMITS_LOW,  np.full(6, -10.0)]).astype(np.float32)
        obs_high = np.concatenate([JOINT_LIMITS_HIGH, np.full(6,  10.0)]).astype(np.float32)
        goal_low  = np.array([-2.0, -2.0, -0.5], dtype=np.float32)
        goal_high = np.array([ 2.0,  2.0,  2.0], dtype=np.float32)

        self.observation_space = spaces.Dict({
            'observation':   spaces.Box(obs_low,   obs_high,   dtype=np.float32),
            'achieved_goal': spaces.Box(goal_low,  goal_high,  dtype=np.float32),
            'desired_goal':  spaces.Box(goal_low,  goal_high,  dtype=np.float32),
        })
        self.action_space = spaces.Box(-1.0, 1.0, shape=(6,), dtype=np.float32)

    # Required by HER 
    def compute_reward(self, achieved_goal, desired_goal, info):
        """
        Vectorised reward for HER relabelling.
        achieved_goal / desired_goal can be (3,) or (N, 3).
        """
        dist    = np.linalg.norm(achieved_goal - desired_goal, axis=-1)
        reward  = -dist * POTENTIAL_SCALE / MAX_STEPS   # normalised dense term
        success = dist < self._reach_threshold
        reward  = np.where(success, reward + REACH_BONUS, reward)
        return reward

    # Gymnasium API 
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._step_count  = 0
        self._ep_reward   = 0.0
        self._prev_action = np.zeros(6, dtype=np.float32)

        # Curriculum threshold update
        if self._total_successes >= CURRICULUM_SUCCESSES:
            if self._reach_threshold != REACH_THRESHOLD_HARD:
                self._ros.get_logger().info(
                    f'[Curriculum] threshold: {REACH_THRESHOLD_EASY}m → '
                    f'{REACH_THRESHOLD_HARD}m  ({self._total_successes} successes)'
                )
            self._reach_threshold = REACH_THRESHOLD_HARD
        else:
            self._reach_threshold = REACH_THRESHOLD_EASY

        self._ros.send_joint_goal(HOME_POSITIONS, duration_sec=2.0)
        time.sleep(0.3)

        rng = self.np_random
        self._target = np.array([
            rng.uniform(*TARGET_X_RANGE),
            rng.uniform(*TARGET_Y_RANGE),
            rng.uniform(*TARGET_Z_RANGE),
        ], dtype=np.float32)

        ee = self._ros.get_ee_position()
        self._prev_ee_pos = ee if ee is not None else np.zeros(3, dtype=np.float32)

        if not self.headless:
            self._ros.publish_target_marker(self._target, self._reach_threshold)

        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        self._step_count += 1
        action = np.clip(action, -1.0, 1.0).astype(np.float32)

        pos, _ = self._ros.get_joint_state()
        new_pos = np.clip(pos + action * ACTION_SCALE, JOINT_LIMITS_LOW, JOINT_LIMITS_HIGH)
        self._ros.send_joint_goal(new_pos, duration_sec=0.5)

        obs      = self._get_obs()
        ee_pos   = obs['achieved_goal']
        dist     = float(np.linalg.norm(ee_pos - self._target))
        prev_dist = float(np.linalg.norm(self._prev_ee_pos - self._target))

        # Potential-based shaping: positive when closer, negative when further
        r_shaping = (prev_dist - dist) * POTENTIAL_SCALE

        # Sparse bonus
        success  = dist < self._reach_threshold
        r_sparse = REACH_BONUS if success else 0.0

        # Action smoothness: penalise magnitude + jerk (delta from prev action)
        r_smooth = -ACTION_SMOOTH_PENALTY * (
            float(np.linalg.norm(action)) +
            float(np.linalg.norm(action - self._prev_action))
        )

        # Joint limit penalty
        jp = obs['observation'][:6]
        near_limit = (
            np.any(jp <= JOINT_LIMITS_LOW  + 0.05) or
            np.any(jp >= JOINT_LIMITS_HIGH - 0.05)
        )
        r_limit = -JOINT_LIMIT_PENALTY if near_limit else 0.0

        reward           = r_shaping + r_sparse + r_smooth + r_limit - TIME_PENALTY
        self._ep_reward += reward
        self._prev_ee_pos = ee_pos.copy()
        self._prev_action = action.copy()

        terminated = bool(success)
        truncated  = bool(self._step_count >= MAX_STEPS or near_limit)

        if success:
            self._total_successes += 1

        info = {
            'distance':        dist,
            'success':         terminated,
            'is_success':      terminated,   # SB3 EvalCallback + HER key
            'ep_reward':       self._ep_reward,
            'step':            self._step_count,
            'threshold':       self._reach_threshold,
            'total_successes': self._total_successes,
            'r_shaping':       r_shaping,
            'r_sparse':        r_sparse,
        }

        if not self.headless:
            self._ros.publish_thoughts({
                'ep_reward': self._ep_reward,
                'distance':  dist,
                'step':      self._step_count,
                'action':    action,
                'joint_pos': obs['observation'][:6],
            })
            self._ros.publish_target_marker(self._target, self._reach_threshold)

        return obs, float(reward), terminated, truncated, info

    def render(self):
        pass

    def close(self):
        self._executor.shutdown()
        self._ros.destroy_node()

    def _get_obs(self) -> dict:
        pos, vel = self._ros.get_joint_state()
        ee_pos   = self._ros.get_ee_position()
        if ee_pos is None:
            ee_pos = np.zeros(3, dtype=np.float32)
        return {
            'observation':   np.concatenate([pos, vel]).astype(np.float32),
            'achieved_goal': ee_pos.astype(np.float32),
            'desired_goal':  self._target.astype(np.float32),
        }