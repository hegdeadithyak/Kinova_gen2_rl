# kinova_rl — Deep RL for Kinova j2n6s300

Reinforcement learning package for the Kinova j2n6s300 arm in
Ignition Gazebo (Fortress) + ROS 2 Humble.

## Files

| File | Purpose |
|------|---------|
| `jaco_reach_env.py` | Gymnasium environment — reach task |
| `train.py` | PPO training (headless or viz mode) |
| `enjoy.py` | Run a trained policy with visualization |
| `thoughts_node.py` | Live terminal "agent thoughts" dashboard |

---

## Install dependencies

```bash
pip install stable-baselines3[extra] gymnasium torch rich tensorboard
```

## Build

```bash
cd ~/kinova_ws
colcon build --packages-select kinova_rl
source install/setup.bash
```

---

## Headless Training (fast, no GUI)

Best for actual training runs. Gazebo runs without a window.

```bash
# Terminal 1 — start everything including training:
ros2 launch kinova_rl train_headless.launch.py

# OR start Gazebo manually first, then:
ros2 run kinova_rl train -- --headless

# Resume from checkpoint:
ros2 run kinova_rl train -- --headless --resume ~/kinova_rl_logs/best_model
```

---

## Visualization Training (see the robot + agent thoughts)

```bash
# Terminal 1 — start full simulation (Gazebo + RViz):
ros2 launch kinova_bringup kinova_launch.py

# Terminal 2 — start training in viz mode:
ros2 run kinova_rl train -- --viz

# Terminal 3 — watch the live agent thoughts dashboard:
ros2 run kinova_rl thoughts

# Terminal 4 (optional) — watch TensorBoard:
tensorboard --logdir ~/kinova_rl_logs/tensorboard
# Then open http://localhost:6006
```

---

## Run a trained policy

```bash
# Terminal 1:
ros2 launch kinova_bringup kinova_launch.py

# Terminal 2:
ros2 run kinova_rl enjoy -- --model ~/kinova_rl_logs/best_model

# Terminal 3 (watch agent thoughts):
ros2 run kinova_rl thoughts
```

---

## What the "Agent Thoughts" dashboard shows

```
╭─── 🤖 Kinova j2n6s300 — Agent Thoughts ────────────────╮
│ Metric                    │ Value    │ Visual            │
├───────────────────────────┼──────────┼───────────────────┤
│ 🏆 Episode Reward         │ -12.34   │ ████░░░░░░░░░░░   │
│ 📏 Distance to Target     │ 0.1823 m │ ████████░░░░░░░   │
│ ⏱️  Step                  │ 47 / 200 │ █████░░░░░░░░░░   │
│ 💭 Value Estimate         │ +23.4    │ (expects Σ reward)│
│ 🎲 Entropy (exploration)  │ 0.34     │ ████░░░░░░░░░░░   │
├───────────────────────────┴──────────┴───────────────────┤
│ Actions (Δ joint, rad)                                   │
│  J1 ▶  +0.032  ████████░░░░                             │
│  J2 ◀  -0.018  ░░░░████░░░░                             │
├──────────────────────────────────────────────────────────┤
│ Joint Positions (rad)                                    │
│  J1 [0.00, 6.28]  4.71  ████████████████░░░░            │
╰──────────────────────────────────────────────────────────╯
```

- **Value Estimate** — what the agent *expects* to earn from this state onward. If high, the agent thinks the current pose is close to success.
- **Entropy** — how uncertain / exploratory the agent is. High early in training (exploring), low when converged (confident).
- **Actions** — the actual joint deltas being applied this step.

---

## TensorBoard metrics

Open `http://localhost:6006` after running `tensorboard --logdir ~/kinova_rl_logs/tensorboard`

| Metric | Meaning |
|--------|---------|
| `rollout/ep_rew_mean` | Mean episode reward (go up) |
| `rollout/ep_len_mean` | Mean steps per episode (go down) |
| `custom/mean_distance` | Mean distance to target at end (go down) |
| `custom/success_rate` | % episodes that reached target (go up) |
| `train/entropy_loss` | Exploration entropy (goes down as policy converges) |
| `train/value_loss` | How well the critic predicts returns |
| `train/explained_variance` | Should approach 1.0 as training progresses |

---

## Observation space (18-dim)

| Dims | Content |
|------|---------|
| 0–5 | Joint positions (rad) |
| 6–11 | Joint velocities (rad/s) |
| 12–14 | End-effector XYZ (m) in base frame |
| 15–17 | Target XYZ (m) in base frame |

## Action space (6-dim continuous)

Joint position deltas, clipped to [-1, 1] and scaled by `ACTION_SCALE=0.05 rad`.

## Reward function

```
r = -dist(ee, target)           dense: penalise distance every step
  + 100                         if dist < 0.05 m (success bonus)
  - 50                          if joint limit breached (safety)
  - 0.01                        time penalty (encourage speed)
  + (prev_dist - dist) * 10     progress reward (encourage moving closer)
```
