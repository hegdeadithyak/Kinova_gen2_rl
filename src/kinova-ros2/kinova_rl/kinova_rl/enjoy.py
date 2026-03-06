"""
enjoy.py
────────
Run a trained policy in visualization mode.
Shows the robot moving in Gazebo/RViz and streams agent thoughts
to the terminal dashboard (thoughts_node).

Usage
─────
  ros2 run kinova_rl enjoy -- --model ~/kinova_rl_logs/best_model

  # Run N episodes then stop
  ros2 run kinova_rl enjoy -- --model ~/kinova_rl_logs/best_model --episodes 10

  # Deterministic policy (no exploration noise)
  ros2 run kinova_rl enjoy -- --model ~/kinova_rl_logs/best_model --deterministic
"""

import argparse
import time
import numpy as np

from stable_baselines3 import PPO
from kinova_rl.jaco_reach_env import JacoReachEnv


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model', type=str, required=True,
                   help='Path to saved model (.zip or without extension)')
    p.add_argument('--episodes', type=int, default=0,
                   help='Number of episodes (0 = run forever)')
    p.add_argument('--deterministic', action='store_true', default=True)
    p.add_argument('--stochastic', dest='deterministic', action='store_false',
                   help='Use stochastic policy (exploration)')
    return p.parse_args()


def main():
    args = parse_args()

    print(f'\n[enjoy] Loading model: {args.model}')
    print('[enjoy] Creating visualization environment...')
    print('[enjoy] Make sure Gazebo + RViz are running.')
    print('[enjoy] Run in another terminal: ros2 run kinova_rl thoughts\n')

    env   = JacoReachEnv(headless=False)
    model = PPO.load(args.model, env=env)

    ep          = 0
    total_steps = 0
    successes   = []

    try:
        while args.episodes == 0 or ep < args.episodes:
            obs, _ = env.reset()
            done   = False
            ep_reward = 0.0
            ep_steps  = 0

            while not done:
                action, state = model.predict(
                    obs, deterministic=args.deterministic,
                )

                # Inject value estimate into thoughts via env
                value_estimate = 0.0
                try:
                    import torch
                    obs_t = model.policy.obs_to_tensor(obs)[0]
                    with torch.no_grad():
                        value_estimate = float(
                            model.policy.predict_values(obs_t).cpu().numpy()[0]
                        )
                    # Also get entropy
                    dist = model.policy.get_distribution(obs_t)
                    entropy = float(dist.entropy().cpu().numpy()[0])
                except Exception:
                    entropy = 0.0

                # Publish thoughts with value + entropy
                env._ros.publish_thoughts({
                    'ep_reward': ep_reward,
                    'distance':  env._last_dist,
                    'step':      ep_steps,
                    'value':     value_estimate,
                    'entropy':   entropy,
                    'action':    action,
                    'joint_pos': obs[:6],
                })

                obs, reward, terminated, truncated, info = env.step(action)
                done       = terminated or truncated
                ep_reward += reward
                ep_steps  += 1
                total_steps += 1

            ep += 1
            success = info.get('success', False)
            successes.append(success)

            status = '✅ SUCCESS' if success else '❌ TIMEOUT'
            print(
                f'[enjoy] Ep {ep:4d} | {status} | '
                f'reward={ep_reward:8.2f} | '
                f'dist={info["distance"]:.4f}m | '
                f'steps={ep_steps:3d} | '
                f'success_rate={np.mean(successes)*100:.1f}%'
            )

    except KeyboardInterrupt:
        print('\n[enjoy] Stopped.')
    finally:
        env.close()

    print(f'\n[enjoy] Ran {ep} episodes | '
          f'Overall success rate: {np.mean(successes)*100:.1f}%')


if __name__ == '__main__':
    main()
