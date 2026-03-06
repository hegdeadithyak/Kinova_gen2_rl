"""
train.py
────────
Train a PPO agent on the Kinova j2n6s300 reach task.

Usage
─────
  # Headless (fast, no GUI needed):
  ros2 run kinova_rl train -- --headless

  # Visualization (see robot move in Gazebo + RViz + thoughts dashboard):
  ros2 run kinova_rl train -- --viz

  # Resume from a checkpoint:
  ros2 run kinova_rl train -- --headless --resume ~/kinova_rl_logs/best_model

Outputs
───────
  ~/kinova_rl_logs/
    tensorboard/          TensorBoard event files
    checkpoints/          Model saved every --save-freq steps
    best_model.zip        Best model seen so far
    train_config.yaml     Config snapshot for reproducibility

TensorBoard
───────────
  tensorboard --logdir ~/kinova_rl_logs/tensorboard
  # Then open http://localhost:6006

What you will see
─────────────────
  - rollout/ep_rew_mean      : mean episode reward
  - rollout/ep_len_mean      : mean episode length
  - custom/mean_distance     : mean distance to target at end of episode
  - custom/success_rate      : fraction of episodes that reached target
  - train/value_loss         : critic loss
  - train/policy_gradient_loss
  - train/entropy_loss       : exploration entropy
  - train/explained_variance : how well value fn predicts returns
"""

import os
import sys
import argparse
import yaml
import time
from pathlib import Path

import numpy as np

# ── SB3 + Gymnasium ───────────────────────────────────────────────────
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.logger import configure as sb3_configure
from stable_baselines3.common.monitor import Monitor

from kinova_rl.jaco_reach_env import JacoReachEnv


# ─────────────────────────────── CONFIG ───────────────────────────────
DEFAULT_CONFIG = {
    # Training
    'total_timesteps':    1_000_000,
    'save_freq':          10_000,
    'eval_freq':          5_000,
    'n_eval_episodes':    5,
    'log_dir':            os.path.expanduser('~/kinova_rl_logs'),

    # PPO hyper-parameters
    'learning_rate':      3e-4,
    'n_steps':            2048,
    'batch_size':         64,
    'n_epochs':           10,
    'gamma':              0.99,
    'gae_lambda':         0.95,
    'clip_range':         0.2,
    'ent_coef':           0.01,
    'vf_coef':            0.5,
    'max_grad_norm':      0.5,

    # Network
    'net_arch':           [256, 256],
}


# ─────────────────────────────── CALLBACKS ────────────────────────────

class RichLoggingCallback(BaseCallback):
    """
    Prints a live rich-formatted table every N episodes showing:
    - Episode number
    - Total reward
    - Mean distance to target
    - Success rate
    - Entropy (exploration measure — the agent's "confidence")
    - Value estimate (the agent's predicted future reward — its "thought")
    """

    def __init__(self, print_freq: int = 10, verbose: int = 0):
        super().__init__(verbose)
        self.print_freq   = print_freq
        self._ep_count    = 0
        self._rewards     = []
        self._distances   = []
        self._successes   = []
        self._start_time  = time.time()

        try:
            from rich.console import Console
            from rich.table import Table
            from rich.live import Live
            self._rich = True
            self._console = Console()
        except ImportError:
            self._rich = False
            print('[train] rich not installed — using plain logging. '
                  'Install with: pip install rich')

    def _on_step(self) -> bool:
        # Collect episode stats from info buffer
        for info in self.locals.get('infos', []):
            if 'episode' in info:
                self._ep_count += 1
                self._rewards.append(info['episode']['r'])
            if 'distance' in info:
                self._distances.append(info['distance'])
            if 'success' in info and info['success']:
                self._successes.append(1)
            elif 'success' in info:
                self._successes.append(0)

        # Log custom metrics to TensorBoard every step
        if len(self._distances) > 0:
            self.logger.record('custom/mean_distance',
                               float(np.mean(self._distances[-100:])))
        if len(self._successes) > 0:
            self.logger.record('custom/success_rate',
                               float(np.mean(self._successes[-100:])))

        # Print rich table every N episodes
        if self._ep_count > 0 and self._ep_count % self.print_freq == 0:
            self._print_table()

        return True

    def _print_table(self):
        elapsed = time.time() - self._start_time
        recent_r   = self._rewards[-self.print_freq:]
        recent_d   = self._distances[-self.print_freq:] if self._distances else [0]
        recent_s   = self._successes[-self.print_freq:] if self._successes else [0]

        # Get agent thoughts from SB3 internals
        entropy = 0.0
        value   = 0.0
        try:
            entropy = float(self.locals.get('entropy_losses', [0])[-1] or 0)
            value   = float(self.locals.get('values', np.zeros(1)).mean())
        except Exception:
            pass

        if self._rich:
            from rich.table import Table
            from rich.panel import Panel
            t = Table(
                title=f'[bold cyan]Kinova j2n6s300 RL Training[/bold cyan] — '
                      f'Episode {self._ep_count} | '
                      f'Steps {self.num_timesteps:,} | '
                      f'Elapsed {elapsed:.0f}s',
                show_header=True,
                header_style='bold magenta',
            )
            t.add_column('Metric',         style='cyan',  width=30)
            t.add_column('Last Episode',   style='white', width=20)
            t.add_column(f'Mean ({self.print_freq} ep)', style='green', width=20)

            t.add_row('🏆 Episode Reward',
                      f'{recent_r[-1]:.2f}',
                      f'{np.mean(recent_r):.2f}')
            t.add_row('📏 Distance to Target (m)',
                      f'{recent_d[-1]:.4f}' if recent_d else '–',
                      f'{np.mean(recent_d):.4f}' if recent_d else '–')
            t.add_row('✅ Success Rate',
                      '✓' if (recent_s and recent_s[-1]) else '✗',
                      f'{np.mean(recent_s)*100:.1f}%' if recent_s else '0%')
            t.add_row('🎲 Entropy (exploration)',
                      f'{entropy:.4f}', '–')
            t.add_row('💭 Value Estimate (expected Σr)',
                      f'{value:.3f}', '–')
            t.add_row('⚡ Steps/sec',
                      f'{self.num_timesteps / max(elapsed, 1):.0f}', '–')

            self._console.print(t)
        else:
            print(
                f'[Ep {self._ep_count:5d}] '
                f'rew={np.mean(recent_r):7.2f} | '
                f'dist={np.mean(recent_d):.4f}m | '
                f'success={np.mean(recent_s)*100:.1f}% | '
                f'entropy={entropy:.4f} | '
                f'value={value:.3f} | '
                f'steps={self.num_timesteps:,}'
            )


class SuccessRateCallback(BaseCallback):
    """Records success rate to TensorBoard for plotting."""

    def __init__(self):
        super().__init__()
        self._successes = []

    def _on_step(self) -> bool:
        for info in self.locals.get('infos', []):
            if 'success' in info:
                self._successes.append(float(info['success']))
        if self._successes:
            self.logger.record(
                'custom/success_rate_tb',
                float(np.mean(self._successes[-200:])),
            )
        return True


# ─────────────────────────────── MAIN ─────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='Train PPO on Kinova reach task')
    mode = p.add_mutually_exclusive_group()
    mode.add_argument('--headless', action='store_true', default=True,
                      help='Train without visualization (default)')
    mode.add_argument('--viz', action='store_true',
                      help='Train with Gazebo + RViz visualization and '
                           'agent-thoughts publishing (slower)')
    p.add_argument('--resume', type=str, default=None,
                   help='Path to a saved model to resume training from')
    p.add_argument('--timesteps', type=int,
                   default=DEFAULT_CONFIG['total_timesteps'])
    p.add_argument('--log-dir', type=str,
                   default=DEFAULT_CONFIG['log_dir'])
    p.add_argument('--save-freq', type=int,
                   default=DEFAULT_CONFIG['save_freq'])
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    headless = not args.viz

    cfg = DEFAULT_CONFIG.copy()
    cfg['log_dir']     = args.log_dir
    cfg['save_freq']   = args.save_freq
    cfg['total_timesteps'] = args.timesteps

    log_dir    = Path(cfg['log_dir'])
    tb_dir     = log_dir / 'tensorboard'
    ckpt_dir   = log_dir / 'checkpoints'
    best_model = log_dir / 'best_model'

    for d in [log_dir, tb_dir, ckpt_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Save config snapshot
    with open(log_dir / 'train_config.yaml', 'w') as f:
        yaml.dump({**cfg, 'headless': headless, 'seed': args.seed}, f)

    print(f'\n{"="*60}')
    print(f'  Kinova j2n6s300 RL Training')
    print(f'  Mode     : {"HEADLESS" if headless else "VISUALIZATION"}')
    print(f'  Log dir  : {log_dir}')
    print(f'  Timesteps: {cfg["total_timesteps"]:,}')
    print(f'{"="*60}\n')
    if not headless:
        print('  ► Make sure Gazebo + RViz are running (kinova_launch.py)')
        print('  ► Run: ros2 run kinova_rl thoughts  (in another terminal)')
        print(f'  ► TensorBoard: tensorboard --logdir {tb_dir}\n')

    # ── Environment ───────────────────────────────────────────────────
    def make_env():
        env = JacoReachEnv(headless=headless)
        env = Monitor(env)
        return env

    # NOTE: n_envs=1 because we have one Gazebo instance.
    # For parallel training you would need multiple Gazebo instances.
    env = make_vec_env(make_env, n_envs=1, seed=args.seed)

    # Eval env always headless (faster) — separate instance
    eval_env = make_vec_env(
        lambda: Monitor(JacoReachEnv(headless=True)),
        n_envs=1, seed=args.seed + 1,
    )

    # ── Model ─────────────────────────────────────────────────────────
    policy_kwargs = dict(net_arch=cfg['net_arch'])

    if args.resume:
        print(f'[train] Resuming from: {args.resume}')
        model = PPO.load(
            args.resume, env=env,
            learning_rate=cfg['learning_rate'],
        )
    else:
        model = PPO(
            'MlpPolicy',
            env,
            learning_rate    = cfg['learning_rate'],
            n_steps          = cfg['n_steps'],
            batch_size       = cfg['batch_size'],
            n_epochs         = cfg['n_epochs'],
            gamma            = cfg['gamma'],
            gae_lambda       = cfg['gae_lambda'],
            clip_range       = cfg['clip_range'],
            ent_coef         = cfg['ent_coef'],
            vf_coef          = cfg['vf_coef'],
            max_grad_norm    = cfg['max_grad_norm'],
            policy_kwargs    = policy_kwargs,
            tensorboard_log  = str(tb_dir),
            verbose          = 1,
            seed             = args.seed,
        )

    # ── Callbacks ─────────────────────────────────────────────────────
    callbacks = [
        CheckpointCallback(
            save_freq=cfg['save_freq'],
            save_path=str(ckpt_dir),
            name_prefix='ppo_kinova',
        ),
        EvalCallback(
            eval_env,
            best_model_save_path=str(log_dir),
            log_path=str(log_dir),
            eval_freq=cfg['eval_freq'],
            n_eval_episodes=cfg['n_eval_episodes'],
            deterministic=True,
            render=False,
        ),
        RichLoggingCallback(print_freq=10),
        SuccessRateCallback(),
    ]

    # ── Train ─────────────────────────────────────────────────────────
    try:
        model.learn(
            total_timesteps=cfg['total_timesteps'],
            callback=callbacks,
            reset_num_timesteps=args.resume is None,
            tb_log_name='PPO_kinova_reach',
        )
    except KeyboardInterrupt:
        print('\n[train] Interrupted — saving model...')
    finally:
        final_path = str(log_dir / 'final_model')
        model.save(final_path)
        print(f'[train] Model saved to {final_path}.zip')
        env.close()
        eval_env.close()

    print('\n[train] Done.')
    print(f'  Best model : {best_model}.zip')
    print(f'  TensorBoard: tensorboard --logdir {tb_dir}')


if __name__ == '__main__':
    main()
