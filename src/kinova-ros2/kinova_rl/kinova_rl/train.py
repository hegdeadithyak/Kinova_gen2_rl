"""
train.py  (v2 — SAC + HER)
───────────────────────────
Trains a SAC agent with Hindsight Experience Replay on the
Kinova j2n6s300 reach task.

Why SAC instead of PPO?
  SAC is off-policy and works with HER. PPO is on-policy and cannot
  use HER. For sparse goal-reaching tasks like this, SAC+HER is the
  standard approach and typically converges 5-10x faster.

Why HER?
  After every failed episode, HER replays the trajectory pretending
  the place the arm actually ended up was the intended goal.
  This means the agent gets a positive learning signal from EVERY
  episode, even when it never touches the real target sphere.

Usage
─────
  # Headless training (recommended):
  ros2 run kinova_rl train -- --headless

  # Visualization + thoughts dashboard:
  ros2 run kinova_rl train -- --viz

  # Resume:
  ros2 run kinova_rl train -- --headless --resume ~/kinova_rl_logs/best_model

TensorBoard
───────────
  tensorboard --logdir ~/kinova_rl_logs/tensorboard
  Open http://localhost:6006

  Key plots to watch:
    custom/success_rate      → should climb toward 1.0
    custom/mean_distance     → should fall toward 0.05 m
    rollout/ep_rew_mean      → should increase
    train/actor_loss         → should be stable/decreasing
    train/critic_loss        → should decrease then plateau
    train/ent_coef           → entropy coefficient (auto-tuned by SAC)
"""

import os
import argparse
import time
import yaml
from pathlib import Path

import numpy as np

from stable_baselines3 import SAC
from stable_baselines3.her.her_replay_buffer import HerReplayBuffer
from stable_baselines3.common.callbacks import (
    BaseCallback, CheckpointCallback, EvalCallback,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

from kinova_rl.jaco_reach_env import JacoReachEnv


# ─────────────────────────────── CONFIG ───────────────────────────────
DEFAULT_CONFIG = {
    'total_timesteps': 500_000,      # SAC+HER converges much faster than PPO
    'save_freq':       10_000,
    'eval_freq':       5_000,
    'n_eval_episodes': 5,
    'log_dir':         os.path.expanduser('~/kinova_rl_logs'),

    # SAC hyper-parameters (well-tuned for continuous reach tasks)
    'learning_rate':        3e-4,
    'buffer_size':          200_000,  # replay buffer size
    'learning_starts':      1_0,    # random exploration steps before training
    'batch_size':           256,
    'tau':                  0.005,    # soft update coefficient
    'gamma':                0.98,     # slightly lower γ for shorter horizons
    'train_freq':           1,
    'gradient_steps':       1,
    'ent_coef':             'auto',   # SAC auto-tunes entropy

    # HER
    'her_n_sampled_goal':   4,        # HER relabels 4 extra goals per transition
    'her_goal_selection_strategy': 'future',  # best strategy for reach tasks

    # Network
    'net_arch': [256, 256, 256],      # slightly deeper for HER
}


# ─────────────────────────────── CALLBACKS ────────────────────────────

class RichLoggingCallback(BaseCallback):
    """
    Prints a live table every N episodes.
    Shows: reward, distance, success rate, SAC entropy coefficient,
    Q-value estimates (critic's belief about how good each state is).
    """

    def __init__(self, print_freq: int = 10, verbose: int = 0):
        super().__init__(verbose)
        self.print_freq  = print_freq
        self._ep_count   = 0
        self._rewards    = []
        self._distances  = []
        self._successes  = []
        self._shapings   = []
        self._start_time = time.time()

        try:
            from rich.console import Console
            self._rich    = True
            self._console = Console()
        except ImportError:
            self._rich = False

    def _on_step(self) -> bool:
        for info in self.locals.get('infos', []):
            if 'episode' in info:
                self._ep_count += 1
                self._rewards.append(info['episode']['r'])
            if 'distance'  in info: self._distances.append(info['distance'])
            if 'is_success' in info: self._successes.append(float(info['is_success']))
            if 'r_shaping'  in info: self._shapings.append(info['r_shaping'])

        if self._distances:
            self.logger.record('custom/mean_distance',
                               float(np.mean(self._distances[-200:])))
        if self._successes:
            self.logger.record('custom/success_rate',
                               float(np.mean(self._successes[-200:])))

        if self._ep_count > 0 and self._ep_count % self.print_freq == 0:
            self._print()
        return True

    def _print(self):
        elapsed   = time.time() - self._start_time
        recent_r  = self._rewards[-self.print_freq:]
        recent_d  = self._distances[-self.print_freq:] if self._distances else [0]
        recent_s  = self._successes[-self.print_freq:] if self._successes else [0]
        recent_sh = self._shapings[-self.print_freq:]  if self._shapings  else [0]

        ent_coef = 0.0
        q_val    = 0.0
        try:
            ent_coef = float(self.model.ent_coef_tensor.item())
            # Approximate mean Q-value from logger
            q_val = float(
                self.locals.get('critic_values', np.zeros(1)).mean()
                if hasattr(self.locals.get('critic_values', None), 'mean')
                else 0.0
            )
        except Exception:
            pass

        if self._rich:
            from rich.table import Table
            t = Table(
                title=(
                    f'[bold cyan]Kinova j2n6s300 — SAC+HER Training[/bold cyan]  '
                    f'Ep {self._ep_count} | Steps {self.num_timesteps:,} | '
                    f'{elapsed:.0f}s'
                ),
                show_header=True, header_style='bold magenta',
            )
            t.add_column('Metric',                  style='cyan',  width=36)
            t.add_column('Last episode',            style='white', width=18)
            t.add_column(f'Mean ({self.print_freq} ep)', style='green', width=18)

            t.add_row('🏆 Episode Reward',
                      f'{recent_r[-1]:.2f}', f'{np.mean(recent_r):.2f}')
            t.add_row('📏 Distance to Target (m)',
                      f'{recent_d[-1]:.4f}' if recent_d else '–',
                      f'{np.mean(recent_d):.4f}')
            t.add_row('✅ Success Rate',
                      '✓' if (recent_s and recent_s[-1]) else '✗',
                      f'{np.mean(recent_s)*100:.1f}%')
            t.add_row('🔥 Reward Shaping (Φ term)',
                      f'{recent_sh[-1]:.3f}' if recent_sh else '–',
                      f'{np.mean(recent_sh):.3f}' if recent_sh else '–')
            t.add_row('🎲 Entropy coef (α, auto-tuned)',
                      f'{ent_coef:.5f}', '→ decreases as policy converges')
            t.add_row('💭 Q-value estimate',
                      f'{q_val:.3f}', '→ critic\'s belief about state value')
            t.add_row('⚡ Steps/sec',
                      f'{self.num_timesteps / max(elapsed, 1):.0f}', '–')
            self._console.print(t)
        else:
            print(
                f'[Ep {self._ep_count:5d}] '
                f'rew={np.mean(recent_r):7.2f} | '
                f'dist={np.mean(recent_d):.4f}m | '
                f'success={np.mean(recent_s)*100:.1f}% | '
                f'ent_coef={ent_coef:.5f} | '
                f'steps={self.num_timesteps:,}'
            )




def parse_args():
    p = argparse.ArgumentParser(description='Train SAC+HER on Kinova reach task')
    mode = p.add_mutually_exclusive_group()
    mode.add_argument('--headless', action='store_true', default=True)
    mode.add_argument('--viz',      action='store_true',
                      help='Train with Gazebo + RViz + thoughts dashboard')
    p.add_argument('--resume',    type=str, default=None)
    p.add_argument('--timesteps', type=int, default=DEFAULT_CONFIG['total_timesteps'])
    p.add_argument('--log-dir',   type=str, default=DEFAULT_CONFIG['log_dir'])
    p.add_argument('--seed',      type=int, default=42)
    return p.parse_args()


def main():
    args     = parse_args()
    headless = not args.viz
    cfg      = DEFAULT_CONFIG.copy()
    cfg['log_dir']         = args.log_dir
    cfg['total_timesteps'] = args.timesteps

    log_dir  = Path(cfg['log_dir'])
    tb_dir   = log_dir / 'tensorboard'
    ckpt_dir = log_dir / 'checkpoints'
    for d in [log_dir, tb_dir, ckpt_dir]:
        d.mkdir(parents=True, exist_ok=True)

    with open(log_dir / 'train_config.yaml', 'w') as f:
        yaml.dump({**cfg, 'headless': headless, 'seed': args.seed,
                   'algorithm': 'SAC+HER'}, f)

    print(f'\n{"="*62}')
    print(f'  Kinova j2n6s300 RL — SAC + Hindsight Experience Replay')
    print(f'  Mode      : {"HEADLESS" if headless else "VISUALIZATION"}')
    print(f'  Log dir   : {log_dir}')
    print(f'  Timesteps : {cfg["total_timesteps"]:,}')
    print(f'  Algorithm : SAC + HER (future strategy, {cfg["her_n_sampled_goal"]} goals/step)')
    print(f'{"="*62}')
    if not headless:
        print('  ► Gazebo + RViz must already be running (kinova_launch.py)')
        print('  ► Run: ros2 run kinova_rl thoughts   (live dashboard)')
        print(f'  ► TensorBoard: tensorboard --logdir {tb_dir}\n')

    # ── Environment ───────────────────────────────────────────────────
    # NOTE: HerReplayBuffer requires n_envs=1 (SB3 limitation)
    env = Monitor(JacoReachEnv(headless=headless))

    eval_env = Monitor(JacoReachEnv(headless=True))

    # ── Model: SAC + HER ──────────────────────────────────────────────
    policy_kwargs = dict(net_arch=cfg['net_arch'])

    if args.resume:
        print(f'[train] Resuming from: {args.resume}')
        model = SAC.load(args.resume, env=env)
    else:
        model = SAC(
            'MultiInputPolicy',         # required for Dict observation spaces
            env,
            replay_buffer_class=HerReplayBuffer,
            replay_buffer_kwargs={
                'n_sampled_goal':        cfg['her_n_sampled_goal'],
                'goal_selection_strategy': cfg['her_goal_selection_strategy'],
            },
            learning_rate   = cfg['learning_rate'],
            buffer_size     = cfg['buffer_size'],
            learning_starts = cfg['learning_starts'],
            batch_size      = cfg['batch_size'],
            tau             = cfg['tau'],
            gamma           = cfg['gamma'],
            train_freq      = cfg['train_freq'],
            gradient_steps  = cfg['gradient_steps'],
            ent_coef        = cfg['ent_coef'],
            policy_kwargs   = policy_kwargs,
            tensorboard_log = str(tb_dir),
            verbose         = 1,
            seed            = args.seed,
        )

    print(f'\n[train] Replay buffer: {cfg["buffer_size"]:,} transitions')
    print(f'[train] HER: {cfg["her_n_sampled_goal"]} relabelled goals per real transition')
    print(f'[train] Exploration starts after {cfg["learning_starts"]:,} random steps\n')

    # ── Callbacks ─────────────────────────────────────────────────────
    callbacks = [
        CheckpointCallback(
            save_freq=cfg['save_freq'],
            save_path=str(ckpt_dir),
            name_prefix='sac_her_kinova',
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
    ]

    # ── Train ─────────────────────────────────────────────────────────
    try:
        model.learn(
            total_timesteps=cfg['total_timesteps'],
            callback=callbacks,
            reset_num_timesteps=args.resume is None,
            tb_log_name='SAC_HER_kinova_reach',
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print('\n[train] Interrupted — saving...')
    finally:
        final = str(log_dir / 'final_model')
        model.save(final)
        print(f'\n[train] Saved → {final}.zip')
        env.close()
        eval_env.close()

    print(f'\n[train] Best model : {log_dir}/best_model.zip')
    print(f'[train] TensorBoard: tensorboard --logdir {tb_dir}')


if __name__ == '__main__':
    main()