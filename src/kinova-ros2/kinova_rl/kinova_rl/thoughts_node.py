"""
thoughts_node.py
────────────────
A ROS2 node that subscribes to /rl/agent_thoughts and renders a live
terminal dashboard showing exactly what the RL agent is "thinking"
at every step.

Run this in a separate terminal alongside train.py --viz or enjoy.py:

    ros2 run kinova_rl thoughts

What you will see
─────────────────
  ┌─────────────────────────────────────────────────────────┐
  │       🤖  Kinova j2n6s300 — Agent Thoughts              │
  ├──────────────────┬──────────────────────────────────────┤
  │ Episode Reward   │  ████████░░░░  -12.34                 │
  │ Distance to Goal │  0.1823 m  ████████████░░░░░░        │
  │ Step             │  47 / 200                             │
  │ Value Estimate   │  +23.4   (agent expects this reward)  │
  │ Entropy          │  0.34    (higher = more exploration)  │
  ├──────────────────┴──────────────────────────────────────┤
  │ Actions (joint deltas, rad)                             │
  │  J1 ▶  +0.032  ████████░░░░░                           │
  │  J2 ▶  -0.018  ░░░░████░░░░░                           │
  │  ...                                                    │
  ├─────────────────────────────────────────────────────────┤
  │ Joint Positions (rad)                                   │
  │  J1  4.71  ████████████████░░░                         │
  │  ...                                                    │
  └─────────────────────────────────────────────────────────┘
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray

import time
import threading
import numpy as np

try:
    from rich.console import Console
    from rich.table import Table
    from rich.live import Live
    from rich.panel import Panel
    from rich.progress import BarColumn, Progress, TextColumn
    from rich.layout import Layout
    from rich.text import Text
    from rich import box
    HAS_RICH = True
except ImportError:
    HAS_RICH = False
    print('[thoughts] rich not installed — install with: pip install rich')
    print('[thoughts] Falling back to plain text output.\n')

from kinova_rl.jaco_reach_env import (
    ARM_JOINTS, MAX_STEPS, JOINT_LIMITS_LOW, JOINT_LIMITS_HIGH,
)

# Thought message layout (must match _ROSBridge.publish_thoughts):
# [0]    ep_reward
# [1]    distance
# [2]    step
# [3]    value estimate
# [4]    entropy
# [5:11] action (6 joints)
# [11:17] joint positions


def _bar(value: float, min_val: float, max_val: float,
         width: int = 20, fill: str = '█', empty: str = '░') -> str:
    """Return an ASCII progress bar string."""
    frac = max(0.0, min(1.0, (value - min_val) / max(max_val - min_val, 1e-6)))
    filled = int(frac * width)
    return fill * filled + empty * (width - filled)


class ThoughtsDashboard:
    """Renders the live terminal dashboard using rich."""

    def __init__(self):
        self._data = None
        self._lock = threading.Lock()
        self._last_update = time.time()

    def update(self, data: list):
        with self._lock:
            self._data = data
            self._last_update = time.time()

    def _make_table(self) -> 'Table':
        with self._lock:
            data = self._data
            age  = time.time() - self._last_update

        if data is None or len(data) < 17:
            t = Table(box=box.ROUNDED)
            t.add_column('Status')
            t.add_row('[yellow]Waiting for agent data on /rl/agent_thoughts ...[/yellow]')
            return t

        ep_reward = data[0]
        distance  = data[1]
        step      = int(data[2])
        value     = data[3]
        entropy   = data[4]
        actions   = data[5:11]
        joints    = data[11:17]

        stale = age > 2.0

        t = Table(
            title=(
                f'[bold cyan]🤖 Kinova j2n6s300 — Agent Thoughts[/bold cyan]'
                f'{"  [red][STALE][/red]" if stale else ""}'
            ),
            box=box.ROUNDED,
            expand=True,
        )
        t.add_column('Metric',  style='bold cyan', width=26)
        t.add_column('Value',   style='white',     width=14)
        t.add_column('Visual',  style='green',     min_width=30)

        # Episode reward
        rew_bar = _bar(ep_reward, -MAX_STEPS * 5, MAX_STEPS)
        rew_col = f'[green]{ep_reward:+.2f}[/green]' if ep_reward > 0 \
                  else f'[red]{ep_reward:+.2f}[/red]'
        t.add_row('🏆 Episode Reward', rew_col, rew_bar)

        # Distance
        dist_pct = max(0, 1 - distance / 1.0)
        dist_bar = _bar(dist_pct, 0, 1)
        dist_col = f'[green]{distance:.4f} m[/green]' if distance < 0.1 \
                   else f'[red]{distance:.4f} m[/red]'
        t.add_row('📏 Distance to Target', dist_col, dist_bar)

        # Step
        step_bar = _bar(step, 0, MAX_STEPS)
        t.add_row('⏱️  Step', f'{step} / {MAX_STEPS}', step_bar)

        # Value estimate
        val_col = f'[cyan]{value:+.3f}[/cyan]'
        val_bar = _bar(value, -200, 200)
        t.add_row(
            '💭 Value Estimate\n   (expected Σ reward)',
            val_col, val_bar,
        )

        # Entropy
        ent_bar = _bar(entropy, 0, 3)
        ent_col = f'[magenta]{entropy:.4f}[/magenta]'
        t.add_row(
            '🎲 Entropy\n   (0=exploit, high=explore)',
            ent_col, ent_bar,
        )

        t.add_section()
        t.add_row('[bold]Actions (Δ joint, rad)', '', '')
        for i, (a, jname) in enumerate(zip(actions, ARM_JOINTS)):
            direction = '▶' if a >= 0 else '◀'
            bar = _bar(a, -1, 1, width=20)
            col = f'[green]{a:+.4f}[/green]' if abs(a) > 0.01 \
                  else f'[dim]{a:+.4f}[/dim]'
            t.add_row(f'  J{i+1} {direction}', col, bar)

        t.add_section()
        t.add_row('[bold]Joint Positions (rad)', '', '')
        for i, (j, lo, hi) in enumerate(
                zip(joints, JOINT_LIMITS_LOW, JOINT_LIMITS_HIGH)):
            bar = _bar(j, lo, hi)
            near_limit = (j < lo + 0.1) or (j > hi - 0.1)
            col = f'[red]{j:.4f}[/red]' if near_limit \
                  else f'[white]{j:.4f}[/white]'
            t.add_row(f'  J{i+1}  [{lo:.2f}, {hi:.2f}]', col, bar)

        return t

    def get_plain_text(self) -> str:
        with self._lock:
            data = self._data
        if data is None or len(data) < 17:
            return 'Waiting for agent data on /rl/agent_thoughts ...'

        ep_reward = data[0]
        distance  = data[1]
        step      = int(data[2])
        value     = data[3]
        entropy   = data[4]
        actions   = data[5:11]
        joints    = data[11:17]

        lines = [
            f'=== Agent Thoughts ===',
            f'  Ep Reward : {ep_reward:+.2f}',
            f'  Distance  : {distance:.4f} m',
            f'  Step      : {step}/{MAX_STEPS}',
            f'  Value Est : {value:+.3f}  (expected sum of future rewards)',
            f'  Entropy   : {entropy:.4f}  (higher = more exploratory)',
            f'  Actions   : {[f"{a:+.4f}" for a in actions]}',
            f'  Joints    : {[f"{j:.4f}" for j in joints]}',
        ]
        return '\n'.join(lines)


class ThoughtsNode(Node):

    def __init__(self):
        super().__init__('kinova_rl_thoughts')
        self._dashboard = ThoughtsDashboard()
        self._sub = self.create_subscription(
            Float64MultiArray,
            '/rl/agent_thoughts',
            self._cb,
            10,
        )
        self.get_logger().info(
            '[thoughts] Listening on /rl/agent_thoughts ...\n'
            '           Make sure train.py --viz or enjoy.py is running.'
        )

    def _cb(self, msg: Float64MultiArray):
        self._dashboard.update(list(msg.data))

    @property
    def dashboard(self):
        return self._dashboard


def main():
    rclpy.init()
    node = ThoughtsNode()
    dashboard = node.dashboard

    spin_thread = threading.Thread(
        target=lambda: rclpy.spin(node), daemon=True,
    )
    spin_thread.start()

    if HAS_RICH:
        console = Console()
        with Live(
            dashboard._make_table(),
            console=console,
            refresh_per_second=4,
            screen=False,
        ) as live:
            try:
                while rclpy.ok():
                    live.update(dashboard._make_table())
                    time.sleep(0.25)
            except KeyboardInterrupt:
                pass
    else:
        try:
            while rclpy.ok():
                print(dashboard.get_plain_text())
                print()
                time.sleep(1.0)
        except KeyboardInterrupt:
            pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
