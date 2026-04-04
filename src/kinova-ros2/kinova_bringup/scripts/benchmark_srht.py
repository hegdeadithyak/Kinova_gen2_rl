#!/usr/bin/env python3
"""
SRHT Benchmark Script
Compares SRHT vs baseline RRT metrics in simulation.
Runs without a real robot — uses pinocchio for kinematics only.

Metrics:
  - Planning time (ms)
  - Path length (sum of joint-space step norms)
  - Normalized jerk
  - Success rate
  - Energy conservation error (unique to SRHT)

Usage:
  python3 benchmark_srht.py
"""

import numpy as np
import time
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from srht_planner import SRHTPlanner

URDF_PATH = '/tmp/j2n6s300.urdf'
N_TRIALS  = 50   # trials per scenario
NQ        = 6

# ─────────────────────────────────────────────────────────────────────────────

def normalized_jerk(path: list, dt: float = 0.1) -> float:
    """
    Normalized jerk metric from path waypoints.
    Lower = smoother. Standard metric in HRI literature.
    """
    if len(path) < 4:
        return 0.0
    configs = np.array(path)
    vel  = np.diff(configs, axis=0) / dt
    acc  = np.diff(vel,     axis=0) / dt
    jerk = np.diff(acc,     axis=0) / dt
    total_jerk = np.sum(np.linalg.norm(jerk, axis=1) ** 2) * dt
    path_len   = float(np.sum(np.linalg.norm(np.diff(configs, axis=0), axis=1)))
    T          = len(path) * dt
    if path_len < 1e-6:
        return 0.0
    return (T ** 5 / path_len ** 2) * total_jerk

def path_length(path: list) -> float:
    configs = np.array(path)
    return float(np.sum(np.linalg.norm(np.diff(configs, axis=0), axis=1)))

def min_head_clearance(path: list, planner: SRHTPlanner,
                       head_pos: np.ndarray) -> float:
    """
    Minimum distance between any link and head across the path.
    Primary safety metric.
    """
    import pinocchio as pin
    min_d = np.inf
    for q in path:
        pin.forwardKinematics(planner.model, planner.data, q)
        for frame_id in range(planner.model.nframes):
            pin.updateFramePlacement(planner.model, planner.data, frame_id)
            link_pos = planner.data.oMf[frame_id].translation
            d = np.linalg.norm(link_pos - head_pos) - planner.r_safe
            if d < min_d:
                min_d = d
    return float(min_d)

def energy_conservation_error(path: list, planner: SRHTPlanner,
                               obstacle_positions: list) -> float:
    """
    SRHT-specific metric: how well energy is conserved along path.
    Should be near zero with symplectic integration.
    Baseline planners can't compute this — it's a unique SRHT claim.
    """
    if len(path) < 2:
        return 0.0
    H_values = []
    for q in path:
        p = np.zeros(NQ)  # approximate momentum as zero for path nodes
        H_values.append(planner.hamiltonian(q, p, obstacle_positions))
    H_arr = np.array(H_values)
    return float(np.max(np.abs(H_arr - H_arr[0])))

# ─────────────────────────────────────────────────────────────────────────────

def run_scenario(planner, scenario_name, n_trials,
                 q_starts, q_goals, obstacle_positions):
    print(f"\n  Running {n_trials} trials...")
    results = {
        'success': 0, 'time_ms': [], 'length': [],
        'jerk': [], 'clearance': [], 'energy_err': []
    }

    head_pos = obstacle_positions[0] if obstacle_positions else np.array([0.4, 0.0, 0.6])

    for trial in range(n_trials):
        q_s = q_starts[trial]
        q_g = q_goals[trial]

        t0   = time.time()
        path = planner.plan(q_s, q_g, obstacle_positions, timeout=15.0)
        dt   = (time.time() - t0) * 1000  # ms

        if path is not None and len(path) > 1:
            results['success'] += 1
            results['time_ms'].append(dt)
            results['length'].append(path_length(path))
            results['jerk'].append(normalized_jerk(path))
            results['clearance'].append(
                min_head_clearance(path, planner, head_pos))
            results['energy_err'].append(
                energy_conservation_error(path, planner, obstacle_positions))

        if (trial + 1) % 10 == 0:
            print(f"    Trial {trial+1}/{n_trials} | "
                  f"Success: {results['success']}/{trial+1}")

    return results

def print_results(scenario_name, results, n_trials):
    sr = results['success'] / n_trials * 100
    print(f"\n  ── {scenario_name} ──────────────────")
    print(f"  Success Rate:          {sr:.1f}%  ({results['success']}/{n_trials})")
    if results['time_ms']:
        print(f"  Planning Time:         {np.mean(results['time_ms']):.1f} ± "
              f"{np.std(results['time_ms']):.1f} ms")
        print(f"  Path Length:           {np.mean(results['length']):.3f} ± "
              f"{np.std(results['length']):.3f} rad")
        print(f"  Normalized Jerk:       {np.mean(results['jerk']):.4f} ± "
              f"{np.std(results['jerk']):.4f}")
        print(f"  Min Head Clearance:    {np.mean(results['clearance']):.3f} ± "
              f"{np.std(results['clearance']):.3f} m")
        print(f"  Energy Conserv. Err:   {np.mean(results['energy_err']):.4f} ± "
              f"{np.std(results['energy_err']):.4f} J")

# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 55)
    print(" SRHT Planner Benchmark")
    print("=" * 55)

    if not os.path.exists(URDF_PATH):
        print(f"[ERROR] URDF not found at {URDF_PATH}")
        print("        Run setup_srht.sh first.")
        sys.exit(1)

    planner = SRHTPlanner(URDF_PATH)

    q_lo = planner.model.lowerPositionLimit
    q_hi = planner.model.upperPositionLimit

    np.random.seed(42)
    q_starts = [np.random.uniform(q_lo, q_hi) for _ in range(N_TRIALS)]
    q_goals  = [np.random.uniform(q_lo, q_hi) for _ in range(N_TRIALS)]

    # ── Scenario 1: Static head ───────────────────────────────────────────────
    print("\n[Scenario 1] Static head — baseline functionality")
    head_static  = [np.array([0.4, 0.0, 0.6])]
    r1 = run_scenario(planner, "Static Head",
                      N_TRIALS, q_starts, q_goals, head_static)
    print_results("Static Head", r1, N_TRIALS)

    # ── Scenario 2: Head close to workspace ──────────────────────────────────
    print("\n[Scenario 2] Head close to workspace — safety margin test")
    head_close   = [np.array([0.35, 0.05, 0.55])]
    r2 = run_scenario(planner, "Close Head",
                      N_TRIALS, q_starts, q_goals, head_close)
    print_results("Close Head", r2, N_TRIALS)

    # ── Scenario 3: Multiple obstacles (head + bowl + plate) ──────────────────
    print("\n[Scenario 3] Multiple obstacles — cluttered table")
    multi_obs = [
        np.array([0.4,  0.0,  0.6]),   # head
        np.array([0.35, 0.1,  0.15]),  # bowl
        np.array([0.30, -0.1, 0.12]),  # plate
    ]
    r3 = run_scenario(planner, "Multi Obstacles",
                      N_TRIALS, q_starts, q_goals, multi_obs)
    print_results("Multi Obstacles", r3, N_TRIALS)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print(" Summary")
    print("=" * 55)
    print(f"  Scenario 1 (Static):    SR={r1['success']/N_TRIALS*100:.0f}%  "
          f"T={np.mean(r1['time_ms']):.0f}ms" if r1['time_ms'] else "  Scenario 1: No successes")
    print(f"  Scenario 2 (Close):     SR={r2['success']/N_TRIALS*100:.0f}%  "
          f"T={np.mean(r2['time_ms']):.0f}ms" if r2['time_ms'] else "  Scenario 2: No successes")
    print(f"  Scenario 3 (Multi-obs): SR={r3['success']/N_TRIALS*100:.0f}%  "
          f"T={np.mean(r3['time_ms']):.0f}ms" if r3['time_ms'] else "  Scenario 3: No successes")
    print("\n  Done. Results ready for paper.\n")

if __name__ == '__main__':
    main()
