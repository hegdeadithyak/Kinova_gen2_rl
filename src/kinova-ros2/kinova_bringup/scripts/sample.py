import sys, os, numpy as np
sys.path.insert(0, os.path.expanduser(
    '~/kinova_ws/src/kinova-ros2/kinova_bringup/scripts'))

from elliptic_delivery import EllipticDelivery

p = EllipticDelivery('/tmp/j2n6s300.urdf', lift_height=0.10,
                      n_waypoints=40, total_time=4.0)

q_cur = np.zeros(6)
A     = np.array([0.30,  0.10, 0.30])   # fork above plate
B     = np.array([0.45, -0.05, 0.55])   # mouth position

result = p.plan(q_cur, A, B)

arc = result['cartesian_arc']
print(f"\n  Arc apex (highest point): z={arc[:,2].max():.3f}m")
print(f"  Arc start: {np.round(arc[0], 3)}")
print(f"  Arc end:   {np.round(arc[-1], 3)}")

# Check minimum-jerk profile shape
ts = result['timestamps']
dt = np.diff(ts)
print(f"\n  Time profile (should be slow-fast-slow):")
print(f"    First interval: {dt[0]*1000:.1f}ms")
print(f"    Middle interval:{dt[len(dt)//2]*1000:.1f}ms  ← fastest here")
print(f"    Last interval:  {dt[-1]*1000:.1f}ms")

print(f"\n  Joint configs: {len(result['joint_configs'])} waypoints")
print("\nSmoke test passed.")