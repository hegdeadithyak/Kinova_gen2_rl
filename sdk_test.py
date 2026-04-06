#!/usr/bin/env python3
import ctypes
import os
import time
import sys
import termios
import tty

# ── SDK PATH ─────────────────────────────────────────────
SDK_PATH = "/home/amma/kinova_ws/rl_v2-master/src/kinova-ros2/kinova_driver/lib/x86_64-linux-gnu/USBCommandLayerUbuntu.so"

# ── LOAD SDK ─────────────────────────────────────────────
try:
    lib = ctypes.CDLL(SDK_PATH)
    print("✅ SDK loaded successfully")
except OSError as e:
    print(f"❌ Failed to load SDK: {e}")
    exit(1)

# ── STRUCTS (CORRECT KINOVA FORMAT) ──────────────────────

class AngularInfo(ctypes.Structure):
    _fields_ = [(f'Actuator{i}', ctypes.c_float) for i in range(1, 8)]

class FingersPosition(ctypes.Structure):
    _fields_ = [
        ("Finger1", ctypes.c_float),
        ("Finger2", ctypes.c_float),
        ("Finger3", ctypes.c_float),
    ]

class AngularPosition(ctypes.Structure):
    _fields_ = [
        ("Actuators", AngularInfo),
        ("Fingers", FingersPosition),
    ]

class TrajectoryPoint(ctypes.Structure):
    _fields_ = [
        ("PositionType", ctypes.c_int),
        ("Actuator1", ctypes.c_float),
        ("Actuator2", ctypes.c_float),
        ("Actuator3", ctypes.c_float),
        ("Actuator4", ctypes.c_float),
        ("Actuator5", ctypes.c_float),
        ("Actuator6", ctypes.c_float),
        ("Finger1", ctypes.c_float),
        ("Finger2", ctypes.c_float),
        ("Finger3", ctypes.c_float),
    ]

# ── HELPER FUNCTIONS ─────────────────────────────────────

def check_result(name, result):
    if result != 1:
        print(f"❌ {name} failed with code: {result}")
        return False
    print(f"✅ {name} success")
    return True

def get_key():
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        return sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)

def send_joint_command(lib, pos):
    pt = TrajectoryPoint()
    pt.PositionType = 2  # ANGULAR_POSITION

    pt.Actuator1 = pos.Actuators.Actuator1
    pt.Actuator2 = pos.Actuators.Actuator2
    pt.Actuator3 = pos.Actuators.Actuator3
    pt.Actuator4 = pos.Actuators.Actuator4
    pt.Actuator5 = pos.Actuators.Actuator5
    pt.Actuator6 = pos.Actuators.Actuator6

    pt.Finger1 = pos.Fingers.Finger1
    pt.Finger2 = pos.Fingers.Finger2
    pt.Finger3 = pos.Fingers.Finger3

    lib.SendBasicTrajectory(pt)

# ── MAIN ────────────────────────────────────────────────

def main():

    # Init
    res = lib.InitAPI()
    if not check_result("InitAPI", res):
        return

    time.sleep(1)

    # Start control
    res = lib.StartControlAPI()
    if not check_result("StartControlAPI", res):
        lib.CloseAPI()
        return

    time.sleep(1)

    # Set angular control
    if hasattr(lib, "SetAngularControl"):
        res = lib.SetAngularControl()
        if not check_result("SetAngularControl", res):
            lib.StopControlAPI()
            lib.CloseAPI()
            return
    else:
        print("⚠️ SetAngularControl not found")

    time.sleep(1)

    # Get current joint state
    pos = AngularPosition()
    res = lib.GetAngularPosition(ctypes.byref(pos))

    if not check_result("GetAngularPosition", res):
        print("❌ Cannot read joint state")
        lib.StopControlAPI()
        lib.CloseAPI()
        return

    print("\n📡 Initial Joint Values:")
    print(f"J5: {pos.Actuators.Actuator5:.2f}")

    print("\n🎮 Control Joint 5")
    print("i → increase")
    print("k → decrease")
    print("q → quit")

    STEP = 5.0  # degrees

    # ── CONTROL LOOP ─────────────────────────

    while True:
        key = get_key()

        if key == 'q':
            break

        elif key == 'i':
            pos.Actuators.Actuator5 += STEP

        elif key == 'k':
            pos.Actuators.Actuator5 -= STEP

        else:
            continue

        # keep in [0, 360)
        pos.Actuators.Actuator5 = pos.Actuators.Actuator5 % 360

        send_joint_command(lib, pos)

        print(f"J5 → {pos.Actuators.Actuator5:.2f}")

    # ── CLEANUP ─────────────────────────────

    lib.StopControlAPI()
    lib.CloseAPI()
    print("\n🔌 API closed")

# ── ENTRY POINT ────────────────────────────

if __name__ == "__main__":
    main()