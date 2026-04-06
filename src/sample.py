#!/usr/bin/env python3
import ctypes, time

SDK = ctypes.CDLL(
    '/home/amma/Downloads/rl_v2-master-master/src/kinova-ros2/kinova_driver/lib/x86_64-linux-gnu/USBCommandLayerUbuntu.so'
)

class AngularInfo(ctypes.Structure):
    _fields_ = [(f'Actuator{i}', ctypes.c_float) for i in range(1, 9)]

class FingersPosition(ctypes.Structure):
    _fields_ = [('Finger1', ctypes.c_float),
                ('Finger2', ctypes.c_float),
                ('Finger3', ctypes.c_float)]

class CartesianInfo(ctypes.Structure):
    _fields_ = [('X',      ctypes.c_float), ('Y',      ctypes.c_float),
                ('Z',      ctypes.c_float), ('ThetaX', ctypes.c_float),
                ('ThetaY', ctypes.c_float), ('ThetaZ', ctypes.c_float)]

class UserPosition(ctypes.Structure):
    _fields_ = [('Type',              ctypes.c_int),
                ('Delay',             ctypes.c_float),
                ('CartesianPosition', CartesianInfo),
                ('Actuators',         AngularInfo),
                ('HandMode',          ctypes.c_int),
                ('Fingers',           FingersPosition)]

class Limitation(ctypes.Structure):
    _fields_ = [('speedParameter1',        ctypes.c_float),
                ('speedParameter2',        ctypes.c_float),
                ('speedParameter3',        ctypes.c_float),
                ('forceParameter1',        ctypes.c_float),
                ('forceParameter2',        ctypes.c_float),
                ('forceParameter3',        ctypes.c_float),
                ('accelerationParameter1', ctypes.c_float),
                ('accelerationParameter2', ctypes.c_float),
                ('accelerationParameter3', ctypes.c_float)]

class TrajectoryPoint(ctypes.Structure):
    _fields_ = [('Position',          UserPosition),
                ('LimitationsActive', ctypes.c_int),
                ('SynchroType',       ctypes.c_int),
                ('Limitations',       Limitation)]

print(f"Struct sizes — UserPosition:{ctypes.sizeof(UserPosition)} "
      f"TrajectoryPoint:{ctypes.sizeof(TrajectoryPoint)}")

print("Initialising API...")
SDK.InitAPI()
time.sleep(1.0)

# Clear any existing trajectory
ret = SDK.EraseAllTrajectories()
print(f"EraseAllTrajectories: {ret}")
time.sleep(0.5)

ret = SDK.SetAngularControl()
print(f"SetAngularControl: {ret}")
time.sleep(0.5)

# Read current
angles = AngularInfo()
SDK.GetAngularPosition(ctypes.byref(angles))
print(f"\nCurrent: J1:{angles.Actuator1:.2f}  J2:{angles.Actuator2:.2f}  "
      f"J3:{angles.Actuator3:.2f}  J4:{angles.Actuator4:.2f}  "
      f"J5:{angles.Actuator5:.2f}  J6:{angles.Actuator6:.2f}")

point = TrajectoryPoint()
point.Position.Type                = 2   # ANGULAR_POSITION
point.Position.Delay               = 0.0
point.Position.Actuators.Actuator1 = angles.Actuator1 
point.Position.Actuators.Actuator2 = angles.Actuator2 +5.0
point.Position.Actuators.Actuator3 = angles.Actuator3
point.Position.Actuators.Actuator4 = angles.Actuator4
point.Position.Actuators.Actuator5 = angles.Actuator5
point.Position.Actuators.Actuator6 = angles.Actuator6
point.Position.Actuators.Actuator7 = 0.0
point.Position.Actuators.Actuator8 = 0.0
point.Position.HandMode            = 0
point.LimitationsActive            = 0
point.SynchroType                  = 0

print(f"\nTarget J1: {point.Position.Actuators.Actuator1:.2f}")
input("Press ENTER to execute (Ctrl+C to abort)...")

ret = SDK.SendBasicTrajectory(ctypes.byref(point))
print(f"SendBasicTrajectory return: {ret}  (1=success)")

print("\nMonitoring...")
for _ in range(40):
    SDK.GetAngularPosition(ctypes.byref(angles))
    print(f"  J1:{angles.Actuator1:7.2f}  J2:{angles.Actuator2:7.2f}  "
          f"J3:{angles.Actuator3:7.2f}  J4:{angles.Actuator4:7.2f}  "
          f"J5:{angles.Actuator5:7.2f}  J6:{angles.Actuator6:7.2f}")
    time.sleep(0.2)

SDK.CloseAPI()
