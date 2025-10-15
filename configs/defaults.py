import numpy as np
import os

ROOT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

ROBOT_MODEL_FILE = "assets/harpy_sophisticated_complete.xml"

ACTUATED_JOINT_RANGE = np.array([
    [
        # 10 Motor Actions (ordered as used in __set_cassie_out)
        -0.2618,  # 0: left-hip-roll: -15° to 22.5° (mpos[0] → leftLeg.hipRollDrive)
        -0.3927,  # 1: left-hip-yaw: -22.5° to 22.5° (mpos[1] → leftLeg.hipYawDrive)
        -0.8727,  # 2: left-hip-pitch: -50° to 80° (mpos[2] → leftLeg.hipPitchDrive)
        -2.8623,  # 3: left-knee: -164° to -37° (mpos[3] → leftLeg.kneeDrive)
        -2.4435,  # 4: left-foot: -140° to -30° (mpos[4] → leftLeg.footDrive)
        -0.3927,  # 5: right-hip-roll: -22.5° to 15° (mpos[5] → rightLeg.hipRollDrive)
        -0.3927,  # 6: right-hip-yaw: -22.5° to 22.5° (mpos[6] → rightLeg.hipYawDrive)
        -0.8727,  # 7: right-hip-pitch: -50° to 80° (mpos[7] → rightLeg.hipPitchDrive)
        -2.8623,  # 8: right-knee: -164° to -37° (mpos[8] → rightLeg.kneeDrive)
        -2.4435,  # 9: right-foot: -140° to -30° (mpos[9] → rightLeg.footDrive)
        # 2 Thruster Force Actions (site-based forces)
        0.0,      # 10: left thruster force (0-1 normalized)
        0.0,      # 11: right thruster force (0-1 normalized)
        # 2 Thruster Pitch Actions (joints)
        -1.5708,  # 12: thruster_left_pitch: -90° to 90° (joint)
        -1.5708,  # 13: thruster_right_pitch: -90° to 90° (joint)
    ],
    [
        # 10 Motor Actions (ordered as used in __set_cassie_out)
        0.3927,   # 0: left-hip-roll: -15° to 22.5°
        0.3927,   # 1: left-hip-yaw: -22.5° to 22.5°
        1.3963,   # 2: left-hip-pitch: -50° to 80°
        -0.6458,  # 3: left-knee: -164° to -37°
        -0.5236,  # 4: left-foot: -140° to -30°
        0.2618,   # 5: right-hip-roll: -22.5° to 15°
        0.3927,   # 6: right-hip-yaw: -22.5° to 22.5°
        1.3963,   # 7: right-hip-pitch: -50° to 80°
        -0.6458,  # 8: right-knee: -164° to -37°
        -0.5236,  # 9: right-foot: -140° to -30°
        # 2 Thruster Force Actions (site-based forces)
        1.0,      # 10: left thruster force (0-1 normalized)
        1.0,      # 11: right thruster force (0-1 normalized)
        # 2 Thruster Pitch Actions (joints)
        1.5708,   # 12: thruster_left_pitch: -90° to 90°
        1.5708,   # 13: thruster_right_pitch: -90° to 90°
    ],
])


FALLING_THRESHOLD = 0.55
TARSUS_HITGROUND_THRESHOLD = 0.15

DEFAULT_PGAIN = np.array([
    # Left leg motors
    400, 200, 200, 500, 20,  # left-hip-roll, left-hip-yaw, left-hip-pitch, left-knee, left-foot
    # Right leg motors  
    400, 200, 200, 500, 20,  # right-hip-roll, right-hip-yaw, right-hip-pitch, right-knee, right-foot
    # Thruster pitch motors
    100, 100,  # thruster_left_pitch, thruster_right_pitch (lower gains for stability)
])
DEFAULT_DGAIN = np.array([
    # Left leg motors
    4, 4, 10, 20, 4,  # left-hip-roll, left-hip-yaw, left-hip-pitch, left-knee, left-foot
    # Right leg motors
    4, 4, 10, 20, 4,  # right-hip-roll, right-hip-yaw, right-hip-pitch, right-knee, right-foot
    # Thruster pitch motors
    2, 2,  # thruster_left_pitch, thruster_right_pitch (lower damping for smoother control)
])

# NOTE: this safe torque range is used in the cassiemujoco_ctypes, hardcoded, this is not used in the python env
# Torque limits for all 12 motorized joints
TORQUE_LB = np.array([
    # Left leg motors
    -80.0, -60.0, -80.0, -190.0, -45.0,  # left-hip-roll, left-hip-yaw, left-hip-pitch, left-knee, left-foot
    # Right leg motors
    -80.0, -60.0, -80.0, -190.0, -45.0,  # right-hip-roll, right-hip-yaw, right-hip-pitch, right-knee, right-foot
    # Thruster pitch motors (lower torque limits for safety)
    -20.0, -20.0,  # thruster_left_pitch, thruster_right_pitch
])

TORQUE_UB = np.array([
    # Left leg motors
    80.0, 60.0, 80.0, 190.0, 45.0,  # left-hip-roll, left-hip-yaw, left-hip-pitch, left-knee, left-foot
    # Right leg motors
    80.0, 60.0, 80.0, 190.0, 45.0,  # right-hip-roll, right-hip-yaw, right-hip-pitch, right-knee, right-foot
    # Thruster pitch motors (lower torque limits for safety)
    20.0, 20.0,  # thruster_left_pitch, thruster_right_pitch
])


STANDING_POSE = np.array(
    [
        0.0,
        0.0,
        0.95,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.4544,
        -1.21,
        -1.643,
        0.0,
        0.0,
        0.4544,
        -1.21,
        -1.643,
    ]
)
