import numpy as np
from rlenv.gait_library import GaitLibrary
from rlenv.cmd_generator import CmdGenerateor
from configs.defaults import STANDING_POSE

'''
A generator for reference motion, which will be used in cassie_env.py. 
The env script only look at the ref_dict returned from this class. 
ref_dict contains all the info of the refernece motion, 
including reference motor position, motor velocity, base translational/rotational position/velocity
The ReferenceGenerator composes the data from cmd_generator.py & gait_libraray.py
and randomly set the robot to stand for a random timespan. 
'''


class ReferenceGenerator:
    def __init__(
        self,
        env_max_timesteps,
        secs_per_env_step,
        config,
    ):
        self.cmd_generator = CmdGenerateor(env_max_timesteps, secs_per_env_step, config)
        self.gait_library = GaitLibrary(secs_per_env_step)
        self.time_stand_transit_cooling = 3.0  # allow 3 sec to transit to standing
        self.norminal_standing = np.copy(STANDING_POSE)
        self.add_standing = config["add_standing"]
        
        # Initialize hopping mode if enabled
        self.hopping_mode = config.get('hopping_mode', False)
        if self.hopping_mode:
            self.__init_hopping_reference(config)
        
        self.reset()

    def reset(self):
        self.time_in_sec = 0.0
        self.cmd_generator.reset()
        self.gait_library.reset()
        if self.add_standing:
            stand_flag = np.random.choice([True, False], p=[0.9, 0.1])
            if stand_flag:
                self.time_standing_start = np.random.uniform(5.0, 30.0)
            else:
                self.time_standing_start = 10000.0
        else:
            self.time_standing_start = 10000.0
        self.standing_span = np.random.uniform(2.0, 30.0)
        self.start_standing = False
        self.end_standing = False
        self.last_ref_gaitparams = np.array([0.0, 0.0, 0.98])
        self.last_ref_rotparams = np.array([0.0, 0.0, 0.0])
        init_stand_flag = np.random.choice([True, False], p=[0.5, 0.5])
        if init_stand_flag:
            self.init_standing_flag = True  # stand at the first time
            self.last_standing_flag = False
        else:
            self.init_standing_flag = False  # jump at the first time
            self.last_standing_flag = True

    def update_ref_env(self, time_in_sec, base_xy_g, base_yaw):
        self.time_in_sec = time_in_sec
        if (
            not self.start_standing
            and self.time_in_sec >= self.time_standing_start
            and self.time_in_sec < self.time_standing_start + self.standing_span
        ):
            self.start_standing = True
        if (
            self.start_standing
            and self.time_in_sec >= self.time_standing_start + self.standing_span
        ):
            self.end_standing = True
            self.start_standing = False
            self.cmd_generator.clear_stand_mode()
        if self.start_standing:
            self.cmd_generator.start_stand_mode()
        self.cmd_generator.update_cmd_env(time_in_sec)
        ref_gaitparams = self.cmd_generator.curr_ref_gaitparams
        ref_rotparams = self.cmd_generator.curr_ref_rotcmds
        self.gait_library.update_gaitlib_env(
            gait_param=ref_gaitparams, time_in_sec=time_in_sec
        )
        if abs(self.last_ref_gaitparams[0] - ref_gaitparams[0]) >= 0.01:
            self.cmd_generator.set_ref_global_pos(xy=base_xy_g)
        if abs(self.last_ref_rotparams[-1] - ref_rotparams[-1]) >= 0.002:  # 0.1 deg
            self.cmd_generator.set_ref_global_yaw(yaw=base_yaw)
        self.last_ref_gaitparams = ref_gaitparams
        self.last_ref_rotparams = ref_rotparams

    def get_init_pose(self):
        # For hopping mode, use the standard standing pose
        if self.hopping_mode:
            # Use the standard Cassie standing pose that respects constraints
            ref_base_pos = np.array([0.0, 0.0, 0.95])  # Standard height
            ref_base_rot = np.array([[0.0, 0.0, 0.0]])  # No rotation
            ref_mpos = np.array([
                0.0,     # left hip roll
                0.0,     # left hip yaw  
                0.4544,  # left hip pitch (standard standing)
                -1.21,   # left knee (standard standing)
                -1.643,  # left foot (standard standing)
                0.0,     # right hip roll
                0.0,     # right hip yaw
                0.4544,  # right hip pitch (standard standing)
                -1.21,   # right knee (standard standing)
                -1.643,  # right foot (standard standing)
            ])
            return ref_base_pos, ref_base_rot, ref_mpos
        
        # Original logic for normal walking mode
        init_gait_params = self.gait_library.get_random_init_gaitparams()
        ref_mpos = self.gait_library.get_ref_states(init_gait_params)
        ref_base_pos_from_cmd, _ = self.cmd_generator.get_ref_base_global()
        ref_base_pos = np.array(
            [
                ref_base_pos_from_cmd[0],
                ref_base_pos_from_cmd[1],
                init_gait_params[-1],
            ]
        )
        norminal_pose_flag = np.random.choice([True, False], p=[0.5, 0.5])
        ref_base_rot = np.array(
            [
                np.radians(np.random.uniform(-2.0, 2.0)),
                np.radians(np.random.uniform(-5.0, 5.0)),
                np.radians(np.random.uniform(-10.0, 10.0)),
            ]
        ).reshape((1, 3))
        if norminal_pose_flag:
            ref_base_pos, _, ref_mpos = self.norminal_pose
            stand_abduction_offset = np.radians(np.random.uniform(-1.0, 7.5, (2,)))
            ref_mpos[0] = ref_mpos[0] + stand_abduction_offset[0]
            ref_mpos[5] = ref_mpos[5] - stand_abduction_offset[1]
            stand_knee_offset = np.radians(np.random.uniform(-5.0, 5.0, (2,)))
            stand_thigh_offset = np.radians(np.random.uniform(-5.0, 5.0, (2,)))
            ref_mpos[3] += stand_knee_offset[0]
            ref_mpos[3 + 5] += stand_knee_offset[1]
            ref_mpos[2] += stand_thigh_offset[0]
            ref_mpos[2 + 5] += stand_thigh_offset[1]
            ref_base_pos[2] += np.random.uniform(-0.05, 0.05)
        return ref_base_pos, ref_base_rot, ref_mpos

    def get_ref_motion(self, look_forward=0):
        if self.hopping_mode:
            # Use hopping reference motion
            return self.get_hopping_reference(self.time_in_sec)
        
        ref_dict = dict()
        ref_gait_params = self.cmd_generator.curr_ref_gaitparams
        ref_rot_params = self.cmd_generator.curr_ref_rotcmds
        (
            ref_base_pos_from_cmd,
            ref_base_rot_from_cmd,
        ) = self.cmd_generator.get_ref_base_global()
        ref_mpos = self.gait_library.get_ref_states(ref_gait_params, look_forward)
        ref_dict["base_pos_global"] = np.array(
            [*ref_base_pos_from_cmd, ref_gait_params[-1]]
        )
        ref_dict["base_rot_global"] = ref_base_rot_from_cmd
        ref_dict["base_vel_local"] = np.array(
            [ref_gait_params[0], ref_gait_params[1], ref_rot_params[-1]]
        )  # vx vy vyaw
        if self.start_standing:
            ref_dict["motor_pos"] = self.norminal_mpos
            ref_dict["motor_vel"] = np.zeros((10,))
        else:
            ref_dict["motor_pos"] = ref_mpos
            ref_dict["motor_vel"] = np.zeros((10,))  # ref_mvel
        return ref_dict

    def get_curr_params(self):
        ref_gait_params = self.cmd_generator.curr_ref_gaitparams
        ref_rot_params = self.cmd_generator.curr_ref_rotcmds
        return ref_gait_params, ref_rot_params

    @property
    def norminal_pose(self):
        ref_base_pos = self.norminal_base_pos
        ref_base_rot = self.norminal_base_rot
        ref_mpos = self.norminal_mpos
        return ref_base_pos, ref_base_rot, ref_mpos

    @property
    def norminal_base_pos(self):
        return np.copy(self.norminal_standing[[0, 1, 2]])

    @property
    def norminal_base_rot(self):
        return np.copy(self.norminal_standing[[3, 4, 5]])

    @property
    def norminal_mpos(self):
        return np.copy(self.norminal_standing[6:])

    @property
    def in_transit_to_stand(self):
        return (
            self.start_standing
            and self.time_in_sec
            <= self.time_standing_start + self.time_stand_transit_cooling
        )

    @property
    def in_stand_mode(self):
        return self.start_standing

    def __init_hopping_reference(self, config):
        """Initialize jumping-specific reference motion parameters"""
        self.jump_target_height = config.get('hop_target_height', 1.5)  # Keep same parameter name for compatibility
        self.jump_preparation_time = 3.0  # Time to prepare for jump
        self.jump_execution_time = 4.0    # Time for jump execution and landing
        self.jump_phase = 0.0  # Current jumping phase
        self.jump_amplitude = (self.jump_target_height - 1.0) / 2.0  # Height amplitude
        
        # Define jumping phases for single explosive jump
        self.jump_phases = {
            'stand': 0,           # Standing still, preparing
            'crouch': 1,          # Compress legs for jump
            'explode': 2,         # Explosive leg extension + thruster boost
            'flight': 3,          # Air phase with thruster stabilization
            'landing_prep': 4,    # Prepare legs for landing
            'landing': 5,         # Land with leg and thruster assistance
            'stabilize': 6        # Return to stable standing
        }
        
        # Thruster reference values for jumping (scaled to match model limits)
        self.thruster_explosive_force = 8.0  # N - scaled to model's 10N limit
        self.thruster_landing_force = 6.0    # N - for controlled landing
        self.thruster_stabilization_angle = 0.1  # rad - minimal angle for stability

    def __calculate_kinematic_pose(self, phase_weight, phase_type):
        """
        Calculate proper joint angles using kinematics principles
        :param phase_weight: 0-1 weight for phase transition
        :param phase_type: 'stand', 'crouch', 'explode', 'flight', 'landing_prep', 'landing', 'stabilize'
        :return: joint angles for symmetric leg movement
        """
        # Base standing pose joint angles
        base_pose = np.copy(STANDING_POSE)
        
        if phase_type == 'stand':
            # Standing still, preparing for jump
            return {
                'left_hip_pitch': base_pose[3],
                'left_knee': base_pose[4],
                'right_hip_pitch': base_pose[8],
                'right_knee': base_pose[9]
            }
            
        elif phase_type == 'crouch':
            # Compress legs symmetrically for jump preparation
            # Use inverse kinematics to find joint angles for compressed stance
            compression_factor = 0.1 * phase_weight  # Much more conservative
            
            # Symmetric leg compression (much more conservative)
            left_hip_compression = -compression_factor
            left_knee_compression = compression_factor * 1.0  # Reduced from 2.5
            right_hip_compression = -compression_factor  
            right_knee_compression = compression_factor * 1.0  # Reduced from 2.5
            
            return {
                'left_hip_pitch': base_pose[3] + left_hip_compression,
                'left_knee': base_pose[4] + left_knee_compression,
                'right_hip_pitch': base_pose[8] + right_hip_compression,
                'right_knee': base_pose[9] + right_knee_compression
            }
            
        elif phase_type == 'explode':
            # Explosive leg extension for maximum jump power
            # Use forward kinematics to ensure both legs extend equally and powerfully
            extension_factor = 0.2 * phase_weight  # Much more conservative
            
            left_hip_extension = extension_factor
            left_knee_extension = -extension_factor * 0.5  # Much more conservative
            right_hip_extension = extension_factor
            right_knee_extension = -extension_factor * 0.5  # Much more conservative
            
            return {
                'left_hip_pitch': base_pose[3] + left_hip_extension,
                'left_knee': base_pose[4] + left_knee_extension,
                'right_hip_pitch': base_pose[8] + right_hip_extension,
                'right_knee': base_pose[9] + right_knee_extension
            }
            
        elif phase_type == 'flight':
            # Fold legs symmetrically for flight phase
            # Ensure both legs fold equally to maintain balance and reduce drag
            fold_factor = 0.4 * phase_weight
            
            left_hip_fold = -fold_factor
            left_knee_fold = fold_factor * 2.0  # More aggressive folding
            right_hip_fold = -fold_factor
            right_knee_fold = fold_factor * 2.0
            
            return {
                'left_hip_pitch': base_pose[3] + left_hip_fold,
                'left_knee': base_pose[4] + left_knee_fold,
                'right_hip_pitch': base_pose[8] + right_hip_fold,
                'right_knee': base_pose[9] + right_knee_fold
            }
            
        elif phase_type == 'landing_prep':
            # Extend legs symmetrically for landing preparation
            # Use inverse kinematics to prepare for impact absorption
            extension_factor = 0.5 * phase_weight
            
            left_hip_extension = extension_factor
            left_knee_extension = -extension_factor * 1.2
            right_hip_extension = extension_factor
            right_knee_extension = -extension_factor * 1.2
            
            return {
                'left_hip_pitch': base_pose[3] + left_hip_extension,
                'left_knee': base_pose[4] + left_knee_extension,
                'right_hip_pitch': base_pose[8] + right_hip_extension,
                'right_knee': base_pose[9] + right_knee_extension
            }
            
        elif phase_type == 'landing':
            # Compress legs symmetrically for landing impact
            # Use forward kinematics to absorb impact equally
            compression_factor = 0.2 * phase_weight
            
            left_hip_compression = -compression_factor
            left_knee_compression = compression_factor * 1.5
            right_hip_compression = -compression_factor
            right_knee_compression = compression_factor * 1.5
            
            return {
                'left_hip_pitch': base_pose[3] + left_hip_compression,
                'left_knee': base_pose[4] + left_knee_compression,
                'right_hip_pitch': base_pose[8] + right_hip_compression,
                'right_knee': base_pose[9] + right_knee_compression
            }
            
        else:  # stabilize
            # Return to standing pose symmetrically
            # Use inverse kinematics to ensure both legs return to exact standing position
            return {
                'left_hip_pitch': base_pose[3],
                'left_knee': base_pose[4],
                'right_hip_pitch': base_pose[8],
                'right_knee': base_pose[9]
            }

    def get_hopping_reference(self, time_in_sec):
        """Generate coordinated leg and thruster jumping reference motion using kinematics"""
        # Calculate jumping phase based on time
        total_jump_time = self.jump_preparation_time + self.jump_execution_time
        
        # Start with standing pose as base
        ref_pose = np.copy(STANDING_POSE)
        
        # Determine phase and apply coordinated leg + thruster action
        if time_in_sec < 2.0:  # Extended stabilization phase - just stand still
            # Keep standing pose for first 2 seconds to stabilize
            thruster_force = 0.0
            thruster_angle = 0.0
        elif time_in_sec < self.jump_preparation_time:  # Preparation phase
            phase_weight = (time_in_sec - 2.0) / (self.jump_preparation_time - 2.0)
            joint_angles = self.__calculate_kinematic_pose(phase_weight, 'crouch')
            
            # Apply symmetric joint angles
            ref_pose[3] = joint_angles['left_hip_pitch']
            ref_pose[4] = joint_angles['left_knee']
            ref_pose[8] = joint_angles['right_hip_pitch']
            ref_pose[9] = joint_angles['right_knee']
            thruster_force = 0.0
            thruster_angle = 0.0
            
        elif time_in_sec < self.jump_preparation_time + 0.5:  # Explosive jump phase
            phase_weight = (time_in_sec - self.jump_preparation_time) / 0.5
            joint_angles = self.__calculate_kinematic_pose(phase_weight, 'explode')
            
            # Apply symmetric joint angles
            ref_pose[3] = joint_angles['left_hip_pitch']
            ref_pose[4] = joint_angles['left_knee']
            ref_pose[8] = joint_angles['right_hip_pitch']
            ref_pose[9] = joint_angles['right_knee']
            thruster_force = self.thruster_explosive_force * phase_weight
            thruster_angle = self.thruster_stabilization_angle * phase_weight
            
        elif time_in_sec < self.jump_preparation_time + 1.5:  # Flight phase
            phase_weight = (time_in_sec - self.jump_preparation_time - 0.5) / 1.0
            joint_angles = self.__calculate_kinematic_pose(phase_weight, 'flight')
            
            # Apply symmetric joint angles
            ref_pose[3] = joint_angles['left_hip_pitch']
            ref_pose[4] = joint_angles['left_knee']
            ref_pose[8] = joint_angles['right_hip_pitch']
            ref_pose[9] = joint_angles['right_knee']
            thruster_force = self.thruster_explosive_force * 0.3  # reduced thrust for flight
            thruster_angle = 0.0
            
        elif time_in_sec < self.jump_preparation_time + 2.5:  # Landing preparation
            phase_weight = (time_in_sec - self.jump_preparation_time - 1.5) / 1.0
            joint_angles = self.__calculate_kinematic_pose(phase_weight, 'landing_prep')
            
            # Apply symmetric joint angles
            ref_pose[3] = joint_angles['left_hip_pitch']
            ref_pose[4] = joint_angles['left_knee']
            ref_pose[8] = joint_angles['right_hip_pitch']
            ref_pose[9] = joint_angles['right_knee']
            thruster_force = self.thruster_landing_force * phase_weight
            thruster_angle = -self.thruster_stabilization_angle * phase_weight
            
        elif time_in_sec < self.jump_preparation_time + 3.5:  # Landing phase
            phase_weight = (time_in_sec - self.jump_preparation_time - 2.5) / 1.0
            joint_angles = self.__calculate_kinematic_pose(phase_weight, 'landing')
            
            # Apply symmetric joint angles
            ref_pose[3] = joint_angles['left_hip_pitch']
            ref_pose[4] = joint_angles['left_knee']
            ref_pose[8] = joint_angles['right_hip_pitch']
            ref_pose[9] = joint_angles['right_knee']
            thruster_force = self.thruster_landing_force * (1.0 - phase_weight)
            thruster_angle = 0.0
            
        else:  # Stabilization phase
            phase_weight = min(1.0, (time_in_sec - self.jump_preparation_time - 3.5) / 0.5)
            joint_angles = self.__calculate_kinematic_pose(phase_weight, 'stabilize')
            
            # Apply symmetric joint angles
            ref_pose[3] = joint_angles['left_hip_pitch']
            ref_pose[4] = joint_angles['left_knee']
            ref_pose[8] = joint_angles['right_hip_pitch']
            ref_pose[9] = joint_angles['right_knee']
            thruster_force = 0.0
            thruster_angle = 0.0
        
        # Create reference motion dictionary
        ref_motion = {
            'base_pos_global': ref_pose[:3],
            'base_rot_global': ref_pose[3:6],
            'motor_pos': ref_pose[6:],
            'motor_vel': np.zeros(10),  # Zero velocity for hopping
            'base_vel_local': np.zeros(3),  # Zero velocity for stationary hopping
            'thruster_forces': [thruster_force, thruster_force],  # [left, right]
            'thruster_angles': [thruster_angle, thruster_angle]   # [left, right]
        }
        
        return ref_motion
