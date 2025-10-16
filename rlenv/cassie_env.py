from rlenv.cassiemujoco import *
from rlenv.cassiemujoco_ctypes import *
from rlenv.reference_generator import ReferenceGenerator
from rlenv.action_filter import ActionFilterButter
from rlenv.env_randomlizer import EnvRandomlizer
from rlenv.perturbation_generator import PerturbationGenerator
from utility.cassie_fk import CassieFK
from gym import spaces
from configs.defaults import *
from utility.utility import *
from collections import deque


class CassieEnv:
    def __init__(
        self,
        config,
    ):
        self.max_timesteps = config["max_timesteps"]
        self.action_bounds = np.copy(ACTUATED_JOINT_RANGE)
        self.safe_action_bounds = np.copy(self.action_bounds)
        # for safe joint limits on hardware
        self.safe_action_bounds[:, [0, 5]] *= 0.5
        self.safe_action_bounds[:, [1, 6]] *= 0.8
        # Thruster actions don't need safety scaling
        self.base_idx = [0, 1, 2, 3, 4, 5, 6]
        self.thruster_pitch_idx = [7, 8]
        self.thruster_pitch_vel_idx = [6, 7]

        self.motor_idx = [9, 10, 11, 16, 22, 23, 24, 25, 30, 36]
        self.motor_vel_idx = [8, 9, 10, 14, 20, 21, 22, 23, 27, 33]
        self.all_motor_idx = self.thruster_pitch_idx + self.motor_idx
        self.all_motor_vel_idx = self.thruster_pitch_vel_idx + self.motor_vel_idx

        self.pGain = np.copy(DEFAULT_PGAIN)
        self.dGain = np.copy(DEFAULT_DGAIN)
        self.offset_footJoint2midFoot = np.sqrt(0.01762**2 + 0.05219**2)
        self.sim = CassieSim(ROBOT_MODEL_FILE)
        if config["is_visual"]:
            self.is_visual = True
            self.vis = CassieVis(self.sim, ROBOT_MODEL_FILE)
            if config["cam_track_robot"]:
                self.vis.set_cam("cassie-pelvis", 2.5, 90, 0)
        else:
            self.is_visual = False
        self.step_zerotorque = config["step_zerotorque"]
        self.cassie_fk = CassieFK()
        # init variables
        self.qpos = np.copy(self.sim.qpos())
        self.qvel = np.copy(self.sim.qvel())
        self.qacc = np.copy(self.sim.qacc())
        self.obs_cassie_state = state_out_t()
        self.cassie_out = cassie_out_t()
        self.u = self._init_u()
        self.num_leg_motors = 10
        self.num_thruster_pitch_motors = 2
        self.num_motor = 12  # Total motors
        self.num_thruster_forces = 2  # Site-based forces (if still using)
        self.num_total_actions = self.num_motor + self.num_thruster_forces  # 14 or 12
        
        # Initialize thruster variables
        self.thruster_forces = [0.0, 0.0]  # [left, right]
        self.thruster_pitches = [0.0, 0.0]  # [left, right]
        self.last_thruster_forces = [0.0, 0.0]  # For stability reward calculation
        
        # Initialize waypoint system
        self.waypoint_pos = np.array([0.0, 0.0, 0.9])  # [x, y, z] - default ground level
        self.waypoint_type = "ground"  # "ground", "aerial", "mixed"
        self.waypoint_reached = False
        self.waypoint_threshold = 0.3  # meters - distance to consider waypoint reached
        self.waypoint_reward_weight = 1.0  # weight for waypoint distance reward
        
        # Store config for later use
        self.config = config
        
        # Waypoint switching configuration
        self.waypoint_strategy = config.get('waypoint_strategy', 'random')  # 'random', 'sequence', 'adaptive'
        self.sequence_type = config.get('sequence_type', 'mixed')  # 'mixed', 'spiral', 'obstacle_course'
        self.waypoint_switch_probability = config.get('waypoint_switch_probability', 0.1)  # Probability of switching waypoints
        self.sim_freq = 2000
        self.appx_env_freq = 30  # 30 hz but the real control frequency is not exactly 30Hz because we round up the num_sims_per_env_step
        self.num_sims_per_env_step = self.sim_freq // self.appx_env_freq
        self.secs_per_env_step = self.num_sims_per_env_step / self.sim_freq
        self.real_env_freq = int(1 / self.secs_per_env_step)
        self.history_len_vf = 4
        self.history_len_pol = 2 * self.real_env_freq
        # observation
        self.observation_space_pol = None
        self.observation_space_vf = None 
        self.observation_space_pol_cnn = None
        self.extrinsics_dim = None
        # the low & high does not actually limit the actions output from MLP network, manually clip instead
        self.action_space = spaces.Box(low=-100, high=100, shape=(self.num_total_actions,))
        self.previous_obs = deque(maxlen=self.history_len_vf)
        self.previous_acs = deque(maxlen=self.history_len_vf)
        self.long_history = deque(maxlen=self.history_len_pol)
        # reference motion
        self.reference_generator = ReferenceGenerator(
            env_max_timesteps=self.max_timesteps,
            secs_per_env_step=self.secs_per_env_step,
            config=config,
        )
        # action filter
        self.action_filter_order = 2
        self.__init_action_filter()
        # dynamics randomization
        self.__set_env_type(config)
        # set reward
        self.__init_reward_func()
        self.__init_env_randomlizer()
        self.__init_perturbation_generator()
        # init step
        self.__init_step_func()
        # reset and init others
        self.reset()
        # self.sim.set_geom_name_pos('table', [0.6, 0.85, -0.265]) # 0.5 is the max

    ##########################################
    #            Init and Reset              #
    ##########################################
    def __set_env_type(self, config):
        # NOTE: set true to use minimal rand range -> only rand floor friction
        self.minimal_rand = config["minimal_rand"]
        # NOTE: set true to add noise on observation
        self.noisy = config["is_noisy"]  
        # NOTE: set true to add external perturbation
        self.perturbation = config["add_perturbation"]

    def __init_reward_func(self):
        # Check if in hopping mode to use different reward structure
        if hasattr(self, 'hopping_mode') and self.hopping_mode:
            self.reward_names = [
                "r_mpos",
                "r_mvel", 
                "r_ptrans",
                "r_ptrans_velx",
                "r_ptrans_vely",
                "r_prot",
                "r_prot_vel",
                "r_ptrans_z",
                "r_torque",
                "r_foot_force",
                "r_acc",
                "r_footpos",
                "r_delta_acs",
                "r_hop_height",
                "r_landing_smoothness",
                "r_thruster_efficiency",
                "r_stability",
            ]
        else:
            self.reward_names = [
                "r_mpos",
                "r_mvel",
                "r_ptrans",
                "r_ptrans_velx",
                "r_ptrans_vely",
                "r_prot",
                "r_prot_vel",
                "r_ptrans_z",
                "r_torque",
                "r_foot_force",
                "r_acc",
                "r_footpos",
                "r_delta_acs",
                "r_waypoint",
                "r_thruster",
            ]
        w_mpos = 15.0
        w_mvel_nonstand = 0.0
        w_mvel_stand = 15.0
        w_ptrans_pos = 6.0
        w_ptrans_velx = 7.5
        w_ptrans_vely = 7.5
        w_prot_pos = 12.5
        w_prot_vel = 3.0
        w_ptrans_z = 5.0
        w_torque = 3.0
        w_foot_force = 10.0
        w_acc = 3.0
        w_footpos = 3.0
        w_delta_acs_nonstand = 3.0
        w_delta_acs_stand = 10.0
        # Check if in hopping mode for different reward weights
        if hasattr(self, 'hopping_mode') and self.hopping_mode:
            # Hopping-specific reward weights
            w_hop_height = 10.0  # Reward for achieving hop height
            w_landing_smoothness = 15.0  # Reward for smooth landing
            w_thruster_efficiency = 5.0  # Reward for efficient thruster use
            w_stability = 8.0  # Reward for maintaining stability
            
            w_array_nonstand = np.array(
                [
                    w_mpos * 0.5,  # Reduce motor position importance for hopping
                    w_mvel_nonstand,
                    w_ptrans_pos * 0.3,  # Reduce position tracking importance
                    w_ptrans_velx * 0.3,
                    w_ptrans_vely * 0.3,
                    w_prot_pos * 2.0,  # Increase orientation importance for stability
                    w_prot_vel * 2.0,
                    w_ptrans_z * 0.5,  # Reduce Z position tracking
                    w_torque * 0.5,  # Reduce torque penalty
                    w_foot_force * 0.3,  # Reduce foot force importance
                    w_acc * 0.5,  # Reduce acceleration penalty
                    w_footpos * 0.5,  # Reduce foot position importance
                    w_delta_acs_nonstand * 0.5,
                    w_hop_height,  # Hop height reward
                    w_landing_smoothness,  # Landing smoothness reward
                    w_thruster_efficiency,  # Thruster efficiency reward
                    w_stability,  # Stability reward
                ]
            )
            w_array_stand = w_array_nonstand.copy()  # Same weights for hopping
        else:
            w_waypoint = 5.0  # Waypoint reward weight
            w_thruster = 2.0  # Thruster reward weight

            w_array_nonstand = np.array(
                [
                    w_mpos,
                    w_mvel_nonstand,
                    w_ptrans_pos,
                    w_ptrans_velx,
                    w_ptrans_vely,
                    w_prot_pos,
                    w_prot_vel,
                    w_ptrans_z,
                    w_torque,
                    w_foot_force,
                    w_acc,
                    w_footpos,
                    w_delta_acs_nonstand,
                    w_waypoint,  # Waypoint reward weight
                    w_thruster,  # Thruster reward weight
                ]
            )
            w_array_stand = np.array(
                [
                    w_mpos,
                    w_mvel_stand,
                    w_ptrans_pos,
                    w_ptrans_velx,
                    w_ptrans_vely,
                    w_prot_pos,
                    w_prot_vel,
                    w_ptrans_z,
                    w_torque,
                    w_foot_force,
                    w_acc,
                    w_footpos,
                    w_delta_acs_stand,
                    w_waypoint,  # Waypoint reward weight
                    w_thruster,  # Thruster reward weight
                ]
            )

        self.reward_weights = w_array_nonstand / np.sum(w_array_nonstand)
        self.reward_weights_stand = w_array_stand / np.sum(w_array_stand)
        self.reward_scales = dict()
        # to balance unit: meter/radians/N/Nm/etc, for reward: exp(-scale*norm)
        self.reward_scales["mpos"] = 5.0
        self.reward_scales["mvel"] = 1e-2
        self.reward_scales["mvel_stand"] = 50.0
        self.reward_scales["prot"] = 10.0
        self.reward_scales["prot_vel"] = 1.0
        self.reward_scales["meter"] = 3.0
        self.reward_scales["meter_z"] = 10.0
        self.reward_scales["meter_velx"] = 2.5
        self.reward_scales["meter_vely"] = 5.0
        self.reward_scales["torque"] = 5e-7
        self.reward_scales["foot_force"] = 2.5e-5
        self.reward_scales["double_foot_force"] = 1.5e-5
        self.reward_scales["side_foot_force"] = 1.5e-3
        self.reward_scales["ppos_acc"] = 2e-5
        self.reward_scales["prot_acc"] = 1e-5
        self.reward_scales["macc"] = 1e-9
        self.reward_scales["foot_pos"] = 100
        self.reward_scales["delta_acs"] = 2.0

        # add a negative sign
        for key in self.reward_scales:
            self.reward_scales[key] *= -1.0

    def __init_step_func(self):
        if self.step_zerotorque:
            self.__sim_step = self.__step_sim_zerotorque
        else:
            self.__sim_step = self.__step_sim_nominal
        self.__get_pTargets = self.acs_norm2actual

    def _init_u(self):
        # p_paras = [400, 200, 200, 500, 20]
        # d_paras = [4, 4, 10, 20, 4]
        # feed forward torque
        torque_fwd = [0, 0, 0, 0, 0]
        self.pd_uncertainty = np.ones((20,))
        # init controller class u
        u = pd_in_t()
        # assign PD controller parameters
        for i in range(5):
            # left leg
            u.leftLeg.motorPd.torque[i] = 0  # torque_fwd[i]  # Feedforward torque
            u.leftLeg.motorPd.pTarget[i] = 0  # p_target[i]
            u.leftLeg.motorPd.pGain[i] = self.pGain[i]
            u.leftLeg.motorPd.dTarget[i] = 0
            u.leftLeg.motorPd.dGain[i] = self.dGain[i]
        u.rightLeg.motorPd = u.leftLeg.motorPd
        return u

    def _update_pd(self):
        self.u.leftLeg.motorPd.pGain[:5] = self.pGain[:5] * self.pd_uncertainty[:5]
        self.u.leftLeg.motorPd.dGain[:5] = self.dGain[:5] * self.pd_uncertainty[10:15]
        self.u.rightLeg.motorPd.pGain[:5] = self.pGain[5:10] * self.pd_uncertainty[5:10]
        self.u.rightLeg.motorPd.dGain[:5] = self.dGain[5:10] * self.pd_uncertainty[15:20]

    def __init_robot_pos(self):
        self.reference_generator.reset()
        (
            ref_base_pos,
            ref_base_rot,
            ref_mpos,
        ) = self.reference_generator.get_init_pose()
        ref_q = euler2quat(ref_base_rot.reshape((3,)))
        ref_base = np.hstack([ref_base_pos.reshape((1, 3)), ref_q.reshape((1, 4))])
        
        # Use the proper constraint-satisfying method for all modes
        self.set_motor_base_pos(ref_mpos, ref_base)
        self._init_obs_cassie_state(ref_mpos.ravel(), ref_base.ravel())

    def _init_obs_cassie_state(self, ref_mpos, ref_base):
        cassie_out = self.__set_cassie_out(ref_mpos, ref_base)
        self.obs_cassie_state = self.sim.estimate_state(cassie_out)
        self.__add_obs_noise()
        # print("init obs_cassie_state:", np.array(self.obs_cassie_state.pelvis.translationalAcceleration).ravel())

    def __init_action_filter(self):
        self.action_filter = ActionFilterButter(
            lowcut=None,
            highcut=[4],
            sampling_rate=self.real_env_freq,
            order=self.action_filter_order,
            num_joints=12,
        )

    def __init_env_randomlizer(self):
        self.env_randomlizer = EnvRandomlizer(self.sim, self.sim_freq)
        self.__set_dynamics_properties()

    def __init_perturbation_generator(self):
        self.pertubration_generator = PerturbationGenerator(self.real_env_freq, 0.2)

    def __init_hopping_system(self, config):
        """Initialize hopping-specific variables and tracking"""
        self.hopping_mode = True
        self.thruster_enabled = config.get('thruster_enabled', True)
        self.hop_target_height = config.get('hop_target_height', 1.5)
        self.landing_smoothness_threshold = config.get('landing_smoothness_threshold', 0.1)
        
        # Hopping state tracking
        self.hop_phase = "ground"  # "ground", "ascent", "descent", "landing"
        self.max_height_reached = 1.0  # Starting height
        self.hop_started = False
        self.landing_started = False
        self.successful_hops = 0
        self.total_landing_velocity = 0.0
        self.landing_count = 0
        
        # Thruster usage tracking
        self.thruster_usage_history = []
        self.efficiency_score = 0.0
        
        # Reset hopping-specific rewards
        self.hop_height_reward = 0.0
        self.landing_smoothness_reward = 0.0
        self.thruster_efficiency_reward = 0.0

    def __reset_action_filter(self):
        self.action_filter.reset()
        noise_rot_magnitute, _, _, _ = self.env_randomlizer.get_rand_nosie()
        default_action = (
            self.qpos[self.all_motor_idx]
            + np.random.normal(size=self.num_motor) * noise_rot_magnitute
        )
        self.action_filter.init_history(default_action)

    def reset(self):
        self.__reset_env()
        self.action_filter.reset()
        self.pertubration_generator.reset()
        self.env_randomlizer.randomize_dynamics()
        self.__set_dynamics_properties()
        self.__update_data(step=False)
        self.__reset_consts()
        self.__reset_action_filter()
        
        # Initialize hopping system if in hopping mode
        if self.config.get('hopping_mode', False):
            self.__init_hopping_system(self.config)
        else:
            # Initialize waypoint switching system
            self.waypoint_reached_count = 0
            
            # Choose waypoint strategy (can be configured)
            waypoint_strategy = getattr(self, 'waypoint_strategy', 'random')  # Default to random
            
            if waypoint_strategy == 'sequence':
                # Use waypoint sequences for more complex navigation
                sequence_type = getattr(self, 'sequence_type', 'mixed')
                sequence = self.create_waypoint_sequence(sequence_type)
                self.set_waypoint_sequence(sequence)
            else:
                # Use random waypoint generation
                waypoint_type = np.random.choice(["ground", "aerial"], p=[0.7, 0.3])  # 70% ground, 30% aerial
                self.generate_waypoint(waypoint_type)
        
        obs_vf, obs_pol = self.__get_observation(step=False)
        if self.observation_space_vf is None:
            self.__set_obs_space(obs_vf, obs_pol)
        return obs_vf, obs_pol

    def __reset_env(self):
        self.sim.full_reset()
        self.sim.set_time(0)
        self.timestep = 0
        self.time_in_sec = 0.0
        self.previous_obs = deque(maxlen=self.history_len_vf)
        self.previous_acs = deque(maxlen=self.history_len_vf)
        self.long_history = deque(maxlen=self.history_len_pol)

        self.__init_robot_pos()

        self.reward = None
        self.done = None
        self.info = {}
        self.fall_flag = False

    def __reset_consts(self):
        self.init_xy = self.qpos[[0, 1]]
        self.height = self.qpos[2]
        self.foot_pos = np.empty(6)
        self.sim.foot_pos(self.foot_pos)
        self.init_foot_pos = self.foot_pos
        self.applied_force = np.zeros((6,))
        self.last_acs = self.qpos[self.all_motor_idx]

    def __set_dynamics_properties(self):
        if self.minimal_rand:
            self.sim.set_geom_friction(
                self.env_randomlizer.get_rand_floor_friction(), "floor"
            )
        else:
            self.sim.set_dof_damping(self.env_randomlizer.get_rand_damping())
            self.sim.set_dof_stiffness(self.env_randomlizer.get_rand_stiffness())
            self.sim.set_body_mass(self.env_randomlizer.get_rand_mass())
            self.sim.set_body_ipos(self.env_randomlizer.get_rand_ipos())
            self.sim.set_body_inertia(self.env_randomlizer.get_rand_inertia())
            self.sim.set_geom_friction(
                self.env_randomlizer.get_rand_floor_friction(), "floor"
            )
            self.pd_uncertainty = self.env_randomlizer.get_pd_uncertainty()
            self._update_pd()
            # self.sim.set_geom_quat(self.env_randomlizer.get_floor_slope(), 'floor')

    def __set_obs_space(self, obs_vf, obs_pol):
        self.observation_space_vf = spaces.Box(
            low=-np.inf, high=np.inf, shape=obs_vf.shape
        )
        self.observation_space_pol = spaces.Box(
            low=-np.inf, high=np.inf, shape=obs_pol[0].shape
        )
        self.observation_space_pol_cnn = obs_pol[1].shape
        print("obs space has init!")

    ##########################################
    #                Step                    #
    ##########################################
    def __step_sim_nominal(self, actual_pTs_filtered):
        # 1 control_step = 0.0005s / 2kHz
        self.u.leftLeg.motorPd.pTarget[:5] = actual_pTs_filtered[:5]
        self.u.rightLeg.motorPd.pTarget[:5] = actual_pTs_filtered[5:10]

        thruster_left = actual_pTs_filtered[10]
        thruster_right = actual_pTs_filtered[11]
        for _ in range(self.num_sims_per_env_step):
            # give pTargets to motors
            self.obs_cassie_state = self.sim.estimate_state(
                self.sim.step_pd_without_estimation(self.u)
            )
            qpos = np.copy(self.sim.qpos())
            qpos[7] = thruster_left
            qpos[8] = thruster_right
            self.sim.set_qpos(qpos)

    def __step_sim_zerotorque(self, actual_pTs_filtered):
        # 1 control_step = 0.0005s / 2kHz
        self.u.leftLeg.motorPd.pTarget[:5] = 0.0 * actual_pTs_filtered[:5]
        self.u.rightLeg.motorPd.pTarget[:5] = 0.0 * actual_pTs_filtered[5:]
        for _ in range(self.num_sims_per_env_step):
            # give pTargets to motors
            self.obs_cassie_state = self.sim.estimate_state(
                self.sim.step_pd_without_estimation(self.u)
            )

    def step(self, acs, restore=False):
        """
        :param act: a dict {control index: pTarget, ...}
        :return:
        """
        assert acs.shape[0] == self.num_total_actions and np.all(-1.0 <= acs) and np.all(acs <= 1.0)
        
        # Separate motor and thruster actions
        motor_acs = acs[:self.num_motor]
        thruster_acs = acs[self.num_motor:]
        
        actual_pTs = self.__get_pTargets(motor_acs)
        actual_pTs_filtered = self.action_filter.filter(actual_pTs)
        
        # Process thruster actions
        self.__apply_thruster_actions(thruster_acs)

        if self.perturbation:
            self.__apply_perturbation()
        # simulated env
        self.__sim_step(actual_pTs_filtered)
        
        # Check for simulation instability and reset if needed
        if self.__check_simulation_stability():
            print("⚠️ Simulation instability detected, resetting episode")
            obs_vf, obs_pol = self.reset()
            # Ensure info dict has required keys with proper reward structure
            self.info["reward_dict"] = dict(zip(self.reward_names, [0.0] * len(self.reward_names)))
            return obs_vf, obs_pol, -10.0, True, self.info
        
        if self.noisy:
            self.__add_obs_noise()

        self.__update_data(step=True)
        
        # Check if waypoint is reached and switch if needed
        if self.is_waypoint_reached():
            if hasattr(self, 'waypoint_sequence') and self.waypoint_sequence:
                # Use sequence-based switching
                if self.advance_waypoint_sequence():
                    if hasattr(self, 'waypoint_reached_count'):
                        self.waypoint_reached_count += 1
                    else:
                        self.waypoint_reached_count = 1
            else:
                # Use random switching
                self.switch_waypoint(strategy="random")
                if hasattr(self, 'waypoint_reached_count'):
                    self.waypoint_reached_count += 1
                else:
                    self.waypoint_reached_count = 1
        
        obs_vf, obs_pol = self.__get_observation(acs=actual_pTs, step=True)
        reward, reward_dict = self.__get_reward(acs=actual_pTs)
        done = self.__is_done() if not restore else False
        self.info["reward_dict"] = reward_dict
        self.info["waypoint_reached"] = self.waypoint_reached
        self.info["waypoint_type"] = self.waypoint_type
        self.info["waypoint_pos"] = self.waypoint_pos.copy()
        self.last_acs = actual_pTs
        return obs_vf, obs_pol, reward, done, self.info

    def __apply_thruster_actions(self, thruster_acs):
        """
        Apply thruster actions to the simulation
        :param thruster_acs: array of 2 thruster actions [left_force, right_force]
        """
        # Convert normalized actions to actual thruster values
        # thruster_acs[0]: left force (0-1 -> 0-10)
        # thruster_acs[1]: right force (0-1 -> 0-10)
        
        left_force = thruster_acs[0] * 10.0  # Scale to 0-10
        right_force = thruster_acs[1] * 10.0  # Scale to 0-10
        
        # Store thruster values for potential use
        self.thruster_forces = [left_force, right_force]

    def __update_data(self, step=True):
        self.qpos = np.copy(self.sim.qpos())
        self.qvel = np.copy(self.sim.qvel())
        self.qacc = np.copy(self.sim.qacc())
        self.height = self.qpos[2]
        self.foot_pos = np.empty(6)
        self.sim.foot_pos(self.foot_pos)
        self.curr_rpy_gt = quat2euler(self.qpos[3:7])
        self.curr_rpy_obs = quat2euler(self.obs_cassie_state.pelvis.orientation)
        if step:
            self.timestep += 1
            self.time_in_sec = (
                self.timestep * self.num_sims_per_env_step
            ) / self.sim_freq
            self.reference_generator.update_ref_env(
                self.time_in_sec, self.qpos[:2], self.curr_rpy_gt[-1]
            )
        self.ref_dict = self.reference_generator.get_ref_motion()

    def __add_obs_noise(self):
        noise_vec = np.random.normal(size=(32,))  # 20+3+3+3+3
        (
            noise_rot,
            noise_rot_vel,
            noise_linear_acc,
            noise_linear_vel,
        ) = self.env_randomlizer.get_rand_nosie()
        obs_euler = quat2euler(self.obs_cassie_state.pelvis.orientation)
        self.obs_cassie_state.pelvis.translationalVelocity[:3] += (
            noise_vec[:3] * noise_linear_vel
        )
        self.obs_cassie_state.pelvis.translationalAcceleration[:3] += (
            noise_vec[3:6] * noise_linear_acc
        )
        self.obs_cassie_state.pelvis.rotationalVelocity[:3] += (
            noise_vec[6:9] * noise_rot_vel
        )
        obs_euler[:3] += noise_vec[9:12] * noise_rot
        obs_quat = euler2quat(obs_euler)
        self.obs_cassie_state.pelvis.orientation[:4] = obs_quat[:4]
        self.obs_cassie_state.motor.position[:10] += noise_vec[12:22] * noise_rot
        self.obs_cassie_state.motor.velocity[:10] += noise_vec[22:32] * noise_rot_vel

    def __set_cassie_out(self, mpos, base_pos):
        cassie_out = cassie_out_t()
        cassie_out.pelvis.vectorNav.orientation[:4] = base_pos[3:7]
        # motor, leftleg
        cassie_out.leftLeg.hipRollDrive.position = mpos[0]
        cassie_out.leftLeg.hipYawDrive.position = mpos[1]
        cassie_out.leftLeg.hipPitchDrive.position = mpos[2]
        cassie_out.leftLeg.kneeDrive.position = mpos[3]
        cassie_out.leftLeg.footDrive.position = mpos[4]
        # motor, rightleg
        cassie_out.rightLeg.hipRollDrive.position = mpos[5]
        cassie_out.rightLeg.hipYawDrive.position = mpos[6]
        cassie_out.rightLeg.hipPitchDrive.position = mpos[7]
        cassie_out.rightLeg.kneeDrive.position = mpos[8]
        cassie_out.rightLeg.footDrive.position = mpos[9]
        # # motor, thruster pitch
        # cassie_out.thruster_left_pitch.position = mpos[10]  
        # cassie_out.thruster_right_pitch.position = mpos[11]
        return cassie_out

    def __apply_perturbation(self):
        (
            force_to_apply,
            apply_force_flag,
        ) = self.pertubration_generator.apply_perturbation()
        if apply_force_flag:
            self.applied_force = force_to_apply
            self.sim.apply_force(self.applied_force)
            if self.is_visual:
                print("Applied Perturbation: ", self.applied_force)
        else:
            self.applied_force *= 0.0
            self.sim.clear_forces()

    ##########################################
    #              Observation               #
    ##########################################
    def __get_observation(self, acs=np.zeros(12), step=False):
        ref_dict_1 = self.reference_generator.get_ref_motion(look_forward=1)
        ref_dict_4 = self.reference_generator.get_ref_motion(look_forward=4)
        ref_dict_7 = self.reference_generator.get_ref_motion(look_forward=7)

        ob1 = ref_dict_1["motor_pos"]
        ob4 = ref_dict_4["motor_pos"]
        ob7 = ref_dict_7["motor_pos"]

        ob_curr = np.concatenate(
            [
                np.array(self.obs_cassie_state.pelvis.translationalVelocity).ravel(),
                np.array(self.obs_cassie_state.pelvis.orientation).ravel(),
                np.array(self.obs_cassie_state.motor.position).ravel(),
                np.array(self.obs_cassie_state.motor.velocity).ravel(),
            ]
        )

        # ground truth observation
        curr_xy_gt = self.qpos[[0, 1]] - self.init_xy
        ob_curr_gt = np.concatenate(
            [
                self.qvel[[0, 1, 2]],
                self.qacc[[0, 1, 2]],
                self.qpos[3:7],
                self.qpos[self.all_motor_idx],
                self.qvel[self.all_motor_vel_idx], # all 12 velocities
            ]
        )

        # command
        ref_yaw = self.ref_dict["base_rot_global"][-1]
        ob_command = np.concatenate(
            [
                [self.ref_dict["base_pos_global"][-1]],  # walking height
                self.ref_dict["base_vel_local"][[0, 1]],  # local velocity command
                [math.cos(ref_yaw), math.sin(ref_yaw)],  # desired turning yaw angle
            ]
        )  #  height, vx, vy, yaw
        if self.timestep == 0:
            [self.previous_obs.append(ob_curr) for i in range(self.history_len_vf)]
            [
                self.previous_acs.append(np.zeros(12))
                for _ in range(self.history_len_vf)
            ]
            [
                self.long_history.append(
                    np.concatenate([ob_curr, np.zeros(12)])
                )
                for _ in range(self.history_len_pol)
            ]
        ob_prev = np.concatenate(
            [np.array(self.previous_obs).ravel(), np.array(self.previous_acs).ravel()]
        )
        obs_pol_hist = np.flip(np.asarray(self.long_history).T, 1)

        # Add waypoint information to policy observation
        waypoint_relative = self.waypoint_pos - self.qpos[:3]  # Relative waypoint position
        waypoint_distance = np.array([self.get_distance_to_waypoint()])  # Distance to waypoint
        waypoint_type_encoded = np.array([1.0 if self.waypoint_type == "aerial" else 0.0])  # Aerial flag
        
        obs_pol_base = np.concatenate([ob_prev, ob_curr, ob1, ob4, ob7, ob_command, waypoint_relative, waypoint_distance, waypoint_type_encoded])
        obs_pol = (obs_pol_base, obs_pol_hist)

        # Add waypoint information to observation
        waypoint_relative = self.waypoint_pos - self.qpos[:3]  # Relative waypoint position
        waypoint_distance = np.array([self.get_distance_to_waypoint()])  # Distance to waypoint
        waypoint_type_encoded = np.array([1.0 if self.waypoint_type == "aerial" else 0.0])  # Aerial flag
        
        obs_vf = np.concatenate(
            [
                ob_prev,
                ob_curr_gt,
                ob1,
                ob4,
                ob7,
                ob_command,
                self.ref_dict["base_pos_global"][[0, 1]] - curr_xy_gt,
                np.array([self.qpos[2]]),
                self.sim.get_foot_forces(),
                self.env_randomlizer.get_rand_floor_friction(),
                waypoint_relative,  # [x, y, z] relative waypoint position
                waypoint_distance,  # [distance] to waypoint
                waypoint_type_encoded,  # [aerial_flag] 1.0 if aerial, 0.0 if ground
            ]
        )

        if step:
            self.previous_obs.append(ob_curr)
            self.previous_acs.append(acs)
            self.long_history.append(np.concatenate([ob_curr, acs]))
        return obs_vf, obs_pol

    ##########################################
    #                Reward                  #
    ##########################################
    def __get_reward(self, acs):
        # NOTE: reward is using qpos/qvel that don't have noise and delay
        mpos_err = np.sum(
            np.square(self.ref_dict["motor_pos"] - self.qpos[self.all_motor_idx])
        )
        r_mpos = np.exp(self.reward_scales["mpos"] * mpos_err)

        mvel_err = np.sum(
            np.square(self.ref_dict["motor_vel"] - self.qvel[self.all_motor_vel_idx])
        )
        if not self.reference_generator.in_stand_mode:
            r_mvel = np.exp(self.reward_scales["mvel"] * mvel_err)
        else:
            r_mvel = np.exp(self.reward_scales["mvel_stand"] * mvel_err)

        curr_xy = self.qpos[[0, 1]] - self.init_xy
        base_pos_err = np.sum(
            np.square(self.ref_dict["base_pos_global"][[0, 1]] - curr_xy)
        )  # pos global
        r_ptrans = np.exp(self.reward_scales["meter"] * base_pos_err)

        velocity_local = global2local(self.qvel[:3], self.curr_rpy_gt)

        r_ptrans_velx = np.exp(
            self.reward_scales["meter_velx"]
            * abs(self.ref_dict["base_vel_local"][0] - velocity_local[0])
        )
        # print(
        #     "ptrans_velx reward: {:.6f}, desired: {:.3f}, actual: {:.3f}".format(
        #         r_ptrans_velx, self.ref_dict["base_vel_local"][0], velocity_local[0]
        #     ),
        #     end="\r",
        # )

        r_ptrans_vely = np.exp(
            self.reward_scales["meter_vely"]
            * abs(self.ref_dict["base_vel_local"][1] - velocity_local[1])
        )
        # print(
        #     "ptrans_vely reward: {:.6f}, desired: {:.3f}, actual: {:.3f}".format(
        #         r_ptrans_vely, self.ref_dict["base_vel_local"][1], velocity_local[1]
        #     ),
        #     end="\r",
        # )

        base_angle_err = np.sum(
            1.0 - np.cos(self.ref_dict["base_rot_global"] - self.curr_rpy_gt)
        )
        r_prot = np.exp(self.reward_scales["prot"] * base_angle_err)
        # print(
        #     "prot reward:{:.6f}, desired: {:.4f}, actual: {:.4f}".format(
        #         r_prot,
        #         np.rad2deg(self.ref_dict["base_rot_global"][-1]),
        #         np.rad2deg(self.curr_rpy_gt[-1]),
        #     ),
        #     end="\r",
        # )

        base_anglevel_err = np.sum(
            np.square(
                self.qvel[[3, 4, 5]]
                - np.array([0.0, 0.0, self.ref_dict["base_vel_local"][-1]])
            )
        )  # stablize roll and pitch and track yaw vel
        r_prot_vel = np.exp(self.reward_scales["prot_vel"] * base_anglevel_err)

        ptrans_z_err = np.square(self.qpos[2] - self.ref_dict["base_pos_global"][2])
        r_ptrans_z = np.exp(self.reward_scales["meter_z"] * ptrans_z_err)

        torques_err = np.sum(np.square(self._calc_torque(acs)))
        r_torques = np.exp(self.reward_scales["torque"] * torques_err)

        impact_forces = self.sim.get_foot_impact_forces()
        if self.reference_generator.in_stand_mode:
            foot_forces = self.sim.get_foot_forces()
            foot_forces_side = foot_forces[[0, 1, 3, 4]]
            if any(abs(impact_forces) <= 1.0):
                r_foot_force = 0.0
            else:
                foot_force_err = np.sum(np.square(foot_forces_side))
                r_foot_force = np.exp(
                    self.reward_scales["side_foot_force"] * foot_force_err
                )
        else:
            if abs(impact_forces[0]) <= 0.1 and abs(impact_forces[1]) <= 0.1:
                r_foot_force = 0.0
            elif abs(impact_forces[0]) >= 0.1 and abs(impact_forces[1]) >= 0.1:
                foot_force_err = np.square(np.sum(impact_forces) / 2.0)
                r_foot_force = np.exp(
                    self.reward_scales["double_foot_force"] * foot_force_err
                )
            else:
                foot_force_err = np.sum(np.square(impact_forces))
                r_foot_force = np.exp(self.reward_scales["foot_force"] * foot_force_err)

        ref_footpos = self.get_foot_pos_absolute(
            self.ref_dict["base_pos_global"],
            self.ref_dict["base_rot_global"],
            self.ref_dict["motor_pos"][:10],  # [:10] only pass leg motor positions
        )
        foot_pos_err = np.sum(
            np.square(
                ref_footpos[[2, 5]]
                - self.offset_footJoint2midFoot
                - self.foot_pos[[2, 5]]
            )
        )
        r_footpos = np.exp(self.reward_scales["foot_pos"] * foot_pos_err)

        base_xyz_acc_err = np.sum(np.square(self.qacc[0:3]))
        base_rxyz_acc_err = np.sum(np.square(self.qacc[3:6]))
        base_motor_acc_err = np.sum(np.square(self.qacc[self.motor_vel_idx]))
        r_base_xyz_acc = np.exp(self.reward_scales["ppos_acc"] * base_xyz_acc_err)
        r_base_rxyz_acc = np.exp(self.reward_scales["prot_acc"] * base_rxyz_acc_err)
        r_motor_acc = np.exp(self.reward_scales["macc"] * base_motor_acc_err)
        r_acc = (r_base_xyz_acc + r_base_rxyz_acc + r_motor_acc) / 3.0

        r_delta_acs = np.exp(
            self.reward_scales["delta_acs"] * np.sum(np.square(self.last_acs - acs))
        )

        # Check if in hopping mode for different reward calculations
        if hasattr(self, 'hopping_mode') and self.hopping_mode:
            # Hopping-specific rewards
            r_hop_height = self.get_hop_height_reward()
            r_landing_smoothness = self.get_landing_smoothness_reward()
            r_thruster_efficiency = self.get_thruster_efficiency_reward()
            r_stability = self.get_stability_reward()
        else:
            # Waypoint reward
            r_waypoint = self.get_waypoint_reward()
            
            # Thruster reward
            r_thruster = self.get_thruster_rewards()

        # NOTE: should be in the same order with self.reward_weights
        if hasattr(self, 'hopping_mode') and self.hopping_mode:
            rewards = np.array(
                [
                    r_mpos,
                    r_mvel,
                    r_ptrans,
                    r_ptrans_velx,
                    r_ptrans_vely,
                    r_prot,
                    r_prot_vel,
                    r_ptrans_z,
                    r_torques,
                    r_foot_force,
                    r_acc,
                    r_footpos,
                    r_delta_acs,
                    r_hop_height,  # Hop height reward
                    r_landing_smoothness,  # Landing smoothness reward
                    r_thruster_efficiency,  # Thruster efficiency reward
                    r_stability,  # Stability reward
                ]
            )
        else:
            rewards = np.array(
                [
                    r_mpos,
                    r_mvel,
                    r_ptrans,
                    r_ptrans_velx,
                    r_ptrans_vely,
                    r_prot,
                    r_prot_vel,
                    r_ptrans_z,
                    r_torques,
                    r_foot_force,
                    r_acc,
                    r_footpos,
                    r_delta_acs,
                    r_waypoint,  # Waypoint reward
                    r_thruster,  # Thruster reward
                ]
            )

        if not self.reference_generator.in_stand_mode:
            total_reward = np.sum(self.reward_weights * rewards)
            reward_dict = dict(zip(self.reward_names, self.reward_weights * rewards))
        else:
            total_reward = np.sum(self.reward_weights_stand * rewards)
            reward_dict = dict(
                zip(self.reward_names, self.reward_weights_stand * rewards)
            )
        return total_reward, reward_dict

    ##########################################
    #           Early Termination            #
    ##########################################
    def __is_done(self):
        tarsus_pos = self.get_tarsus_pos(
            self.qpos[[0, 1, 2]], self.qpos[4:7], self.qpos[self.motor_idx]
        )
        if self.height < FALLING_THRESHOLD:
            self.fall_flag = True
            # print('below loweset height')
            return True
        elif any(tarsus_pos[[2, 5]] <= TARSUS_HITGROUND_THRESHOLD):
            self.fall_flag = True
            # print('tarsus on the ground')
            return True
        elif self.timestep >= self.max_timesteps:
            # print('max step reached:{}'.format(self.max_timesteps))
            return True
        else:
            return False

    ##########################################
    #                 Utils                  #
    ##########################################
    def acs_actual2norm(self, actual_acs):
        # Bounds for 10 leg motors and 2 thruster pitch motors
        motor_bounds = np.concatenate([
            self.safe_action_bounds[:, :10],  # Leg motors
            self.safe_action_bounds[:, 12:14]  # Thruster pitch motors
        ], axis=1)
        return (actual_acs - motor_bounds[0]) * 2 / (
            motor_bounds[1] - motor_bounds[0]
        ) - 1

    def acs_norm2actual(self, acs):
        # Bounds for 10 leg motors and 2 thruster pitch motors
        motor_bounds = np.concatenate([
            self.safe_action_bounds[:, :10],  # Leg motors
            self.safe_action_bounds[:, 12:14]  # Thruster pitch motors
        ], axis=1)
        return motor_bounds[0] + (acs + 1) / 2.0 * (
            motor_bounds[1] - motor_bounds[0]
        )


    def set_motor_base_pos(self, motor_pos, base_pos, iters=200):
        """
        Kind of hackish.
        This takes a floating base position and some joint positions
        and abuses the MuJoCo solver to get the constrained forward
        kinematics.
        There might be a better way to do this, e.g. using mj_kinematics
        """
        # Reduce iterations and add stability checks
        for i in range(iters):
            qpos = np.copy(self.sim.qpos())
            qvel = np.copy(self.sim.qvel())

            qpos[self.motor_idx] = motor_pos[:10]
            qpos[7] = motor_pos[10]  # left thruster pitch
            qpos[8] = motor_pos[11]  # right thruster pitch
            qpos[self.base_idx] = base_pos

            self.sim.set_qpos(qpos)
            self.sim.set_qvel(0 * qvel)

            self.sim.step_pd_without_estimation(pd_in_t())
            self.obs_cassie_state = self.sim.estimate_state(cassie_out_t())
            self.sim.set_time(self.time_in_sec)
            
            # Check for instability during pose setting - more conservative checks
            if i > 5:  # Allow fewer iterations for convergence
                current_qacc = np.copy(self.sim.qacc())
                if np.any(np.abs(current_qacc) > 50):  # More conservative acceleration threshold
                    print(f"⚠️ Extreme accelerations during pose setting at iteration {i}")
                    break

    def close(self):
        if self.is_visual:
            self.vis.__del__()
        self.sim.__del__()

    def render(self):
        return self.vis.draw(self.sim)

    def _calc_torque(self, actual_pTs):
        leg_pTs = actual_pTs[:10]
        torques = self.pGain[:10] * (
            leg_pTs - np.array(self.obs_cassie_state.motor.position)
        ) - self.dGain[:10] * np.array(self.obs_cassie_state.motor.velocity)
        return torques

    def get_foot_pos_relative(self, base_rot, motor_pos):
        return self.cassie_fk.get_foot_pos(motor_pos, [0, 0, 0], base_rot)

    def get_foot_pos_absolute(self, base_pos, base_rot, motor_pos):
        return self.cassie_fk.get_foot_pos(motor_pos, base_pos, base_rot)

    ##########################################
    #            Waypoint System             #
    ##########################################
    
    def generate_waypoint(self, waypoint_type="ground"):
        """Generate a new waypoint based on type"""
        if waypoint_type == "ground":
            # Ground waypoint: random position at ground level
            x = np.random.uniform(-2.0, 2.0)  # 4m range in x
            y = np.random.uniform(-2.0, 2.0)  # 4m range in y
            z = 0.9  # Ground level (robot height)
        elif waypoint_type == "aerial":
            # Aerial waypoint: random position above ground
            x = np.random.uniform(-1.5, 1.5)  # Smaller range for aerial
            y = np.random.uniform(-1.5, 1.5)
            z = np.random.uniform(1.5, 3.0)  # 1.5m to 3m height
        elif waypoint_type == "mixed":
            # Mixed waypoint: random type
            if np.random.random() < 0.5:
                return self.generate_waypoint("ground")
            else:
                return self.generate_waypoint("aerial")
        else:
            raise ValueError(f"Unknown waypoint type: {waypoint_type}")
        
        self.waypoint_pos = np.array([x, y, z])
        self.waypoint_type = waypoint_type
        self.waypoint_reached = False
        return self.waypoint_pos
    
    def get_distance_to_waypoint(self):
        """Calculate distance to current waypoint"""
        robot_pos = self.qpos[:3]  # [x, y, z]
        distance = np.linalg.norm(robot_pos - self.waypoint_pos)
        return distance
    
    def is_waypoint_reached(self):
        """Check if waypoint has been reached"""
        distance = self.get_distance_to_waypoint()
        if distance < self.waypoint_threshold:
            self.waypoint_reached = True
            return True
        return False
    
    def get_waypoint_reward(self):
        """Calculate waypoint-based reward"""
        distance = self.get_distance_to_waypoint()
        
        # Distance reward (negative - closer is better)
        distance_reward = -distance * self.waypoint_reward_weight
        
        # Success reward (bonus for reaching waypoint)
        success_reward = 0.0
        if self.is_waypoint_reached():
            success_reward = 10.0  # Large bonus for reaching waypoint
        
        # Height tracking reward for aerial waypoints
        height_reward = 0.0
        if self.waypoint_type == "aerial":
            height_diff = abs(self.qpos[2] - self.waypoint_pos[2])
            height_reward = -height_diff * 0.5  # Penalty for height deviation
        
        return distance_reward + success_reward + height_reward
    
    def get_thruster_rewards(self):
        """Calculate thruster-specific rewards"""
        # Thruster force magnitude (penalty for excessive force)
        total_thruster_force = abs(self.thruster_forces[0]) + abs(self.thruster_forces[1])
        force_efficiency_reward = -total_thruster_force * 0.001  # Penalty for high force usage
        
        # Thruster pitch efficiency (penalty for excessive pitch changes)
        total_thruster_pitch = abs(self.thruster_pitches[0]) + abs(self.thruster_pitches[1])
        pitch_efficiency_reward = -total_thruster_pitch * 0.01  # Penalty for excessive pitch
        
        # Thruster stability reward (reward for smooth thruster usage)
        if hasattr(self, 'last_thruster_forces'):
            force_change = abs(self.thruster_forces[0] - self.last_thruster_forces[0]) + \
                          abs(self.thruster_forces[1] - self.last_thruster_forces[1])
            stability_reward = -force_change * 0.1  # Penalty for sudden changes
        else:
            stability_reward = 0.0
        
        # Context-aware thruster rewards
        context_reward = 0.0
        if self.waypoint_type == "ground":
            # For ground waypoints, penalize unnecessary thruster usage
            if total_thruster_force > 50:  # More than 50N total force
                context_reward = -total_thruster_force * 0.002  # Extra penalty
        elif self.waypoint_type == "aerial":
            # For aerial waypoints, reward appropriate thruster usage
            height_diff = self.waypoint_pos[2] - self.qpos[2]
            if height_diff > 0.5:  # Need to go up
                if total_thruster_force > 100:  # Good thruster usage
                    context_reward = 0.1  # Small bonus for appropriate usage
            else:
                context_reward = -total_thruster_force * 0.001  # Penalty for unnecessary force
        
        # Store current thruster forces for next step
        self.last_thruster_forces = self.thruster_forces.copy()
        
        return force_efficiency_reward + pitch_efficiency_reward + stability_reward + context_reward

    def __check_simulation_stability(self):
        """Check for simulation instability (NaN/Inf values) with enhanced checks"""
        try:
            # Check for NaN or Inf values in key simulation states
            if np.any(np.isnan(self.qpos)) or np.any(np.isinf(self.qpos)):
                print(f"⚠️ NaN/Inf detected in qpos: {self.qpos}")
                return True
            if np.any(np.isnan(self.qvel)) or np.any(np.isinf(self.qvel)):
                print(f"⚠️ NaN/Inf detected in qvel: {self.qvel}")
                return True
            if np.any(np.isnan(self.qacc)) or np.any(np.isinf(self.qacc)):
                print(f"⚠️ NaN/Inf detected in qacc: {self.qacc}")
                return True
            
            # Check for extreme values that indicate instability
            if np.any(np.abs(self.qpos) > 50):
                print(f"⚠️ Extreme qpos values: {self.qpos}")
                return True
            if np.any(np.abs(self.qvel) > 50):
                print(f"⚠️ Extreme qvel values: {self.qvel}")
                return True
            
            # Check for thruster-specific instability
            if hasattr(self, 'thruster_forces'):
                if np.any(np.isnan(self.thruster_forces)) or np.any(np.isinf(self.thruster_forces)):
                    print(f"⚠️ NaN/Inf detected in thruster_forces: {self.thruster_forces}")
                    return True
                if np.any(np.abs(self.thruster_forces) > 200):  # Unrealistic force values
                    print(f"⚠️ Extreme thruster_forces values: {self.thruster_forces}")
                    return True
            
            # Check for extreme accelerations (indicates numerical issues)
            if np.any(np.abs(self.qacc) > 1000):
                print(f"⚠️ Extreme qacc values: {self.qacc}")
                return True
                
            return False
        except Exception as e:
            # If we can't check, assume instability
            print(f"⚠️ Exception in stability check: {e}")
            return True

    def switch_waypoint(self, strategy="random"):
        """Switch to a new waypoint based on strategy"""
        if strategy == "random":
            # Randomly choose between ground and aerial waypoints
            waypoint_type = np.random.choice(["ground", "aerial"], p=[0.6, 0.4])  # Slightly favor ground
        elif strategy == "sequential":
            # Alternate between ground and aerial
            if self.waypoint_type == "ground":
                waypoint_type = "aerial"
            else:
                waypoint_type = "ground"
        elif strategy == "adaptive":
            # Switch based on current performance
            if hasattr(self, 'waypoint_reached_count'):
                if self.waypoint_reached_count > 3:  # If reaching waypoints easily
                    waypoint_type = "aerial"  # Make it harder
                else:
                    waypoint_type = "ground"  # Keep it easier
            else:
                waypoint_type = "ground"
        else:
            waypoint_type = "ground"  # Default to ground
        
        # Generate new waypoint
        self.generate_waypoint(waypoint_type)
        
        # Reset waypoint reached flag
        self.waypoint_reached = False
        
        return waypoint_type

    def create_waypoint_sequence(self, sequence_type="mixed"):
        """Create a sequence of waypoints for complex navigation"""
        if sequence_type == "mixed":
            # Create a mixed sequence: ground -> aerial -> ground
            sequence = [
                {"type": "ground", "pos": [2.0, 0.0, 0.9]},
                {"type": "aerial", "pos": [4.0, 0.0, 1.5]},
                {"type": "ground", "pos": [6.0, 0.0, 0.9]},
                {"type": "aerial", "pos": [8.0, 0.0, 2.0]},
            ]
        elif sequence_type == "spiral":
            # Create a spiral pattern
            sequence = []
            for i in range(5):
                angle = i * 0.5 * np.pi
                radius = 1.0 + i * 0.5
                x = radius * np.cos(angle)
                y = radius * np.sin(angle)
                z = 0.9 + i * 0.3
                sequence.append({
                    "type": "aerial" if i % 2 == 1 else "ground",
                    "pos": [x, y, z]
                })
        elif sequence_type == "obstacle_course":
            # Create an obstacle course pattern
            sequence = [
                {"type": "ground", "pos": [1.0, 0.0, 0.9]},
                {"type": "aerial", "pos": [2.0, 0.0, 1.2]},
                {"type": "ground", "pos": [3.0, 1.0, 0.9]},
                {"type": "aerial", "pos": [4.0, 1.0, 1.5]},
                {"type": "ground", "pos": [5.0, 0.0, 0.9]},
            ]
        else:
            # Default mixed sequence
            sequence = [
                {"type": "ground", "pos": [2.0, 0.0, 0.9]},
                {"type": "aerial", "pos": [4.0, 0.0, 1.5]},
            ]
        
        return sequence

    def set_waypoint_sequence(self, sequence):
        """Set a sequence of waypoints to follow"""
        self.waypoint_sequence = sequence
        self.current_waypoint_index = 0
        self.waypoint_reached_count = 0
        
        # Set the first waypoint
        if sequence:
            first_waypoint = sequence[0]
            self.waypoint_type = first_waypoint["type"]
            self.waypoint_pos = np.array(first_waypoint["pos"])
            self.waypoint_reached = False

    def advance_waypoint_sequence(self):
        """Advance to the next waypoint in the sequence"""
        if hasattr(self, 'waypoint_sequence') and self.waypoint_sequence:
            self.current_waypoint_index += 1
            
            if self.current_waypoint_index < len(self.waypoint_sequence):
                # Move to next waypoint in sequence
                next_waypoint = self.waypoint_sequence[self.current_waypoint_index]
                self.waypoint_type = next_waypoint["type"]
                self.waypoint_pos = np.array(next_waypoint["pos"])
                self.waypoint_reached = False
                return True
            else:
                # Sequence completed, generate new random sequence
                new_sequence = self.create_waypoint_sequence("mixed")
                self.set_waypoint_sequence(new_sequence)
                return True
        
        return False

    def get_tarsus_pos(self, base_pos, base_rot, motor_pos):
        return self.cassie_fk.get_tarsus_pos(motor_pos, base_pos, base_rot)

    ##########################################
    #            Hopping Reward System        #
    ##########################################
    
    def update_hopping_state(self):
        """Update hopping phase and tracking variables"""
        current_height = self.qpos[2]
        vertical_velocity = self.qvel[2] if len(self.qvel) > 2 else 0
        
        # Update maximum height reached
        if current_height > self.max_height_reached:
            self.max_height_reached = current_height
        
        # Determine hopping phase
        if self.hop_phase == "ground" and current_height > 1.05:
            self.hop_phase = "ascent"
            self.hop_started = True
        elif self.hop_phase == "ascent" and vertical_velocity < -0.1:
            self.hop_phase = "descent"
        elif self.hop_phase == "descent" and current_height < 1.05:
            self.hop_phase = "landing"
            self.landing_started = True
            self.landing_count += 1
            self.total_landing_velocity += abs(vertical_velocity)
        elif self.hop_phase == "landing" and abs(vertical_velocity) < 0.05:
            self.hop_phase = "ground"
            self.landing_started = False
            
            # Check if hop was successful
            hop_height = self.max_height_reached - 1.0
            if hop_height > 0.3:  # At least 30cm hop
                self.successful_hops += 1
            
            # Reset for next hop
            self.max_height_reached = current_height

    def get_hop_height_reward(self):
        """Calculate reward for achieving hop height"""
        self.update_hopping_state()
        
        current_height = self.qpos[2]
        hop_height = current_height - 1.0  # Height above ground
        
        if hop_height < 0:
            return 0.0
        
        # Reward for reaching target height
        if hop_height >= (self.hop_target_height - 1.0):
            return 1.0  # Maximum reward for reaching target
        else:
            # Gradual reward based on height achieved
            return (hop_height / (self.hop_target_height - 1.0)) * 0.8

    def get_landing_smoothness_reward(self):
        """Calculate reward for smooth landing"""
        self.update_hopping_state()
        
        vertical_velocity = abs(self.qvel[2]) if len(self.qvel) > 2 else 0
        
        if self.hop_phase == "landing":
            # Reward for smooth landing (low vertical velocity)
            if vertical_velocity <= self.landing_smoothness_threshold:
                return 1.0  # Maximum reward for very smooth landing
            else:
                # Gradual penalty for rough landing
                return max(0.0, 1.0 - (vertical_velocity / 1.0))
        
        return 0.0  # No reward when not landing

    def get_thruster_efficiency_reward(self):
        """Calculate reward for efficient thruster usage"""
        if not hasattr(self, 'thruster_forces'):
            return 0.0
        
        # Calculate total thruster force
        total_force = abs(self.thruster_forces[0]) + abs(self.thruster_forces[1])
        
        # Reward for appropriate thruster usage during different phases
        if self.hop_phase == "ascent":
            # Reward for using thrusters during ascent
            if total_force > 50:  # Using thrusters for jumping
                return 0.5
        elif self.hop_phase == "landing":
            # Reward for using thrusters for smooth landing
            if total_force > 30:  # Using thrusters for landing
                return 0.3
        elif self.hop_phase == "ground":
            # Penalty for unnecessary thruster usage on ground
            if total_force > 20:
                return -0.1
        
        # Efficiency penalty for excessive force
        if total_force > 150:  # Very high force usage
            return -0.2
        
        return 0.0

    def get_stability_reward(self):
        """Calculate reward for maintaining stability during hopping"""
        # Check orientation stability
        orientation_error = np.sum(np.square(self.curr_rpy_gt))
        
        # Check if robot is upright (not fallen)
        if self.qpos[2] < 0.5:  # Fallen
            return -1.0
        
        # Reward for maintaining upright orientation
        stability_reward = max(0.0, 1.0 - orientation_error * 2.0)
        
        # Additional reward for maintaining stability during different phases
        if self.hop_phase in ["ascent", "descent"]:
            stability_reward *= 1.5  # Higher importance during flight
        
        return stability_reward