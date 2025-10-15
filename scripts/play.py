from baselines.common import tf_util as U
import tensorflow as tf
import numpy as np
from rlenv.cassie_env import CassieEnv
import ppo.policies as policies
from configs.env_config import config_play, config_hopping
from configs.defaults import ROOT_PATH

# to ignore specific deprecation warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import argparse

model_folder = ROOT_PATH + "/ckpts/"

def get_args():
    parser = argparse.ArgumentParser(description="Leonardo Hopping Reference Test")

    parser.add_argument(
        "--test_model", type=str, default=None, help="checkpoint to test (default None)"
    )
    parser.add_argument(
        "--test_episode_len", type=int, default=20000, help="episode length to test"
    )
    parser.add_argument(
        "--visual", action="store_true", help="enable visualization"
    )
    parser.add_argument(
        "--hop_height", type=float, default=1.0, help="target hop height"
    )
    parser.add_argument(
        "--hop_freq", type=float, default=0.5, help="hopping frequency"
    )
    args = parser.parse_args()

    return args


args = get_args()
# model name to test
test_model = args.test_model
test_episode_len = args.test_episode_len


def main():
    """
    Test Leonardo hopping reference motion
    """
    
    # Create Leonardo hopping configuration
    leonardo_config = config_hopping.copy()
    leonardo_config.update({
        'model_path': 'assets/cassie_clean.xml',
        'thruster_enabled': True,
        'hop_target_height': args.hop_height,
        'hop_frequency': args.hop_freq,
        'is_visual': args.visual,
        'add_standing': True
    })
    
    env = CassieEnv(config=leonardo_config)
    
    # If no model specified, just run reference motion
    if test_model is None:
        print("🚀 Testing Leonardo Hopping Reference Motion")
        print(f"   - Target hop height: {args.hop_height}m")
        print(f"   - Hopping frequency: {args.hop_freq}Hz")
        print(f"   - Visualization: {'ON' if args.visual else 'OFF'}")
        
        # Run reference motion without trained policy
        if args.visual:
            draw_state = env.render()
            while draw_state:
                obs_vf, obs_pol = env.reset()
                for _ in range(test_episode_len):
                    if not env.vis.ispaused():
                        # Use reference motion (zero actions for now)
                        ac = np.zeros(env.action_space.shape[0])
                        obs_vf, obs_pol, reward, done, info = env.step(ac)
                        draw_state = env.render()
                        if done:
                            env.reset()
                            break
                    else:
                        while env.vis.ispaused() and draw_state:
                            draw_state = env.render()
        else:
            print("Running reference motion without visualization...")
            obs_vf, obs_pol = env.reset()
            total_reward = 0
            step_count = 0
            
            for step in range(test_episode_len):
                # Use reference motion (zero actions for now)
                ac = np.zeros(env.action_space.shape[0])
                obs_vf, obs_pol, reward, done, info = env.step(ac)
                total_reward += reward
                step_count += 1
                
                if step % 100 == 0:
                    print(f"Step {step}: Reward = {reward:.3f}, Total = {total_reward:.3f}")
                
                if done:
                    print(f"Episode finished after {step_count} steps. Total reward: {total_reward:.3f}")
                    obs_vf, obs_pol = env.reset()
                    total_reward = 0
                    step_count = 0
                    break
            
            print("Reference motion test completed successfully!")
        return
    
    # Load trained model if specified
    model_dir = model_folder + test_model
    latest_checkpoint = tf.train.latest_checkpoint(model_dir)
    model_path = latest_checkpoint
    config = tf.ConfigProto(device_count={"GPU": 0})

    ob_space_pol = env.observation_space_pol
    ac_space = env.action_space
    ob_space_vf = env.observation_space_vf
    ob_space_pol_cnn = env.observation_space_pol_cnn
    pi = policies.MLPCNNPolicy(
        name="pi",
        ob_space_vf=ob_space_vf,
        ob_space_pol=ob_space_pol,
        ob_space_pol_cnn=ob_space_pol_cnn,
        ac_space=ac_space,
        hid_size=512,
        num_hid_layers=2,
    )

    U.make_session(config=config)
    U.load_state(model_path)

    # Check if visualization is enabled
    if leonardo_config.get("is_visual", False):
        draw_state = env.render()
        while draw_state:
            obs_vf, obs_pol = env.reset()
            for _ in range(test_episode_len):
                if not env.vis.ispaused():
                    ac = pi.act(stochastic=False, ob_vf=obs_vf, ob_pol=obs_pol)[0]
                    obs_vf, obs_pol, reward, done, info = env.step(ac)
                    draw_state = env.render()
                    if done:
                        env.reset()
                        break
                else:
                    while env.vis.ispaused() and draw_state:
                        draw_state = env.render()
    else:
        # Run without visualization
        print("Running simulation without visualization...")
        obs_vf, obs_pol = env.reset()
        total_reward = 0
        step_count = 0
        
        for step in range(test_episode_len):
            ac = pi.act(stochastic=False, ob_vf=obs_vf, ob_pol=obs_pol)[0]
            obs_vf, obs_pol, reward, done, info = env.step(ac)
            total_reward += reward
            step_count += 1
            
            if step % 100 == 0:
                print(f"Step {step}: Reward = {reward:.3f}, Total = {total_reward:.3f}")
            
            if done:
                print(f"Episode finished after {step_count} steps. Total reward: {total_reward:.3f}")
                obs_vf, obs_pol = env.reset()
                total_reward = 0
                step_count = 0
                break
        
        print("Simulation completed successfully!")


if __name__ == "__main__":
    main()
