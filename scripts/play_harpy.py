#!/usr/bin/env python3
"""
Play script for Harpy robot - similar to play.py but for the Harpy environment
"""

import numpy as np
import argparse
import time
import sys
import os

# Add the parent directory to the path to import rlenv
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rlenv.harpy_env import HarpyEnv, create_harpy_config

def get_args():
    parser = argparse.ArgumentParser(description="Harpy Robot Visualization and Testing")

    parser.add_argument(
        "--test_episode_len", type=int, default=5000, help="episode length to test"
    )
    parser.add_argument(
        "--visual", action="store_true", help="enable visualization"
    )
    parser.add_argument(
        "--thrust_force", type=float, default=8.0, help="thruster force (N)"
    )
    parser.add_argument(
        "--thrust_freq", type=float, default=0.5, help="thrusting frequency (Hz)"
    )
    parser.add_argument(
        "--model", type=str, default="harpy_cassie_legs", choices=["harpy_simple", "harpy_sketch", "harpy_cassie_legs"], 
        help="which Harpy model to use"
    )
    parser.add_argument(
        "--mode", type=str, default="hopping", choices=["standing", "hopping", "hovering"],
        help="behavior mode"
    )
    args = parser.parse_args()

    return args


def main():
    """
    Test Harpy robot with different behaviors
    """
    
    args = get_args()
    
    # Create Harpy configuration
    harpy_config = create_harpy_config()
    harpy_config.update({
        'model_path': f'assets/{args.model}.xml',
        'visual': args.visual,
        'max_time': 30.0,  # 30 second episodes
        'thrust_force': args.thrust_force,
        'thrust_frequency': args.thrust_freq
    })
    
    print("🤖 Harpy Robot Visualization")
    print("=" * 50)
    print(f"   - Model: {args.model}")
    print(f"   - Mode: {args.mode}")
    print(f"   - Thruster force: {args.thrust_force}N")
    print(f"   - Thrust frequency: {args.thrust_freq}Hz")
    print(f"   - Visualization: {'ON' if args.visual else 'OFF'}")
    print(f"   - Episode length: {args.test_episode_len} steps")
    
    env = HarpyEnv(config=harpy_config)
    
    if args.visual:
        print("\n🎮 Starting interactive visualization...")
        print("   - The robot will perform the selected behavior")
        print("   - Use Ctrl+C to stop")
        
        try:
            obs = env.reset()
            step_count = 0
            
            while step_count < args.test_episode_len:
                # Generate action based on mode
                action = generate_action(env, args.mode, step_count, args.thrust_force, args.thrust_freq)
                
                obs, reward, done, info = env.step(action)
                
                # Print status every 100 steps
                if step_count % 100 == 0:
                    print(f"Step {step_count:4d}: Height={info['torso_height']:6.3f}m, "
                          f"Velocity={info['torso_velocity']:6.3f}m/s, Reward={reward:7.3f}")
                
                step_count += 1
                
                if done:
                    print(f"Episode ended at step {step_count}")
                    obs = env.reset()
                    step_count = 0
                
                # Small delay for better visualization
                time.sleep(0.01)
                
        except KeyboardInterrupt:
            print("\n👋 Visualization stopped by user")
        finally:
            env.close()
            
    else:
        print("\n🏃 Running simulation without visualization...")
        obs = env.reset()
        total_reward = 0
        step_count = 0
        
        for step in range(args.test_episode_len):
            # Generate action based on mode
            action = generate_action(env, args.mode, step, args.thrust_force, args.thrust_freq)
            
            obs, reward, done, info = env.step(action)
            total_reward += reward
            step_count += 1
            
            if step % 200 == 0:
                print(f"Step {step:4d}: Height={info['torso_height']:6.3f}m, "
                      f"Velocity={info['torso_velocity']:6.3f}m/s, Reward={reward:7.3f}")
            
            if done:
                print(f"Episode ended at step {step}")
                obs = env.reset()
                step_count = 0
        
        print(f"\n📊 Final Results:")
        print(f"   - Total reward: {total_reward:.3f}")
        print(f"   - Average reward: {total_reward/args.test_episode_len:.3f}")
        print(f"   - Steps completed: {step_count}")


def generate_action(env, mode, step, thrust_force, thrust_freq):
    """Generate action based on the selected mode"""
    
    action = np.zeros(env.num_total_actions)
    
    if mode == "standing":
        # Just maintain standing pose
        action[0] = 0.0   # left_hip_roll
        action[1] = 0.4   # left_hip_pitch
        action[2] = -0.8  # left_knee
        action[3] = 0.0   # right_hip_roll
        action[4] = 0.4   # right_hip_pitch
        action[5] = -0.8  # right_knee
        action[6] = 0.0   # thruster_pitch
        # No thruster forces
        action[7] = 0.0   # left_thruster
        action[8] = 0.0   # right_thruster
        
    elif mode == "hopping":
        # Hopping motion with periodic thrust
        phase = (step * 0.01) * 2 * np.pi * thrust_freq
        
        # Leg motion for hopping
        hip_amplitude = 0.3
        hip_offset = 0.4
        hip_pitch = hip_offset + hip_amplitude * np.sin(phase)
        
        knee_amplitude = 0.4
        knee_offset = -0.8
        knee_angle = knee_offset - knee_amplitude * np.sin(phase)
        
        action[0] = 0.0        # left_hip_roll
        action[1] = hip_pitch  # left_hip_pitch
        action[2] = knee_angle # left_knee
        action[3] = 0.0        # right_hip_roll
        action[4] = hip_pitch  # right_hip_pitch
        action[5] = knee_angle # right_knee
        
        # Thruster control - periodic thrust
        if np.sin(phase) > 0.5:  # Thrust during upward motion
            action[7] = thrust_force   # left_thruster
            action[8] = thrust_force   # right_thruster
            action[6] = 0.1 * np.sin(phase)  # slight pitch
        else:
            action[7] = 0.0   # left_thruster
            action[8] = 0.0   # right_thruster
            action[6] = 0.0   # thruster_pitch
            
    elif mode == "hovering":
        # Continuous hovering with thrust
        height = env.data.qpos[2] if hasattr(env, 'data') else 0.9
        
        # Maintain standing leg pose
        action[0] = 0.0   # left_hip_roll
        action[1] = 0.4   # left_hip_pitch
        action[2] = -0.8  # left_knee
        action[3] = 0.0   # right_hip_roll
        action[4] = 0.4   # right_hip_pitch
        action[5] = -0.8  # right_knee
        action[6] = 0.0   # thruster_pitch
        
        # Adjust thrust based on height
        target_height = 1.0
        if height < target_height - 0.1:
            # Need more thrust
            action[7] = thrust_force * 1.5   # left_thruster
            action[8] = thrust_force * 1.5   # right_thruster
        elif height > target_height + 0.1:
            # Too high, reduce thrust
            action[7] = thrust_force * 0.5   # left_thruster
            action[8] = thrust_force * 0.5   # right_thruster
        else:
            # Maintain altitude
            action[7] = thrust_force         # left_thruster
            action[8] = thrust_force         # right_thruster
    
    return action


if __name__ == "__main__":
    main()
