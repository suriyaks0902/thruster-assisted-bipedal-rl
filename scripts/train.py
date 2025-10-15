import os
import sys
import tensorflow as tf
import numpy as np
from typing import Optional, Callable, Dict, Any
import argparse
import logging
from pathlib import Path

# Configure TensorFlow for GPU
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        # Enable memory growth to avoid allocating all GPU memory at once
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
        print(f"✅ GPU devices found: {len(physical_devices)}")
        print(f"✅ Using GPU: {tf.config.list_physical_devices('GPU')[0]}")
    except RuntimeError as e:
        print(f"⚠️ GPU memory growth setting failed: {e}")
else:
    print("⚠️ No GPU devices found, using CPU")

# Set TensorFlow logging level
tf.get_logger().setLevel(logging.ERROR)

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from ppo.policies import MLPCNNPolicy
from rlenv.cassie_env import CassieEnv
from configs.defaults import ROOT_PATH
import mpi4py.MPI as MPI

# Configure environment
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

model_folder = ROOT_PATH + "/ckpts/"


def get_args():
    parser = argparse.ArgumentParser(description="Leonardo Hopping RL Training")

    parser.add_argument(
        "--train_name", type=str, default="leonardo_hopping", help="Training session name"
    )

    parser.add_argument(
        "--rnd_seed", type=int, default=42, help="Random seed (default 42)"
    )

    parser.add_argument(
        "--max_iters", type=int, default=5000, help="Max iterations (default 5000)"
    )

    parser.add_argument(
        "--restore_from",
        type=str,
        default=None,
        help="Restore from previous checkpoint (default None)",
    )

    parser.add_argument(
        "--restore_cont",
        type=int,
        default=None,
        help="Continuation count for resumed training (default None)",
    )

    parser.add_argument(
        "--save_interval",
        type=int,
        default=100,
        help="Save checkpoint every N iterations",
    )

    parser.add_argument(
        "--use_gpu",
        action="store_true",
        default=True,
        help="Use GPU for training (default True)",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=512,
        help="Batch size for training (default 512)",
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate (default 1e-4)",
    )

    parser.add_argument(
        "--timesteps_per_batch",
        type=int,
        default=4096,
        help="Timesteps per actor batch (default 4096)",
    )

    parser.add_argument(
        "--hop_target_height",
        type=float,
        default=1.5,
        help="Target hopping height in meters (default 1.5)",
    )

    parser.add_argument(
        "--hop_frequency",
        type=float,
        default=0.5,
        help="Hopping frequency in Hz (default 0.5)",
    )

    parser.add_argument(
        "--visual",
        action="store_true",
        default=False,
        help="Enable visualization during training",
    )

    return parser.parse_args()


def setup_logging(train_name: str, restore_cont: Optional[int] = None):
    """Setup logging and model naming"""
    if restore_cont and args.restore_from:
        saved_model = f"{train_name}_rnds{args.rnd_seed}_cont{restore_cont}"
    else:
        saved_model = f"{train_name}_rnds{args.rnd_seed}"

    log_dir = Path(ROOT_PATH) / "logs" / saved_model
    log_dir.mkdir(parents=True, exist_ok=True)
    
    os.environ["OPENAI_LOGDIR"] = str(log_dir)
    print(f"[Train]: MODEL_TO_SAVE: {saved_model}")
    print(f"[Train]: LOG_DIR: {log_dir}")
    
    return saved_model


def create_policy_fn(hid_size: int = 512, num_hid_layers: int = 2):
    """Create policy function with modern TensorFlow 2.x"""
    def policy_fn(name, ob_space_vf, ob_space_pol, ob_space_pol_cnn, ac_space):
        return MLPCNNPolicy(
            name=name,
            ob_space_vf=ob_space_vf,
            ob_space_pol=ob_space_pol,
            ob_space_pol_cnn=ob_space_pol_cnn,
            ac_space=ac_space,
            hid_size=hid_size,
            num_hid_layers=num_hid_layers,
        )
    return policy_fn


def train(max_iters: int, use_gpu: bool = True, callback: Optional[Callable] = None):
    """Modern training function with TensorFlow 2.x"""
    
    # Configure TensorFlow session
    if not use_gpu:
        tf.config.set_visible_devices([], 'GPU')
        print("🖥️ Using CPU for training")
    else:
        print("🚀 Using GPU for training")
        # Verify GPU is available
        if not tf.config.list_physical_devices('GPU'):
            print("⚠️ GPU requested but not available, falling back to CPU")
            tf.config.set_visible_devices([], 'GPU')

    # Set random seeds for reproducibility
    tf.random.set_seed(args.rnd_seed)
    np.random.seed(args.rnd_seed)
    
    # Create policy function
    policy_fn = create_policy_fn()
    
    # Import environment config and create environment
    from configs.env_config import config_train
    
    # Create hopping configuration
    hopping_config = config_train.copy()
    hopping_config.update({
        "hopping_mode": True,
        "thruster_enabled": True,
        "hop_target_height": args.hop_target_height,
        "hop_frequency": args.hop_frequency,
        "is_visual": args.visual,
        "max_timesteps": 1000,  # Shorter episodes for hopping
    })
    
    env = CassieEnv(config=hopping_config)
    
    # Import PPO implementation
    from ppo import ppo_sgd_cnn as ppo_sgd
    
    print(f"🎯 Starting Leonardo Hopping Training with:")
    print(f"   - Max iterations: {max_iters}")
    print(f"   - Batch size: {args.batch_size}")
    print(f"   - Learning rate: {args.learning_rate}")
    print(f"   - Timesteps per batch: {args.timesteps_per_batch}")
    print(f"   - Target hop height: {args.hop_target_height}m")
    print(f"   - Hopping frequency: {args.hop_frequency}Hz")
    print(f"   - GPU enabled: {use_gpu}")
    print(f"   - Visualization: {args.visual}")

    # Start PPO training
    pi = ppo_sgd.learn(
        env,
        policy_fn,
        max_iters=max_iters,
        timesteps_per_actorbatch=args.timesteps_per_batch,
        clip_param=0.2,
        entcoeff=0,
        optim_epochs=2,
        optim_stepsize=args.learning_rate,
        optim_batchsize=args.batch_size,
        gamma=0.98,
        lam=0.95,
        callback=callback,
        schedule="constant",
        continue_from=args.restore_from,
    )
    
    return pi


def training_callback(locals_: Dict[str, Any], globals_: Dict[str, Any]):
    """Enhanced training callback with better logging"""
    saver_ = locals_["saver"]
    sess_ = tf.compat.v1.get_default_session()
    timesteps_so_far_ = locals_["timesteps_so_far"]
    iters_so_far_ = locals_["iters_so_far"]
    
    model_dir = Path(model_folder) / saved_model
    model_dir.mkdir(parents=True, exist_ok=True)
    
    if MPI.COMM_WORLD.Get_rank() == 0 and iters_so_far_ % args.save_interval == 0:
        checkpoint_path = model_dir / "model"
        saver_.save(sess_, str(checkpoint_path), global_step=timesteps_so_far_)
        print(f"💾 Saved checkpoint at iteration {iters_so_far_}")
    
    return True


if __name__ == "__main__":
    # Parse arguments
    args = get_args()
    
    # Setup logging and model naming
    saved_model = setup_logging(args.train_name, args.restore_cont)
    
    # Configure MPI logging
    if MPI.COMM_WORLD.Get_rank() == 0:
        print("🚀 Starting Leonardo Hopping RL Training with Modern TensorFlow")
        print(f"🎯 Training configuration:")
        print(f"   - Name: {args.train_name}")
        print(f"   - Random seed: {args.rnd_seed}")
        print(f"   - Max iterations: {args.max_iters}")
        print(f"   - GPU enabled: {args.use_gpu}")
        print(f"   - Restore from: {args.restore_from}")
        print(f"   - Save interval: {args.save_interval}")
        print(f"   - Target hop height: {args.hop_target_height}m")
        print(f"   - Hopping frequency: {args.hop_frequency}Hz")
    
    # Start training
    train(
        max_iters=args.max_iters, 
        use_gpu=args.use_gpu, 
        callback=training_callback
    )
