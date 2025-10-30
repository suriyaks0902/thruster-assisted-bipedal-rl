# Harpy: Thruster-Assisted Bipedal Jumping

A reinforcement learning system for training the Harpy robot (Cassie + thrusters) to perform thruster-assisted jumping and hopping maneuvers. Based on the modern Cassie RL framework with TensorFlow 2.x, Python 3.10+, and GPU acceleration.

## 🚀 Features

- **Thruster-Assisted Jumping**: Train the robot to perform vertical jumps using combined leg power and thrusters
- **Dual-History RL Architecture**: Short-term (0.1s) and long-term (2s) memory for adaptive control
- **Modern TensorFlow 2.x**: Updated from TensorFlow 1.15 to TensorFlow 2.12+ with GPU support
- **14-DOF Control**: 10 leg motors + 2 thruster forces + 2 thruster pitch angles
- **Hopping Mode**: Specialized reward system for jump height, soft landings, and thruster efficiency
- **GPU Acceleration**: Full support for NVIDIA GPUs with CUDA 12.2
- **Reference Motion System**: Gait library + jump reference generator with 5-phase jumping

## 🎯 Training Objectives

The system trains Harpy to achieve three primary goals:

1. **Maximum Jump Height** 🚀: Use thrusters to exceed leg-only jumping capability (target: 1.0-1.5m apex)
2. **Soft Landings** 🛬: Minimize impact forces through retro-propulsive thrust braking before touchdown
3. **Energy Efficiency** ⚡: Use minimum thrust needed for each jump task (short, targeted bursts)

## 🖥️ System Requirements

- **Python**: 3.10 or higher
- **GPU**: NVIDIA GPU with CUDA support (recommended for faster training)
- **OS**: Linux (tested on Ubuntu 20.04+)
- **Memory**: 8GB RAM minimum, 16GB recommended
- **Storage**: 10GB free space

## 📦 Installation

### Quick Setup (Recommended)

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd cassie_rl_walking-master
   ```

2. **Run the automated setup script**:
   ```bash
   ./setup_env.sh
   ```

3. **Activate the environment**:
   ```bash
   source cassie_env/bin/activate
   ```

### Manual Setup

1. **Create a virtual environment**:
   ```bash
   python3 -m venv cassie_env
   source cassie_env/bin/activate
   ```

2. **Install dependencies**:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   pip install -e .
   ```

3. **Verify installation**:
   ```bash
   python -c "import tensorflow as tf; print(f'TensorFlow {tf.__version__}'); print(f'GPU available: {len(tf.config.list_physical_devices(\"GPU\")) > 0}')"
   ```

## 🎯 Usage

### Training Curriculum (3 Stages)

The system follows a progressive training curriculum based on the Cassie research paper:

#### **Stage 1: Core Jump Skill Learning** (5-10M timesteps)
Learn basic thruster-assisted jump with a single target height (e.g., 0.5-1.0m).

```bash
# Start Stage 1 training
python scripts/train.py --stage 1 --target_height 0.5 --max_iters 5000
```

**Focus**: Master coordinated leg extension + thruster firing + soft landing

#### **Stage 2: Task Randomization** (10-20M timesteps)
Generalize to varied jump heights (0.3m → 1.5m) and distances.

```bash
# Stage 2 with random jump targets
python scripts/train.py --stage 2 --max_iters 10000 --randomize_targets
```

**Focus**: Learn to use optimal thrust for each scenario, exceed reference limitations

#### **Stage 3: Dynamics Randomization** (10-20M timesteps)
Add robustness through thruster-specific randomization.

```bash
# Stage 3 with full randomization
python scripts/train.py --stage 3 --max_iters 10000 --dynamics_randomization
```

**Focus**: Handle thrust variability, mass changes, sensor noise

### Quick Start (Test Mode)

```bash
# Quick test run with minimal iterations
python scripts/train.py --test_mode --max_iters 100

# Monitor training with TensorBoard
tensorboard --logdir logs/
```

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--stage` | `1` | Training stage (1, 2, or 3) |
| `--target_height` | `1.0` | Target jump height in meters |
| `--max_iters` | `5000` | Maximum training iterations |
| `--timesteps_per_batch` | `4096` | Timesteps per batch |
| `--learning_rate` | `1e-5` | PPO learning rate |
| `--randomize_targets` | `False` | Enable target randomization (Stage 2) |
| `--dynamics_randomization` | `False` | Enable dynamics randomization (Stage 3) |
| `--test_mode` | `False` | Run with minimal iterations for testing |

### Monitoring Training

1. **TensorBoard** (recommended):
   ```bash
   tensorboard --logdir logs/
   ```
   Then open http://localhost:6006 in your browser

2. **Check logs**:
   ```bash
   tail -f logs/<training_name>/log.txt
   ```

### Playing Trained Models

```bash
python scripts/play.py --model_path ckpts/<training_name>/model-<step>
```

## 🔧 Configuration

### Environment Configuration

Edit `configs/env_config.py` to modify:
- Robot parameters
- Training environment settings
- Reward functions
- Observation spaces

### GPU Configuration

The system automatically detects and configures GPU usage. To force CPU-only mode:

```bash
python train_modern.py --train_name cpu_training  # GPU flag defaults to True
```

## 📊 Performance

### GPU Performance (RTX 4050)
- **Training Speed**: ~2-3x faster than CPU
- **Memory Usage**: ~4-6GB GPU memory
- **Batch Size**: Up to 1024 recommended

### CPU Performance
- **Training Speed**: Slower but functional
- **Memory Usage**: ~8-12GB RAM
- **Batch Size**: 256-512 recommended

## 🐛 Troubleshooting

### Common Issues

1. **Import Errors**:
   ```bash
   pip install -r requirements.txt
   pip install -e .
   ```

2. **GPU Not Detected**:
   ```bash
   nvidia-smi  # Check GPU availability
   python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
   ```

3. **Memory Issues**:
   - Reduce batch size: `--batch_size 256`
   - Enable memory growth (automatic)
   - Close other GPU applications

4. **CUDA Version Mismatch**:
   ```bash
   pip uninstall tensorflow tensorflow-gpu
   pip install tensorflow[gpu]>=2.12.0
   ```

### Debug Mode

Run with verbose logging:
```bash
python train_modern.py --train_name debug --test_mode --max_iters 5
```

## 📁 Project Structure

```
cassie_rl_walking-master/
├── assets/                       # Robot models (harpy_sophisticated_complete.xml)
│   └── research_idea.md          # Research paper analysis and methods
├── ckpts/                        # Model checkpoints
├── configs/                      # Configuration files
│   ├── defaults.py               # Joint limits, PD gains, torque limits
│   └── env_config.py             # Environment parameters
├── logs/                         # Training logs and TensorBoard data
├── ppo/                          # PPO implementation
│   ├── policies.py               # Actor-critic networks
│   └── ppo_sgd_mlp.py            # PPO with dual-history architecture
├── rlenv/                        # RL environment
│   ├── cassie_env.py             # Main environment (hopping mode enabled)
│   ├── reference_generator.py   # Jump reference motion (5 phases)
│   ├── gait_library.py           # Bezier curve gait parameterization
│   └── cassiemujoco.py           # MuJoCo simulation wrapper
├── scripts/                      # Training scripts
│   └── train.py                  # Main training entry point
├── training_config.json          # Training hyperparameters
└── README.md                     # This file
```

## 🎖️ Reward System

The hopping mode uses 17 reward components focused on jumping performance:

### Core Rewards (from Cassie)
- Motor position/velocity tracking
- Pelvis position/velocity/orientation
- Torque minimization
- Foot force smoothing
- Action smoothness

### Jumping-Specific Rewards
1. **Hop Height** (weight: 10.0): Reward for achieving target apex height
2. **Landing Smoothness** (weight: 15.0): Penalize hard impacts, encourage retro-propulsive braking
3. **Thruster Efficiency** (weight: 5.0): Minimize thrust usage while achieving goals
4. **Stability** (weight: 8.0): Maintain upright orientation during flight

### Reward Weight Adjustments for Jumping
- ↓ Motor tracking: 0.5× (allow more freedom from reference)
- ↑ Orientation: 2.0× (critical for stable landing)
- ↓ Torque penalty: 0.5× (allow explosive leg movements)
- ↓ Foot force: 0.3× (expect harder landings than walking)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the Creative Commons Attribution ShareAlike 4.0 License.

## 🙏 Acknowledgments

- **Cassie RL Framework**: Li et al. (2024) - "Reinforcement Learning for Versatile, Dynamic, and Robust Bipedal Locomotion Control" (arXiv:2401.16889v2)
- **Harpy Robot**: Dangol et al. (2021) - "Control of Thruster-Assisted, Bipedal Legged Locomotion" (Frontiers in Robotics and AI)
- MuJoCo physics engine
- TensorFlow team for GPU support
- OpenAI Baselines for RL algorithms

## 📞 Support

For issues and questions:
1. Check the troubleshooting section
2. Search existing issues
3. Create a new issue with detailed information

---

**Happy Training! 🚀**
