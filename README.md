# Modern Cassie RL Walking

A modernized version of the Cassie RL Walking project with TensorFlow 2.x, Python 3.10+, and GPU acceleration support.

## 🚀 Features

- **Modern TensorFlow 2.x**: Updated from TensorFlow 1.15 to TensorFlow 2.12+ with GPU support
- **Python 3.10+**: Compatible with modern Python versions
- **GPU Acceleration**: Full support for NVIDIA GPUs with CUDA 12.2
- **Improved Dependencies**: Updated all dependencies to stable, modern versions
- **Better Error Handling**: Enhanced error messages and debugging
- **Modern Code Structure**: Cleaner, more maintainable codebase

## 🖥️ System Requirements

- **Python**: 3.10 or higher
- **GPU**: NVIDIA GPU with CUDA support (optional, CPU fallback available)
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

### Training

#### Quick Start
```bash
# Basic training with GPU
python train_modern.py --train_name my_training --use_gpu

# Test mode (few iterations)
python train_modern.py --train_name test_run --test_mode --use_gpu
```

#### Advanced Training Options
```bash
python train_modern.py \
    --train_name advanced_training \
    --max_iters 10000 \
    --batch_size 1024 \
    --learning_rate 5e-5 \
    --timesteps_per_batch 8192 \
    --save_interval 50 \
    --use_gpu \
    --rnd_seed 42
```

#### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--train_name` | `modern_training` | Name for the training session |
| `--max_iters` | `5000` | Maximum training iterations |
| `--batch_size` | `512` | Batch size for training |
| `--learning_rate` | `1e-4` | Learning rate |
| `--timesteps_per_batch` | `4096` | Timesteps per actor batch |
| `--save_interval` | `100` | Save checkpoint every N iterations |
| `--use_gpu` | `True` | Enable GPU acceleration |
| `--rnd_seed` | `42` | Random seed for reproducibility |
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
├── assets/                 # Robot models and meshes
├── ckpts/                  # Model checkpoints
├── configs/                # Configuration files
├── logs/                   # Training logs
├── ppo/                    # PPO implementation
├── rlenv/                  # RL environment
├── scripts/                # Training and utility scripts
├── utility/                 # Utility functions
├── requirements.txt         # Python dependencies
├── setup.py                # Package setup
├── setup_env.sh            # Environment setup script
├── train_modern.py         # Modern training script
└── README.md               # This file
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the Creative Commons Attribution ShareAlike 4.0 License.

## 🙏 Acknowledgments

- Original implementation by Zhongyu Li
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
