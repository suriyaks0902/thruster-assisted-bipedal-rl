#!/bin/bash

# Modern Cassie RL Walking Setup Script
# Compatible with Python 3.10+ and TensorFlow 2.x with GPU support

set -e

echo "🚀 Setting up Modern Cassie RL Walking Environment"
echo "=================================================="

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
REQUIRED_VERSION="3.10"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" = "$REQUIRED_VERSION" ]; then
    echo "✅ Python version $PYTHON_VERSION is compatible"
else
    echo "❌ Python version $PYTHON_VERSION is too old. Please install Python 3.10+"
    exit 1
fi

# Check if CUDA is available
if command -v nvidia-smi &> /dev/null; then
    echo "✅ NVIDIA GPU detected"
    CUDA_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits | head -1)
    echo "   Driver version: $CUDA_VERSION"
else
    echo "⚠️ No NVIDIA GPU detected, will use CPU"
fi

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv cassie_env
source cassie_env/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install TensorFlow (GPU support included by default)
echo "🤖 Installing TensorFlow with GPU support..."
pip install tensorflow>=2.12.0

# Install other dependencies
echo "📚 Installing dependencies..."
pip install -r requirements.txt

# Install the package in development mode
echo "🔧 Installing package in development mode..."
pip install -e .

# Verify installation
echo "🔍 Verifying installation..."
python3 -c "
import tensorflow as tf
import numpy as np
import gym
import mujoco
import kinpy
import mpi4py

print('✅ TensorFlow version:', tf.__version__)
print('✅ GPU available:', len(tf.config.list_physical_devices('GPU')) > 0)
if tf.config.list_physical_devices('GPU'):
    print('✅ GPU devices:', tf.config.list_physical_devices('GPU'))
print('✅ NumPy version:', np.__version__)
print('✅ Gym version:', gym.__version__)
print('✅ MuJoCo version:', mujoco.__version__)
print('✅ All dependencies installed successfully!')
"

echo ""
echo "🎉 Setup complete! To activate the environment:"
echo "   source cassie_env/bin/activate"
echo ""
echo "🚀 To start training:"
echo "   python train_modern.py --train_name my_training --use_gpu"
echo ""
echo "📊 To monitor training:"
echo "   tensorboard --logdir logs/"
