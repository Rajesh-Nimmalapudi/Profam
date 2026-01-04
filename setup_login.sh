#!/bin/bash
set -e

echo "======================================"
echo "[LOGIN NODE] Base environment setup"
echo "======================================"

module load miniconda/3

# Clean old env if exists
conda env remove -n profam-env -y || true

# Create environment
conda create -n profam-env python=3.10 -y

# Activate
source $(conda info --base)/etc/profile.d/conda.sh
conda activate profam-env

# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install base requirements (NO CUDA EXTENSIONS HERE)
pip install -r requirements.txt

# Install PyTorch (pip wheels – HPC safe)
pip install \
  torch==2.3.1+cu121 \
  torchvision==0.18.1+cu121 \
  torchaudio==2.3.1+cu121 \
  --index-url https://download.pytorch.org/whl/cu121

echo "--------------------------------------"
echo "LOGIN setup complete."
echo "NEXT: request GPU node and run setup_gpu.sh"
echo "--------------------------------------"

