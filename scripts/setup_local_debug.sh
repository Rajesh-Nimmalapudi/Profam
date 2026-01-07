#!/bin/bash
set -e

echo "======================================"
echo "[LOCAL DEBUG] Setting up 'profam-debug'"
echo "GPU: RTX 3090 | CUDA: 12.1/13.0 Compatible"
echo "======================================"

# 1. Clean Slate
conda env remove -n profam-debug -y || true
conda create -n profam-debug python=3.10 -y

# 1.5 Install C++ Compiler (Critical for compiling extensions)
# "g++ failed" error happens because system g++ is missing or too old.
conda install -n profam-debug gxx_linux-64 -c conda-forge -y

# 2. Activate
source $(conda info --base)/etc/profile.d/conda.sh
conda activate profam-debug

# 3. Base Dependecies
pip install --upgrade pip
pip install -r requirements.txt

# 4. PyTorch (Stable 2.4 for RTX 30 Series)
# We use cu121 because it's the most stable wheel for 3090s usually.
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 5. Mamba & Causal Conv1d (Pre-built wheels or source)
# Since we are local, we try standard pip. If it fails, SDPA fallback saves us.
pip install causal-conv1d>=1.4.0
pip install mamba-ssm>=2.2.2

echo "--------------------------------------"
echo "Setup Complete! Activate with:"
echo "conda activate profam-debug"
echo "--------------------------------------"
