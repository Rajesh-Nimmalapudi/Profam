#!/bin/bash
# ============================================
# LOGIN NODE SETUP (NO PYTORCH)
# ============================================

echo "============================================"
echo " [LOGIN NODE] Creating profam-env (CPU only)"
echo "============================================"

module load miniconda/3

# Remove old env if exists
echo "Removing old profam-env if present..."
conda env remove -n profam-env -y >/dev/null 2>&1

# Create fresh env
echo "Creating environment with Python 3.10..."
conda create -n profam-env python=3.10 -y

# Activate safely
source $(conda info --base)/etc/profile.d/conda.sh
conda activate profam-env

# Upgrade pip
pip install --upgrade pip

# Install CPU-only dependencies
echo "Installing requirements.txt (NO torch)..."
pip install -r requirements.txt

echo "--------------------------------------------"
echo "Login-node setup complete."
echo "NEXT:"
echo "  srun --qos=normal --gres=gpu:1 --cpus-per-task=4 --mem=32G --pty bash"
echo "  then run setup_gpu.sh"
echo "--------------------------------------------"

conda deactivate

