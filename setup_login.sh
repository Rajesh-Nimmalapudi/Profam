#!/bin/bash
# ============================================
# Login Node Setup Script (CPU-safe only)
# ============================================

echo "============================================"
echo " [LOGIN NODE] Setting up profam-env"
echo "============================================"

# 1. Load Miniconda module
module load miniconda/3

# 2. Remove old environment if exists
echo "Removing old profam-env (if any)..."
conda env remove -n profam-env -y >/dev/null 2>&1

# 3. Create fresh environment (Python 3.10)
echo "Creating new environment (Python 3.10)..."
conda create -n profam-env python=3.10 -y

# 4. Activate environment (script-safe)
source $(conda info --base)/etc/profile.d/conda.sh
conda activate profam-env

# 5. Install PyTorch (CUDA runtime included, GPU not required here)
echo "Installing PyTorch 2.3.1 + CUDA 12.1 runtime..."
conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 \
              pytorch-cuda=12.1 -c pytorch -c nvidia -y

# 6. Install remaining CPU / Python dependencies
echo "Installing Python dependencies from requirements.txt..."
pip install --upgrade pip
pip install -r requirements.txt

# 7. Done
echo "------------------------------------------------"
echo "Login-node setup complete."
echo "NEXT STEP:"
echo "  1) Request a GPU node using srun"
echo "  2) Run setup_gpu.sh inside that node"
echo "------------------------------------------------"

conda deactivate
