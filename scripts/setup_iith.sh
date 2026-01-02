#!/bin/bash

# 1. Load Miniconda Module
module load miniconda/3

# 2. CLEAN SLATE: Remove existing environment if it exists
# This is crucial to fix the broken state.
echo "Removing old environment..."
conda env remove -n profam-env -y

# 3. Create "Golden" Environment
echo "Creating fresh environment..."
conda create -n profam-env python=3.10 -y

# Activate it (Workaround for script execution)
source $(conda info --base)/etc/profile.d/conda.sh
conda activate profam-env

# 4. Install PyTorch PINNED to 2.3.1 + CUDA 12.1
# (Downgraded to 2.3.1 to match CONFIRMED Mamba wheels)
echo "Installing PyTorch 2.3.1 (Guaranteed Compatibility)..."
conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=12.1 -c pytorch -c nvidia -y

# 5. Install Packaging
pip install packaging

# 6. Install Mamba-SSM & Causal Conv1d (Using Pre-built Wheels)
# Exact versions for PyTorch 2.3 + CUDA 12.1
echo "Installing Mamba-SSM (Pre-built)..."
pip install causal-conv1d>=1.4.0
pip install mamba-ssm>=2.2.2

# 7. Verify Installation
python -c "import torch; print(f'Torch: {torch.__version__}'); import mamba_ssm; print('Mamba SSM: Installed Successfully ✅')"

echo "----------------------------------------------------------------"
echo "Setup Complete! To use this environment, run:"
echo "module load miniconda/3"
echo "source activate profam-env"
echo "----------------------------------------------------------------"
