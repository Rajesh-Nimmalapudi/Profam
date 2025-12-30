#!/bin/bash
# IITH Server Setup Script for ProFam-v2
# Usage: source scripts/setup_iith.sh

echo "Setting up ProFam-v2 Environment on IITH..."

# 1. Load Anaconda (Standard Module on most HPCs)
if command -v module &> /dev/null; then
    module load anaconda/3
    echo "Loaded anaconda module."
fi

# 2. Create Environment
# Check if env exists
if conda info --envs | grep -q "profam-env"; then
    echo "Environment profam-env already exists."
else
    echo "Creating conda environment 'profam-env'..."
    conda create -n profam-env python=3.10 -y
fi

# 3. Activate
source activate profam-env || conda activate profam-env

# 4. Install Core Dependencies
echo "Installing Dependencies..."
# PyTorch with CUDA 11.8 or 12.1 (Adjust based on A100 driver)
# Assuming A100 supports CUDA 12.1 which is standard now
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# ProFam Dependencies
pip install -r requirements.txt
pip install lightning transformers hydra-core rootutils rich wandb pandas numpy

# 5. Install Mamba Kernels (Optimized for A100)
echo "Installing Mamba Optimized Kernels..."
pip install causal-conv1d>=1.2.0
pip install mamba-ssm>=1.2.0

# 6. Install Flash Attention 2 (Essential for A100)
echo "Installing Flash Attention 2..."
pip install flash-attn --no-build-isolation

# 7. Install Dev Requirements
if [ -f "requirements-dev.txt" ]; then
    echo "Installing Dev Requirements..."
    pip install -r requirements-dev.txt
fi

echo "Environment Setup Complete!"
echo "To activate: conda activate profam-env"
