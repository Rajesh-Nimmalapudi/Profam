#!/bin/bash
# ============================================
# GPU NODE SETUP (PYTORCH + MAMBA)
# ============================================

echo "============================================"
echo " [GPU NODE] Installing PyTorch + Mamba"
echo "============================================"

module load miniconda/3
source $(conda info --base)/etc/profile.d/conda.sh
conda activate profam-env

# Install PyTorch via pip (CUDA 12.1 wheels)
echo "Installing PyTorch 2.3.1 (pip, CUDA 12.1)..."
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 \
  --index-url https://download.pytorch.org/whl/cu121

# Verify torch BEFORE proceeding
echo "Verifying PyTorch..."
python - <<EOF
import torch
print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
assert torch.cuda.is_available(), "CUDA NOT AVAILABLE"
EOF

# Install CUDA extensions
echo "Installing causal-conv1d..."
pip install causal-conv1d==1.4.0

echo "Installing mamba-ssm..."
pip install mamba-ssm==2.2.2

# Final verification
echo "Final verification..."
python - <<EOF
import torch
import mamba_ssm
print("Torch OK:", torch.cuda.is_available())
print("Mamba-SSM OK ✅")
EOF

echo "--------------------------------------------"
echo "GPU setup complete."
echo "Environment profam-env is READY."
echo "--------------------------------------------"

conda deactivate

