#!/bin/bash
set -e

echo "======================================"
echo "[GPU NODE] CUDA extension installation"
echo "======================================"

module load cuda/13.0
module load miniconda/3

source $(conda info --base)/etc/profile.d/conda.sh
conda activate profam-env

# Sanity check
python - <<EOF
import torch
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
EOF

echo "Installing causal-conv1d..."
pip install causal-conv1d==1.4.0 --no-build-isolation

echo "Installing mamba-ssm..."
pip install mamba-ssm==2.2.2 --no-build-isolation

echo "Installing flash-attn..."
pip install flash-attn --no-build-isolation

echo "--------------------------------------"
echo "Final verification"
echo "--------------------------------------"

python - <<EOF
import torch, mamba_ssm
from flash_attn import flash_attn_func
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("Mamba-SSM: OK")
print("Flash-Attn: OK")
EOF

echo "======================================"
echo "GPU setup COMPLETE. Environment READY."
echo "======================================"

