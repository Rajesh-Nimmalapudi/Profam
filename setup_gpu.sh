#!/bin/bash
# ============================================
# GPU Node Setup Script (CUDA extensions)
# ============================================

echo "============================================"
echo " [GPU NODE] Installing CUDA extensions"
echo "============================================"

# 1. Load Miniconda
module load miniconda/3

# 2. Activate environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate profam-env

# 3. Sanity check: GPU visibility
echo "Checking GPU availability..."
python - <<EOF
import torch
print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
EOF

# 4. Install CUDA-native libraries (PRE-BUILT WHEELS)
echo "Installing causal-conv1d..."
pip install causal-conv1d==1.4.0

echo "Installing mamba-ssm..."
pip install mamba-ssm==2.2.2

# 5. Final verification
echo "Verifying Mamba installation..."
python - <<EOF
import torch
import mamba_ssm
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("Mamba-SSM import: SUCCESS ✅")
EOF

echo "------------------------------------------------"
echo "GPU setup complete."
echo "Environment profam-env is READY for training."
echo "------------------------------------------------"

conda deactivate
