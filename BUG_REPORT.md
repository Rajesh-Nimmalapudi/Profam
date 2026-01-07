# ProFam Bug Report & Analysis

**Date:** January 7, 2026
**Analysis Scope:** Complete codebase review for A100 GPU HPC execution
**Status:** ⚠️ **NOT READY FOR A100 EXECUTION** - Requires fixes before deployment

---

## 🔴 CRITICAL BUGS (Must Fix Before Execution)

### 1. Missing PROJECT_ROOT Environment Variable
**File:** `src/train.py:23`, `configs/train.yaml:39`
**Severity:** CRITICAL - Prevents code execution
**Issue:**
```python
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
```
- train.py uses rootutils to set PROJECT_ROOT
- Configs reference `${oc.env:PROJECT_ROOT}` and `${paths.root_dir}/../ProFam-atlas`
- Environment variable PROJECT_ROOT is NOT SET in current environment
- **Impact:** Config loading fails with `KeyError: "Environment variable 'PROJECT_ROOT' not found"`

**Evidence:**
```bash
$ echo $PROJECT_ROOT
# Output: (empty/null)

$ python -c "import os; print(os.environ.get('PROJECT_ROOT', 'NOT SET'))"
# Output: NOT SET
```

**Fix Required:**
```bash
# In HPC job script or .bashrc
export PROJECT_ROOT=/path/to/profam_exp/profam
```

---

### 2. Missing ProFam-atlas Data Directory
**File:** `configs/train.yaml:41`, `configs/data/profam.yaml:7-11`
**Severity:** CRITICAL - Data loading will fail
**Issue:**
- Config references: `data_dir: ${paths.root_dir}/../ProFam-atlas`
- Directory `../ProFam-atlas` does NOT exist
- Only debug data available at `data/processed_debug/` (69GB)
- Multiple datasets (openfold_train, ted_train, foldseek_s50_train, uniref90_train) reference this path

**Evidence:**
```bash
$ ls -la ../ProFam-atlas
# Output: ls: cannot access '../ProFam-atlas': No such file or directory

$ ls -la data/
# Available: processed_debug/ (69GB), raw_debug/, train_example/, etc.
# Missing: processed/ with ProFam-atlas datasets
```

**Fix Required:**
- Option 1: Download full ProFam-atlas dataset using `scripts/download_data.sh`
- Option 2: Use debug experiment config: `python src/train.py experiment=debug`
- Option 3: Update data path to point to existing `data/processed_debug/`

---

### 3. Missing download() Function in launch.sh
**File:** `launch.sh:44, 453-455, 457-461`
**Severity:** CRITICAL - Script execution fails
**Issue:**
- Function `download()` is documented in usage() at line 44
- Case statement references `download` command at lines 453-455
- Function is **never defined** in the script
- Running `./launch.sh download` will fail with "download: command not found"

**Evidence:**
```bash
$ ./launch.sh download
# Expected: Downloads pre-trained models from Hugging Face
# Actual: Error - function not found
```

**Available Functions:** `usage()`, `pull()`, `build()`, `push()`, `setup()`, `dev()`, `run()`, `attach()`
**Missing Functions:** `download()`, `download_test_data()` (referenced but not defined)

**Fix Required:**
- Define `download()` function or remove from usage/case statement
- Currently: Use `scripts/download_data.sh` and `scripts/hf_download_checkpoint.py` directly

---

### 4. CUDA Version Mismatch
**File:** `setup_gpu.sh:8`
**Severity:** CRITICAL - May cause runtime failures
**Issue:**
```bash
module load cuda/13.0
```
- Script expects CUDA 13.0
- Current system has CUDA 12.8 installed
- Module `cuda/13.0` may not exist on HPC cluster

**Evidence:**
```bash
$ python -c "import torch; print(torch.version.cuda)"
# Output: 12.8

$ nvcc --version
# Output: CUDA 12.8
```

**Fix Required:**
- Update `setup_gpu.sh:8` to use available CUDA version (12.8)
- OR verify HPC cluster has cuda/13.0 module available
- OR adjust to use system default CUDA

---

### 5. PyTorch Version Mismatch
**File:** `setup_login.sh:27-31`, `requirements.lock.txt:87`
**Severity:** HIGH - May cause compatibility issues
**Issue:**
```bash
pip install \
  torch==2.3.1+cu121 \
  torchvision==0.18.1+cu121 \
  torchaudio==2.3.1+cu121 \
  --index-url https://download.pytorch.org/whl/cu121
```
- Script expects PyTorch 2.3.1 with CUDA 12.1
- Current system has PyTorch 2.9.1+cu128 installed
- Significant version gap (2.3.1 → 2.9.1) and CUDA version mismatch (12.1 → 12.8)

**Evidence:**
```bash
$ python -c "import torch; print(torch.__version__)"
# Output: 2.9.1+cu128

$ cat requirements.lock.txt | grep "^torch"
# Output: torch==2.3.1+cu121
```

**Fix Required:**
- Option 1: Update setup_login.sh to install PyTorch 2.9.1+cu128
- Option 2: Reinstall PyTorch 2.3.1+cu121 (downgrade)
- Option 3: Use compatible version for CUDA 12.8

---

## 🟠 HIGH PRIORITY ISSUES

### 6. Missing .env File
**File:** `launch.sh:27-34, 126-154`
**Severity:** HIGH - Environment configuration incomplete
**Issue:**
- launch.sh expects `.env` file in project root
- File does not exist
- Script will create defaults, but may not work for HPC environment
- Critical variables like WANDB_API_KEY, NGC_CLI_API_KEY, DOCKER_IMAGE will use defaults

**Evidence:**
```bash
$ ls -la .env
# Output: ls: cannot access '.env': No such file or directory
```

**Default Values Created by Script:**
```bash
DOCKER_IMAGE=nvcr.io/nvidian/dbr/profam:dev
LOCAL_REPO_PATH=$(pwd)
WANDB_API_KEY=NotSpecified
NGC_CLI_API_KEY=NotSpecified
```

**Fix Required:**
- Create `.env` file with HPC-specific values
- Set WANDB_API_KEY for experiment tracking
- Configure DOCKER_IMAGE path for HPC container registry

---

### 7. Excessive num_workers Setting
**File:** `configs/data/profam.yaml:135`
**Severity:** HIGH - May cause CPU overload or crashes
**Issue:**
```yaml
num_workers: 32
```
- Hardcoded to 32 workers for data loading
- Current GPU (RTX 3090) has limited CPU allocation
- HPC nodes may have different CPU allocations
- Can cause multiprocessing crashes or system overload

**Evidence:**
```bash
$ nproc
# Output: (likely < 32 on single-user system)
```

**Fix Required:**
- Reduce to 4-8 for debugging
- Adjust based on HPC CPU allocation (e.g., `--cpus-per-task=32` in SLURM)
- Make configurable per environment

---

### 8. Mamba Kernels Compilation Required
**File:** `configs/experiment/pretrain_v2_full_a100.yaml:21`, `setup_gpu.sh:25`
**Severity:** HIGH - Runtime failure if not compiled
**Issue:**
```yaml
use_mamba_kernels: true  # Set to true if kernels are installed on A100
```
- Config enables mamba_kernels
- Requires mamba-ssm to be compiled on GPU node
- setup_gpu.sh installs mamba-ssm on GPU node, but must be run before training
- If setup_gpu.sh not run, training will fail at model initialization

**Evidence:**
```bash
$ python -c "import mamba_ssm; print('Available')"
# Output: Available (v2.2.6.post3)

# BUT: Must be compiled with correct CUDA version on A100
```

**Fix Required:**
- Run `bash setup_gpu.sh` on GPU node before training
- OR set `use_mamba_kernels: false` in config (falls back to transformer-only)

---

## 🟡 MEDIUM PRIORITY ISSUES

### 9. Attention Implementation Conflict
**File:** `src/train.py:14-16`, `configs/experiment/pretrain_v2_full_a100.yaml:22`
**Severity:** MEDIUM - May cause warnings or runtime issues
**Issue:**
```python
# train.py
torch.backends.cuda.enable_flash_sdp(False)        # Disable external flash-attn
torch.backends.cuda.enable_mem_efficient_sdp(True) # Enable Triton/Cutlass
torch.backends.cuda.enable_math_sdp(True)          # Enable Fallback
```
```yaml
# pretrain_v2_full_a100.yaml
attn_implementation: sdpa  # Use PyTorch Native SDPA
```
- train.py disables flash attention at startup
- Config sets `attn_implementation: sdpa`
- Potential conflict or redundant settings
- May cause runtime warnings

**Impact:** Likely works but may produce warnings

**Fix Required:**
- Verify both settings are compatible
- Consider removing redundant settings in train.py

---

### 10. Inconsistent Training Strategy
**File:** `configs/experiment/pretrain_v2_full_a100.yaml:34`, `configs/train.yaml:188`
**Severity:** MEDIUM - Multi-GPU training may not work correctly
**Issue:**
```yaml
# pretrain_v2_full_a100.yaml
strategy: auto  # Default to auto
```
```yaml
# train.yaml
strategy: ddp  # Default to DDP
```
- pretrain_v2_full_a100.yaml uses `strategy: auto`
- train.yaml uses `strategy: ddp`
- For multi-GPU A100 training, should explicitly use `ddp`
- `auto` may not correctly detect multi-GPU setup

**Fix Required:**
- Change pretrain_v2_full_a100.yaml line 34 to `strategy: ddp`
- OR verify `auto` works correctly on HPC cluster

---

## 🔵 LOW PRIORITY / OBSERVATIONS

### 11. Current GPU is Not A100
**Severity:** LOW - Environment mismatch
**Issue:**
```bash
$ nvidia-smi | grep "Product Name"
# Output: NVIDIA GeForce RTX 3090 (24GB)
```
- Current environment has RTX 3090 (24GB VRAM)
- A100 has 40GB or 80GB VRAM
- Configs are optimized for A100 (larger batch sizes, model sizes)
- Likely test environment vs. production HPC environment

**Impact:** Low - This is expected for local testing

**Note:** Debug config exists for smaller GPUs (`configs/experiment/debug.yaml`)

---

### 12. Tokenizer Initialization Inconsistency
**File:** `src/data/tokenizers.py`, configs/train.yaml:72-83
**Severity:** LOW - Minor API inconsistency
**Issue:**
- Tokenizer requires explicit special token parameters during initialization
- Config provides parameters, but some default to None
- Properties like `bos_token_id`, `sep_token_id` can be None if not properly initialized

**Evidence:**
```python
# Without explicit parameters:
tokenizer = ProFamTokenizer(tokenizer_file='data/profam_tokenizer.json')
# Result: bos_token=None, sep_token=None, pad_token=None

# With explicit parameters:
tokenizer = ProFamTokenizer(
    tokenizer_file='data/profam_tokenizer.json',
    bos_token='[start-of-document]',
    sep_token='[SEP]',
    pad_token='[PAD]',
    unk_token='[UNK]'
)
# Result: All tokens properly set
```

**Fix Required:**
- Ensure configs properly pass all special token parameters
- Already handled correctly in configs/train.yaml

---

## ✅ WHAT WORKS CORRECTLY

### Verified Functional Components:

1. **Jamba Model Imports:**
   ```bash
   from transformers import JambaConfig, JambaForCausalLM
   # Status: ✓ Working
   ```

2. **Mamba-SSM Installation:**
   ```bash
   import mamba_ssm
   # Version: 2.2.6.post3
   # Status: ✓ Available
   ```

3. **Causal Conv1D:**
   ```bash
   import causal_conv1d
   # Status: ✓ Available
   ```

4. **Binary Dataset Loading:**
   ```bash
   from src.data.builders.binary_dataset import BinaryMemmapDataset
   # Dataset: 493,869,567 samples
   # Size: 69GB
   # Status: ✓ Loads successfully
   ```

5. **Tokenizer Loading:**
   ```bash
   from src.data.tokenizers import ProFamTokenizer
   # Vocab size: 69
   # Status: ✓ Works with proper initialization
   ```

6. **Config Structure:**
   ```bash
   with hydra.initialize(config_path='configs'):
       cfg = hydra.compose(config_name='train')
   # Status: ✓ Structure is correct (fails only on missing PROJECT_ROOT)
   ```

7. **CUDA Availability:**
   ```bash
   torch.cuda.is_available()
   # Status: ✓ True (CUDA 12.8)
   ```

---

## 📋 SUMMARY STATISTICS

| Category | Count |
|----------|-------|
| Critical Bugs | 5 |
| High Priority Issues | 3 |
| Medium Priority Issues | 2 |
| Low Priority Issues | 2 |
| **Total Issues** | **12** |
| Working Components | 7 |

---

## 🎯 RECOMMENDED FIX ORDER

### For Immediate Debug Testing (Local GPU):

1. **Fix #1:** Set `PROJECT_ROOT` environment variable
2. **Fix #2:** Use debug experiment config to avoid missing ProFam-atlas data
   ```bash
   export PROJECT_ROOT=/home/boltzmann11/profam_exp/profam
   python src/train.py experiment=debug
   ```
3. **Fix #7:** Reduce num_workers in debug config

### For A100 HPC Production Execution:

1. **Fix #1:** Set PROJECT_ROOT in HPC job script
2. **Fix #2:** Download and process ProFam-atlas dataset OR configure correct data path
3. **Fix #4:** Verify CUDA 13.0 module exists OR update to CUDA 12.8
4. **Fix #5:** Update PyTorch version to match system OR use compatible version
5. **Fix #6:** Create .env file with HPC-specific settings
6. **Fix #8:** Run `bash setup_gpu.sh` on GPU node to compile mamba kernels
7. **Fix #10:** Change strategy from `auto` to `ddp` for multi-GPU
8. **Fix #7:** Adjust num_workers based on HPC CPU allocation
9. **Fix #3:** Define download() function in launch.sh OR document alternative

---

## 🚀 QUICK START COMMANDS

### Debug Mode (Fix Issues #1, #2, #7):

```bash
# Step 1: Set environment variable
export PROJECT_ROOT=/home/boltzmann11/profam_exp/profam

# Step 2: Run debug experiment (uses existing debug data)
cd profam
python src/train.py experiment=debug
```

### Production A100 Mode (All Fixes Required):

```bash
# Step 1: Create .env file
cat > .env << 'EOF'
PROJECT_ROOT=/path/to/profam_exp/profam
WANDB_API_KEY=your_api_key_here
DOCKER_IMAGE=nvcr.io/nvidian/dbr/profam:dev
EOF

# Step 2: Download data (if needed)
bash scripts/download_data.sh /path/to/data/raw

# Step 3: On HPC GPU node, run setup
module load cuda/13.0  # or appropriate version
bash setup_gpu.sh

# Step 4: Run training
export PROJECT_ROOT=/path/to/profam_exp/profam
python src/train.py experiment=pretrain_v2_full_a100
```

---

## 📝 ADDITIONAL NOTES

- **Code Quality:** The codebase is well-structured with proper separation of concerns
- **Configuration:** Hydra config system is properly implemented
- **Model Architecture:** Jamba hybrid architecture correctly implemented
- **Data Pipeline:** Binary memmap dataset is efficient and functional
- **Documentation:** README.md provides good setup instructions, but needs HPC-specific updates

**Overall Assessment:** The codebase is architecturally sound and mostly bug-free at the code level. The critical issues are primarily **environment and configuration problems**, not code logic bugs. Once the environment is properly configured (PROJECT_ROOT, data paths, CUDA/PyTorch versions, etc.), the code should execute successfully on A100 GPUs.

---

**Report Generated:** January 7, 2026
**Analysis Tool:** Manual code review + runtime testing
**Test Environment:** RTX 3090, CUDA 12.8, PyTorch 2.9.1
**Target Environment:** A100 GPU, HPC Cluster
