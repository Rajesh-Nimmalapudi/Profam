#!/bin/bash
set -e

# Config
SRC_RAW="../raw"
DEST_DEBUG="data/raw_debug"
NUM_FILES=5000

echo "======================================"
echo "[DEBUG DATA PREP] Extracting $NUM_FILES files"
echo "Src: $SRC_RAW -> Dest: $DEST_DEBUG"
echo "======================================"

# 1. Cleanup old debug data
if [ -d "$DEST_DEBUG" ]; then
    echo "Cleaning up old $DEST_DEBUG..."
    rm -rf "$DEST_DEBUG"
fi
mkdir -p "$DEST_DEBUG"

# 2. Use Python for Robust File Selection (No Broken Pipes)
python3 - <<EOF
import os
import random
import glob
import shutil
from pathlib import Path

src_dir = Path("$SRC_RAW")
dest_dir = Path("$DEST_DEBUG")
num_files = $NUM_FILES

print("Scanning for .sequences files (this may take a moment)...")
# Iterate RECURSIVELY to find all .sequences
all_files = list(src_dir.rglob("*.sequences"))

print(f"Found {len(all_files)} files.")

if len(all_files) == 0:
    print("WARNING: No .sequences files found! Check your ../raw path.")
    exit(1)

# Sample Randomly
selected_files = random.sample(all_files, min(len(all_files), num_files))
print(f"Selected {len(selected_files)} files. Copying...")

for src_file in selected_files:
    # Compute relative path to keep structure (e.g. uniref90/train/file.sequences)
    rel_path = src_file.relative_to(src_dir)
    target_path = dest_dir / rel_path
    
    # Create parent dirs
    target_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Copy
    shutil.copy2(src_file, target_path)

print("Copy complete.")
EOF

echo "--------------------------------------"
echo "Done! Debug data ready in $DEST_DEBUG"
echo "--------------------------------------"
