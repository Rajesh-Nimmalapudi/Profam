#!/bin/bash
# ============================================================
# Safe & Resumable Downloader for ProFam Raw Datasets
# Author: You
# Usage: ./download_data_safe.sh [destination_folder]
# ============================================================

set -e  # stop on unexpected errors

DEST_DIR="${1:-./data/raw}"
ZENODO_BASE="https://zenodo.org/record/17713590/files"

FILES=(
  ec.tar.gz
  foldseek_s50.tar.gz
  funfams_s50.tar.gz
  OpenFold_OpenProteinSet.tar.gz
  ted.tar.gz
  uniref90.tar.gz
)

echo "============================================================"
echo " ProFam dataset downloader (SAFE + RESUMABLE)"
echo " Destination: $DEST_DIR"
echo "============================================================"
echo ""

# ------------------------------------------------------------
# Step 1: Fresh start (intentional full cleanup)
# ------------------------------------------------------------
echo "[1/4] Cleaning destination directory..."
rm -rf "$DEST_DIR"
mkdir -p "$DEST_DIR"

# ------------------------------------------------------------
# Step 2: Show available disk space
# ------------------------------------------------------------
echo "[2/4] Checking available disk space..."
df -h "$DEST_DIR"
echo ""

# ------------------------------------------------------------
# Step 3: Download (resumable)
# ------------------------------------------------------------
echo "[3/4] Downloading datasets (resumable)..."
cd "$DEST_DIR"

for f in "${FILES[@]}"; do
    echo "--------------------------------------------"
    echo "Downloading: $f"
    wget -c "${ZENODO_BASE}/${f}"
done

echo ""
echo "All downloads completed (or resumed)."

# ------------------------------------------------------------
# Step 4: Integrity check
# ------------------------------------------------------------
echo ""
echo "[4/4] Verifying archives..."
FAILED=0
for f in *.tar.gz; do
    echo "Testing $f"
    if ! tar -tzf "$f" > /dev/null 2>&1; then
        echo "❌ Corrupted: $f"
        FAILED=1
    else
        echo "✅ OK: $f"
    fi
done

echo ""
if [ "$FAILED" -eq 1 ]; then
    echo "❌ Some archives are corrupted."
    echo "→ Delete only the corrupted files and re-run the script."
    exit 1
else
    echo "✅ All archives verified successfully."
fi

echo ""
echo "============================================================"
echo " NEXT STEP (run manually when ready):"
echo " for f in *.tar.gz; do tar -xzf \"\$f\"; done"
echo "============================================================"
