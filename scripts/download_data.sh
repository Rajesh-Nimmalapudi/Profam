#!/bin/bash
# Script to download raw datasets for ProFam
# Usage: ./download_data.sh [destination_folder]

DEST_DIR="${1:-./data/raw}"
mkdir -p "$DEST_DIR"

echo "Downloading datasets to $DEST_DIR..."

wget -P "$DEST_DIR" https://zenodo.org/record/17713590/files/ec.tar.gz
wget -P "$DEST_DIR" https://zenodo.org/record/17713590/files/foldseek_s50.tar.gz
wget -P "$DEST_DIR" https://zenodo.org/record/17713590/files/funfams_s50.tar.gz
wget -P "$DEST_DIR" https://zenodo.org/record/17713590/files/OpenFold_OpenProteinSet.tar.gz
wget -P "$DEST_DIR" https://zenodo.org/record/17713590/files/ted.tar.gz
wget -P "$DEST_DIR" https://zenodo.org/record/17713590/files/uniref90.tar.gz

echo "Download complete."
echo "CRITICAL: You MUST extract these repositories before training."
echo "Recommended command: for f in $DEST_DIR/*.tar.gz; do tar -xzf "\$f" -C $DEST_DIR; done"
