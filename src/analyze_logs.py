
import pandas as pd
import argparse
import sys

def summarize_logs(csv_path):
    print(f"Reading logs from: {csv_path}")
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    # Filter for relevant columns
    cols = ['epoch', 'step', 'train/loss', 'train/aa_accuracy']
    # Check if columns exist
    existing_cols = [c for c in cols if c in df.columns]
    if not existing_cols:
        print("No relevant columns found in CSV.")
        print("Available columns:", df.columns.tolist())
        return

    subset = df[existing_cols].copy()
    
    # Drop rows where 'train/loss' is NaN
    if 'train/loss' in subset.columns:
        subset = subset.dropna(subset=['train/loss'])

    if subset.empty:
        print("No rows with valid training loss found.")
        return

    # Group by epoch and calculate means
    summary = subset.groupby('epoch').agg({
        'train/loss': 'mean',
        'train/aa_accuracy': 'mean',
        'step': 'max' # Last step of the epoch
    }).reset_index()

    # Rename for clarity
    summary = summary.rename(columns={
        'train/loss': 'Avg Loss',
        'train/aa_accuracy': 'Avg Accuracy',
        'step': 'End Step'
    })

    print("\n" + "="*40)
    print("       TRAINING SUMMARY (PER EPOCH)")
    print("="*40)
    print(summary.to_string(index=False))
    print("="*40 + "\n")

    # Save summary
    output_path = csv_path.replace('metrics.csv', 'summary_metrics.csv')
    summary.to_csv(output_path, index=False)
    print(f"Summary saved to: {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        # Default path if not provided (convenience for the specific run)
        print("Usage: python src/analyze_logs.py <path_to_metrics.csv>")
        print("Defaulting to local search...")
        # Try to find a metrics.csv in logs/
        import glob
        files = glob.glob("logs/**/metrics.csv", recursive=True)
        if files:
            print(f"Found {len(files)} metrics files. Using the most recent: {files[0]}")
            summarize_logs(files[0])
        else:
            print("No metrics.csv found.")
    else:
        summarize_logs(sys.argv[1])
