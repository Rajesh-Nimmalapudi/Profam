import os
import glob
import numpy as np
from tqdm import tqdm
from tokenizers import Tokenizer
import argparse
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

def process_file_chunk(file_path, tokenizer_path):
    """
    Reads a .sequences file and tokenizes it.
    Returns: (list_of_token_arrays, list_of_family_ids)
    """
    try:
        from src.data.tokenizers import ProFamTokenizer
        # Re-initialize tokenizer in worker (handles parallelism better)
        tokenizer = ProFamTokenizer(tokenizer_file=tokenizer_path, add_document_token=False, add_bos_token=False)
        
        # Extract family ID (e.g. "PF12345" from path or filename)
        # Assuming path structure like: .../PF12345.sequences OR .../PF12345/something.sequences
        # For now, we will hash the filename to get a simplified ID or use a placeholder if not strictly required for pretraining
        # Ideally, we map names to IDs. For this script, we'll focus on sequences.
        
        token_arrays = []
        
        with open(file_path, 'r') as f:
            sequences = [line.strip() for line in f if line.strip()]
            
        if not sequences:
            return None
            
        # Bulk encoding
        encodings = tokenizer.encode_batch(sequences)
        
        for enc in encodings:
            # Drop special tokens if needed, but ProFamTokenizer usually handles this.
            # Using uint8 to save space (Vocab size < 256)
            token_arrays.append(np.array(enc.ids, dtype=np.uint8))
            
        return token_arrays
    except Exception as e:
        print(f"Error in {file_path}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Destructive Binary Preprocessor for ProFam")
    parser.add_argument("--raw_dir", type=str, required=True, help="Path to raw .sequences files")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for .bin files")
    parser.add_argument("--tokenizer_file", type=str, required=True, help="Path to tokenizer.json")
    parser.add_argument("--num_workers", type=int, default=8)
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    tokens_file = os.path.join(args.output_dir, "tokens.bin")
    offsets_file = os.path.join(args.output_dir, "offsets.bin")
    
    # Check for existing processing
    if os.path.exists(tokens_file):
        print("WARNING: Output files exist. Re-running will APPEND or OVERWRITE. Please clean directory first.")
        # In a real run, we might want to strict fail to avoid corruption.
    
    # Find all files
    search_pattern = os.path.join(args.raw_dir, "**", "*.sequences")
    files = glob.glob(search_pattern, recursive=True)
    print(f"Found {len(files)} files to process.")
    
    # Initialize pointers
    current_offset = 0
    
    # Open binary files for APPENDING/WRITING
    # We use 'ab' (append binary) or 'wb' (write binary). 'wb' clears it.
    # Let's use 'wb' to start fresh.
    with open(tokens_file, "wb") as f_tokens, open(offsets_file, "wb") as f_offsets:
        
        # Write initial offset (0)
        # Using int64 (8 bytes) for offsets because 137GB > 4GB
        f_offsets.write(np.int64(0).tobytes())
        
        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            # Submit all tasks
            futures = [executor.submit(process_file_chunk, f, args.tokenizer_file) for f in files]
            
            for future in tqdm(futures, total=len(files), desc="Tokenizing to Bin"):
                result = future.result()
                if result:
                    token_arrays = result
                    
                    for tokens in token_arrays:
                        # 1. Write tokens bytes
                        f_tokens.write(tokens.tobytes())
                        
                        # 2. Update and write offset
                        current_offset += len(tokens)
                        f_offsets.write(np.int64(current_offset).tobytes())
                        
    print("----------------------------------------------------------------")
    print(f"Done! Created:")
    print(f"  - {tokens_file}")
    print(f"  - {offsets_file}")
    print("You can now verify the size with 'du -sh'.")
    print("----------------------------------------------------------------")

if __name__ == "__main__":
    main()
