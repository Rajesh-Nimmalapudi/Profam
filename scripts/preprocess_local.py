
import argparse
import glob
import os
import shutil
import warnings
from concurrent.futures import ProcessPoolExecutor
from typing import List, Tuple

import datasets
import numpy as np
from datasets import Dataset, Features, Sequence, Value
from tqdm import tqdm

# Add src to path
import sys

sys.path.append(os.getcwd())

from src.data.tokenizers import ProFamTokenizer
from src.data.objects import ProteinDocument


def process_sequence_file(
    file_path: str, output_dir: str, tokenizer_path: str = None
) -> str:
    """
    Process a single .sequences file:
    1. Read lines (accession, sequence)
    2. Tokenize sequences
    3. Save as HuggingFace Dataset (Arrow format)
    """
    # Process to avoid pickling tokenizer
    try:
        if tokenizer_path:
             tokenizer = ProFamTokenizer(tokenizer_file=tokenizer_path, add_document_token=False, add_bos_token=False, sep_token="[SEP]", unk_token="[UNK]", pad_token="[PAD]", mask_token="?", cls_token="[CLS]")
        else:
             tokenizer = ProFamTokenizer(add_document_token=False, add_bos_token=False, sep_token="[SEP]", unk_token="[UNK]", pad_token="[PAD]", mask_token="?", cls_token="[CLS]")


        accessions = []
        input_ids = []
        
        # Check if output already exists (Resumable Logic)
        filename = os.path.basename(file_path).replace(".sequences", ".arrow")
        output_path = os.path.join(output_dir, filename)
        
        if os.path.exists(output_path):
             return file_path
             
        # Read file
        with open(file_path, "r") as f:
            lines = f.readlines()
        
        if len(lines) % 2 != 0:
            print(f"Warning: File {file_path} has odd number of lines ({len(lines)}). Skipping.")
            return None

        # Parse lines
        for i in range(0, len(lines), 2):
            acc_line = lines[i].strip()
            seq_line = lines[i+1].strip()
            
            # Basic validation
            if not acc_line.startswith(">"):
                print(f"Warning: Line {i} in {file_path} does not start with '>'. Skipping pair.")
                continue
                
            accession = acc_line[1:] # Remove '>'
            sequence = seq_line
            
            # Tokenize
            # We treat each sequence as a single protein document for basic tokenization
            # We do NOT add [RAW] or [FAM] tokens here essentially if we want to keep them raw-ish
            # BUT the tokenizer adds special tokens by default.
            # Let's check ProFamTokenizer defaults: add_bos_token=True, add_document_token=True
            # If we want to mimic the Runtime behavior, runtime does:
            #   preprocessor -> apply_transforms -> tokenizer.encode
            # Transforms might map amino acids.
            # Since this is "Raw to Token ID" conversion, we should probably stick to just mapping AA to ID.
            # If we bake in [BOS] and [EOS] now, we save time later.
            
            # Use tokenizer to encode.
            # Note: We want individual sequence tokens, not packed documents yet.
            prot_doc = ProteinDocument(sequences=[sequence], accessions=[accession], identifier="dummy")
            
            # We disable document tokens for individual sequences to keep them clean
            # We will handle document structure at runtime collation
            tokenized = tokenizer.encode(
                prot_doc,
                document_token=None,  # Do not add [RAW] yet
                add_final_sep=False, # Do not add [SEP] yet, or maybe yes?
                # If we add stats here, we can't easily concatenate them later without stripping.
                # Ideally, we just want the AA ids.
            )
            
            # The tokenizer encode adds BOS/EOS/FAM if configured.
            # Let's manually convert to IDs to be safe and raw?
            # Or use tokenizer but configure it to be minimal.
            # For now, let's use the tokenizer's convert_tokens_to_ids which handles vocab.
            
            # Actually, `tokenizer.encode` logic is:
            # concatenated_seqs = ... joined ...
            # tokenized = self(concatenated_seqs)
            
            # If we want just the AA ids:
            ids = tokenizer.convert_tokens_to_ids(list(sequence))
            # But we must handle unknown tokens if allow_unk=False?
            # ProFamTokenizer handles UNK check.
            
            # Let's use the full tokenizer but Strip the special tokens if they are added?
            # Or better: Configure tokenizer to NOT add special tokens for this pass.
            # ProFamTokenizer init has flags. But we used default init above.
            
            ids = tokenizer.encode(
                prot_doc,
                document_token=None,
                add_final_sep=False,
            ).input_ids
            
            # Remove BOS if added (default tokenizer adds it?)
            # tokenizer.add_bos_token defaults to True.
            # Check if first token is BOS
            if tokenizer.bos_token_id is not None and len(ids) > 0 and ids[0] == tokenizer.bos_token_id:
                ids = ids[1:]
                
            input_ids.append(ids)
            accessions.append(accession)

        # Create Dataset with explicit features to save space
        # Use uint8 for tokens (save 4x space vs int32) - max vocab is small so this is safe
        features = Features({
            "input_ids": Sequence(Value("uint8")),
            # "accession": Value("string") # Commented out to save space if not needed for training
        })
        
        # Accessions are useful for debugging but take space. 
        # If strict size limit, drop them. Let's keep them but use uint8 for IDs.
        
        ds = Dataset.from_dict(
            {"accession": accessions, "input_ids": input_ids},
            features=Features({"accession": Value("string"), "input_ids": Sequence(Value("uint8"))})
        )
        
        # Save to output dir
        filename = os.path.basename(file_path).replace(".sequences", ".arrow")
        output_path = os.path.join(output_dir, filename)
        
        # Save arrow
        ds.save_to_disk(output_path)
        
        return file_path
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Preprocess ProFam raw data to binary Arrow format.")
    parser.add_argument("--raw_dir", type=str, required=True, help="Root directory containing raw .sequences and .mapping files")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for processed data")
    parser.add_argument("--tokenizer_file", type=str, default=None, help="Path to profam_tokenizer.json")
    parser.add_argument("--num_workers", type=int, default=os.cpu_count(), help="Number of worker processes")
    
    args = parser.parse_args()
    
    # If not provided, try to find it in default location relative to script
    if args.tokenizer_file is None:
        default_tok = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "profam_tokenizer.json")
        if os.path.exists(default_tok):
            print(f"Using default tokenizer: {default_tok}")
            args.tokenizer_file = default_tok
        else:
            print("Warning: No tokenizer file provided and default not found. Using defaults.")
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        
    # Find all .sequences files recursively
    # The structure is usually [Dataset]/[Split]/*.sequences
    # We want to preserve the subdirectory structure in the output
    
    search_pattern = os.path.join(args.raw_dir, "**", "*.sequences")
    sequence_files = glob.glob(search_pattern, recursive=True)
    
    print(f"Found {len(sequence_files)} .sequences files in {args.raw_dir}")
    
    # Also find .mapping files
    mapping_pattern = os.path.join(args.raw_dir, "**", "*.mapping")
    mapping_files = glob.glob(mapping_pattern, recursive=True)
    print(f"Found {len(mapping_files)} .mapping files.")
    
    # Process sequences
    # We need to preserve relative paths
    tasks = []
    
    for seq_file in sequence_files:
        rel_path = os.path.relpath(seq_file, args.raw_dir)
        parent_dir = os.path.dirname(rel_path)
        out_subdir = os.path.join(args.output_dir, parent_dir)
        
        if not os.path.exists(out_subdir):
            os.makedirs(out_subdir, exist_ok=True)
            
        tasks.append((seq_file, out_subdir))
        
    print(f"Starting processing with {args.num_workers} workers...")
    
    # Use ProcessPoolExecutor
    # Note: Tokenizer might fork-bomb if not careful, but usually fine.
    # We pass None for tokenizer_path to let function init it.
    
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = [executor.submit(process_sequence_file, t[0], t[1], args.tokenizer_file) for t in tasks]
        
        for future in tqdm(futures, total=len(tasks), desc="Processing files"):
            res = future.result()
            
    print("Processing complete. Copying mapping files...")
    
    for map_file in tqdm(mapping_files, desc="Copying mappings"):
        rel_path = os.path.relpath(map_file, args.raw_dir)
        out_path = os.path.join(args.output_dir, rel_path)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        shutil.copy2(map_file, out_path)
        
    print("Done! Data is ready for transfer.")
    print(f"Output Directory: {args.output_dir}")
    print("Suggested Transfer Command:")
    print(f"tar -I zstd -cvf profam_processed.tar.zst -C {args.output_dir} .")

if __name__ == "__main__":
    main()
