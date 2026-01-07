import torch
import numpy as np
import os
from torch.utils.data import Dataset

class BinaryMemmapDataset(Dataset):
    """
    A Dataset backing a raw binary file of tokens (uint8) and an offsets file (int64).
    This implementation uses numpy.memmap for zero-copy access to massive datasets.
    """
    def __init__(self, data_dir: str, name: str = "binary_dataset", bos_token_id: int = 0):
        super().__init__()
        self.data_dir = data_dir
        self.name = name
        self.bos_token_id = bos_token_id
        
        tokens_path = os.path.join(data_dir, "tokens.bin")
        offsets_path = os.path.join(data_dir, "offsets.bin")
        
        if not os.path.exists(tokens_path) or not os.path.exists(offsets_path):
            raise FileNotFoundError(f"Missing binary files in {data_dir}. Expected tokens.bin and offsets.bin")
            
        # Load offsets into RAM (It's just integers, usually safe. 100M sequences * 8 bytes = 800MB RAM)
        # If offsets are too huge, we can memmap them too, but RAM is faster for lookups.
        try:
            self.offsets = np.fromfile(offsets_path, dtype=np.int64)
            self.num_samples = len(self.offsets) - 1
        except Exception as e:
            raise RuntimeError(f"Failed to load offsets: {e}")
            
        # Memmap the tokens file (The Big Data)
        # mode='r' means read-only, changes won't be saved to disk
        self.tokens_mmap = np.memmap(tokens_path, dtype=np.uint8, mode='r')
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        if idx < 0 or idx >= self.num_samples:
            raise IndexError(f"Index {idx} out of range (0..{self.num_samples-1})")
            
        start_ptr = self.offsets[idx]
        end_ptr = self.offsets[idx + 1]
        
        # Slicing a memmap returns a new memmap object (view), which is fast.
        # We convert to np.array -> torch.tensor
        # .copy() ensures we have a real array not a view of the disk file
        token_ids = np.array(self.tokens_mmap[start_ptr:end_ptr], dtype=np.int64) # Convert to long for PyTorch embedding
        
        # Ensure BOS token is present for packing
        if len(token_ids) == 0 or token_ids[0] != self.bos_token_id:
            token_ids = np.concatenate(([self.bos_token_id], token_ids))
        
        return {
            "input_ids": torch.tensor(token_ids),
            "ds_name": self.name,
            "identifier": f"{self.name}_{idx}"
        }

# Example usage for debugging/testing
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        ds = BinaryMemmapDataset(sys.argv[1])
        print(f"Loaded dataset with {len(ds)} samples.")
        print(f"Sample 0: {ds[0]}")
