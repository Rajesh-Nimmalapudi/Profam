import glob
import os
from functools import lru_cache
from typing import Any, List, Optional

import numpy as np
import datasets
from torch.utils.data import Dataset

from src.data.objects import ProteinDocument
from src.data.processors import ProteinDocumentPreprocessor
from src.data.tokenizers import ProFamTokenizer

from ..text_memmap_datasets import TextMemMapDataset


class MappingProteinFamilyMemmapDataset(TextMemMapDataset):
    """
    A *.mapping FASTA dataset, holding family id and mapping of sequences files and corresponding indices (per file), for each family.
    """

    def __init__(
        self,
        dataset_root: str,
        workers=None,
        sort_dataset_paths=True,
        index_mapping_dir=None,
    ):
        """
        Args:
            dataset_root: point to the root directory of the dataset (i.e., train, val, test)
            workers: number of workers to use for parallel data indexing (on first run)
            sort_dataset_paths: whether to sort dataset paths by name
            index_mapping_dir: directory to store index mapping cached files
        """
        dataset_paths = glob.glob(f"{dataset_root}/*.mapping")
        super().__init__(
            dataset_paths=dataset_paths,
            newline_int=ord(">"),
            header_lines=1,  # skip first line since it is an empty sequence
            workers=workers,
            sort_dataset_paths=sort_dataset_paths,
            index_mapping_dir=index_mapping_dir,
        )

        self._data_sep = "\n"

    def _build_data_from_text(self, text):
        """Allows child-classes to modify the parsing of raw text, prior to tokenization"""
        # tokenize sequences
        _build_data_from_text = super()._build_data_from_text
        # extract id and sequence and tokenize (if needed)
        text_fields = text.split(self._data_sep)

        fam_id = text_fields[0]
        sample_indices = {}
        for line in text_fields[1:]:
            line = line.strip()
            if not line:
                continue
            seq_fname, seq_ind = line.split(":")
            seq_ind = [int(i) for i in seq_ind.split(",")]
            sample_indices[seq_fname] = seq_ind

        data = {
            "fam_id": fam_id,
            "sample_indices": sample_indices,
        }

        return data


class SequencesProteinFamilyMemmapDataset(Dataset):
    """
    A *.sequences FASTA dataset, holding accession and sequence for all families.
    We treat each line in the *.sequences files independently even though every 2 lines create a sample with accession + sqeuence. We do so to be able to read sequence size efficiently.
    """

    def __init__(
        self,
        dataset_root: str,
        workers=None,
        sort_dataset_paths=True,
        index_mapping_dir=None,
        load_precomputed=False,
    ):
        """
        Args:
            dataset_root: point to the root directory of the dataset (i.e., train, val, test)
            workers: number of workers to use for parallel data indexing (on first run)
            sort_dataset_paths: whether to sort dataset paths by name
            index_mapping_dir: directory to store index mapping cached files
            load_precomputed: if True, load tokens from .arrow files instead of parsing .sequences
        """
        self.load_precomputed = load_precomputed
        if self.load_precomputed:
             # Look for .arrow files
             dataset_paths = glob.glob(f"{dataset_root}/*.arrow")
             if not dataset_paths:
                 # fallback or error? A user might have preprocessed into subdirs or similar.
                 # Using the same recursive search logic as preprocess_local might be better,
                 # but here we usually expect a flat struct or consistent struct.
                 # Let's assume standard structure: {dataset_root}/{files}.arrow
                 pass
             
             # We assume the user has replaced .sequences with .arrow at the same location/name
             # But TextMemMapDataset expects text files.
             # We will NOT use TextMemMapDataset if load_precomputed is True.
             # We just need to map filename -> arrow dataset.
             self.dataset_paths = sorted(dataset_paths) if sort_dataset_paths else dataset_paths
             self._file_map = {}
             self._arrow_datasets = {} # Cache for open datasets
             for path in self.dataset_paths:
                 fn = os.path.basename(path).replace(".arrow", ".sequences") # Map back to expected mapping name
                 self._file_map[fn] = path
                 
        else:
            dataset_paths = glob.glob(f"{dataset_root}/*.sequences")
            # We read the sequences files as text lines, so we can use TextMemMapDataset
            self.lines_ds = TextMemMapDataset(
                dataset_paths=dataset_paths,
                newline_int=ord("\n"),
                header_lines=0,  # no header lines in sequences files
                workers=workers,
                sort_dataset_paths=sort_dataset_paths,
                index_mapping_dir=index_mapping_dir,
            )

            if len(self.lines_ds) % 2 != 0:
                raise ValueError(
                    "The number of lines in the sequences files must be even (each sequence has an accession and a sequence line)."
                )

            # build mapping from file name to base index to support relative indices for each sequences file
            self._file_to_base_idx = {}
            for base_idx, fn_path in zip(
                [0] + list(self.lines_ds.midx_bins), self.lines_ds._files_list
            ):
                fn = os.path.basename(fn_path)
                self._file_to_base_idx[fn] = base_idx

            # build mapping from file name to file index to support fast access to each sequences file
            self._file_to_file_idx = {}
            for file_idx, fn_path in enumerate(self.lines_ds._files_list):
                fn = os.path.basename(fn_path)
                self._file_to_file_idx[fn] = file_idx

    def __len__(self):
        """Return the number of sequences in the dataset."""
        # Each sequence is represented by 2 lines (accession and sequence)
        return len(self.lines_ds) // 2

    def __getitem__(self, idx):
        """Return the sequence and its accession for the given index."""
        # Get the text lines for the accession and sequence
        accession_line = self.lines_ds[idx * 2]
        sequence_line = self.lines_ds[idx * 2 + 1]

        # Build data from text lines
        data = {
            # skip the first character (">") in the accession line
            "accession": accession_line[1:].strip(),
            "sequence": sequence_line.strip(),
        }

        return data

    def get_sequences_from_file(self, fn: str, indices: List[int], load_precomputed: bool = False):
        """
        Extract Sequences efficiently from a specific file.
        Args:
            fn: Filename (basename)
            indices: List of local indices (relative to the file)
            load_precomputed: whether to load from Arrow
        """
        if load_precomputed:
             # Load from Arrow
             # Arrow filename derived from .sequences name in __init__
             arrow_path = self._file_map.get(fn)
             if not arrow_path:
                 # Could try with .arrow extension replacement explicitly if map fails
                 arrow_path = fn.replace(".sequences", ".arrow") 
                 # This path needs to be absolute or relative to root?
                 # Handled by _file_map. If miss, we might be in trouble.
                 if fn not in self._file_map:
                     # Attempt to find it?
                     pass
                     
             if fn not in self._arrow_datasets:
                try:
                    self._arrow_datasets[fn] = datasets.load_from_disk(arrow_path).with_format("numpy")
                except Exception as e:
                    print(f"Failed to load arrow dataset {arrow_path}: {e}")
                    raise e
                    
             ds = self._arrow_datasets[fn]
             # Arrow supports batch indexing
             batch = ds[indices]
             return batch["input_ids"] # List of numpy arrays
        else:
             # Use text memmap
             # Reuse __getitem__ logic but batched?
             # __getitem__ calls lines_ds[idx] which is one by one.
             # but we can use list indexing if TextMemMap supported it. 
             # TextMemMap doesn't support list indexing natively usually.
             # So we loop.
             
             # Need global indices
             # We assume indices are 0-based index of SEQUENCE (so line 2*i)
             base_idx = self._file_to_base_idx[fn]
             global_indices = [idx + (base_idx // 2) for idx in indices]
             
             result = []
             for idx in global_indices:
                 # self[idx] calls __getitem__ which returns dict
                 result.append(self[idx]["sequence"])
             return result


    def get_global_sequence_indices(self, fn, local_indices):
        """
        Get the absolute index of the sequence in the dataset given relative index and file name.
        """
        # get the base index for the file
        base_idx = self._file_to_base_idx[fn]
        # return the absolues index
        return [idx + (base_idx // 2) for idx in local_indices]

        return sizes

    def get_sequence_sizes(self, fn: str, local_indices: list):
        """
        Compute and return the number of tokens in each sequence.
        """
        if self.load_precomputed:
             # Ensure dataset is open
             if fn not in self._arrow_datasets:
                  # Force load via helper (cleaner way would be a `get_ds` method)
                  # But access via map
                  arrow_path = self._file_map.get(fn)
                  self._arrow_datasets[fn] = datasets.load_from_disk(arrow_path).with_format("numpy")
             
             ds = self._arrow_datasets[fn]
             # Access lengths without loading full data if possible?
             # Arrow format numpy conversion might verify shape?
             # ds[local_indices] returns batch.
             batch = ds[local_indices]
             return [len(x) for x in batch["input_ids"]]
        else:
            sizes = []
            file_dx = self._file_to_file_idx[fn]
            _, midx = self.lines_ds.mdata_midx_list[file_dx]
            # return sizes for the given indices
            for idx in local_indices:
                sizes.append(midx[idx * 2 + 1] - midx[idx * 2] - 1)

            return sizes


class ProteinFamilyMemmapDataset(Dataset):
    def __init__(
        self,
        name: str,
        dataset_root: str,
        preprocessor: ProteinDocumentPreprocessor,
        tokenizer: ProFamTokenizer,
        max_tokens_per_family: Optional[
            int
        ] = None,  # CAUTION: caching results in same sequences being sampled from the family across epoch, we recommend setting max_tokens_per_example in the preprocessor instead
        max_families: Optional[int] = None,
        shuffle_family_sequences: bool = True,
        sample_cache_size: int = 1000,
        seed: Optional[int] = 1,
        load_precomputed: bool = False,
        **kwargs,
    ):
        """
        Args:
            name: name of the dataset
            dataset_root: point to the root directory of the dataset (i.e., train, val, test)
            tokenizer: tokenizer to use to convert sequences to tokens.
            max_families: maximum number of families to use (useful for validation)
            kwargs: additional arguments to pass to the dataset
        """
        super().__init__()
        self.name = name
        self.preprocessor = preprocessor
        self.tokenizer = tokenizer
        self.max_tokens_per_family = max_tokens_per_family
        self.max_families = max_families
        self.shuffle_family_sequences = shuffle_family_sequences
        self.sample_cache_size = sample_cache_size
        self.seed = seed
        self.load_precomputed = load_precomputed
        self.mapping_ds = MappingProteinFamilyMemmapDataset(
            dataset_root=dataset_root,
            # make sure order of files is deterministic
            sort_dataset_paths=True,
            **kwargs,
        )
        self.sequences_ds = SequencesProteinFamilyMemmapDataset(
            dataset_root=dataset_root,
            # make sure order of files is deterministic
            sort_dataset_paths=True,
            load_precomputed=load_precomputed,
            **kwargs,
        )

        self.local_rng = np.random.RandomState(seed=self.seed)

        # NOTE: MAKE __getitem__ A CACHED FUNCTION!!!
        # We need it since sampler will also load samples from the dataset to compute samples size.
        if self.sample_cache_size is not None and self.sample_cache_size > 0:
            self.__getitem__ = lru_cache(maxsize=self.sample_cache_size, typed=False)(
                self.__getitem__
            )

    def __len__(self):
        length = len(self.mapping_ds)
        if self.max_families is not None:
            length = min(length, self.max_families)
        return length

    def __getitem__(self, idx):
        mapping_data = self.mapping_ds[idx]
        sequence_indices = []
        sequence_sizes = []
        # collect samples from all files
        # NOTE: If load_precomputed is True, we optimize this!
        
        # Pre-pass: Collect all indices per file
        # mapping_data["sample_indices"] is {fn: [indices]}
        
        # 1. Get sizes (needed for filtering)
        # This part assumes we need sizes for ALL sequences in family to shuffle and filter.
        # This is fast via get_sequence_sizes (now supported for precomputed)
        for fn, indices in mapping_data["sample_indices"].items():
            sequence_sizes.extend(self.sequences_ds.get_sequence_sizes(fn, indices))
            # project each relative index to absolute index (Needed only for old text flow?)
            # If load_precomputed, we track (fn, local_idx) instead of global index.
            # But the logic below uses a flat list "sequence_indices".
            # We can pack tuples into it? Or just keep it as flat list of tuples?
            if not self.load_precomputed:
                sequence_indices.extend(
                    self.sequences_ds.get_global_sequence_indices(fn, indices)
                )
            else:
                # Store tuples (fn, idx)
                sequence_indices.extend([(fn, i) for i in indices])

        # randomize order of sequences within a family
        if self.shuffle_family_sequences:
            family_idx = list(range(len(sequence_indices)))
            self.local_rng.shuffle(family_idx)
        else:
            family_idx = list(range(len(sequence_indices)))

        # Limit tokens per family if specified
        if self.max_tokens_per_family is not None:
            cur_tokens = 0
            new_family_idx_len = 0
            for cur_family_i in family_idx:
                cur_tokens += sequence_sizes[cur_family_i]
                new_family_idx_len += 1
                if cur_tokens > self.max_tokens_per_family:
                    break
            family_idx = family_idx[:new_family_idx_len]

        # reorder and subset the family sequences
        sequence_indices = [sequence_indices[i] for i in family_idx]
        
        # Fetch Data
        if self.load_precomputed:
            # sequence_indices is list of (fn, idx)
            # Group by filename to minimize file access?
            # Or just fetch one by one? Arrow is decent at random access? 
            # Actually Arrow is column oriented. Batch fetch is better.
            # But we shuffled...
            # We can re-group.
            
            # Map for batch fetching
            to_fetch = {}
            # track order
            # (fn, idx) -> position in result
            
            flat_input_ids = []
            
            # Simple iteration might be slow if we switch files often.
            # But "family" usually spans few files?
            # Actually, family likely in ONE file unless huge.
            # Let's iterate.
            
            # Or:
            # Since we have the Full Tokenized Tensors now, we just concat them.
            # And add special tokens.
            
            # [FAM]
            tokenized_seqs = []
            
            # We need to fetch the data first.
            # Optimization: Sort by filename to batch fetch, then restore order?
            # Given limited family size, maybe just fetch.
            
            for fn, local_idx in sequence_indices:
                # get_sequences_from_file expects list of indices, returns list of arrays
                 batch = self.sequences_ds.get_sequences_from_file(fn, [local_idx], load_precomputed=True)
                 tokenized_seqs.append(batch[0])
            
            # Now assemble
            # Logic similar to tokenizers.py encode.
            # [FAM] (optional) + [SEQ1] [SEP] [SEQ2] [SEP] ... [SEP] (optional)
            
            # Get IDs
            fam_id = self.tokenizer.fam_token_id if self.tokenizer.add_fam_token else None
            sep_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.sep_token) # usually [SEP]
            # bos/eos? ProFamTokenizer adds them?
            # Our preprocess_local STRIPPED them? 
            # In preprocess_local.py, we did:
            # ids = tokenizer.encode(..., add_final_sep=False).input_ids
            # And removed BOS if present.
            # So we have raw AA IDs.
            
            # Reconstruct:
            # [FAM] [AA...] [SEP] [AA...] [SEP]
            
            final_ids = []
            if fam_id is not None:
                final_ids.append(np.array([fam_id]))
                
            for seq in tokenized_seqs:
                # Add [BOS]? The tokenizer normally adds it?
                # "get_sequence_of_sequences" joins with [SEP].
                # It does NOT add [BOS] to every sequence unless configured?
                # It adds [BOS] at START of document.
                # So: [BOS] [FAM] [SEQ1] [SEP] [SEQ2] [SEP] ...
                
                # Check tokenizer config
                # It has add_bos_token.
                # If we want to exact match runtime:
                # runtime: tokenizer.encode(ProteinDocument)
                # tokenizer.encode -> get_sequence_of_sequences -> tokenizer(concat_str)
                # tokenizer(concat_str) adds BOS if configured.
                
                final_ids.append(seq)
                final_ids.append(np.array([sep_id]))
                
            # Add BOS at start?
            if self.tokenizer.add_bos_token and self.tokenizer.bos_token_id is not None:
                final_ids.insert(0, np.array([self.tokenizer.bos_token_id]))
                
            # Document Token? [RAW]
            # Runtime adds it if add_document_token is True.
            if self.tokenizer.add_document_token:
                 # Which document token? cfg.document_token.
                 # Where is cfg? It is in self.preprocessor.cfg.
                 doc_token_str = self.preprocessor.cfg.document_token
                 doc_token_id = self.tokenizer.convert_tokens_to_ids(doc_token_str)
                 # It is added BEFORE sequences but after BOS?
                 # get_sequence_of_sequences: bos + doc + fam + seqs
                 # So insert at index 1 (after BOS)
                 final_ids.insert(1, np.array([doc_token_id]))

            merged = np.concatenate(final_ids)
            
            # Apply max length check
            if self.preprocessor.cfg.max_tokens_per_example is not None:
                if len(merged) > self.preprocessor.cfg.max_tokens_per_example:
                     merged = merged[:self.preprocessor.cfg.max_tokens_per_example]
            
            processed = {
                "input_ids": merged,
                "ds_name": self.name
            }
            # We might need attention_mask?
            # Collator usually handles padding and mask creation.
            return processed

        else:
            # get the actual sequence data for the selected indices
            sequences_data = [self.sequences_ds[i] for i in sequence_indices]
            protein_doc = ProteinDocument(
                sequences=[sd["sequence"] for sd in sequences_data],
                identifier=mapping_data["fam_id"],
                accessions=[sd["accession"] for sd in sequences_data],
            )
            processed = self.preprocessor.preprocess_protein_data(
                protein_doc,
                tokenizer=self.tokenizer,
            )
            processed["ds_name"] = self.name
            return processed
