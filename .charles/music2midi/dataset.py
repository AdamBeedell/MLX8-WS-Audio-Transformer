# dataset.py

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from dotenv import load_dotenv
import logging
import colorlog

load_dotenv()

# Setup logger
logger = colorlog.getLogger(__name__)

PREPROCESSED_DIR = os.getenv("PREPROCESSED_DATA_DIR", "../.data/preprocessed")
TOKENIZER_DIR = os.path.join(PREPROCESSED_DIR, "qwen_with_abc_tokenizer")
PARQUET_PATH = os.path.join(PREPROCESSED_DIR, "music_dataset.parquet")
MAX_SEQUENCE_LENGTH = int(os.getenv("MAX_SEQUENCE_LENGTH", 512))

class MusicDataset(Dataset):
    """
    Dataset for music transcription that loads audio waveforms and ABC notation pairs.
    """
    def __init__(self, parquet_path=PARQUET_PATH, tokenizer=None, tokenizer_dir=TOKENIZER_DIR):
        """
        Initialize the dataset.
        
        Args:
            parquet_path: Path to the parquet file containing the dataset
            tokenizer: Pre-loaded tokenizer to use (if None, loads from tokenizer_dir)
            tokenizer_dir: Directory containing tokenizer files (fallback)
        """
        # Use provided tokenizer or load from directory
        if tokenizer is not None:
            logger.info("Using provided tokenizer for dataset")
            self.tokenizer = tokenizer
        else:
            logger.info(f"Loading tokenizer from: {tokenizer_dir}")
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, trust_remote_code=True)
        
        # Load the parquet file
        logger.info(f"Loading dataset from: {parquet_path}")
        self.df = pd.read_parquet(parquet_path)
        
        # Filter out failed processing entries
        original_length = len(self.df)
        self.df = self.df[self.df['processing_success'] == True].reset_index(drop=True)
        filtered_length = len(self.df)
        
        if original_length != filtered_length:
            logger.warning(f"Filtered out {original_length - filtered_length} failed entries")
        
        logger.info(f"Loaded {filtered_length} successful audio-ABC pairs")
        
        # Set pad token if not exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            logger.info("Set pad_token to eos_token")
        
        # Log tokenizer info
        logger.info(f"Dataset using tokenizer with vocab size: {len(self.tokenizer)}")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Get waveform (already as numpy array from parquet)
        waveform = np.array(row['waveform'], dtype=np.float32)
        sampling_rate = int(row['sampling_rate'])
        
        # Get ABC notation
        abc_string = row['abc_string']
        
        # Tokenize ABC notation
        encoded = self.tokenizer(
            abc_string,
            truncation=True,
            padding='max_length',
            max_length=MAX_SEQUENCE_LENGTH,
            return_tensors='pt'
        )
        
        return {
            'waveform': waveform,
            'sampling_rate': sampling_rate,
            'input_ids': encoded['input_ids'].squeeze(0),  # Remove batch dimension
            'attention_mask': encoded['attention_mask'].squeeze(0),
            'abc_string': abc_string,  # Keep original for debugging
            'filename': row['filename']
        }