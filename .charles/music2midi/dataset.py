# dataset.py

import os
from pathlib import Path

import pandas as pd
import torch
from dotenv import load_dotenv
from torch.utils.data import Dataset
from transformers import AutoTokenizer

load_dotenv()

PREPROCESSED_DIR = Path(os.getenv("PREPROCESSED_DATA_DIR", "../.data/preprocessed"))
PARQUET_PATH = PREPROCESSED_DIR / "music_dataset.parquet"
TOKENIZER_DIR = PREPROCESSED_DIR / "qwen_with_abc_tokenizer"

class MusicDataset(Dataset):
    """
    PyTorch Dataset for loading our preprocessed music data.
    """
    def __init__(self, parquet_path=PARQUET_PATH, tokenizer_dir=TOKENIZER_DIR, max_length=512):
        print("Loading dataset from Parquet file...")
        self.df = pd.read_parquet(parquet_path)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, trust_remote_code=True)
        self.max_length = max_length
        self.sampling_rate = int(os.getenv("SAMPLE_RATE", 16000))
        
        # Add a padding token if it doesn't exist
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            print("Added [PAD] as a special token.")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        waveform = row["waveform"]
        abc_string = row["abc_string"]

        # Tokenize the target ABC string
        tokenized_output = self.tokenizer(
            abc_string,
            max_length=self.max_length,
            padding="max_length", # Pad to a fixed length
            truncation=True,
            return_tensors="pt"
        )
        
        # Squeeze tensors to remove the batch dimension of 1
        input_ids = tokenized_output.input_ids.squeeze(0)
        attention_mask = tokenized_output.attention_mask.squeeze(0)

        return {
            "waveform": waveform,
            "sampling_rate": self.sampling_rate,
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }