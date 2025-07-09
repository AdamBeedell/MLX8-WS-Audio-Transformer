# train.py

import os
from typing import Dict, List

import numpy as np
import torch
import wandb
from dotenv import load_dotenv
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import MusicDataset
from model import MusicTranscriptionModel

# --- Setup ---
load_dotenv()
torch.manual_seed(int(os.getenv("SEED", 42)))

# Config
EPOCHS = int(os.getenv("EPOCHS", 10))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 4))
LR_ADAPTER = float(os.getenv("LR_ADAPTER", 1e-4))
LR_QWEN = float(os.getenv("LR_QWEN", 2e-5))
WEIGHT_DECAY = float(os.getenv("WEIGHT_DECAY", 1e-4))
MODELS_DIR = os.getenv("MODELS_OUTPUT_DIR", "../.data/models/music2midi")
os.makedirs(MODELS_DIR, exist_ok=True)

# W&B
WANDB_API_KEY = os.getenv("WANDB_API_KEY")
WANDB_ENTITY = os.getenv("WANDB_ENTITY")
WANDB_PROJECT = os.getenv("WANDB_PROJECT")

def collate_fn(batch: List[Dict]) -> Dict:
    """
    Custom collate function to handle batching of our dataset items.
    """
    # Waveforms are now passed as a list of numpy arrays, which the processor handles.
    waveforms = [item["waveform"] for item in batch]
    sampling_rates = [item["sampling_rate"] for item in batch]
    
    # Stack the tokenized tensors
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_masks = torch.stack([item["attention_mask"] for item in batch])
    
    # Ensure all items in the batch have the same sampling rate
    assert all(sr == sampling_rates[0] for sr in sampling_rates)

    return {
        "waveforms": waveforms,
        "sampling_rate": sampling_rates[0],
        "input_ids": input_ids,
        "attention_mask": attention_masks
    }

def main():
    """Main training function."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Init W&B
    if WANDB_API_KEY:
        wandb.login(key=WANDB_API_KEY)
        wandb.init(
            project=WANDB_PROJECT,
            entity=WANDB_ENTITY,
            name=f"music-transcription-{os.getenv('QWEN_MODEL').split('/')[-1]}",
            config={
                "epochs": EPOCHS, "batch_size": BATCH_SIZE, "lr_qwen": LR_QWEN,
                "top_k_layers": os.getenv("TOP_K_QWEN_LAYERS")
            }
        )

    # Model, Dataset, DataLoader
    model = MusicTranscriptionModel().to(device)
    dataset = MusicDataset()
    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    # Optimizer (targeting only trainable parameters)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"Number of trainable parameters: {sum(p.numel() for p in trainable_params):,}")
    optimizer = torch.optim.AdamW(trainable_params, lr=LR_QWEN, weight_decay=WEIGHT_DECAY)

    # Training Loop
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        pbar = tqdm(data_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}")
        
        for batch in pbar:
            optimizer.zero_grad()
            
            # Move tensors to the correct device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            # Forward pass
            loss = model(
                waveform=batch["waveforms"],
                sampling_rate=batch["sampling_rate"],
                target_token_ids=input_ids,
                attention_mask=attention_mask
            )
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            if WANDB_API_KEY:
                wandb.log({"batch_loss": loss.item()})

        avg_loss = total_loss / len(data_loader)
        print(f"Epoch {epoch + 1} finished. Average Loss: {avg_loss:.4f}")
        if WANDB_API_KEY:
            wandb.log({"epoch": epoch + 1, "average_loss": avg_loss})
            
        # Save checkpoint
        checkpoint_path = os.path.join(MODELS_DIR, f"checkpoint_epoch_{epoch + 1}.pt")
        # Save the state dict of the text_decoder part of the model
        torch.save(model.text_decoder.state_dict(), checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")
        
    if WANDB_API_KEY:
        wandb.finish()
    print("Training complete.")

if __name__ == "__main__":
    main()