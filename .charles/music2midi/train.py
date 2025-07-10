# train.py

import os
from typing import Dict, List
import logging

import numpy as np
import torch
import torch.nn.functional as F
import wandb
import colorlog
from dotenv import load_dotenv
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from transformers import AutoTokenizer

from dataset import MusicDataset
from model import MusicTranscriptionModel

# --- Setup ---
load_dotenv()
torch.manual_seed(int(os.getenv("SEED", 42)))

# --- Logger Setup ---
SUCCESS_LEVEL_NUM = 25
logging.addLevelName(SUCCESS_LEVEL_NUM, "SUCCESS")
def success(self, message, *args, **kws):
    if self.isEnabledFor(SUCCESS_LEVEL_NUM): 
        self._log(SUCCESS_LEVEL_NUM, message, args, **kws)
logging.Logger.success = success

handler = colorlog.StreamHandler()
handler.setFormatter(colorlog.ColoredFormatter(
    "%(log_color)s%(levelname)-8s%(reset)s | %(message)s",
    log_colors={
        'DEBUG': 'white',
        'INFO': 'cyan', 
        'SUCCESS': 'green',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'bold_red'
    }
))
logger = colorlog.getLogger()
if not logger.handlers: 
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Config
EPOCHS = int(os.getenv("EPOCHS_MUSIC2MIDI", 3))
BATCH_SIZE = int(os.getenv("BATCH_SIZE_MUSIC2MIDI", 4))
LR_ADAPTER = float(os.getenv("LR_ADAPTER", 1e-4))
LR_QWEN = float(os.getenv("LR_QWEN", 2e-5))
WEIGHT_DECAY = float(os.getenv("WEIGHT_DECAY", 1e-4))
GRAD_CLIP_NORM = float(os.getenv("GRAD_CLIP_NORM", 1.0))
SAVE_EVERY_N_EPOCHS = int(os.getenv("SAVE_EVERY_N_EPOCHS", 1))
VALIDATION_SPLIT = float(os.getenv("VALIDATION_SPLIT", 0.1))
MODELS_DIR = os.getenv("MODELS_OUTPUT_DIR", "../.data/models/music2midi")
CUSTOM_TOKENIZER_DIR = os.getenv("CUSTOM_TOKENIZER_DIR", "../.data/preprocessed/qwen_with_abc_tokenizer")
os.makedirs(MODELS_DIR, exist_ok=True)

# W&B
WANDB_API_KEY = os.getenv("WANDB_API_KEY")
WANDB_ENTITY = os.getenv("WANDB_ENTITY")
WANDB_PROJECT = os.getenv("WANDB_PROJECT")

def calculate_model_size(model):
    """Calculate detailed model parameter breakdown and memory usage."""
    logger.info("🔍 Calculating model parameter breakdown...")
    
    # Component breakdown
    whisper_params = 0
    qwen_params = 0
    adapter_params = 0
    trainable_params = 0
    frozen_params = 0
    
    component_breakdown = {
        'whisper_encoder': 0,
        'qwen_embeddings': 0,
        'qwen_layers': 0,
        'qwen_norm_head': 0,
        'cross_attention_adapter': 0
    }
    
    for name, param in model.named_parameters():
        param_count = param.numel()
        
        # Component categorization
        if 'audio_encoder' in name:
            whisper_params += param_count
            component_breakdown['whisper_encoder'] += param_count
        elif 'text_decoder.model.embed_tokens' in name or 'text_decoder.lm_head' in name:
            qwen_params += param_count
            component_breakdown['qwen_embeddings'] += param_count
        elif 'text_decoder.model.layers' in name:
            qwen_params += param_count
            component_breakdown['qwen_layers'] += param_count
        elif 'text_decoder.model.norm' in name:
            qwen_params += param_count
            component_breakdown['qwen_norm_head'] += param_count
        elif 'cross_attention_adapter' in name:
            adapter_params += param_count
            component_breakdown['cross_attention_adapter'] += param_count
        
        # Training status
        if param.requires_grad:
            trainable_params += param_count
        else:
            frozen_params += param_count
    
    total_params = trainable_params + frozen_params
    
    # Memory calculations (assuming float32)
    param_memory_gb = total_params * 4 / (1024**3)
    trainable_memory_gb = trainable_params * 4 / (1024**3)
    
    # Display results
    logger.info("=" * 60)
    logger.info("🎯 MODEL PARAMETER ANALYSIS")
    logger.info("=" * 60)
    
    # Component breakdown
    logger.info("📊 Component Breakdown:")
    logger.info(f"   Whisper Encoder:        {whisper_params:>12,} params ({whisper_params/1e6:.1f}M)")
    logger.info(f"   Qwen Embeddings/Head:   {component_breakdown['qwen_embeddings']:>12,} params ({component_breakdown['qwen_embeddings']/1e6:.1f}M)")
    logger.info(f"   Qwen Transformer Layers:{component_breakdown['qwen_layers']:>12,} params ({component_breakdown['qwen_layers']/1e6:.1f}M)")
    logger.info(f"   Qwen Norm/Head:         {component_breakdown['qwen_norm_head']:>12,} params ({component_breakdown['qwen_norm_head']/1e6:.1f}M)")
    logger.info(f"   Cross-Attention Adapter:{adapter_params:>12,} params ({adapter_params/1e6:.1f}M)")
    
    logger.info("-" * 60)
    
    # Training status breakdown
    logger.info("🎓 Training Status:")
    logger.info(f"   Trainable Parameters:   {trainable_params:>12,} params ({trainable_params/1e6:.1f}M)")
    logger.info(f"   Frozen Parameters:      {frozen_params:>12,} params ({frozen_params/1e6:.1f}M)")
    logger.info(f"   Total Parameters:       {total_params:>12,} params ({total_params/1e6:.1f}M)")
    
    logger.info("-" * 60)
    
    # Memory usage
    logger.info("💾 Memory Usage (Float32):")
    logger.info(f"   Total Model Memory:     {param_memory_gb:>8.2f} GB")
    logger.info(f"   Trainable Memory:       {trainable_memory_gb:>8.2f} GB")
    logger.info(f"   Training Efficiency:    {(trainable_params/total_params)*100:>7.1f}% of model is trainable")
    
    # Model architecture info
    if hasattr(model.text_decoder, 'config'):
        config = model.text_decoder.config
        logger.info("-" * 60)
        logger.info("🏗️  Architecture Details:")
        logger.info(f"   Qwen Hidden Size:       {config.hidden_size:>12,}")
        logger.info(f"   Qwen Vocab Size:        {config.vocab_size:>12,}")
        logger.info(f"   Qwen Layers:            {config.num_hidden_layers:>12,}")
        logger.info(f"   Qwen Attention Heads:   {config.num_attention_heads:>12,}")
    
    if hasattr(model.audio_encoder.encoder, 'config'):
        whisper_config = model.audio_encoder.encoder.config
        logger.info(f"   Whisper Hidden Size:    {whisper_config.d_model:>12,}")
        logger.info(f"   Whisper Layers:         {whisper_config.encoder_layers:>12,}")
        logger.info(f"   Whisper Attention Heads:{whisper_config.encoder_attention_heads:>12,}")
    
    logger.info("=" * 60)
    
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'frozen_params': frozen_params,
        'whisper_params': whisper_params,
        'qwen_params': qwen_params,
        'adapter_params': adapter_params,
        'param_memory_gb': param_memory_gb,
        'trainable_memory_gb': trainable_memory_gb,
        'component_breakdown': component_breakdown
    }

def load_custom_tokenizer():
    """Load the custom tokenizer with extended ABC vocabulary."""
    tokenizer_path = CUSTOM_TOKENIZER_DIR
    fallback_model = os.getenv("QWEN_MODEL", "Qwen/Qwen3-0.6B-Base")
    
    if os.path.exists(tokenizer_path) and os.path.exists(os.path.join(tokenizer_path, "tokenizer.json")):
        logger.info(f"Loading custom tokenizer from: {tokenizer_path}")
        try:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
            logger.success(f"Custom tokenizer loaded successfully. Vocab size: {len(tokenizer)}")
            
            # Check for ABC tokens
            abc_tokens = [token for token in tokenizer.get_vocab().keys() if any(
                abc_marker in token for abc_marker in ['[', ']', '|', 'M:', 'K:', 'Q:', 'T:']
            )]
            logger.info(f"Found {len(abc_tokens)} ABC-related tokens in vocabulary")
            return tokenizer, True
            
        except Exception as e:
            logger.error(f"Failed to load custom tokenizer: {e}")
            logger.warning(f"Falling back to base model: {fallback_model}")
    else:
        logger.warning(f"Custom tokenizer not found at {tokenizer_path}")
        logger.info(f"Using base model tokenizer: {fallback_model}")
    
    # Fallback to base model
    tokenizer = AutoTokenizer.from_pretrained(fallback_model, trust_remote_code=True)
    logger.info(f"Base tokenizer loaded. Vocab size: {len(tokenizer)}")
    return tokenizer, False

def collate_fn(batch: List[Dict]) -> Dict:
    """
    Custom collate function to handle batching of our dataset items.
    """
    # Waveforms are passed as a list of numpy arrays
    waveforms = [item["waveform"] for item in batch]
    sampling_rates = [item["sampling_rate"] for item in batch]
    
    # Stack the tokenized tensors
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_masks = torch.stack([item["attention_mask"] for item in batch])
    
    # Ensure all items in the batch have the same sampling rate
    assert all(sr == sampling_rates[0] for sr in sampling_rates), "All samples must have same sampling rate"

    return {
        "waveforms": waveforms,  # Fixed: matches model.forward parameter name
        "sampling_rate": sampling_rates[0],
        "input_ids": input_ids,
        "attention_mask": attention_masks,
        "filenames": [item["filename"] for item in batch]  # For debugging
    }

def setup_optimizers(model):
    """Setup optimizer with simplified parameter grouping."""
    trainable_params = []
    frozen_params = []
    
    # Collect all parameters and categorize them
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_params.append((name, param))
        else:
            frozen_params.append((name, param))
    
    # Print parameter breakdown
    logger.info("=== Optimizer Parameter Analysis ===")
    logger.info(f"Trainable parameters: {sum(p.numel() for _, p in trainable_params):,}")
    logger.info(f"Frozen parameters: {sum(p.numel() for _, p in frozen_params):,}")
    
    # Group trainable parameters by component
    adapter_params = []
    qwen_params = []
    
    for name, param in trainable_params:
        if 'cross_attention_adapter' in name:
            adapter_params.append(param)
        elif 'text_decoder' in name:
            qwen_params.append(param)
        else:
            logger.warning(f"Unknown trainable parameter: {name}")
            adapter_params.append(param)  # Default to adapter group
    
    # Verify no Whisper parameters are trainable
    whisper_trainable = [name for name, _ in trainable_params if 'audio_encoder' in name]
    if whisper_trainable:
        logger.warning(f"⚠️  WARNING: Found trainable Whisper parameters: {whisper_trainable}")
    else:
        logger.success("✅ Whisper encoder is properly frozen")
    
    # Create parameter groups
    param_groups = [
        {'params': adapter_params, 'lr': LR_ADAPTER, 'name': 'adapter'},
        {'params': qwen_params, 'lr': LR_QWEN, 'name': 'qwen'}
    ]
    
    optimizer = torch.optim.AdamW(param_groups, weight_decay=WEIGHT_DECAY)
    
    logger.info(f"Adapter parameters: {sum(p.numel() for p in adapter_params):,}")
    logger.info(f"Qwen parameters: {sum(p.numel() for p in qwen_params):,}")
    logger.info(f"Total trainable: {sum(p.numel() for p in adapter_params + qwen_params):,}")
    
    return optimizer

def save_checkpoint(model, optimizer, epoch, loss, is_best=False):
    """Save checkpoint excluding frozen Whisper weights to save space."""
    
    # Get only trainable state dict (excludes frozen Whisper weights)
    trainable_state_dict = {
        name: param for name, param in model.state_dict().items() 
        if any(p is param for p in model.parameters() if p.requires_grad)
    }
    
    # Also save critical frozen components that might be needed for loading
    critical_frozen_components = {}
    for name, param in model.state_dict().items():
        # Save text decoder config-related weights even if frozen
        if 'text_decoder.model.embed_tokens' in name or 'text_decoder.lm_head' in name:
            critical_frozen_components[name] = param
    
    checkpoint = {
        'epoch': epoch,
        'trainable_state_dict': trainable_state_dict,
        'critical_frozen_state_dict': critical_frozen_components,
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'config': {
            'lr_adapter': LR_ADAPTER,
            'lr_qwen': LR_QWEN,
            'batch_size': BATCH_SIZE,
            'model_name': os.getenv('QWEN_MODEL'),
            'whisper_model': os.getenv('WHISPER_MODEL'),
            'custom_tokenizer_dir': CUSTOM_TOKENIZER_DIR,
            'vocab_size': model.text_decoder.config.vocab_size if hasattr(model.text_decoder, 'config') else None,
            'top_k_layers': os.getenv("TOP_K_QWEN_LAYERS"),
            'note': 'Whisper weights excluded to save space - model will reload them from base model'
        }
    }
    
    # Regular checkpoint
    checkpoint_path = os.path.join(MODELS_DIR, f"checkpoint_epoch_{epoch}.pt")
    torch.save(checkpoint, checkpoint_path)
    
    # Calculate space saved
    full_size = sum(p.numel() for p in model.parameters()) * 4  # Assuming float32
    saved_size = sum(p.numel() for p in trainable_state_dict.values()) * 4
    space_saved = (full_size - saved_size) / (1024**3)  # GB
    
    logger.info(f"Checkpoint saved: {checkpoint_path}")
    logger.info(f"Space saved by excluding frozen weights: {space_saved:.2f} GB")
    
    # Best model checkpoint
    if is_best:
        best_path = os.path.join(MODELS_DIR, "best_model.pt")
        torch.save(checkpoint, best_path)
        logger.success(f"Saved best model to {best_path}")
    
    return checkpoint_path

def load_checkpoint(model, optimizer, checkpoint_path):
    """Load checkpoint with proper handling of excluded Whisper weights."""
    logger.info(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Load trainable parameters
    if 'trainable_state_dict' in checkpoint:
        missing_keys, unexpected_keys = model.load_state_dict(
            checkpoint['trainable_state_dict'], strict=False
        )
        logger.info(f"Loaded trainable parameters. Missing: {len(missing_keys)}, Unexpected: {len(unexpected_keys)}")
        
        # Load critical frozen components if available
        if 'critical_frozen_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['critical_frozen_state_dict'], strict=False)
            logger.info("Loaded critical frozen components")
    else:
        # Fallback for old checkpoint format
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        logger.info("Loaded full model state (old format)")
    
    # Load optimizer
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logger.info("Loaded optimizer state")
    
    return checkpoint.get('epoch', 0), checkpoint.get('loss', float('inf'))

def validate_model(model, val_loader, device):
    """Run validation and return average loss."""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            loss = model(
                waveforms=batch["waveforms"],
                sampling_rate=batch["sampling_rate"],
                target_token_ids=input_ids,
                attention_mask=attention_mask
            )
            
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches if num_batches > 0 else float('inf')

def main():
    """Main training function."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"🚀 Using device: {device}")
    
    # Load custom tokenizer first
    tokenizer, using_custom_tokenizer = load_custom_tokenizer()
    
    # Init W&B with improved run name
    if WANDB_API_KEY:
        wandb.login(key=WANDB_API_KEY)
        
        # Create descriptive run name
        qwen_model_name = os.getenv('QWEN_MODEL', 'qwen').split('/')[-1]
        tokenizer_suffix = "custom-abc" if using_custom_tokenizer else "base"
        run_name = f"music2midi-{qwen_model_name}-{tokenizer_suffix}"
        
        wandb.init(
            project=WANDB_PROJECT,
            entity=WANDB_ENTITY,
            name=run_name,
            config={
                "epochs": EPOCHS, "batch_size": BATCH_SIZE, 
                "lr_qwen": LR_QWEN, "lr_adapter": LR_ADAPTER,
                "weight_decay": WEIGHT_DECAY, "grad_clip_norm": GRAD_CLIP_NORM,
                "top_k_layers": os.getenv("TOP_K_QWEN_LAYERS"),
                "whisper_model": os.getenv("WHISPER_MODEL"),
                "qwen_model": os.getenv("QWEN_MODEL"),
                "using_custom_tokenizer": using_custom_tokenizer,
                "tokenizer_vocab_size": len(tokenizer),
                "custom_tokenizer_dir": CUSTOM_TOKENIZER_DIR if using_custom_tokenizer else None,
                "checkpoint_excludes_whisper": True
            }
        )

    # Dataset and DataLoader
    logger.info("📚 Loading dataset...")
    dataset = MusicDataset(tokenizer=tokenizer)
    
    # Split into train/validation
    val_size = int(len(dataset) * VALIDATION_SPLIT)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    
    logger.info(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")

    # Model and Optimizer - pass tokenizer to model
    logger.info("🏗️  Initializing model...")
    model = MusicTranscriptionModel(tokenizer=tokenizer).to(device)
    
    # Calculate and display model size
    model_stats = calculate_model_size(model)
    
    optimizer = setup_optimizers(model)
    
    # Print tokenizer info
    logger.info(f"🔤 Model using tokenizer with vocab size: {len(tokenizer)}")
    if using_custom_tokenizer:
        logger.success("✅ Using custom tokenizer with extended ABC vocabulary")
    else:
        logger.warning("⚠️  Using base tokenizer - custom tokenizer not found")
    
    # Log model stats to W&B
    if WANDB_API_KEY:
        wandb.log({
            "model/total_params": model_stats['total_params'],
            "model/trainable_params": model_stats['trainable_params'],
            "model/frozen_params": model_stats['frozen_params'],
            "model/whisper_params": model_stats['whisper_params'],
            "model/qwen_params": model_stats['qwen_params'],
            "model/adapter_params": model_stats['adapter_params'],
            "model/param_memory_gb": model_stats['param_memory_gb'],
            "model/trainable_memory_gb": model_stats['trainable_memory_gb'],
            "model/training_efficiency": (model_stats['trainable_params']/model_stats['total_params'])*100
        })
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    best_val_loss = float('inf')
    
    # Training Loop
    logger.info("🎯 Starting training loop...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}")
        
        for batch_idx, batch in enumerate(pbar):
            optimizer.zero_grad()
            
            # Move tensors to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            # Forward pass
            loss = model(
                waveforms=batch["waveforms"],
                sampling_rate=batch["sampling_rate"],
                target_token_ids=input_ids,
                attention_mask=attention_mask
            )
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "avg_loss": f"{total_loss/num_batches:.4f}"
            })
            
            # Log to W&B
            if WANDB_API_KEY and batch_idx % 10 == 0:
                wandb.log({
                    "train/loss": loss.item(),
                    "train/avg_loss": total_loss/num_batches,
                    "train/epoch": epoch + 1,
                    "train/batch": batch_idx
                })
        
        # Validation
        if len(val_loader) > 0:
            val_loss = validate_model(model, val_loader, device)
            scheduler.step(val_loss)
            
            logger.info(f"Epoch {epoch + 1} - Train Loss: {total_loss/num_batches:.4f}, Val Loss: {val_loss:.4f}")
            
            # Log validation metrics
            if WANDB_API_KEY:
                wandb.log({
                    "val/loss": val_loss,
                    "val/epoch": epoch + 1,
                    "lr/adapter": optimizer.param_groups[0]['lr'],
                    "lr/qwen": optimizer.param_groups[1]['lr'] if len(optimizer.param_groups) > 1 else optimizer.param_groups[0]['lr']
                })
            
            # Save best model
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
            
            # Save checkpoints
            if (epoch + 1) % SAVE_EVERY_N_EPOCHS == 0 or is_best:
                save_checkpoint(model, optimizer, epoch + 1, val_loss, is_best)
        else:
            logger.info(f"Epoch {epoch + 1} - Train Loss: {total_loss/num_batches:.4f}")
            
            # Save regular checkpoints even without validation
            if (epoch + 1) % SAVE_EVERY_N_EPOCHS == 0:
                save_checkpoint(model, optimizer, epoch + 1, total_loss/num_batches)
    
    logger.success("🎉 Training completed!")
    if WANDB_API_KEY:
        wandb.finish()

if __name__ == "__main__":
    main()