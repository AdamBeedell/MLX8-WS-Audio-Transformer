import os
import pandas as pd
import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from dotenv import load_dotenv
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt

import logging
import colorlog
import wandb
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, classification_report, confusion_matrix
import seaborn as sns
import math

# Setup colorlog logger with emojis
SUCCESS_LEVEL_NUM = 25
logging.addLevelName(SUCCESS_LEVEL_NUM, "SUCCESS")

def success(self, message, *args, **kws):
    if self.isEnabledFor(SUCCESS_LEVEL_NUM):
        self._log(SUCCESS_LEVEL_NUM, message, args, **kws)
logging.Logger.success = success

handler = colorlog.StreamHandler()
handler.setFormatter(colorlog.ColoredFormatter(
    "%(log_color)s%(levelname)s %(message)s",
    log_colors={
        'DEBUG':    'white',
        'INFO':     'white',
        'SUCCESS':  'green',
        'WARNING':  'yellow',
        'ERROR':    'red',
        'CRITICAL': 'bold_red',
    },
    secondary_log_colors={},
    style='%'
))

logger = colorlog.getLogger()
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)

# Load environment variables
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

# Environment variables (define these in .env)
DATA_ROOT = os.getenv("DATA_ROOT", "./.data/UrbanSound8K")
METADATA_CSV = os.getenv("METADATA_CSV", "./.data/UrbanSound8K/metadata/UrbanSound8K.csv")
PROCESSED_PARQUET_PATH = os.getenv("PROCESSED_PARQUET_PATH", "./.data/UrbanSound8K/processed")
EVAL_PARQUET_PATH = os.getenv("EVAL_PARQUET_PATH", "./.data/UrbanSound8K/eval")
SAMPLE_RATE = int(os.getenv("SAMPLE_RATE", 16000))
N_MELS = int(os.getenv("N_MELS", 128))
N_FFT = int(os.getenv("N_FFT", 1024))
HOP_LENGTH = int(os.getenv("HOP_LENGTH", 512))
FMIN = int(os.getenv("FMIN", 0))
FMAX = int(os.getenv("FMAX", 8000))
MONO = os.getenv("MONO", "True") == "True"
DURATION = float(os.getenv("DURATION", 4.0))  # seconds
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 32))
EPOCHS = int(os.getenv("EPOCHS", 10))
LR = float(os.getenv("LR", 3e-4))
DROPOUT = float(os.getenv("DROPOUT", 0.3))
WEIGHT_DECAY = float(os.getenv("WEIGHT_DECAY", 1e-4))

# Add transformer hyperparameters from env
TRANSFORMER_DIM = int(os.getenv("TRANSFORMER_DIM", 128))
TRANSFORMER_HEADS = int(os.getenv("TRANSFORMER_HEADS", 4))
TRANSFORMER_LAYERS = int(os.getenv("TRANSFORMER_LAYERS", 2))
TRANSFORMER_DROPOUT = float(os.getenv("TRANSFORMER_DROPOUT", 0.1))
TRANSFORMER_MLP_DIM = int(os.getenv("TRANSFORMER_MLP_DIM", 256))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.success(f"Using device: {device}")

mel_spectrogram = torchaudio.transforms.MelSpectrogram(
    sample_rate=SAMPLE_RATE,
    n_fft=N_FFT,
    hop_length=HOP_LENGTH,
    n_mels=N_MELS,
    f_min=FMIN,
    f_max=FMAX,
    power=2.0,
).to(device)

MODEL_ROOT = os.getenv("MODEL_ROOT", "./.data/UrbanSound8K/models").replace('"', '').replace(",", "")
WANDB_API_KEY = os.getenv("WANDB_API_KEY")
WANDB_ENTITY = os.getenv("WANDB_ENTITY")
WANDB_PROJECT = os.getenv("WANDB_PROJECT")

def get_processed_parquet_filename():
    """Generate processed parquet filename with embedded hyperparameters."""
    return f"urbansound8k_processed_mels{N_MELS}_hop{HOP_LENGTH}.parquet"

def get_processed_parquet_path():
    """Get full path for processed parquet file."""
    return os.path.join(PROCESSED_PARQUET_PATH, get_processed_parquet_filename())

def get_test_parquet_filename():
    """Generate test parquet filename with embedded hyperparameters."""
    return f"urbansound8k_test_fold10_mels{N_MELS}_hop{HOP_LENGTH}_batch{BATCH_SIZE}_epochs{EPOCHS}.parquet"

def get_test_parquet_path():
    """Get full path for test parquet file."""
    return os.path.join(EVAL_PARQUET_PATH, get_test_parquet_filename())

def get_checkpoint_filename(model_type="cnn"):
    """Generate checkpoint filename with embedded hyperparameters."""
    if model_type == "cnn":
        model_str = "cnn"
    elif model_type == "transformer":
        model_str = "transformer"
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    return f"urbansound8k_{model_str}_final_mels{N_MELS}_hop{HOP_LENGTH}_batch{BATCH_SIZE}_epochs{EPOCHS}_lr{LR}_dropout{DROPOUT}.pt"

def preprocess_to_parquet():
    """
    Preprocess UrbanSound8K audio files to log-mel spectrograms and save as Parquet.
    """
    parquet_path = get_processed_parquet_path()
    
    # Check if file exists and ask for overwrite
    if os.path.exists(parquet_path):
        response = input(f"Processed parquet file already exists at {parquet_path}. Overwrite? (y/N): ")
        if response.lower() not in ['y', 'yes']:
            logger.info("Preprocessing cancelled.")
            return
    
    df = pd.read_csv(METADATA_CSV)
    records = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing audio"):
        fold = row['fold']
        file_name = row['slice_file_name']
        class_id = row['classID']
        class_name = row['class']
        rel_path = os.path.join("audio", f"fold{fold}", file_name)
        abs_path = os.path.join(DATA_ROOT, rel_path)
        try:
            waveform, sr = torchaudio.load(abs_path)
            if MONO and waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            # Resample if needed
            if sr != SAMPLE_RATE:
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=SAMPLE_RATE)
                waveform = resampler(waveform)
            # Pad or trim to fixed duration
            num_samples = int(SAMPLE_RATE * DURATION)
            if waveform.shape[1] < num_samples:
                pad = num_samples - waveform.shape[1]
                waveform = torch.nn.functional.pad(waveform, (0, pad))
            else:
                waveform = waveform[:, :num_samples]
            
            # Move waveform to the same device as mel_spectrogram
            waveform = waveform.to(device)
            mel = mel_spectrogram(waveform)
            log_mel = torch.log(mel + 1e-6)
            
            # Move back to CPU and convert to numpy for storage
            log_mel_np = log_mel.squeeze(0).cpu().numpy().astype(np.float32)  # shape: [n_mels, time]
            records.append({
                "rel_path": rel_path,
                "fold": fold,
                "class_id": class_id,
                "class_name": class_name,
                "log_mel_flat": log_mel_np.flatten(),  # Flatten to 1D array
                "log_mel_shape": log_mel_np.shape,     # Store original shape
            })
        except Exception as e:
            logger.error(f"❌ Error processing {abs_path}: {e}")

    out_df = pd.DataFrame(records)
    # Convert shape tuples to lists for Parquet compatibility
    out_df["log_mel_shape"] = out_df["log_mel_shape"].apply(lambda x: list(x))
    os.makedirs(os.path.dirname(parquet_path), exist_ok=True)
    out_df.to_parquet(parquet_path, index=False)
    logger.success(f"✅ Saved processed dataset to {parquet_path}")

class UrbanSoundDataSet(Dataset):
    """
    PyTorch Dataset for UrbanSound8K, loading from processed Parquet file.
    """
    def __init__(self, parquet_path=None, folds=None):
        """
        Args:
            parquet_path: Path to processed Parquet file. If None, uses default processed path.
            folds: List of folds to include (e.g., [1,2,3,4,5,6,7,8])
        """
        if parquet_path is None:
            parquet_path = get_processed_parquet_path()
        self.df = pd.read_parquet(parquet_path)
        if folds is not None:
            self.df = self.df[self.df["fold"].isin(folds)].reset_index(drop=True)
        self.n_mels = N_MELS

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # Reconstruct the log-mel spectrogram from flattened data
        log_mel_flat = np.array(row["log_mel_flat"], dtype=np.float32)
        log_mel_shape = tuple(row["log_mel_shape"])
        log_mel = log_mel_flat.reshape(log_mel_shape)
        # shape: [n_mels, time]
        label = int(row["class_id"])
        return torch.tensor(log_mel), label

def preprocess_audio_for_cnn(waveform, sr):
    """
    Preprocess audio the same way as CNN training: pad or truncate to fixed duration.
    Returns both original and processed waveforms.
    """
    if MONO and waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    # Resample if needed
    if sr != SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=SAMPLE_RATE)
        waveform = resampler(waveform)
        sr = SAMPLE_RATE
    
    # Create CNN-ready version (pad or truncate to fixed duration)
    num_samples = int(SAMPLE_RATE * DURATION)
    waveform_cnn = waveform.clone()
    
    if waveform_cnn.shape[1] < num_samples:
        # Pad with zeros
        pad = num_samples - waveform_cnn.shape[1]
        waveform_cnn = torch.nn.functional.pad(waveform_cnn, (0, pad))
    else:
        # Truncate
        waveform_cnn = waveform_cnn[:, :num_samples]
    
    return waveform, waveform_cnn, sr

def plot_waveform_image(waveform, sr, out_path, title=None):
    """
    Plot and save a waveform image, showing both original and CNN-processed (4s fixed) versions.
    Red box indicates the fixed 4s window that goes to CNN classifier.
    """
    waveform_orig, waveform_cnn, sr = preprocess_audio_for_cnn(waveform, sr)
    
    # Original waveform
    waveform_orig_np = waveform_orig.squeeze().cpu().numpy()
    time_axis_orig = np.linspace(0, len(waveform_orig_np) / sr, len(waveform_orig_np))
    
    # CNN waveform (fixed 4s)
    waveform_cnn_np = waveform_cnn.squeeze().cpu().numpy()
    time_axis_cnn = np.linspace(0, DURATION, len(waveform_cnn_np))
    
    plt.figure(figsize=(12, 6))
    
    # Plot original waveform
    plt.subplot(2, 1, 1)
    plt.plot(time_axis_orig, waveform_orig_np, 'b-', alpha=0.7, label='Original')
    
    # Add red box for 4s window
    plt.axvspan(0, min(DURATION, time_axis_orig[-1]), alpha=0.2, color='red', label='CNN Input (4s)')
    if time_axis_orig[-1] > DURATION:
        plt.axvline(DURATION, color='red', linestyle='--', alpha=0.8, label='4s cutoff')
    
    plt.title(f"{title if title else 'Waveform'} - Original")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot CNN-processed waveform (fixed 4s)
    plt.subplot(2, 1, 2)
    plt.plot(time_axis_cnn, waveform_cnn_np, 'r-', alpha=0.8)
    
    # Red box around entire CNN input
    plt.axvspan(0, DURATION, alpha=0.1, color='red', label='CNN Input (Fixed 4s)')
    plt.title("CNN Input - Fixed 4s (padded/truncated)")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_spectrogram_image(waveform, sr, out_path, title=None):
    """
    Plot and save a log-mel spectrogram image, showing both original and CNN-processed (4s fixed) versions.
    Red box indicates the fixed 4s window that goes to CNN classifier.
    """
    waveform_orig, waveform_cnn, sr = preprocess_audio_for_cnn(waveform, sr)
    
    # Original spectrogram
    waveform_orig = waveform_orig.to(device)
    mel_orig = mel_spectrogram(waveform_orig)
    log_mel_orig = torch.log(mel_orig + 1e-6).squeeze().cpu().numpy()
    n_frames_orig = log_mel_orig.shape[-1]
    time_axis_orig = np.linspace(0, n_frames_orig * HOP_LENGTH / SAMPLE_RATE, n_frames_orig)
    
    # CNN spectrogram (fixed 4s)
    waveform_cnn = waveform_cnn.to(device)
    mel_cnn = mel_spectrogram(waveform_cnn)
    log_mel_cnn = torch.log(mel_cnn + 1e-6).squeeze().cpu().numpy()
    n_frames_cnn = log_mel_cnn.shape[-1]
    time_axis_cnn = np.linspace(0, DURATION, n_frames_cnn)
    
    # Use same color scale for both plots
    vmin = min(log_mel_orig.min(), log_mel_cnn.min())
    vmax = max(log_mel_orig.max(), log_mel_cnn.max())
    
    plt.figure(figsize=(12, 8))
    
    # Plot original spectrogram
    plt.subplot(2, 1, 1)
    im1 = plt.imshow(log_mel_orig, aspect='auto', origin='lower', 
                     extent=[0, time_axis_orig[-1], 0, log_mel_orig.shape[0]], 
                     cmap='viridis', vmin=vmin, vmax=vmax)
    
    # Add red box outline only (no fill to avoid color pollution)
    if time_axis_orig[-1] > DURATION:
        # Show cutoff line
        plt.axvline(DURATION, color='red', linestyle='--', linewidth=2, alpha=0.8, label='4s cutoff')
        # Add red box outline for first 4s
        plt.plot([0, DURATION, DURATION, 0, 0], 
                 [0, 0, log_mel_orig.shape[0], log_mel_orig.shape[0], 0], 
                 'r-', linewidth=2, alpha=0.8, label='CNN Input (4s)')
    else:
        # Entire signal is ≤4s, outline the whole thing
        plt.plot([0, time_axis_orig[-1], time_axis_orig[-1], 0, 0], 
                 [0, 0, log_mel_orig.shape[0], log_mel_orig.shape[0], 0], 
                 'r-', linewidth=2, alpha=0.8, label='CNN Input (≤4s)')
    
    plt.colorbar(im1, format='%+2.0f dB')
    plt.title(f"{title if title else 'Log-Mel Spectrogram'} - Original")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Mel bins")
    plt.legend()
    
    # Plot CNN spectrogram (fixed 4s)
    plt.subplot(2, 1, 2)
    im2 = plt.imshow(log_mel_cnn, aspect='auto', origin='lower', 
                     extent=[0, DURATION, 0, log_mel_cnn.shape[0]], 
                     cmap='viridis', vmin=vmin, vmax=vmax)
    
    # Red box outline around entire CNN input (no fill)
    plt.plot([0, DURATION, DURATION, 0, 0], 
             [0, 0, log_mel_cnn.shape[0], log_mel_cnn.shape[0], 0], 
             'r-', linewidth=2, alpha=0.8, label='CNN Input (Fixed 4s)')
    
    plt.colorbar(im2, format='%+2.0f dB')
    plt.title("CNN Input - Fixed 4s (padded/truncated)")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Mel bins")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()

def export_sample_waveforms():
    """
    For each class (0 to 9), randomly select 2 wav files and export their waveform images.
    Images are named as sample-waveform-<class_name>-1.png and sample-waveform-<class_name>-2.png.
    """
    df = pd.read_csv(METADATA_CSV)
    processed_dir = PROCESSED_PARQUET_PATH
    os.makedirs(processed_dir, exist_ok=True)
    rng = np.random.default_rng(seed=42)
    
    for class_id in sorted(df["classID"].unique()):
        class_df = df[df["classID"] == class_id]
        if len(class_df) < 2:
            logger.warning(f"⚠️ Not enough samples for class {class_id}, skipping.")
            continue
        
        class_name = class_df.iloc[0]["class"]
        sample_rows = class_df.sample(n=2, random_state=42)
        
        for i, (_, row) in enumerate(sample_rows.iterrows(), 1):
            fold = row['fold']
            file_name = row['slice_file_name']
            rel_path = os.path.join("audio", f"fold{fold}", file_name)
            abs_path = os.path.join(DATA_ROOT, rel_path)
            
            try:
                waveform, sr = torchaudio.load(abs_path)
                out_img = os.path.join(
                    processed_dir,
                    f"sample-waveform-<{class_id}>-({class_name})-{i}.png"
                )
                plot_waveform_image(
                    waveform, sr, out_img,
                    title=f"Waveform - Class: {class_name} | File: {file_name}"
                )
                logger.info(f"🌊 Saved {out_img}")
            except Exception as e:
                logger.error(f"❌ Error processing {abs_path}: {e}")

def export_sample_spectrograms():
    """
    For each class (0 to 9), randomly select 2 wav files and export their log-mel spectrogram images.
    Images are named as sample-spectrogram-<class_name>-1.png and sample-spectrogram-<class_name>-2.png.
    """
    df = pd.read_csv(METADATA_CSV)
    processed_dir = PROCESSED_PARQUET_PATH
    os.makedirs(processed_dir, exist_ok=True)
    rng = np.random.default_rng(seed=42)
    for class_id in sorted(df["classID"].unique()):
        class_df = df[df["classID"] == class_id]
        if len(class_df) < 2:
            logger.warning(f"⚠️ Not enough samples for class {class_id}, skipping.")
            continue
        class_name = class_df.iloc[0]["class"]
        sample_rows = class_df.sample(n=2, random_state=42)
        for i, (_, row) in enumerate(sample_rows.iterrows(), 1):
            fold = row['fold']
            file_name = row['slice_file_name']
            rel_path = os.path.join("audio", f"fold{fold}", file_name)
            abs_path = os.path.join(DATA_ROOT, rel_path)
            try:
                waveform, sr = torchaudio.load(abs_path)
                out_img = os.path.join(
                    processed_dir,
                    f"sample-spectrogram-<{class_id}>-({class_name})-{i}.png"
                )
                plot_spectrogram_image(
                    waveform, sr, out_img,
                    title=f"Spectrogram - Class: {class_name} | File: {file_name}"
                )
                logger.info(f"🖼️ Saved {out_img}")
            except Exception as e:
                logger.error(f"❌ Error processing {abs_path}: {e}")

# NOTE: UrbanSound8K sample WAV files are already pre-sliced according to the 'start' and 'end' times in the metadata.
#       You do NOT need to further slice the audio using these columns; each WAV file is already the correct excerpt.


class CNNUrbanSound8KClassifier(torch.nn.Module):
    """
    1D CNN classifier for UrbanSound8K log-mel spectrograms.
    Input shape: [batch_size, n_mels, n_frames] = [B, 64, 126]
    """
    def __init__(self, n_classes=10, n_mels=64, dropout=DROPOUT):
        super(CNNUrbanSound8KClassifier, self).__init__()
        
        # Feature extraction layers (treat mel bins as channels, frames as sequence)
        self.conv_layers = torch.nn.Sequential(
            # First conv block
            torch.nn.Conv1d(n_mels, 128, kernel_size=3, padding=1),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(kernel_size=2, stride=2),  # 126 -> 63 frames
            torch.nn.Dropout(dropout),
            
            # Second conv block
            torch.nn.Conv1d(128, 256, kernel_size=3, padding=1),
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(kernel_size=2, stride=2),  # 63 -> 31 frames
            torch.nn.Dropout(dropout),
            
            # Third conv block
            torch.nn.Conv1d(256, 512, kernel_size=3, padding=1),
            torch.nn.BatchNorm1d(512),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(kernel_size=2, stride=2),  # 31 -> 15 frames
            torch.nn.Dropout(dropout),
            
            # Fourth conv block
            torch.nn.Conv1d(512, 512, kernel_size=3, padding=1),
            torch.nn.BatchNorm1d(512),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool1d(1),  # Global average pooling -> [B, 512, 1]
        )
        
        # Classification head
        self.classifier = torch.nn.Sequential(
            torch.nn.Flatten(),  # [B, 512, 1] -> [B, 512]
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(128, n_classes)
        )
    
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape [batch_size, n_mels, n_frames]
        Returns:
            logits: Output tensor of shape [batch_size, n_classes]
        """
        # Extract features
        features = self.conv_layers(x)  # [B, 512, 1]
        
        # Classify
        logits = self.classifier(features)  # [B, n_classes]
        
        return logits
    
    def get_feature_embeddings(self, x):
        """
        Extract feature embeddings before classification.
        Useful for visualization or transfer learning.
        """
        with torch.no_grad():
            features = self.conv_layers(x)
            embeddings = torch.flatten(features, 1)  # [B, 512]
        return embeddings

    def train_model(
        self,
        parquet_path=None,
        model_dir=MODEL_ROOT,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        lr=LR,
        n_classes=10,
        n_mels=N_MELS,
        device=device,
        wandb_run=None,
        run_name="urbansound8K_spectrogram_cnn_classifier"
    ):
        if parquet_path is None:
            parquet_path = get_processed_parquet_path()
        os.makedirs(model_dir, exist_ok=True)
        train_ds = UrbanSoundDataSet(parquet_path, folds=list(range(1, 9)))
        eval_ds = UrbanSoundDataSet(parquet_path, folds=[9])
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
        eval_loader = DataLoader(eval_ds, batch_size=batch_size, shuffle=False)
        model = self.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
        criterion = torch.nn.CrossEntropyLoss()
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            with tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]") as pbar:
                for xb, yb in pbar:
                    xb, yb = xb.to(device), yb.to(device)
                    optimizer.zero_grad()
                    logits = model(xb)
                    loss = criterion(logits, yb)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item() * xb.size(0)
                    pbar.set_postfix(loss=loss.item())
            avg_loss = total_loss / len(train_loader.dataset)
            logger.success(f"Epoch {epoch+1}: Train loss={avg_loss:.4f}")
            
            # Evaluate on fold 9
            eval_results = eval_or_test_cnn(
                model, eval_loader, eval_ds.df, device, n_classes, return_df=True, show_tqdm=True
            )
            # Only save eval parquet on last epoch
            if epoch == epochs - 1:
                eval_out_path = os.path.join(
                    EVAL_PARQUET_PATH,
                    f"urbansound8k_eval_cnn_fold9_{epoch+1}.parquet"
                )
                os.makedirs(os.path.dirname(eval_out_path), exist_ok=True)
                eval_results.to_parquet(eval_out_path, index=False)
                logger.success(f"Saved eval results: {eval_out_path}")
            
            # Compute metrics
            y_true = eval_results["class_id"].values
            y_pred = eval_results["predicted_class_id"].values
            eval_acc = accuracy_score(y_true, y_pred)
            eval_prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
            eval_rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
            eval_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
            
            logger.success(f"Epoch {epoch+1}: Eval Acc={eval_acc:.4f}, Prec={eval_prec:.4f}, Rec={eval_rec:.4f}, F1={eval_f1:.4f}")
            
            # Log to wandb
            if wandb_run is not None:
                wandb_run.log({
                    "epoch": epoch + 1,
                    "train_loss": avg_loss,
                    "eval_accuracy": eval_acc,
                    "eval_precision": eval_prec,
                    "eval_recall": eval_rec,
                    "eval_f1": eval_f1,
                })
        
        # Save final checkpoint only after all epochs
        final_ckpt_path = os.path.join(model_dir, get_checkpoint_filename(model_type="cnn"))
        torch.save(model.state_dict(), final_ckpt_path)
        logger.success(f"Saved final checkpoint: {final_ckpt_path}")

def train_cnn(
    parquet_path=None,
    model_dir=MODEL_ROOT,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    lr=LR,
    n_classes=10,
    n_mels=N_MELS,
    device=device
):
    if parquet_path is None:
        parquet_path = get_processed_parquet_path()
    run_name = "urbansound8K_cnn_classifier_x"
    wandb_run = None
    if WANDB_API_KEY and WANDB_ENTITY and WANDB_PROJECT:
        wandb.login(key=WANDB_API_KEY)
        wandb_run = wandb.init(
            project=WANDB_PROJECT,
            entity=WANDB_ENTITY,
            name=run_name,
            config={
                "epochs": epochs,
                "batch_size": batch_size,
                "lr": lr,
                "n_classes": n_classes,
                "n_mels": n_mels,
                "dropout": DROPOUT,
                "weight_decay": WEIGHT_DECAY,
            }
        )
    model = CNNUrbanSound8KClassifier(n_classes=n_classes, n_mels=n_mels, dropout=DROPOUT)
    model.train_model(
        parquet_path=parquet_path,
        model_dir=model_dir,
        batch_size=batch_size,
        epochs=epochs,
        lr=lr,
        n_classes=n_classes,
        n_mels=n_mels,
        device=device,
        wandb_run=wandb_run,
        run_name=run_name
    )
    if wandb_run is not None:
        wandb_run.finish()

def eval_or_test_cnn(model, loader, base_df, device, n_classes, return_df=False, show_tqdm=False):
    model.eval()
    preds = []
    all_labels = []
    iterator = loader
    if show_tqdm:
        iterator = tqdm(loader, desc="Eval/Test")
    with torch.no_grad():
        for xb, yb in iterator:
            xb = xb.to(device)
            logits = model(xb)
            pred = torch.argmax(logits, dim=1).cpu().numpy()
            preds.extend(pred)
            all_labels.extend(yb.numpy())
    out_df = base_df.copy().reset_index(drop=True)
    out_df["predicted_class_id"] = preds
    if return_df:
        return out_df
    return preds


 # CNN model that directly processes raw waveforms.
class CNNWaveformClassifier(torch.nn.Module):
    def __init__(self, n_classes=10, dropout=DROPOUT):
        super(CNNWaveformClassifier, self).__init__()
        self.conv_layers = torch.nn.Sequential(
            # Lower-level feature extraction from raw waveform
            torch.nn.Conv1d(1, 64, kernel_size=80, stride=16),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(kernel_size=4),
            
            torch.nn.Conv1d(64, 128, kernel_size=3, padding=1),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(kernel_size=4),
            
            torch.nn.Conv1d(128, 256, kernel_size=3, padding=1),
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(kernel_size=4),
            
            torch.nn.AdaptiveAvgPool1d(1),  # Global pooling
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.Flatten(),  # [B, 256, 1] -> [B, 256]
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(128, n_classes)
        )
        
    def forward(self, x):
        features = self.conv_layers(x)
        logits = self.classifier(features)
        return logits

def train_waveform_classifier():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Use folds 1-8 for training and fold 9 for validation (as in your spectrogram approach)
    train_ds = UrbanSoundRawDataset(METADATA_CSV, folds=list(range(1, 9)))
    val_ds = UrbanSoundRawDataset(METADATA_CSV, folds=[9])
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    model = CNNWaveformClassifier().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = torch.nn.CrossEntropyLoss()
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for xb, yb in tqdm(train_loader, desc=f"Train Epoch {epoch+1}"):
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
        avg_loss = total_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1}: Train Loss = {avg_loss:.4f}")
        
        # Validation step
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                preds = torch.argmax(logits, dim=1)
                correct += (preds == yb).sum().item()
                total += yb.size(0)
        print(f"Epoch {epoch+1}: Val Accuracy = {correct / total:.4f}")
    
    os.makedirs(MODEL_ROOT, exist_ok=True)
    ckpt_path = os.path.join(MODEL_ROOT, "cnn_waveform_classifier.pt")
    torch.save(model.state_dict(), ckpt_path)
    print(f"Saved model to {ckpt_path}")

def compute_detailed_metrics(y_true, y_pred, class_names=None):
    """
    Compute detailed classification metrics including per-class metrics.
    """
    if class_names is None:
        class_names = [f"Class_{i}" for i in range(10)]
    
    # Overall metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    
    # Weighted metrics (account for class imbalance)
    precision_weighted = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    recall_weighted = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    
    # Per-class metrics
    precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Classification report
    report = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)
    
    metrics = {
        "accuracy": accuracy,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "precision_weighted": precision_weighted,
        "recall_weighted": recall_weighted,
        "f1_weighted": f1_weighted,
        "precision_per_class": precision_per_class,
        "recall_per_class": recall_per_class,
        "f1_per_class": f1_per_class,
        "confusion_matrix": cm,
        "classification_report": report,
    }
    
    return metrics
def plot_confusion_matrix(y_true, y_pred, class_names=None, save_path=None, normalize=True):
    """
    Plot and save confusion matrix.
    Each row is normalized to percentages and annotated with total sample counts.
    
    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        class_names: List of class names.
        save_path: Where to save the image.
        normalize: If True, normalize each row to percentages.
    """
    if class_names is None:
        class_names = [f"Class_{i}" for i in range(10)]
    
    cm = confusion_matrix(y_true, y_pred)
    row_totals = cm.sum(axis=1)
    
    if normalize:
        cm_normalized = cm.astype('float') / row_totals[:, None] * 100
        fmt = '.1f'
    else:
        cm_normalized = cm
        fmt = 'd'
    
    # Create new y-axis labels with total counts
    class_names_with_totals = [f"{name}\n(n={total})" for name, total in zip(class_names, row_totals)]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_normalized, annot=True, fmt=fmt, cmap='Blues',
                xticklabels=class_names, yticklabels=class_names_with_totals)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix (Normalized %)' if normalize else 'Confusion Matrix')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"📊 Saved confusion matrix to {save_path}")
    
    plt.close()

def test_cnn(
    parquet_path=None,
    model_dir=MODEL_ROOT,
    n_classes=10,
    n_mels=N_MELS,
    device=device
):
    if parquet_path is None:
        parquet_path = get_processed_parquet_path()
    
    # Look for final checkpoint with hyperparameters
    final_ckpt_name = get_checkpoint_filename(model_type="cnn")
    final_ckpt_path = os.path.join(model_dir, final_ckpt_name)
    
    if os.path.exists(final_ckpt_path):
        ckpt_path = final_ckpt_path
        logger.success(f"Loading final checkpoint: {ckpt_path}")
    else:
        # Fallback to old naming pattern if new one doesn't exist
        checkpoints = [
            f for f in os.listdir(model_dir)
            if f.endswith("_urbansound8k_cnn.pt")
        ]
        if not checkpoints:
            logger.error("No checkpoints found for testing.")
            return
        last_ckpt = sorted(checkpoints, key=lambda x: int(x.split("_")[0]))[-1]
        ckpt_path = os.path.join(model_dir, last_ckpt)
        logger.success(f"Loading checkpoint: {ckpt_path}")
    
    test_ds = UrbanSoundDataSet(parquet_path, folds=[10])
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)
    model = CNNUrbanSound8KClassifier(n_classes=n_classes, n_mels=n_mels, dropout=DROPOUT).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    
    # Run test with tqdm
    test_results = eval_or_test_cnn(
        model, test_loader, test_ds.df, device, n_classes, return_df=True, show_tqdm=True
    )
    test_out_path = get_test_parquet_path()
    os.makedirs(os.path.dirname(test_out_path), exist_ok=True)
    test_results.to_parquet(test_out_path, index=False)
    logger.success(f"Saved test results: {test_out_path}")
    
    # Compute detailed metrics
    y_true = test_results["class_id"].values
    y_pred = test_results["predicted_class_id"].values
    
    # UrbanSound8K class names
    class_names = [
        "air_conditioner", "car_horn", "children_playing", "dog_bark", "drilling",
        "engine_idling", "gun_shot", "jackhammer", "siren", "street_music"
    ]
    
    detailed_metrics = compute_detailed_metrics(y_true, y_pred, class_names)
    
    # Print detailed results
    logger.success(f"🎯 Test Results:")
    logger.success(f"   Accuracy: {detailed_metrics['accuracy']:.4f}")
    logger.success(f"   Precision (macro): {detailed_metrics['precision_macro']:.4f}")
    logger.success(f"   Recall (macro): {detailed_metrics['recall_macro']:.4f}")
    logger.success(f"   F1-score (macro): {detailed_metrics['f1_macro']:.4f}")
    logger.success(f"   Precision (weighted): {detailed_metrics['precision_weighted']:.4f}")
    logger.success(f"   Recall (weighted): {detailed_metrics['recall_weighted']:.4f}")
    logger.success(f"   F1-score (weighted): {detailed_metrics['f1_weighted']:.4f}")
    
    # Print classification report
    logger.info(f"📋 Classification Report:")
    print(detailed_metrics['classification_report'])
    
    # Save confusion matrix inside Eval folder with hyperparameter info.
    cm_filename = f"confusion_matrix_{os.path.basename(ckpt_path)[:-3]}.png"
    cm_path = os.path.join(EVAL_PARQUET_PATH, cm_filename)
    plot_confusion_matrix(y_true, y_pred, class_names, save_path=cm_path, normalize=True)

    if WANDB_API_KEY and WANDB_ENTITY and WANDB_PROJECT:
        wandb.login(key=WANDB_API_KEY)
        wandb_run = wandb.init(
            project=WANDB_PROJECT,
            entity=WANDB_ENTITY,
            name="urbansound8K_cnn_classifier_test",
            config={
                "test_checkpoint": os.path.basename(ckpt_path),
                "n_classes": n_classes,
                "n_mels": n_mels,
                "dropout": DROPOUT,
                "weight_decay": WEIGHT_DECAY,
            }
        )
        
        # Log all metrics to wandb
        wandb_run.log({
            "test_accuracy": detailed_metrics['accuracy'],
            "test_precision_macro": detailed_metrics['precision_macro'],
            "test_recall_macro": detailed_metrics['recall_macro'],
            "test_f1_macro": detailed_metrics['f1_macro'],
            "test_precision_weighted": detailed_metrics['precision_weighted'],
            "test_recall_weighted": detailed_metrics['recall_weighted'],
            "test_f1_weighted": detailed_metrics['f1_weighted'],
        })
        
        # Log per-class metrics
        for i, class_name in enumerate(class_names):
            wandb_run.log({
                f"test_precision_{class_name}": detailed_metrics['precision_per_class'][i],
                f"test_recall_{class_name}": detailed_metrics['recall_per_class'][i],
                f"test_f1_{class_name}": detailed_metrics['f1_per_class'][i],
            })
        
        # Log confusion matrix as image
        wandb_run.log({"confusion_matrix": wandb.Image(cm_path)})
        
        wandb_run.finish()

class TransformerUrbanSound8KClassifier(torch.nn.Module):
    """
    Transformer encoder classifier for UrbanSound8K log-mel spectrograms.
    Input shape: [batch_size, n_mels, n_frames]
    """
    def __init__(
        self,
        n_classes=10,
        n_mels=N_MELS,
        n_frames=None,  # Optionally specify for positional embedding
        dim=TRANSFORMER_DIM,
        depth=TRANSFORMER_LAYERS,
        heads=TRANSFORMER_HEADS,
        mlp_dim=TRANSFORMER_MLP_DIM,
        dropout=TRANSFORMER_DROPOUT
    ):
        super().__init__()
        # We'll infer n_frames at runtime if not provided
        self.n_mels = n_mels
        self.dim = dim

        # Project each frame (n_mels) to transformer dim
        self.input_proj = torch.nn.Linear(n_mels, dim)

        # Positional encoding (learnable)
        self.pos_embed = None
        self.n_frames = n_frames

        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=depth)

        self.dropout = torch.nn.Dropout(dropout)
        self.norm = torch.nn.LayerNorm(dim)

        # Classification head
        self.head = torch.nn.Sequential(
            torch.nn.Linear(dim, mlp_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(mlp_dim, n_classes)
        )

    def forward(self, x):
        """
        Args:
            x: [batch_size, n_mels, n_frames]
        Returns:
            logits: [batch_size, n_classes]
        """
        # Transpose to [batch, n_frames, n_mels]
        x = x.transpose(1, 2)
        B, T, _ = x.shape

        # Project to transformer dimension
        x = self.input_proj(x)  # [B, T, dim]

        # Positional embedding (initialize if needed)
        if (self.pos_embed is None) or (self.n_frames != T):
            # Create learnable positional embedding for this sequence length
            self.n_frames = T
            self.pos_embed = torch.nn.Parameter(torch.zeros(1, T, self.dim, device=x.device))
            torch.nn.init.trunc_normal_(self.pos_embed, std=0.02)

        x = x + self.pos_embed  # [B, T, dim]
        x = self.dropout(x)

        # Transformer encoder
        x = self.encoder(x)  # [B, T, dim]
        x = self.norm(x)

        # Pooling: mean over time
        x = x.mean(dim=1)  # [B, dim]

        logits = self.head(x)  # [B, n_classes]
        return logits

    def get_feature_embeddings(self, x):
        # Returns the pooled transformer features before classification head
        x = x.transpose(1, 2)
        x = self.input_proj(x)
        if (self.pos_embed is None) or (self.n_frames != x.shape[1]):
            self.n_frames = x.shape[1]
            self.pos_embed = torch.nn.Parameter(torch.zeros(1, self.n_frames, self.dim, device=x.device))
            torch.nn.init.trunc_normal_(self.pos_embed, std=0.02)
        x = x + self.pos_embed
        x = self.dropout(x)
        x = self.encoder(x)
        x = self.norm(x)
        return x.mean(dim=1)

def train_transformer(
    parquet_path=None,
    model_dir=MODEL_ROOT,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    lr=LR,
    n_classes=10,
    n_mels=N_MELS,
    device=device
):
    if parquet_path is None:
        parquet_path = get_processed_parquet_path()
    run_name = "urbansound8K_transformer_classifier_x"
    wandb_run = None
    if WANDB_API_KEY and WANDB_ENTITY and WANDB_PROJECT:
        wandb.login(key=WANDB_API_KEY)
        wandb_run = wandb.init(
            project=WANDB_PROJECT,
            entity=WANDB_ENTITY,
            name=run_name,
            config={
                "epochs": epochs,
                "batch_size": batch_size,
                "lr": lr,
                "n_classes": n_classes,
                "n_mels": n_mels,
                "dropout": TRANSFORMER_DROPOUT,
                "weight_decay": WEIGHT_DECAY,
                "dim": TRANSFORMER_DIM,
                "depth": TRANSFORMER_LAYERS,
                "heads": TRANSFORMER_HEADS,
                "mlp_dim": TRANSFORMER_MLP_DIM,
            }
        )
    model = TransformerUrbanSound8KClassifier(
        n_classes=n_classes,
        n_mels=n_mels,
        dropout=TRANSFORMER_DROPOUT
    ).to(device)
    
    # Prepare data loaders
    train_ds = UrbanSoundDataSet(parquet_path, folds=list(range(1, 9)))
    eval_ds = UrbanSoundDataSet(parquet_path, folds=[9])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    eval_loader = DataLoader(eval_ds, batch_size=batch_size, shuffle=False)
    
    # Optimizer and criterion
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    criterion = torch.nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]") as pbar:
            for xb, yb in pbar:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * xb.size(0)
                pbar.set_postfix(loss=loss.item())
        avg_loss = total_loss / len(train_loader.dataset)
        logger.success(f"Epoch {epoch+1}: Train loss={avg_loss:.4f}")
        
        # Evaluate on fold 9
        eval_results = eval_or_test_cnn(
            model, eval_loader, eval_ds.df, device, n_classes, return_df=True, show_tqdm=True
        )
        
        # Only save eval parquet on last epoch
        if epoch == epochs - 1:
            eval_out_path = os.path.join(
                EVAL_PARQUET_PATH,
                f"urbansound8k_eval_transformer_fold9_{epoch+1}.parquet"
            )
            os.makedirs(os.path.dirname(eval_out_path), exist_ok=True)
            eval_results.to_parquet(eval_out_path, index=False)
            logger.success(f"Saved eval results: {eval_out_path}")
        
        # Compute metrics
        y_true = eval_results["class_id"].values
        y_pred = eval_results["predicted_class_id"].values
        eval_acc = accuracy_score(y_true, y_pred)
        eval_prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
        eval_rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
        eval_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        
        logger.success(f"Epoch {epoch+1}: Eval Acc={eval_acc:.4f}, Prec={eval_prec:.4f}, Rec={eval_rec:.4f}, F1={eval_f1:.4f}")
        
        # Log to wandb
        if wandb_run is not None:
            wandb_run.log({
                "epoch": epoch + 1,
                "train_loss": avg_loss,
                "eval_accuracy": eval_acc,
                "eval_precision": eval_prec,
                "eval_recall": eval_rec,
                "eval_f1": eval_f1,
            })
    
    # Save final checkpoint only after all epochs
    final_ckpt_path = os.path.join(model_dir, get_checkpoint_filename(model_type="transformer"))
    torch.save(model.state_dict(), final_ckpt_path)
    logger.success(f"Saved final checkpoint: {final_ckpt_path}")

def test_transformer(
    parquet_path=None,
    model_dir=MODEL_ROOT,
    n_classes=10,
    n_mels=N_MELS,
    device=device
):
    if parquet_path is None:
        parquet_path = get_processed_parquet_path()
    
    # Look for final checkpoint with hyperparameters
    final_ckpt_name = get_checkpoint_filename(model_type="transformer")
    final_ckpt_path = os.path.join(model_dir, final_ckpt_name)
    
    if os.path.exists(final_ckpt_path):
        ckpt_path = final_ckpt_path
        logger.success(f"Loading final checkpoint: {ckpt_path}")
    else:
        # Fallback to old naming pattern if new one doesn't exist
        checkpoints = [
            f for f in os.listdir(model_dir)
            if f.endswith("_urbansound8k_cnn.pt")
        ]
        if not checkpoints:
            logger.error("No checkpoints found for testing.")
            return
        last_ckpt = sorted(checkpoints, key=lambda x: int(x.split("_")[0]))[-1]
        ckpt_path = os.path.join(model_dir, last_ckpt)
        logger.success(f"Loading checkpoint: {ckpt_path}")
    
    test_ds = UrbanSoundDataSet(parquet_path, folds=[10])
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)
    model = TransformerUrbanSound8KClassifier(n_classes=n_classes, n_mels=n_mels, dropout=TRANSFORMER_DROPOUT).to(device)
    # Ignore unexpected keys like pos_embed when loading
    model.load_state_dict(torch.load(ckpt_path, map_location=device), strict=False)
    
    # Run test with tqdm
    test_results = eval_or_test_cnn(
        model, test_loader, test_ds.df, device, n_classes, return_df=True, show_tqdm=True
    )
    test_out_path = get_test_parquet_path()
    os.makedirs(os.path.dirname(test_out_path), exist_ok=True)
    test_results.to_parquet(test_out_path, index=False)
    logger.success(f"Saved test results: {test_out_path}")
    
    # Compute detailed metrics
    y_true = test_results["class_id"].values
    y_pred = test_results["predicted_class_id"].values
    
    # UrbanSound8K class names
    class_names = [
        "air_conditioner", "car_horn", "children_playing", "dog_bark", "drilling",
        "engine_idling", "gun_shot", "jackhammer", "siren", "street_music"
    ]
    
    detailed_metrics = compute_detailed_metrics(y_true, y_pred, class_names)
    
    # Print detailed results
    logger.success(f"🎯 Test Results:")
    logger.success(f"   Accuracy: {detailed_metrics['accuracy']:.4f}")
    logger.success(f"   Precision (macro): {detailed_metrics['precision_macro']:.4f}")
    logger.success(f"   Recall (macro): {detailed_metrics['recall_macro']:.4f}")
    logger.success(f"   F1-score (macro): {detailed_metrics['f1_macro']:.4f}")
    logger.success(f"   Precision (weighted): {detailed_metrics['precision_weighted']:.4f}")
    logger.success(f"   Recall (weighted): {detailed_metrics['recall_weighted']:.4f}")
    logger.success(f"   F1-score (weighted): {detailed_metrics['f1_weighted']:.4f}")
    
    # Print classification report
    logger.info(f"📋 Classification Report:")
    print(detailed_metrics['classification_report'])
    
    # Save confusion matrix inside Eval folder with hyperparameter info.
    cm_filename = f"confusion_matrix_{os.path.basename(ckpt_path)[:-3]}.png"
    cm_path = os.path.join(EVAL_PARQUET_PATH, cm_filename)
    plot_confusion_matrix(y_true, y_pred, class_names, save_path=cm_path, normalize=True)

    if WANDB_API_KEY and WANDB_ENTITY and WANDB_PROJECT:
        wandb.login(key=WANDB_API_KEY)
        wandb_run = wandb.init(
            project=WANDB_PROJECT,
            entity=WANDB_ENTITY,
            name="urbansound8K_transformer_classifier_test",
            config={
                "test_checkpoint": os.path.basename(ckpt_path),
                "n_classes": n_classes,
                "n_mels": n_mels,
                "dropout": TRANSFORMER_DROPOUT,
                "weight_decay": WEIGHT_DECAY,
                "dim": TRANSFORMER_DIM,
                "depth": TRANSFORMER_LAYERS,
                "heads": TRANSFORMER_HEADS,
                "mlp_dim": TRANSFORMER_MLP_DIM,
            }
        )
        
        # Log all metrics to wandb
        wandb_run.log({
            "test_accuracy": detailed_metrics['accuracy'],
            "test_precision_macro": detailed_metrics['precision_macro'],
            "test_recall_macro": detailed_metrics['recall_macro'],
            "test_f1_macro": detailed_metrics['f1_macro'],
            "test_precision_weighted": detailed_metrics['precision_weighted'],
            "test_recall_weighted": detailed_metrics['recall_weighted'],
            "test_f1_weighted": detailed_metrics['f1_weighted'],
        })
        
        # Log per-class metrics
        for i, class_name in enumerate(class_names):
            wandb_run.log({
                f"test_precision_{class_name}": detailed_metrics['precision_per_class'][i],
                f"test_recall_{class_name}": detailed_metrics['recall_per_class'][i],
                f"test_f1_{class_name}": detailed_metrics['f1_per_class'][i],
            })
        
        # Log confusion matrix as image
        wandb_run.log({"confusion_matrix": wandb.Image(cm_path)})
        
        wandb_run.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UrbanSound8K Preprocessing and Visualization")
    parser.add_argument("--preprocess", action="store_true", help="Preprocess dataset to Parquet")
    parser.add_argument("--sample-waveform", action="store_true", help="Export sample waveform images for each class")
    parser.add_argument("--sample-spectrogram", action="store_true", help="Export sample spectrogram images for each class")
    parser.add_argument("--train-cnn", action="store_true", help="Train CNN on folds 1-8, eval on fold 9")
    parser.add_argument("--test-cnn", action="store_true", help="Test CNN on fold 10 using last checkpoint")
    parser.add_argument("--train-transformer", action="store_true", help="Train Transformer on folds 1-8, eval on fold 9")
    parser.add_argument("--test-transformer", action="store_true", help="Test Transformer on fold 10 using last checkpoint")
    args = parser.parse_args()

    if args.preprocess:
        preprocess_to_parquet()
    if getattr(args, 'sample_waveform'):
        export_sample_waveforms()
    if getattr(args, 'sample_spectrogram'):
        export_sample_spectrograms()
    if getattr(args, 'train_cnn'):
        train_cnn()
    if getattr(args, 'test_cnn'):
        test_cnn()
    if getattr(args, 'train_transformer'):
        train_transformer()
    if getattr(args, 'test_transformer'):
        test_transformer()
    elif not any([args.preprocess, getattr(args, 'sample_waveform'), getattr(args, 'sample_spectrogram'), getattr(args, 'train_cnn'), getattr(args, 'test_cnn'), getattr(args, 'train_transformer'), getattr(args, 'test_transformer')]):
        logger.error("No action specified. Use --help for options.")