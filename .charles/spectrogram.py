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
from sklearn.metrics import precision_score, recall_score, accuracy_score

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
PARQUET_PATH = os.getenv("PARQUET_PATH", "./.data/UrbanSound8K/processed/urbansound8k.parquet")
SAMPLE_RATE = int(os.getenv("SAMPLE_RATE", 16000))
N_MELS = int(os.getenv("N_MELS", 64))
N_FFT = int(os.getenv("N_FFT", 1024))
HOP_LENGTH = int(os.getenv("HOP_LENGTH", 512))
FMIN = int(os.getenv("FMIN", 0))
FMAX = int(os.getenv("FMAX", 8000))
MONO = os.getenv("MONO", "True") == "True"
DURATION = float(os.getenv("DURATION", 4.0))  # seconds
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 64))
EPOCHS = int(os.getenv("EPOCHS", 10))
LR = float(os.getenv("LR", 1e-3))
DROPOUT = float(os.getenv("DROPOUT", 0.3))

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

def preprocess_to_parquet():
    """
    Preprocess UrbanSound8K audio files to log-mel spectrograms and save as Parquet.
    """
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
    os.makedirs(os.path.dirname(PARQUET_PATH), exist_ok=True)
    out_df.to_parquet(PARQUET_PATH, index=False)
    logger.success(f"✅ Saved processed dataset to {PARQUET_PATH}")

class UrbanSoundDataSet(Dataset):
    """
    PyTorch Dataset for UrbanSound8K, loading from processed Parquet file.
    """
    def __init__(self, parquet_path=PARQUET_PATH, folds=None):
        """
        Args:
            parquet_path: Path to processed Parquet file.
            folds: List of folds to include (e.g., [1,2,3,4,5,6,7,8])
        """
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

def plot_waveform_image(waveform, sr, out_path, title=None):
    """
    Plot and save a waveform image.
    """
    if MONO and waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    waveform_np = waveform.squeeze().cpu().numpy()
    time_axis = np.linspace(0, len(waveform_np) / sr, len(waveform_np))
    
    plt.figure(figsize=(12, 4))
    plt.plot(time_axis, waveform_np)
    plt.title(title if title else "Waveform")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_spectrogram_image(waveform, sr, out_path, title=None):

    if MONO and waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    # Move waveform to the same device as mel_spectrogram
    waveform = waveform.to(device)
    mel = mel_spectrogram(waveform)
    log_mel = torch.log(mel + 1e-6).squeeze().cpu().numpy()
    plt.figure(figsize=(10, 4))
    plt.imshow(log_mel, aspect='auto', origin='lower')
    plt.colorbar(format='%+2.0f dB')
    plt.title(title if title else "Log-Mel Spectrogram")
    plt.xlabel("Frames")
    plt.ylabel("Mel bins")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def export_sample_waveforms():
    """
    For each class (0 to 9), randomly select 2 wav files and export their waveform images.
    Images are named as sample-waveform-<class_name>-1.png and sample-waveform-<class_name>-2.png.
    """
    df = pd.read_csv(METADATA_CSV)
    processed_dir = os.path.dirname(PARQUET_PATH)
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
    processed_dir = os.path.dirname(PARQUET_PATH)
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
        parquet_path=PARQUET_PATH,
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
        os.makedirs(model_dir, exist_ok=True)
        train_ds = UrbanSoundDataSet(parquet_path, folds=list(range(1, 9)))
        eval_ds = UrbanSoundDataSet(parquet_path, folds=[9])
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
        eval_loader = DataLoader(eval_ds, batch_size=batch_size, shuffle=False)
        model = self.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
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
            # Save checkpoint
            ckpt_path = os.path.join(model_dir, f"{epoch+1}_urbansound8k_cnn.pt")
            torch.save(model.state_dict(), ckpt_path)
            logger.success(f"Saved checkpoint: {ckpt_path}")
            # Evaluate on fold 9
            eval_results = eval_or_test_cnn(
                model, eval_loader, eval_ds.df, device, n_classes, return_df=True, show_tqdm=True
            )
            eval_out_path = os.path.join(
                os.path.dirname(parquet_path),
                f"urbansound8k_eval_fold9_{epoch+1}.parquet"
            )
            eval_results.to_parquet(eval_out_path, index=False)
            logger.success(f"Saved eval results: {eval_out_path}")
            # Compute metrics
            y_true = eval_results["class_id"].values
            y_pred = eval_results["predicted_class_id"].values
            eval_acc = accuracy_score(y_true, y_pred)
            eval_prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
            eval_rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
            # Log to wandb
            if wandb_run is not None:
                wandb_run.log({
                    "epoch": epoch + 1,
                    "train_loss": avg_loss,
                    "eval_accuracy": eval_acc,
                    "eval_precision": eval_prec,
                    "eval_recall": eval_rec,
                })

def train_cnn(
    parquet_path=PARQUET_PATH,
    model_dir=MODEL_ROOT,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    lr=LR,
    n_classes=10,
    n_mels=N_MELS,
    device=device
):
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

def test_cnn(
    parquet_path=PARQUET_PATH,
    model_dir=MODEL_ROOT,
    n_classes=10,
    n_mels=N_MELS,
    device=device
):
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
    test_out_path = os.path.join(
        os.path.dirname(parquet_path),
        "urbansound8k_test_fold10.parquet"
    )
    test_results.to_parquet(test_out_path, index=False)
    logger.success(f"Saved test results: {test_out_path}")
    # Compute metrics
    y_true = test_results["class_id"].values
    y_pred = test_results["predicted_class_id"].values
    test_acc = accuracy_score(y_true, y_pred)
    test_prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    test_rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
    if WANDB_API_KEY and WANDB_ENTITY and WANDB_PROJECT:
        wandb.login(key=WANDB_API_KEY)
        wandb_run = wandb.init(
            project=WANDB_PROJECT,
            entity=WANDB_ENTITY,
            name="urbansound8K_cnn_classifier_test",
            config={
                "test_checkpoint": last_ckpt,
                "n_classes": n_classes,
                "n_mels": n_mels,
                "dropout": DROPOUT,
            }
        )
        wandb_run.log({
            "test_accuracy": test_acc,
            "test_precision": test_prec,
            "test_recall": test_rec,
        })
        wandb_run.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UrbanSound8K Preprocessing and Visualization")
    parser.add_argument("--preprocess", action="store_true", help="Preprocess dataset to Parquet")
    parser.add_argument("--sample-waveform", action="store_true", help="Export sample waveform images for each class")
    parser.add_argument("--sample-spectrogram", action="store_true", help="Export sample spectrogram images for each class")
    parser.add_argument("--train-cnn", action="store_true", help="Train CNN on folds 1-8, eval on fold 9")
    parser.add_argument("--test-cnn", action="store_true", help="Test CNN on fold 10 using last checkpoint")
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
    else:
        logger.error("No action specified. Use --help for options.")