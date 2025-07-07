import os
import pandas as pd
import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset
from dotenv import load_dotenv
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt

import logging
import colorlog

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
            mel = mel_spectrogram(waveform)
            log_mel = torch.log(mel + 1e-6)
            # Store as numpy array for Parquet
            records.append({
                "rel_path": rel_path,
                "fold": fold,
                "class_id": class_id,
                "class_name": class_name,
                "log_mel": log_mel.squeeze(0).numpy().astype(np.float32),  # shape: [n_mels, time]
            })
        except Exception as e:
            logger.error(f"❌ Error processing {abs_path}: {e}")
    out_df = pd.DataFrame(records)
    # Parquet can't store arrays directly, so use list or serialize
    out_df["log_mel"] = out_df["log_mel"].apply(lambda x: x.tolist())
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
        log_mel = np.array(row["log_mel"], dtype=np.float32)
        # shape: [n_mels, time]
        label = int(row["class_id"])
        return torch.tensor(log_mel), label

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

def export_sample_spectrograms():
    """
    For each class (0 to 9), randomly select 2 wav files and export their log-mel spectrogram images.
    Images are named as sample-<class_name>-1.png and sample-<class_name>-2.png.
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
                    f"sample-<{class_id}>-({class_name})-{i}.png"
                )
                plot_spectrogram_image(
                    waveform, sr, out_img,
                    title=f"Class: {class_name} | File: {file_name}"
                )
                logger.info(f"🖼️ Saved {out_img}")
            except Exception as e:
                logger.error(f"❌ Error processing {abs_path}: {e}")

# NOTE: UrbanSound8K sample WAV files are already pre-sliced according to the 'start' and 'end' times in the metadata.
#       You do NOT need to further slice the audio using these columns; each WAV file is already the correct excerpt.

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UrbanSound8K Preprocessing and Visualization")
    parser.add_argument("--preprocess", action="store_true", help="Preprocess dataset to Parquet")
    parser.add_argument("--sampledata", action="store_true", help="Export sample spectrogram images for each fold")
    args = parser.parse_args()

    if args.preprocess:
        preprocess_to_parquet()
    if args.sampledata:
        export_sample_spectrograms()
