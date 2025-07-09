# inference.py

import os
import soundfile as sf
import torch
from dotenv import load_dotenv

from model import MusicTranscriptionModel

# --- Setup ---
load_dotenv()

# Get checkpoint and audio file from .env or use defaults
MODELS_DIR = os.getenv("MODELS_OUTPUT_DIR", "../.data/models/music2midi")
DEFAULT_CKPT = ""
if os.path.exists(MODELS_DIR):
    checkpoints = [f for f in os.listdir(MODELS_DIR) if f.endswith(".pt")]
    if checkpoints:
        DEFAULT_CKPT = os.path.join(MODELS_DIR, sorted(checkpoints)[-1]) # Get latest

INFERENCE_CHECKPOINT = os.getenv("INFERENCE_CHECKPOINT_PATH", DEFAULT_CKPT)
INFERENCE_AUDIO = os.getenv("INFERENCE_AUDIO_FILE")
SAMPLE_RATE = int(os.getenv("SAMPLE_RATE", 16000))

def main():
    """Main inference function."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if not INFERENCE_CHECKPOINT:
        print("Error: No model checkpoint found or specified in .env (INFERENCE_CHECKPOINT_PATH).")
        return
        
    if not INFERENCE_AUDIO or not os.path.exists(INFERENCE_AUDIO):
        print(f"Error: Audio file not found or specified in .env (INFERENCE_AUDIO_FILE).")
        print(f"Checked path: {INFERENCE_AUDIO}")
        return
        
    print("Loading model for inference...")
    # 1. Instantiate the full model architecture
    model = MusicTranscriptionModel()
    
    # 2. Load the fine-tuned weights for the text_decoder part
    print(f"Loading weights from checkpoint: {INFERENCE_CHECKPOINT}")
    model.text_decoder.load_state_dict(torch.load(INFERENCE_CHECKPOINT, map_location=device))
    model.to(device)
    model.eval()

    # 3. Load and preprocess the audio file
    print(f"Loading audio file: {INFERENCE_AUDIO}")
    waveform, sr = sf.read(INFERENCE_AUDIO, dtype='float32')
    if sr != SAMPLE_RATE:
        print(f"Warning: Audio sample rate ({sr}) differs from model's expected rate ({SAMPLE_RATE}). Resampling is needed.")
        # In a real app, you would resample here using torchaudio or librosa.
        # For now, we'll assume it matches.
        
    # 4. Generate the transcription
    print("\n--- Generating Transcription ---")
    generated_abc = model.generate(waveform, sampling_rate=SAMPLE_RATE)
    
    print(f"\nAudio File: {os.path.basename(INFERENCE_AUDIO)}")
    print("="*30)
    print("Predicted ABC Notation:")
    print("="*30)
    print(generated_abc)
    print("="*30)

if __name__ == "__main__":
    main()