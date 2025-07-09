# preprocess_data.py

import os
import re
from pathlib import Path
from typing import List, Set, Dict

import music21
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from dotenv import load_dotenv
from fluidsynth import Synth
from tqdm import tqdm
from transformers import AutoTokenizer

# --- Setup ---
load_dotenv()

MIDI_DIR = Path(os.getenv("MIDI_FILES_DIR", "../.data/midis"))
PREPROCESSED_DIR = Path(os.getenv("PREPROCESSED_DATA_DIR", "../.data/preprocessed"))
TOKENIZER_OUTPUT_DIR = PREPROCESSED_DIR / "qwen_with_abc_tokenizer"
PARQUET_OUTPUT_PATH = PREPROCESSED_DIR / "music_dataset.parquet"
SOUNDFONT_PATH = os.getenv("SOUNDFONT_PATH")

CHUNK_DURATION = float(os.getenv("CHUNK_DURATION_SECONDS", 30.0))
SAMPLE_RATE = int(os.getenv("SAMPLE_RATE", 16000))
QWEN_MODEL = os.getenv("QWEN_MODEL", "Qwen/Qwen3-0.6B-Base")

# Create directories if they don't exist
PREPROCESSED_DIR.mkdir(parents=True, exist_ok=True)
TOKENIZER_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Main Functions ---

def get_midi_files(directory: Path) -> List[Path]:
    """Recursively finds all MIDI files (.mid, .midi) in a directory."""
    print(f"Searching for MIDI files in: {directory}")
    files = list(directory.rglob("*.mid")) + list(directory.rglob("*.midi"))
    print(f"Found {len(files)} MIDI files.")
    return files

def process_midi_file(midi_path: Path, fluid_synth: Synth) -> Dict | None:
    """
    Processes a single MIDI file: chops it, converts to audio and ABC notation.
    """
    try:
        # 1. Load and chop the MIDI file using music21
        score = music21.converter.parse(midi_path)
        
        # Get the duration of the first 30 seconds or the full score if shorter
        duration_in_seconds = min(CHUNK_DURATION, score.duration.quarterLength * 60.0 / score.metronomeMarkBoundaries()[0][-1].number)
        
        # music21's `getElementsByOffset` is perfect for chopping
        chopped_score = score.getElementsByOffset(0, duration_in_seconds, includeEndBoundary=True)

        # 2. Convert chopped MIDI to a temporary file for rendering
        temp_midi_path = PREPROCESSED_DIR / "temp_chopped.mid"
        chopped_score.write('midi', fp=temp_midi_path)

        # 3. Render the temporary MIDI to a WAV file using FluidSynth
        temp_wav_path = PREPROCESSED_DIR / "temp_audio.wav"
        if temp_wav_path.exists():
            temp_wav_path.unlink()
            
        fluid_synth.midi_to_audio(str(temp_midi_path), str(temp_wav_path))

        # 4. Load the audio waveform into a numpy array
        waveform, sr = sf.read(temp_wav_path, dtype='float32')
        assert sr == SAMPLE_RATE, f"Sample rate mismatch: expected {SAMPLE_RATE}, got {sr}"

        # 5. Convert the chopped score to ABC notation
        abc_string = music21.converter.freezeStream(chopped_score, fmt='abc')

        # Clean up temporary files
        temp_midi_path.unlink()
        temp_wav_path.unlink()

        return {
            "waveform": waveform,
            "abc_string": abc_string,
            "sampling_rate": SAMPLE_RATE,
            "original_file": str(midi_path.name)
        }
    except Exception as e:
        print(f"Warning: Could not process {midi_path.name}. Error: {e}")
        return None

def extract_tokens_from_abc(abc_string: str) -> Set[str]:
    """Uses a regex to extract meaningful musical tokens from an ABC string."""
    pattern = re.compile(
        r'\[[^\]]+\]|[=^_]?[a-gA-G][\'`,]*[0-9]*\/*[0-9]*|\|:\||:\||\|\||\||"[^"]+"|[LMNPKRSTVw]:[^\n]*'
    )
    return set(pattern.findall(abc_string))

def create_and_save_tokenizer(all_abc_strings: List[str]):
    """Creates and saves a custom tokenizer extended with ABC notation."""
    print(f"\nLoading base tokenizer from '{QWEN_MODEL}'...")
    tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL, trust_remote_code=True)
    
    print("Extracting unique ABC tokens from the corpus...")
    all_abc_tokens: Set[str] = set()
    for abc_data in tqdm(all_abc_strings, desc="Extracting tokens"):
        tokens = extract_tokens_from_abc(abc_data)
        all_abc_tokens.update(tokens)
    
    print(f"Found {len(all_abc_tokens)} unique potential ABC tokens.")
    
    new_tokens_added = tokenizer.add_tokens(list(all_abc_tokens))
    print(f"Added {new_tokens_added} new tokens to the vocabulary.")
    
    print(f"Saving combined tokenizer to '{TOKENIZER_OUTPUT_DIR}'...")
    tokenizer.save_pretrained(TOKENIZER_OUTPUT_DIR)
    print("✅ Tokenizer saved successfully!")

def main():
    """Main script to run the full preprocessing pipeline."""
    # Check for prerequisites
    if not MIDI_DIR.exists() or not any(MIDI_DIR.iterdir()):
        print(f"Error: MIDI directory '{MIDI_DIR}' is empty or does not exist.")
        print("Please add your .mid files to it before running.")
        return
        
    if not Path(SOUNDFONT_PATH).exists():
        print(f"Error: SoundFont file not found at '{SOUNDFONT_PATH}'")
        print("Please download a .sf2 file and update the SOUNDFONT_PATH in your .env file.")
        return

    # Initialize FluidSynth
    fs = Synth(samplerate=SAMPLE_RATE)
    fs.start()
    sfid = fs.sfload(SOUNDFONT_PATH)
    fs.program_select(0, sfid, 0, 0)
    
    # Process all MIDI files
    midi_files = get_midi_files(MIDI_DIR)
    processed_data = []
    for midi_file in tqdm(midi_files, desc="Processing MIDI files"):
        result = process_midi_file(midi_file, fs)
        if result:
            processed_data.append(result)
            
    # Shutdown FluidSynth
    fs.delete()
    
    if not processed_data:
        print("No MIDI files were successfully processed. Exiting.")
        return
        
    # Create and save the tokenizer based on the processed data
    all_abc = [item['abc_string'] for item in processed_data]
    create_and_save_tokenizer(all_abc)
    
    # Save the final dataset to a Parquet file
    print(f"\nSaving processed data to '{PARQUET_OUTPUT_PATH}'...")
    df = pd.DataFrame(processed_data)
    df.to_parquet(PARQUET_OUTPUT_PATH, index=False)
    print(f"✅ Successfully created dataset with {len(df)} entries.")

if __name__ == "__main__":
    main()