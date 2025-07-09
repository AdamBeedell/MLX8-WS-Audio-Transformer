# preprocess_data.py (Scalable Version)

import os
import re
import shutil
import subprocess
from functools import partial
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Dict, List, Set, Tuple
import argparse
import logging
import colorlog

import pandas as pd
import soundfile as sf
import pyarrow as pa
import pyarrow.parquet as pq
from dotenv import load_dotenv
from tqdm import tqdm
from transformers import AutoTokenizer

import music21

# --- Setup ---
load_dotenv()

# --- Configuration (from .env) ---
MIDI_DIR = Path(os.getenv("MIDI_FILES_DIR", "../.data/midis"))
PREPROCESSED_DIR = Path(os.getenv("PREPROCESSED_DATA_DIR", "../.data/preprocessed"))
TOKENIZER_OUTPUT_DIR = PREPROCESSED_DIR / "qwen_with_abc_tokenizer"
PARQUET_OUTPUT_PATH = PREPROCESSED_DIR / "music_dataset.parquet"
SOUNDFONT_PATH = os.getenv("SOUNDFONT_PATH", "/usr/share/sounds/sf2/FluidR3_GM.sf2")
CHUNK_DURATION = float(os.getenv("CHUNK_DURATION_SECONDS", 30.0))
SAMPLE_RATE = int(os.getenv("SAMPLE_RATE", 16000))
QWEN_MODEL = os.getenv("QWEN_MODEL", "Qwen/Qwen3-0.6B-Base")
# How many files to process in memory before writing to disk
WRITE_BATCH_SIZE = 256 

# Create directories
PREPROCESSED_DIR.mkdir(parents=True, exist_ok=True)
TOKENIZER_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TEMP_AUDIO_DIR = PREPROCESSED_DIR / "temp_audio"
TEMP_AUDIO_DIR.mkdir(exist_ok=True)

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


# --- Core Worker Function (for Parallel Processing) ---
# This function remains the same as the previous version.
def process_single_midi_task(midi_path: Path, temp_dir: Path) -> Tuple[str, Dict | None]:
    try:
        # Using music21, which requires an environment setup for some systems
        import music21
        score = music21.converter.parse(midi_path)
        
        tempo = 120
        mm_boundaries = score.metronomeMarkBoundaries()
        if mm_boundaries:
            tempo = mm_boundaries[0][-1].number
        
        duration_in_seconds = min(CHUNK_DURATION, score.duration.quarterLength * 60.0 / tempo)
        chopped_score = score.getElementsByOffset(0, duration_in_seconds, includeEndBoundary=True)

        pid = os.getpid()
        temp_midi_path = temp_dir / f"temp_chopped_{pid}.mid"
        chopped_score.write('midi', fp=temp_midi_path)

        temp_wav_path = temp_dir / f"temp_audio_{pid}.wav"
        if temp_wav_path.exists():
            temp_wav_path.unlink()

        command = [
            "fluidsynth", "-ni", SOUNDFONT_PATH, str(temp_midi_path),
            "-F", str(temp_wav_path), "-r", str(SAMPLE_RATE)
        ]
        
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        waveform, sr = sf.read(temp_wav_path, dtype='float32')
        assert sr == SAMPLE_RATE

        abc_string = music21.converter.freezeStream(chopped_score, fmt='abc')

        temp_midi_path.unlink()
        temp_wav_path.unlink()
        
        result_dict = {
            "waveform": waveform, "abc_string": abc_string,
            "sampling_rate": SAMPLE_RATE, "original_file": str(midi_path.name)
        }
        return (str(midi_path), result_dict)
    except Exception:
        return (str(midi_path), None)


# --- Helper and Main Functions ---
def get_midi_files(directory: Path) -> List[Path]:
    print(f"Searching for MIDI files in: {directory}")
    files = list(directory.rglob("*.mid")) + list(directory.rglob("*.midi"))
    print(f"Found {len(files)} MIDI files.")
    return files

def extract_tokens_from_abc(abc_string: str) -> Set[str]:
    pattern = re.compile(
        r'\[[^\]]+\]|[=^_]?[a-gA-G][\'`,]*[0-9]*\/*[0-9]*|\|:\||:\||\|\||\||"[^"]+"|[LMNPKRSTVw]:[^\n]*'
    )
    return set(pattern.findall(abc_string))

def create_and_save_tokenizer(all_abc_strings: List[str]):
    # This function can still operate on the full list of strings,
    # as they are not nearly as large as the waveforms.
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


def convert_midi_to_wav(midi_path: Path) -> bool:
    try:
        score = music21.converter.parse(midi_path)
        # Calculate duration and chop if needed
        tempo = 120
        mm_boundaries = score.metronomeMarkBoundaries()
        if mm_boundaries:
            tempo = mm_boundaries[0][-1].number

        duration_in_seconds = min(CHUNK_DURATION, score.duration.quarterLength * 60.0 / tempo)
        chopped_score = score.getElementsByOffset(0, duration_in_seconds, includeEndBoundary=True)

        # Save chopped MIDI temporarily
        pid = os.getpid()
        temp_midi_path = TEMP_AUDIO_DIR / f"temp_chopped_{pid}.mid"
        chopped_score.stream().write('midi', fp=temp_midi_path)

        # Convert to WAV
        wav_output_dir = PREPROCESSED_DIR / "wav_files"
        wav_output_dir.mkdir(exist_ok=True)
        wav_output_path = wav_output_dir / f"{midi_path.stem}.wav"
        command = [
            "fluidsynth", "-ni", SOUNDFONT_PATH, str(temp_midi_path),
            "-F", str(wav_output_path), "-r", str(SAMPLE_RATE)
        ]

        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        temp_midi_path.unlink()
        return True
    except Exception as e:
        logger.error(f"Failed to convert {midi_path}: {e}")
        return False

def midi2wav():
    """Convert MIDI files to WAV files using fluidsynth."""
    logger.info("🎹 Starting MIDI to WAV conversion...")
    
    # Check prerequisites
    if not MIDI_DIR.exists() or not any(MIDI_DIR.iterdir()):
        logger.error(f"MIDI directory '{MIDI_DIR}' is empty or does not exist.")
        return False
    if not Path(SOUNDFONT_PATH).exists():
        logger.error(f"SoundFont file not found at '{SOUNDFONT_PATH}'")
        return False
    
    midi_files = get_midi_files(MIDI_DIR)
    num_workers = cpu_count()
    logger.info(f"Processing {len(midi_files)} MIDI files with {num_workers} workers...")
    
    with Pool(processes=(num_workers-2)) as pool:
        results = list(tqdm(
            pool.imap(convert_midi_to_wav, midi_files),
            total=len(midi_files),
            desc="Converting MIDI to WAV"
        ))
    
    successful = sum(results)
    logger.success(f"✅ Converted {successful}/{len(midi_files)} MIDI files to WAV.")
    return successful > 0

def midi2abc():
    """Convert MIDI files to ABC notation."""
    logger.info("🎼 Starting MIDI to ABC notation conversion...")
    
    # Check if MIDI directory exists
    if not MIDI_DIR.exists() or not any(MIDI_DIR.iterdir()):
        logger.error(f"MIDI directory '{MIDI_DIR}' is empty or does not exist.")
        return False
    
    abc_output_dir = PREPROCESSED_DIR / "abc_files"
    abc_output_dir.mkdir(exist_ok=True)
    
    midi_files = get_midi_files(MIDI_DIR)
    
    def extract_abc_from_midi(midi_path: Path) -> bool:
        try:
            score = music21.converter.parse(midi_path)
            
            # Apply same chopping logic as in wav conversion
            tempo = 120
            mm_boundaries = score.metronomeMarkBoundaries()
            if mm_boundaries:
                tempo = mm_boundaries[0][-1].number
            
            duration_in_seconds = min(CHUNK_DURATION, score.duration.quarterLength * 60.0 / tempo)
            chopped_score = score.getElementsByOffset(0, duration_in_seconds, includeEndBoundary=True)
            
            abc_string = music21.converter.freezeStream(chopped_score, fmt='abc')
            
            # Save ABC notation
            abc_output_path = abc_output_dir / f"{midi_path.stem}.abc"
            with open(abc_output_path, 'w') as f:
                f.write(abc_string)
            return True
        except Exception as e:
            logger.error(f"Failed to extract ABC from {midi_path}: {e}")
            return False
    
    num_workers = cpu_count()
    with Pool(processes=num_workers) as pool:
        results = list(tqdm(
            pool.imap(extract_abc_from_midi, midi_files),
            total=len(midi_files),
            desc="Extracting ABC notation"
        ))
    
    successful = sum(results)
    logger.success(f"✅ Extracted ABC notation for {successful}/{len(midi_files)} files.")
    return successful > 0

def gentokens():
    """Generate tokenizer from ABC files."""
    logger.info("🔤 Starting tokenizer generation...")
    
    abc_dir = PREPROCESSED_DIR / "abc_files"
    if not abc_dir.exists():
        logger.error("ABC files directory not found. Run --midi2abc first.")
        return False
    
    abc_files = list(abc_dir.glob("*.abc"))
    if not abc_files:
        logger.error("No ABC files found.")
        return False
    
    # Read all ABC strings
    all_abc_strings = []
    for abc_file in tqdm(abc_files, desc="Reading ABC files"):
        try:
            with open(abc_file, 'r') as f:
                all_abc_strings.append(f.read())
        except Exception as e:
            logger.error(f"Failed to read {abc_file}: {e}")
    
    if not all_abc_strings:
        logger.error("No ABC strings could be read.")
        return False
    
    create_and_save_tokenizer(all_abc_strings)
    logger.success("✅ Finished tokenizer generation.")
    return True

def genparquent():
    """Create Parquet file from WAV and ABC data."""
    logger.info("📦 Starting Parquet file creation...")
    
    wav_dir = PREPROCESSED_DIR / "wav_files"
    abc_dir = PREPROCESSED_DIR / "abc_files"
    
    if not wav_dir.exists() or not abc_dir.exists():
        logger.error("WAV or ABC directories not found. Run previous stages first.")
        return False
    
    wav_files = list(wav_dir.glob("*.wav"))
    abc_files = list(abc_dir.glob("*.abc"))
    
    # Create mapping
    wav_to_abc = {}
    for wav_file in wav_files:
        abc_file = abc_dir / f"{wav_file.stem}.abc"
        if abc_file in abc_files:
            wav_to_abc[wav_file] = abc_file
    
    if not wav_to_abc:
        logger.error("No matching WAV-ABC pairs found.")
        return False
    
    # Process in batches
    processed_results_batch = []
    writer = None
    
    for wav_file, abc_file in tqdm(wav_to_abc.items(), desc="Creating Parquet dataset"):
        try:
            # Read WAV
            waveform, sr = sf.read(wav_file, dtype='float32')
            assert sr == SAMPLE_RATE
            
            # Read ABC
            with open(abc_file, 'r') as f:
                abc_string = f.read()
            
            result_dict = {
                "waveform": waveform,
                "abc_string": abc_string,
                "sampling_rate": SAMPLE_RATE,
                "original_file": wav_file.name
            }
            processed_results_batch.append(result_dict)
            
            # Write batch when full
            if len(processed_results_batch) >= WRITE_BATCH_SIZE:
                df = pd.DataFrame(processed_results_batch)
                table = pa.Table.from_pandas(df)
                if writer is None:
                    writer = pq.ParquetWriter(PARQUET_OUTPUT_PATH, table.schema)
                writer.write_table(table)
                processed_results_batch = []
                
        except Exception as e:
            logger.error(f"Failed to process {wav_file}: {e}")
    
    # Write remaining batch
    if processed_results_batch:
        df = pd.DataFrame(processed_results_batch)
        table = pa.Table.from_pandas(df)
        if writer is None:
            writer = pq.ParquetWriter(PARQUET_OUTPUT_PATH, table.schema)
        writer.write_table(table)
    
    if writer:
        writer.close()
        logger.success(f"✅ Created Parquet file at '{PARQUET_OUTPUT_PATH}'")
        return True
    else:
        logger.error("No data was written to Parquet file.")
        return False

def main():
    parser = argparse.ArgumentParser(description="Music2MIDI Preprocessing Pipeline")
    parser.add_argument("--midi2wav", action="store_true", help="Converts all chopped MIDIs into WAV files.")
    parser.add_argument("--midi2abc", action="store_true", help="Creates the corresponding ABC notation for each chopped MIDI.")
    parser.add_argument("--gentokens", action="store_true", help="Collects the data, builds the final tokenizer.")
    parser.add_argument("--genparquent", action="store_true", help="Creates the Parquet file.")
    parser.add_argument("--full", action="store_true", help="Run the full pipeline (default behavior).")
    args = parser.parse_args()

    if args.midi2wav:
        midi2wav()
    elif args.midi2abc:
        midi2abc()
    elif args.gentokens:
        gentokens()
    elif args.genparquent:
        genparquent()
    elif args.full or not any([args.midi2wav, args.midi2abc, args.gentokens, args.genparquent]):
        # Remove old parquet file if it exists, as we are starting fresh
        if PARQUET_OUTPUT_PATH.exists():
            os.remove(PARQUET_OUTPUT_PATH)
            logger.info(f"Removed existing Parquet file at {PARQUET_OUTPUT_PATH}")

if __name__ == "__main__":
    main()