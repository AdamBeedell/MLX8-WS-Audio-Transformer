# preprocess_data.py (Final Corrected Version)

import argparse
import logging
import os
import re
import shutil
import subprocess
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Set
import gc
import copy

import colorlog
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf
from dotenv import load_dotenv
from tqdm import tqdm
from transformers import AutoTokenizer

import music21
from music21 import converter, tempo, stream, note

import warnings
from music21.midi.translate import TranslateWarning
warnings.filterwarnings("ignore", category=TranslateWarning)



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

# --- Subdirectories for modular steps ---
CHOPPED_MIDI_DIR = PREPROCESSED_DIR / "chopped_midis"
RENDERED_WAV_DIR = PREPROCESSED_DIR / "rendered_wavs"
ABC_FILES_DIR = PREPROCESSED_DIR / "abc_files"

# --- Logger Setup ---
# (Your colorlog setup code is perfect, keeping it as is)
SUCCESS_LEVEL_NUM = 25
logging.addLevelName(SUCCESS_LEVEL_NUM, "SUCCESS")
def success(self, message, *args, **kws):
    if self.isEnabledFor(SUCCESS_LEVEL_NUM): self._log(SUCCESS_LEVEL_NUM, message, args, **kws)
logging.Logger.success = success
handler = colorlog.StreamHandler()
handler.setFormatter(colorlog.ColoredFormatter(
    "%(log_color)s%(levelname)-8s%(reset)s | %(message)s",
    log_colors={'DEBUG':'white','INFO':'cyan','SUCCESS':'green','WARNING':'yellow','ERROR':'red','CRITICAL':'bold_red'}
))
logger = colorlog.getLogger()
if not logger.handlers: logger.addHandler(handler)
logger.setLevel(logging.INFO)


def get_offset_for_seconds(score, seconds_target):
    """
    Calculates the quarter-length offset that corresponds to a given
    time in seconds, correctly handling all tempo changes.
    """
    mm_boundaries = score.metronomeMarkBoundaries()
    if not mm_boundaries:
        bpm = 120.0
        seconds_per_quarter = 60.0 / bpm
        return seconds_target / seconds_per_quarter

    elapsed_seconds = 0.0
    for start_offset, end_offset, mm_obj in mm_boundaries:
        bpm = mm_obj.number
        if end_offset is None:
            end_offset = score.duration.quarterLength

        segment_duration_ql = end_offset - start_offset
        seconds_per_ql_in_segment = 60.0 / bpm
        segment_duration_seconds = segment_duration_ql * seconds_per_ql_in_segment

        if elapsed_seconds + segment_duration_seconds >= seconds_target:
            seconds_into_segment = seconds_target - elapsed_seconds
            ql_into_segment = seconds_into_segment / seconds_per_ql_in_segment
            return start_offset + ql_into_segment
        
        elapsed_seconds += segment_duration_seconds
    return score.duration.quarterLength


def cut_midi_to_duration(score, cut_length_in_seconds):
    """
    Cuts a MIDI score to a specified length in seconds from the beginning.
    """
    end_offset_ql = get_offset_for_seconds(score, float(cut_length_in_seconds))
    
    new_score = stream.Score()
    if score.metadata:
        new_score.metadata = copy.deepcopy(score.metadata)
        title = score.metadata.title or "MIDI Cut"
        new_score.metadata.title = f"{title} ({cut_length_in_seconds}s)"

    for part in score.parts:
        new_part = stream.Part(id=part.id)
        if part.getInstrument(returnDefault=False):
            new_part.insert(0, copy.deepcopy(part.getInstrument()))
        
        # Flatten the part to get global offsets
        flat_part = part.flatten().notesAndRests
        
        # Filter by global offset
        elements_in_range = flat_part.getElementsByOffset(
            offsetStart=0,
            offsetEnd=end_offset_ql
        )

        # Append deep copies to avoid object identity issues
        for el in elements_in_range:
            new_part.append(copy.deepcopy(el))
            
        new_score.insert(0, new_part)

    return new_score


# --- Worker Functions ---
def process_and_render_task(midi_path: Path) -> bool:
    """Worker task for --midi2wav step using the new chopping logic."""
    try:
        score = music21.converter.parse(midi_path)
        chopped_score = cut_midi_to_duration(score, CHUNK_DURATION)

        output_midi_path = CHOPPED_MIDI_DIR / f"{midi_path.stem}.mid"
        chopped_score.write('midi', fp=output_midi_path)
        
        output_wav_path = RENDERED_WAV_DIR / f"{midi_path.stem}.wav"
        command = [
            "fluidsynth", "-ni", SOUNDFONT_PATH, str(output_midi_path),
            "-F", str(output_wav_path), "-r", str(SAMPLE_RATE)
        ]
        result = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        
        if result.returncode != 0:
            logger.error(f"FluidSynth failed for {midi_path.name}: {result.stderr.decode().strip()}")
            logger.error(f"Command: {' '.join(command)}")
            return False
        
        del score, chopped_score
        gc.collect()

        return True
    except Exception as e:
        logger.error(f"Failed to process {midi_path.name}: {str(e)}")
        return False


def midi_to_abc_task(midi_path: Path) -> bool:
    """Worker task for --midi2abc step using external midi2abc tool."""
    try:
        # Use external midi2abc tool for ABC conversion directly on the chopped MIDI
        abc_output_path = ABC_FILES_DIR / f"{midi_path.stem}.abc"
        
        # Run midi2abc command
        command = ['midi2abc', str(midi_path), '-o', str(abc_output_path)]
        result = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        
        if result.returncode != 0:
            logger.error(f"midi2abc failed for {midi_path.name}: {result.stderr.decode().strip()}")
            logger.error(f"Command: {' '.join(command)}")
            return False
        
        return True
    except (subprocess.CalledProcessError, FileNotFoundError, Exception) as e:
        logger.error(f"Failed to convert {midi_path.name} to ABC: {str(e)}")
        return False


# --- Main Step Functions (Unchanged from your version) ---
def get_midi_files(directory: Path):
    files = list(directory.glob("*.mid")) + list(directory.glob("*.midi")) #.rglob to look in subdirectories!
    return files

def extract_tokens_from_abc(abc_string: str) -> Set[str]:
    pattern = re.compile(
        r'\[[^\]]+\]|[=^_]?[a-gA-G][\'`,]*[0-9]*\/*[0-9]*|\|:\||:\||\|\||\||"[^"]+"|[LMNPKRSTVw]:[^\n]*'
    )
    return set(pattern.findall(str(abc_string)))

def midi2wav():
    logger.info("🎹 STEP 1: Starting MIDI to WAV conversion...")
    CHOPPED_MIDI_DIR.mkdir(exist_ok=True, parents=True)
    RENDERED_WAV_DIR.mkdir(exist_ok=True, parents=True)
    
    midi_files = get_midi_files(MIDI_DIR)
    if not midi_files:
        logger.error(f"No MIDI files found in {MIDI_DIR}. Exiting.")
        return
        
    num_workers = cpu_count() // 2
    success_count = 0
    with Pool(processes=num_workers) as pool:
        with tqdm(total=len(midi_files), desc="Rendering audio") as pbar:
            for success in pool.imap_unordered(process_and_render_task, midi_files):
                if success:
                    success_count += 1
                pbar.update()
    
    logger.success(f"✅ Finished. Successfully created {success_count}/{len(midi_files)} WAV files.")
    logger.info(f"   Chopped MIDIs saved to: {CHOPPED_MIDI_DIR}")
    logger.info(f"   Rendered WAVs saved to: {RENDERED_WAV_DIR}")

def midi2abc():
    logger.info("🎼 STEP 2: Starting MIDI to ABC notation conversion...")
    ABC_FILES_DIR.mkdir(exist_ok=True, parents=True)
    
    # Check if midi2abc tool is available
    try:
        subprocess.run(['midi2abc', '--help'], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.error("midi2abc tool not found. Install with: sudo apt-get install abcmidi")
        return
    
    # Check if chopped MIDI directory exists and has files
    if not CHOPPED_MIDI_DIR.exists() or not any(CHOPPED_MIDI_DIR.iterdir()):
        logger.error(f"No chopped MIDI files found in {CHOPPED_MIDI_DIR}. Run --midi2wav first.")
        return
    
    midi_files = get_midi_files(CHOPPED_MIDI_DIR)
    if not midi_files:
        logger.error(f"No MIDI files found in {CHOPPED_MIDI_DIR}. Run --midi2wav first.")
        return
    
    num_workers = cpu_count() // 4
    success_count = 0
    with Pool(processes=num_workers) as pool:
        with tqdm(total=len(midi_files), desc="Generating ABC") as pbar:
            for success in pool.imap_unordered(midi_to_abc_task, midi_files):
                if success:
                    success_count += 1
                pbar.update()
                
    logger.success(f"✅ Finished. Extracted ABC for {success_count}/{len(midi_files)} files.")
    logger.info(f"   ABC files saved to: {ABC_FILES_DIR}")

def gentokens():
    logger.info("🔤 STEP 3: Starting tokenizer generation...")
    if not ABC_FILES_DIR.exists() or not any(ABC_FILES_DIR.iterdir()):
        logger.error(f"ABC files directory not found. Run --midi2abc first.")
        return
    
    abc_files = list(ABC_FILES_DIR.glob("*.abc"))
    all_abc_strings = [p.read_text() for p in abc_files]
    
    logger.info(f"Loading base tokenizer from '{QWEN_MODEL}'...")
    tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL, trust_remote_code=True)
    
    all_abc_tokens: Set[str] = set()
    for abc_data in tqdm(all_abc_strings, desc="Extracting tokens"):
        tokens = extract_tokens_from_abc(abc_data)
        all_abc_tokens.update(tokens)
    
    new_tokens_added = tokenizer.add_tokens(list(all_abc_tokens))
    logger.info(f"Added {new_tokens_added} new tokens to the vocabulary.")
    
    tokenizer.save_pretrained(TOKENIZER_OUTPUT_DIR)
    logger.success(f"✅ Tokenizer saved to '{TOKENIZER_OUTPUT_DIR}'")

def genparquet():
    logger.info("📦 STEP 4: Starting Parquet file creation...")
    if not RENDERED_WAV_DIR.exists() or not ABC_FILES_DIR.exists():
        logger.error("WAV or ABC directories not found. Run --midi2wav and --midi2abc first.")
        return
    
    abc_map = {p.stem: p for p in ABC_FILES_DIR.glob("*.abc")}
    wav_files = list(RENDERED_WAV_DIR.glob("*.wav"))

    if PARQUET_OUTPUT_PATH.exists():
        os.remove(PARQUET_OUTPUT_PATH)

    writer = None
    processed_count = 0
    
    for wav_file in tqdm(wav_files, desc="Assembling dataset"):
        file_stem = wav_file.stem
        if file_stem in abc_map:
            try:
                waveform, sr = sf.read(wav_file, dtype='float32')
                abc_string = abc_map[file_stem].read_text()
                
                df = pd.DataFrame([{
                    "waveform": waveform,
                    "abc_string": abc_string,
                    "sampling_rate": sr,
                    "original_file": f"{file_stem}.mid"
                }])
                table = pa.Table.from_pandas(df)
                
                if writer is None:
                    writer = pq.ParquetWriter(PARQUET_OUTPUT_PATH, table.schema)
                writer.write_table(table)
                processed_count += 1
            except Exception as e:
                logger.warning(f"Could not process pair for {file_stem}: {e}")
    
    if writer:
        writer.close()
        logger.success(f"✅ Created Parquet file with {processed_count} entries at '{PARQUET_OUTPUT_PATH}'")
    else:
        logger.error("No data was written to the Parquet file.")

def main():
    parser = argparse.ArgumentParser(description="Music2MIDI Preprocessing Pipeline")
    parser.add_argument("--midi2wav", action="store_true", help="STEP 1: Converts all raw MIDIs into chopped WAV files.")
    parser.add_argument("--midi2abc", action="store_true", help="STEP 2: Creates corresponding chopped ABC notation for each MIDI.")
    parser.add_argument("--gentokens", action="store_true", help="STEP 3: Builds the custom tokenizer from the ABC files.")
    parser.add_argument("--genparquet", action="store_true", help="STEP 4: Creates the final Parquet dataset from WAVs and ABCs.")
    parser.add_argument("--all", action="store_true", help="Run all preprocessing steps in sequence.")
    args = parser.parse_args()

    if not MIDI_DIR.exists() or not any(MIDI_DIR.iterdir()):
        logger.error(f"MIDI directory '{MIDI_DIR}' is empty or does not exist.")
        return
    if not Path(SOUNDFONT_PATH).exists():
        logger.error(f"SoundFont file not found at '{SOUNDFONT_PATH}'")
        return

    run_all = args.all or not any(vars(args).values())

    if run_all or args.midi2wav:
        midi2wav()
    if run_all or args.midi2abc:
        midi2abc()
    if run_all or args.gentokens:
        gentokens()
    if run_all or args.genparquet:
        genparquet()
    
    if not any(vars(args).values()):
        logger.info("No steps specified. To run the full pipeline, use --all.")

if __name__ == "__main__":
    # This setup is critical for music21 in a multiprocessing context on some systems
    main()
    music21.environment.set('autoDownload', 'allow')
    main()