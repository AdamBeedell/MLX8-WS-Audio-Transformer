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
import json
import tempfile

import colorlog
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf
from dotenv import load_dotenv
from tqdm import tqdm
from transformers import AutoTokenizer
from tokenizers import Tokenizer, models, trainers, pre_tokenizers

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
BPE_VOCAB_SIZE = int(os.getenv("BPE_VOCAB_SIZE", 2000))

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
    """
    Uses a regular expression to extract meaningful musical tokens from an ABC string.
    This is more robust than simply splitting by space.
    
    The regex captures:
    1. [^\]]+\]      : Full chords like [CEG] or [F/2A/2c/2]
    2. [=^_]?[a-gA-G]['`,]*[0-9]*\/*[0-9]* : Notes with all variations:
       - [=^_]?       : Optional accidental (natural, sharp, flat)
       - [a-gA-G]     : Note name
       - ['`,]*       : Optional octave markers
       - [0-9]*\/*[0-9]* : Optional duration (e.g., C2, D/2, E3/2)
    3. \|:\||\|:\|\|\|\| : All types of bar lines
    4. "[^"]+"       : Chord symbols like "C" or "G7"
    5. [LMNPKRSTVw]:[^\n]* : Header lines like M:4/4, K:Cmaj
    """
    pattern = re.compile(
        r'\[[^\]]+\]|[=^_]?[a-gA-G][\'`,]*[0-9]*\/*[0-9]*|\|:\||:\||\|\||\||"[^"]+"|[LMNPKRSTVw]:[^\n]*'
    )
    tokens = set(pattern.findall(str(abc_string)))
    
    # Filter out problematic header tokens that contain file paths or non-musical data: 
    # e.g. 'T: from ../.data/preprocessed/chopped_midis/052058ffa01a225995931694f729ee7a.mid' 
    # during chopping midi to DURATION phase
    filtered_tokens = set()
    for token in tokens:
        # Skip title headers with file paths
        if token.startswith('T:') and ('/' in token or '\\' in token or '.mid' in token):
            continue
        # Skip other headers that might contain file system references
        if any(unwanted in token.lower() for unwanted in ['.mid', '.midi', 'chopped_midis', 'preprocessed', '../']):
            continue
        # Keep all other tokens
        filtered_tokens.add(token)
    
    return filtered_tokens

def extract_abc_metadata(abc_string: str) -> dict:
    """Extract musical metadata from ABC notation string."""
    metadata = {
        "tempo_bpm": None,
        "key_signature": None,
        "time_signature": None,
        "title": None,
        "token_count": 0
    }
    
    lines = abc_string.split('\n')
    for line in lines:
        line = line.strip()
        if line.startswith('Q:'):
            # Extract tempo (Q:120 or Q:1/4=120)
            tempo_match = re.search(r'(\d+)$', line)
            if tempo_match:
                metadata["tempo_bpm"] = float(tempo_match.group(1))
        elif line.startswith('K:'):
            # Extract key signature
            key_part = line[2:].strip()
            metadata["key_signature"] = key_part if key_part else "C"
        elif line.startswith('M:'):
            # Extract time signature
            time_part = line[2:].strip()
            metadata["time_signature"] = time_part if time_part else "4/4"
        elif line.startswith('T:'):
            # Extract title
            title_part = line[2:].strip()
            metadata["title"] = title_part
    
    # Count tokens
    tokens = extract_tokens_from_abc(abc_string)
    metadata["token_count"] = len(tokens)
    
    return metadata

def calculate_audio_duration(waveform, sample_rate: int) -> float:
    """Calculate duration in seconds from waveform."""
    return len(waveform) / sample_rate

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

# Store raw tokens from ABC files, not BPE training, not saved into QWEN3 tokenizer
def gentokens_raw():
    logger.info("🔤 STEP 3: Starting tokenizer generation...")
    if not ABC_FILES_DIR.exists() or not any(ABC_FILES_DIR.iterdir()):
        logger.error(f"ABC files directory not found. Run --midi2abc first.")
        return
    
    abc_files = list(ABC_FILES_DIR.glob("*.abc"))
    if not abc_files:
        logger.error(f"No ABC files found in {ABC_FILES_DIR}. Run --midi2abc first.")
        return
        
    logger.info(f"Found {len(abc_files)} ABC files to process.")
    
    logger.info(f"Loading base tokenizer from '{QWEN_MODEL}'...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL, trust_remote_code=True)
        original_vocab_size = len(tokenizer)
        logger.info(f"Original vocabulary size: {original_vocab_size}")
    except Exception as e:
        logger.error(f"Failed to load tokenizer from {QWEN_MODEL}: {e}")
        return
    
    all_abc_tokens: Set[str] = set()
    processed_files = 0
    
    for abc_file in tqdm(abc_files, desc="Extracting tokens"):
        try:
            abc_data = abc_file.read_text(encoding='utf-8')
            tokens = extract_tokens_from_abc(abc_data)
            all_abc_tokens.update(tokens)
            processed_files += 1
        except Exception as e:
            logger.warning(f"Could not process {abc_file.name}: {e}")
    
    logger.info(f"Processed {processed_files}/{len(abc_files)} ABC files successfully.")
    logger.info(f"Found {len(all_abc_tokens)} unique ABC tokens in the dataset.")
    
    # Save raw tokens as JSON
    raw_tokens_json_path = TOKENIZER_OUTPUT_DIR / "raw_abc_tokens.json"
    TOKENIZER_OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    with open(raw_tokens_json_path, 'w', encoding='utf-8') as f:
        json.dump(sorted(list(all_abc_tokens)), f, indent=2, ensure_ascii=False)
    logger.info(f"Raw tokens saved to: {raw_tokens_json_path}")
    
    if not all_abc_tokens:
        logger.error("No ABC tokens extracted. Check your ABC files.")
        return
    
    logger.success(f"✅ Raw token extraction complete. Found {len(all_abc_tokens)} unique tokens.")
    logger.info(f"   Raw tokens saved to: {raw_tokens_json_path}")
    logger.info(f"   Note: Tokenizer was NOT modified - only raw tokens extracted.")

def gentokens_with_bpe(target_vocab_size: int = None):
    """STEP 3: Generate tokenizer with BPE training for controlled vocabulary size."""
    if target_vocab_size is None:
        target_vocab_size = BPE_VOCAB_SIZE
        
    logger.info("🔤 STEP 3: Starting BPE tokenizer generation...")
    
    if not ABC_FILES_DIR.exists() or not any(ABC_FILES_DIR.iterdir()):
        logger.error(f"ABC files directory not found. Run --midi2abc first.")
        return
    
    abc_files = list(ABC_FILES_DIR.glob("*.abc"))
    if not abc_files:
        logger.error(f"No ABC files found in {ABC_FILES_DIR}. Run --midi2abc first.")
        return
        
    logger.info(f"Found {len(abc_files)} ABC files for BPE training.")
    
    # 1. Extract raw tokens first (same as original gentokens)
    logger.info("Extracting raw ABC tokens...")
    all_abc_tokens: Set[str] = set()
    processed_files = 0
    
    for abc_file in tqdm(abc_files, desc="Extracting raw tokens"):
        try:
            abc_data = abc_file.read_text(encoding='utf-8')
            tokens = extract_tokens_from_abc(abc_data)
            all_abc_tokens.update(tokens)
            processed_files += 1
        except Exception as e:
            logger.warning(f"Could not process {abc_file.name}: {e}")
    
    logger.info(f"Processed {processed_files}/{len(abc_files)} ABC files successfully.")
    logger.info(f"Found {len(all_abc_tokens)} unique raw ABC tokens in the dataset.")
    
    # Save raw tokens as JSON
    TOKENIZER_OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    raw_tokens_json_path = TOKENIZER_OUTPUT_DIR / "raw_abc_tokens.json"
    with open(raw_tokens_json_path, 'w', encoding='utf-8') as f:
        json.dump(sorted(list(all_abc_tokens)), f, indent=2, ensure_ascii=False)
    logger.info(f"Raw tokens saved to: {raw_tokens_json_path}")
    
    # 2. Load base tokenizer
    logger.info(f"Loading base tokenizer from '{QWEN_MODEL}'...")
    try:
        base_tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL, trust_remote_code=True)
        original_vocab_size = len(base_tokenizer)
        logger.info(f"Original vocabulary size: {original_vocab_size}")
    except Exception as e:
        logger.error(f"Failed to load tokenizer from {QWEN_MODEL}: {e}")
        return
    
    # 3. Initialize BPE tokenizer
    logger.info(f"Training BPE tokenizer with target vocab size: {target_vocab_size}")
    bpe_tokenizer = Tokenizer(models.BPE())
    bpe_tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    
    trainer = trainers.BpeTrainer(
        vocab_size=target_vocab_size,
        special_tokens=["<abc_start>", "<abc_end>", "<abc_pad>"],
        min_frequency=2,
        show_progress=True
    )
    
    # 4. Train on ABC files directly
    abc_file_paths = [str(f) for f in abc_files]
    logger.info("Training BPE on ABC corpus...")
    bpe_tokenizer.train(abc_file_paths, trainer)
    
    # 5. Extract learned BPE tokens
    bpe_vocab = bpe_tokenizer.get_vocab()
    special_tokens = {"<abc_start>", "<abc_end>", "<abc_pad>"}
    abc_bpe_tokens = [token for token in bpe_vocab.keys() if token not in special_tokens]
    
    logger.info(f"BPE training produced {len(abc_bpe_tokens)} ABC subword tokens")
    
    # Save BPE tokens as JSON
    bpe_tokens_json_path = TOKENIZER_OUTPUT_DIR / "bpe_abc_tokens.json"
    bpe_token_data = {
        "vocab_size": target_vocab_size,
        "special_tokens": list(special_tokens),
        "bpe_tokens": sorted(abc_bpe_tokens),
        "bpe_vocab_with_ids": bpe_vocab
    }
    with open(bpe_tokens_json_path, 'w', encoding='utf-8') as f:
        json.dump(bpe_token_data, f, indent=2, ensure_ascii=False)
    logger.info(f"BPE tokens saved to: {bpe_tokens_json_path}")
    
    # 6. Add BPE tokens to base tokenizer
    new_tokens_added = base_tokenizer.add_tokens(abc_bpe_tokens)
    new_vocab_size = len(base_tokenizer)
    
    logger.info(f"Added {new_tokens_added} BPE tokens to vocabulary.")
    logger.info(f"Final vocabulary size: {new_vocab_size} (was {original_vocab_size})")
    
    # 7. Save tokenizer
    try:
        base_tokenizer.save_pretrained(TOKENIZER_OUTPUT_DIR)
        
        # Also save the BPE tokenizer separately
        bpe_tokenizer_path = TOKENIZER_OUTPUT_DIR / "bpe_tokenizer.json"
        bpe_tokenizer.save(str(bpe_tokenizer_path))
        
        logger.success(f"✅ BPE tokenizer saved to '{TOKENIZER_OUTPUT_DIR}'")
        logger.info(f"   - Combined tokenizer: tokenizer.json")
        logger.info(f"   - BPE-only tokenizer: bpe_tokenizer.json")
        logger.info(f"   - Raw tokens: raw_abc_tokens.json")
        logger.info(f"   - BPE tokens: bpe_abc_tokens.json")
    except Exception as e:
        logger.error(f"Failed to save tokenizer: {e}")

def genparquet():
    logger.info("📦 STEP 4: Starting enhanced Parquet file creation...")
    if not RENDERED_WAV_DIR.exists() or not ABC_FILES_DIR.exists():
        logger.error("WAV or ABC directories not found. Run --midi2wav and --midi2abc first.")
        return
    
    abc_map = {p.stem: p for p in ABC_FILES_DIR.glob("*.abc")}
    wav_files = list(RENDERED_WAV_DIR.glob("*.wav"))

    if PARQUET_OUTPUT_PATH.exists():
        os.remove(PARQUET_OUTPUT_PATH)

    # Pre-define consistent schema to avoid mismatches
    schema = pa.schema([
        ('filename', pa.string()),
        ('waveform', pa.list_(pa.float32())),  # Simplified: 1D array of floats
        ('abc_string', pa.string()),
        ('duration_seconds', pa.float64()),
        ('sampling_rate', pa.int64()),
        ('tempo_bpm', pa.float64()),  # Always float64, even if null
        ('key_signature', pa.string()),  # Always string, even if null
        ('time_signature', pa.string()),  # Always string, even if null
        ('title', pa.string()),  # Always string, even if null
        ('token_count', pa.int64()),
        ('num_channels', pa.int64()),
        ('processing_success', pa.bool_()),
        ('chunk_duration_target', pa.float64())
    ])

    processed_count = 0
    failed_count = 0
    batch_size = 100  # Process in batches for better performance
    records_batch = []
    
    def write_batch(records_list):
        """Write a batch of records efficiently."""
        if not records_list:
            return
        
        # Ensure all records have consistent types and handle nulls properly
        for record in records_list:
            # Convert waveform to 1D list and handle None case
            if record.get('waveform') is not None:
                waveform = record['waveform']
                if hasattr(waveform, 'flatten'):  # numpy array
                    record['waveform'] = waveform.flatten().astype('float32').tolist()
                elif isinstance(waveform, list) and len(waveform) > 0 and isinstance(waveform[0], list):
                    # Already nested list, flatten it
                    record['waveform'] = [item for sublist in waveform for item in sublist]
            
            # Ensure nullable fields have proper defaults
            record['tempo_bpm'] = record.get('tempo_bpm') or None
            record['key_signature'] = record.get('key_signature') or None
            record['time_signature'] = record.get('time_signature') or None
            record['title'] = record.get('title') or None
        
        df_batch = pd.DataFrame(records_list)
        table_batch = pa.Table.from_pandas(df_batch, schema=schema)
        return table_batch
    
    writer = pq.ParquetWriter(PARQUET_OUTPUT_PATH, schema)
    
    for wav_file in tqdm(wav_files, desc="Assembling enhanced dataset"):
        file_stem = wav_file.stem
        if file_stem in abc_map:
            try:
                # Load audio data
                waveform, sr = sf.read(wav_file, dtype='float32')
                audio_duration = calculate_audio_duration(waveform, sr)
                
                # Load and process ABC data
                abc_string = abc_map[file_stem].read_text(encoding='utf-8')
                abc_metadata = extract_abc_metadata(abc_string)
                
                # Create enhanced record with consistent types
                record = {
                    "filename": file_stem,
                    "waveform": waveform,  # Will be processed in write_batch
                    "abc_string": abc_string,
                    "duration_seconds": float(audio_duration),
                    "sampling_rate": int(sr),
                    "tempo_bpm": float(abc_metadata["tempo_bpm"]) if abc_metadata["tempo_bpm"] is not None else None,
                    "key_signature": str(abc_metadata["key_signature"]) if abc_metadata["key_signature"] is not None else None,
                    "time_signature": str(abc_metadata["time_signature"]) if abc_metadata["time_signature"] is not None else None,
                    "title": str(abc_metadata["title"]) if abc_metadata["title"] is not None else None,
                    "token_count": int(abc_metadata["token_count"]),
                    "num_channels": int(1 if len(waveform.shape) == 1 else waveform.shape[1]),
                    "processing_success": True,
                    "chunk_duration_target": float(CHUNK_DURATION)
                }
                
                records_batch.append(record)
                processed_count += 1
                
            except Exception as e:
                logger.warning(f"Could not process pair for {file_stem}: {e}")
                failed_count += 1
                
                # Add failed record with consistent schema
                failed_record = {
                    "filename": file_stem,
                    "waveform": None,
                    "abc_string": None,
                    "duration_seconds": None,
                    "sampling_rate": None,
                    "tempo_bpm": None,
                    "key_signature": None,
                    "time_signature": None,
                    "title": None,
                    "token_count": 0,
                    "num_channels": None,
                    "processing_success": False,
                    "chunk_duration_target": float(CHUNK_DURATION)
                }
                records_batch.append(failed_record)
        
        # Write batch when it reaches batch_size
        if len(records_batch) >= batch_size:
            try:
                table_batch = write_batch(records_batch)
                writer.write_table(table_batch)
                records_batch = []  # Reset batch
            except Exception as e:
                logger.error(f"Failed to write batch: {e}")
                records_batch = []  # Reset batch to continue
    
    # Write remaining records in final batch
    if records_batch:
        try:
            table_batch = write_batch(records_batch)
            writer.write_table(table_batch)
        except Exception as e:
            logger.error(f"Failed to write final batch: {e}")
    
    writer.close()
    
    if processed_count > 0 or failed_count > 0:
        logger.success(f"✅ Created enhanced Parquet file with {processed_count} successful entries")
        if failed_count > 0:
            logger.warning(f"   {failed_count} entries failed processing but were recorded")
        logger.info(f"   Enhanced schema includes: filename, duration, tempo, key, time signature, token count")
        logger.info(f"   Dataset saved to: {PARQUET_OUTPUT_PATH}")
        
        # Log some statistics
        try:
            df_stats = pd.read_parquet(PARQUET_OUTPUT_PATH)
            successful_entries = df_stats[df_stats['processing_success'] == True]
            if len(successful_entries) > 0:
                avg_duration = successful_entries['duration_seconds'].mean()
                avg_tokens = successful_entries['token_count'].mean()
                logger.info(f"   Average duration: {avg_duration:.2f} seconds")
                logger.info(f"   Average token count: {avg_tokens:.1f}")
                if successful_entries['tempo_bpm'].notna().any():
                    avg_tempo = successful_entries['tempo_bpm'].mean()
                    logger.info(f"   Average tempo: {avg_tempo:.1f} BPM")
        except Exception as e:
            logger.warning(f"Could not compute dataset statistics: {e}")
    else:
        logger.error("No data was written to the Parquet file.")

def main():
    parser = argparse.ArgumentParser(description="Music2MIDI Preprocessing Pipeline")
    parser.add_argument("--midi2wav", action="store_true", help="STEP 1: Converts all raw MIDIs into chopped WAV files.")
    parser.add_argument("--midi2abc", action="store_true", help="STEP 2: Creates corresponding chopped ABC notation for each MIDI.")
    parser.add_argument("--gentokens-raw", action="store_true", help="STEP 3: Builds the custom tokenizer from the ABC files.")
    parser.add_argument("--gentokens-bpe", action="store_true", help="STEP 3: Builds the custom BPE tokenizer from the ABC files.")
    parser.add_argument("--genparquet", action="store_true", help="STEP 4: Creates the final Parquet dataset from WAVs and ABCs.")
    
    args = parser.parse_args()

    if not MIDI_DIR.exists() or not any(MIDI_DIR.iterdir()):
        logger.error(f"MIDI directory '{MIDI_DIR}' is empty or does not exist.")
        return
    if not Path(SOUNDFONT_PATH).exists():
        logger.error(f"SoundFont file not found at '{SOUNDFONT_PATH}'")
        return

    if args.midi2wav:
        midi2wav()
    if args.midi2abc:
        midi2abc()
    if args.gentokens_raw:
        gentokens_raw()
    if args.gentokens_bpe:
        gentokens_with_bpe()
    if args.genparquet:
        genparquet()
    
    if not any(vars(args).values()):
        logger.info("No steps specified. Available options:")
        logger.info("  --midi2wav      : Convert MIDI to WAV files")
        logger.info("  --midi2abc      : Convert MIDI to ABC notation")
        logger.info("  --gentokens-raw : Extract raw tokens (no BPE)")
        logger.info("  --gentokens-bpe : Train BPE tokenizer")
        logger.info("  --genparquet    : Create final dataset")

if __name__ == "__main__":
    # This setup is critical for music21 in a multiprocessing context on some systems
    music21.environment.set('autoDownload', 'allow')
    main()