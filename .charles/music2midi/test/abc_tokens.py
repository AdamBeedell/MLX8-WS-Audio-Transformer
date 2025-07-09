# abc_tokens.py

import os
import re
from pathlib import Path
from typing import List, Set

import music21
from tqdm import tqdm
from transformers import AutoTokenizer

# --- Configuration ---
# Directory where your MIDI files are stored.
MIDI_DIR = Path("../.data/midis")

# Directory where the new, combined tokenizer will be saved.
TOKENIZER_OUTPUT_DIR = "./qwen_with_abc_tokenizer"

# The base model whose tokenizer we are extending.
BASE_MODEL = "Qwen/Qwen3-0.6B-Base"


def get_midi_files(directory: Path) -> List[Path]:
    """Recursively finds all MIDI files (.mid, .midi) in a directory."""
    print(f"Searching for MIDI files in: {directory}")
    files = list(directory.rglob("*.mid")) + list(directory.rglob("*.midi"))
    print(f"Found {len(files)} MIDI files.")
    return files


def midi_to_abc(midi_path: Path) -> str | None:
    """
    Converts a single MIDI file to its ABC notation string representation.
    Returns None if conversion fails.
    """
    try:
        # music21 is the most robust library for format conversion.
        score = music21.converter.parse(midi_path)
        # The 'abc' format option converts the score to ABC notation.
        return music21.converter.freezeStream(score, fmt='abc')
    except Exception as e:
        # Some MIDI files can be malformed or have unsupported features.
        # It's important to handle these errors gracefully.
        print(f"Warning: Could not process {midi_path}. Error: {e}")
        return None


def extract_tokens_from_abc(abc_string: str) -> Set[str]:
    """
    Uses a regular expression to extract meaningful musical tokens from an ABC string.
    This is more robust than simply splitting by space.
    """
    # This regex is designed to capture the core components of ABC notation:
    # 1. [^\]]+\]      : Catches full chords like [CEG] or [F/2A/2c/2]
    # 2. [=^_]?[a-gA-G]['`,]*[0-9]*\/*[0-9]* : Catches notes with all variations:
    #    - [=^_]?       : Optional accidental (natural, sharp, flat)
    #    - [a-gA-G]     : Note name
    #    - ['`,]*       : Optional octave markers
    #    - [0-9]*\/*[0-9]* : Optional duration (e.g., C2, D/2, E3/2)
    # 3. "|:" or ":|" or "||" or "|" : Catches all types of bar lines
    # 4. "[^"]+"       : Catches chord symbols like "C" or "G7"
    # 5. [LMNPKRSTVw]:[^\n]* : Catches header lines like M:4/4, K:Cmaj
    pattern = re.compile(
        r'\[[^\]]+\]|[=^_]?[a-gA-G][\'`,]*[0-9]*\/*[0-9]*|\|:\||:\||\|\||\||"[^"]+"|[LMNPKRSTVw]:[^\n]*'
    )
    return set(pattern.findall(abc_string))


def create_and_save_tokenizer():
    """Main function to generate and save the combined tokenizer."""
    if not MIDI_DIR.exists():
        print(f"Error: MIDI directory not found at '{MIDI_DIR}'")
        print("Please create it and add your MIDI files.")
        return

    # 1. Load the pre-trained tokenizer from Hugging Face
    print(f"Loading base tokenizer from '{BASE_MODEL}'...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    original_vocab_size = len(tokenizer)
    print(f"Original vocabulary size: {original_vocab_size}")

    # 2. Find all unique musical tokens in our dataset
    midi_files = get_midi_files(MIDI_DIR)
    if not midi_files:
        return
        
    print("Processing MIDI files to extract unique ABC tokens...")
    all_abc_tokens: Set[str] = set()
    for midi_file in tqdm(midi_files, desc="Converting MIDI to ABC"):
        abc_data = midi_to_abc(midi_file)
        if abc_data:
            tokens = extract_tokens_from_abc(abc_data)
            all_abc_tokens.update(tokens)

    print(f"Found {len(all_abc_tokens)} unique potential tokens in the MIDI dataset.")

    # 3. Add the new tokens to the tokenizer's vocabulary
    # `add_tokens` is smart: it will only add tokens that don't already exist.
    new_tokens_added = tokenizer.add_tokens(list(all_abc_tokens))
    new_vocab_size = len(tokenizer)

    print(f"\nAdded {new_tokens_added} new tokens to the vocabulary.")
    print(f"New vocabulary size: {new_vocab_size}")

    # 4. Save the extended tokenizer
    # This saves all necessary files (tokenizer.json, vocab.json, etc.)
    # to the specified directory. This directory now contains your complete tokenizer.
    print(f"Saving combined tokenizer to '{TOKENIZER_OUTPUT_DIR}'...")
    tokenizer.save_pretrained(TOKENIZER_OUTPUT_DIR)
    print("✅ Tokenizer saved successfully!")


# --- HOW TO USE FOR TRAINING AND INFERENCE ---
# The process is the same for both training and inference because the tokenizer
# is a fixed part of your model's definition. You create it once, and then
# load it from the saved directory every time.

def load_model_and_tokenizer_for_use(model_checkpoint_path=None):
    """
    Demonstrates how to load your custom tokenizer and a model for training or inference.
    """
    print("\n--- Example: Loading Tokenizer and Model for Use ---")
    from transformers import AutoModelForCausalLM

    # ALWAYS load the tokenizer from the directory you saved it to.
    print(f"Loading custom tokenizer from: {TOKENIZER_OUTPUT_DIR}")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_OUTPUT_DIR)

    # During training, you start with the base model.
    # During inference, you would load your fine-tuned checkpoint.
    if model_checkpoint_path:
        print(f"Loading fine-tuned model checkpoint from: {model_checkpoint_path}")
        model = AutoModelForCausalLM.from_pretrained(model_checkpoint_path)
    else:
        print(f"Loading base model '{BASE_MODEL}' for initial training.")
        model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)

    # **CRITICAL STEP**: You MUST resize the model's token embeddings to match
    # the new size of your tokenizer. This creates learnable vectors for your new tokens.
    # This needs to be done *before* you load an optimizer or start training.
    # It is also necessary for inference so the model has the correct architecture.
    model.resize_token_embeddings(len(tokenizer))
    
    print(f"Model and tokenizer are ready.")
    print(f"Tokenizer vocab size: {len(tokenizer)}")
    print(f"Model embedding matrix size: {model.get_input_embeddings().weight.shape[0]}")
    
    # Now, `model` and `tokenizer` are a matched pair, ready for your training loop or inference script.
    return model, tokenizer


if __name__ == "__main__":
    create_and_save_tokenizer()
    
    # This demonstrates how you would load everything in another script.
    # In a real project, this loading logic would be in `train.py` or `inference.py`.
    load_model_and_tokenizer_for_use()