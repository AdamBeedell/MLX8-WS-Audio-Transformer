# MIDI File Cutter (`music21_tests.py`)



This Python script provides a robust function to cut a MIDI file to a specified length from the beginning. It correctly handles multi-track MIDI files and files with tempo changes.

## Features

-   **Precise Time-Based Cutting**: Trims a MIDI file to a specific number of seconds.
-   **Tempo-Aware**: Accurately calculates the cut point even if the MIDI file contains multiple tempo changes.
-   **Multi-Track Support**: Correctly processes each track (Part) in a MIDI file, preserving instrument assignments.
-   **Metadata Preservation**: Copies the original file's metadata (like title) to the new cut file.

## Requirements

- For MIDI to ABC conversion, the script uses the `midi2abc` command-line tool, which is provided by the `abcmidi` package.
- Install it on Debian/Ubuntu with:
  ```bash
  sudo apt-get install abcmidi
  ```

## Usage

You can run the script from the command line.

**Example:**

To cut `alan_walker_-_faded.mid` to the first 10 seconds:
```bash
uv run music21_tests.py --cut ../../samples/alan_walker_-_faded.mid 10
```
This will generate a new file named `cut_10s.mid` in the same directory.

## MIDI to ABC Conversion Example

To convert the Alan Walker "Faded" MIDI file to ABC notation using the `midi2abc` tool:

```bash
midi2abc ../../samples/alan_walker_-_faded.mid -o alan_walker_-_faded.abc
```

This will create an `alan_walker_-_faded.abc` file with the ABC notation.

## The Final Working Code

See the `cut_midi` function in `music21_tests.py` for the implementation.

---

The rest of this README describes the key concepts and logic used in the MIDI cutting function, which is now located in `music21_tests.py`.

## How It Works: Key `music21` Concepts

Achieving this seemingly simple task required overcoming several common `music21` hurdles. The final solution is built on these key concepts:

### 1. Converting Seconds to Quarter Lengths (`get_offset_for_seconds`)

MIDI files are structured around beats and quarter lengths, not seconds. To cut by time, we must first convert our target in seconds to a quarter-length offset. This is complicated by tempo changes.

**The Solution**: We manually walk through the tempo map of the score provided by `score.metronomeMarkBoundaries()`. This function gives us a list of every tempo, its start offset, and its end offset. Our helper function `get_offset_for_seconds` iterates through these segments, calculating how many quarter lengths pass for each elapsed second until the target time is reached. This is the most robust way to handle time conversion.

### 2. Global vs. Local Offsets (The Root of the Length Problem)

The most critical issue was understanding `music21`'s offset system.
- A `Note` object inside a `Measure` has a **local offset** relative to the start of its measure (e.g., `2.0` for beat 3 in 4/4 time).
- To cut the whole piece, we need its **global offset** from the very beginning of the track.

**The Solution**: The `.flatten()` method is the key. Calling `part.flatten()` creates a new, temporary stream where all measure containers are removed. In this "flat" view, every note's `.offset` attribute now represents its **global offset**.

### 3. Filtering by Global Offset

Once we have a flattened part where offsets are global, we can use the `.getElementsByOffset()` method.

**The Solution**: `flat_part.getElementsByOffset(0, end_offset_ql)` now works as expected. It selects every note, chord, and rest from the flattened stream whose global start time falls within our desired range.

### 4. Rebuilding the Score

Simply filtering isn't enough; we must construct a valid new `Score` object.

**The Solution**: The script creates a new empty `stream.Score`, then loops through the original parts. For each original part, it creates a new empty `stream.Part`, copies the instrument, performs the flatten-and-filter operation, and appends the resulting notes into the new part. Finally, each new, cut part is inserted into the new score, which is then written to a file. Using `copy.deepcopy()` on the notes ensures that we are adding fresh objects to our new stream, preventing potential context and object identity errors.