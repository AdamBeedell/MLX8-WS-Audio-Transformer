## midiDatasetGen.py

import random
from mido import Message, MidiFile, MidiTrack, bpm2tempo

# Config
N = 5  # how many midi files
NOTES = [60, 62, 64, 65, 67]  # MIDI pitches (C major)
TICKS_PER_BEAT = 480  # MIDI resolution
BPM = 120
TEMPO = bpm2tempo(BPM)

# Convert seconds to ticks
def seconds_to_ticks(seconds, ticks_per_beat=TICKS_PER_BEAT, bpm=BPM):
    beats_per_second = bpm / 60.0
    return int(seconds * ticks_per_beat * beats_per_second)

for i in range(N):
    mid = MidiFile(ticks_per_beat=TICKS_PER_BEAT)
    track = MidiTrack()
    mid.tracks.append(track)

    # Starting point
    track.append(Message('program_change', program=12, time=0))

    current_tick = 0
    for _ in range(5):
        # Random wait before note
        delay_secs = 1.0 + random.uniform(0.1, 0.5)
        delay_ticks = seconds_to_ticks(delay_secs)
        pitch = random.choice(NOTES)

        # Add Note On
        track.append(Message('note_on', note=pitch, velocity=64, time=delay_ticks))
        current_tick += delay_ticks

        # Add Note Off 30 ticks later
        track.append(Message('note_off', note=pitch, velocity=64, time=30))
        current_tick += 30

    mid.save(f'random_notes_{i+1}.mid')