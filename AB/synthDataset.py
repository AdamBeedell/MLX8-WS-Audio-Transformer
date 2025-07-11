### Import stuff

import pretty_midi
import random
import pathlib
import csv
#import pyfluidsynth
import os
import soundfile as sf
import pandas as pd

### Arrange some musical notation

note_names = ['C', 'C#', 'D', 'D#', 'E', 'F',
              'F#', 'G', 'G#', 'A', 'A#', 'B']

def note_number_to_name(n):
    octave = (n // 12) - 1
    name = note_names[n % 12]
    return f"{name}{octave}"

### Init

piano_keys = list(range(21, 109))  #### Valid piano keys
all_notes = {n: note_number_to_name(n) for n in range(128)}  #### Valid midi range
tetrissoundfont = "Tetris SoundFont.sf2"  ### Sound Font, or instrument which will actually render the sound
pianosoundfont = "Mobile_0400_lite.sf2"
sample_rate = 16000  ### in sync with whisper

os.makedirs("MIDI", exist_ok=True)
os.makedirs("MIDI2WAV", exist_ok=True)

def midi_to_wav(midi, sf2_path, output_path):
    midi = pretty_midi.PrettyMIDI(midi)
    audio = midi.fluidsynth(sf2_path=sf2_path)  # path to a SoundFont file
    sf.write(output_path, audio, sample_rate)  # Matching whisper sample rate

# Save to WAV

#midi_to_wav("MIDI/random_notes_1.mid", "Tetris SoundFont.sf2")


def make_midi_dataset(N, sf2_path):

    #N = 1000  # Number of MIDI files
    NOTES = range(21,109) ### "normal" piano, bigger and smaller commonly exist though.
    VELOCITY = 100  # Volume of notes
    StartToken = "<|MIDI|>"
    EndToken = "<|/MIDI|>"
    DurationRange = [0.1, 0.12, 0.15, 0.2, 0.23, 0.3]
    GapRange = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]


    results = []

    for i in range(N):

        midi = pretty_midi.PrettyMIDI()
        instrument = pretty_midi.Instrument(program=0)  # Acoustic Grand Piano, but also default, kinda - Later .sf files tend to modify the program=0 output unless you specify otherwise
        
        current_time = 0.0

        Melody = StartToken + " "

        for _ in range(5):
            # Wait 1.0 + small random delay before next note
            current_time += random.choice(GapRange)
            pitch = random.choice(NOTES)
            duration = random.choice(DurationRange)

            note = pretty_midi.Note(velocity=VELOCITY, pitch=pitch,
                                    start=current_time, end=current_time + duration)
            instrument.notes.append(note)
            pitch = note_number_to_name(pitch)
            Melody += pitch + " "
            current_time += duration + random.choice(GapRange)

        midi.instruments.append(instrument)
        midipath = f"MIDI/random_notes_{i+1}.mid"
        wavpath = f"MIDI2WAV/random_notes_{i+1}.wav"
        midi.write(midipath)
        midi_to_wav(midipath, sf2_path, wavpath)
        Melody += EndToken
        line = {"MidiPath": midipath, 
                "WavPath": wavpath,
                "Labels": Melody
                }
        results.append(line)

    df = pd.DataFrame(results)
    df.to_csv("mididataset.csv", index=False)



make_midi_dataset(10, soundfont)

#midi = pretty_midi.PrettyMIDI("MIDI/random_notes_1.mid")
#audio = midi.fluidsynth(sf2_path="Tetris SoundFont.sf2")  # path to a SoundFont file





#datadir = pathlib.Path("MIDI") ### Loop through dir
#sf.write("MIDI2WAV/random_notes_1.wav", audio, 16000)





def piano_full_range_midi():

    piano_keys = list(range(21, 109))
    all_notes = {n: note_number_to_name(n) for n in range(128)}

    duration = 0.2  # seconds
    velocity = 100

    # Create MIDI object
    midi = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0)  # Acoustic Grand Piano

    # Build notes
    current_time = 0.0
    for note_number in piano_keys:
        note = pretty_midi.Note(
            velocity=velocity,
            pitch=note_number,
            start=current_time,
            end=current_time + duration
        )
        instrument.notes.append(note)
        current_time += duration  # no overlap or gap

    midi.instruments.append(instrument)
    midi.write("piano_full_range.mid")

piano_full_range_midi()