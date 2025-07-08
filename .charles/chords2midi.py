import mido
import re

"""
Raspberry Beret (LP Version)
Prince & The Revolution

"""

# The input data provided by the user
CHORD_DATA = {
    "symbol": ["N","Ab7","F#m","Gmaj7","F#m","Gmaj7","A6","Gmaj7","F#m","Gmaj7","A6","Gmaj7","F#m7","Gmaj7","A6","Gmaj7","F#m7","Gmaj7","F#m","Gmaj7","F#m","G","A6","Gmaj7","F#m","Gmaj7","A6","Gmaj7","F#m","G","F#m","Gmaj7","F#m7","E","A6","Gmaj7","F#m","Gmaj7","F#m","Gmaj7","F#m","G","A6","Gmaj7","F#m","Gmaj7","A6","Gmaj7","F#m","G","F#m","Gmaj7","F#m7","E","F#m7","G6","F#m7","Gmaj7","F#m7","E","A6","Gmaj7","F#m","G","F#m","E","A6","Gmaj7","F#m","Gmaj7","F#m","Gmaj7","A6","Gmaj7","F#m","Gmaj7","A6","N"],
    "timestamp": [0.3715192675590515,4.086711883544922,11.888616561889648,16.161088943481445,18.761722564697266,24.148752212524414,26.749387741088867,28.792743682861328,30.278820037841797,32.04353713989258,34.73705291748047,36.96616744995117,38.080726623535156,40.124080657958984,42.72471618652344,44.95383071899414,46.16127014160156,48.29750442504883,50.34086227416992,53.03437805175781,54.056053161621094,56.00653076171875,58.97868347167969,60.929161071777344,61.85795974731445,64.08707427978516,66.78058624267578,68.91682434082031,69.84562683105469,72.07473754882812,74.48961639404297,76.99736785888672,77.83329010009766,80.15528106689453,82.01287841796875,84.89215087890625,85.91383361816406,88.0500717163086,90.74358367919922,96.22349548339844,98.73124694824219,112.10594177246094,115.07809448242188,116.9356918334961,117.86448669433594,120.0936050415039,122.78711700439453,124.92335510253906,125.85215759277344,128.08126831054688,130.4961395263672,133.00390625,133.9326934814453,136.1618194580078,137.92652893066406,155.94522094726562,156.87400817871094,164.11863708496094,165.7904815673828,167.9267120361328,171.45614624023438,172.9422149658203,173.8710174560547,176.1001434326172,178.42213439941406,183.9949188232422,185.20236206054688,188.9175567626953,189.84634399414062,196.9980926513672,197.83401489257812,200.06312561035156,202.7566375732422,204.89288330078125,205.91455078125,208.05079650878906,210.83718872070312,216.22421264648438]
}

# --- 1. Chord Parser ---

# Map note names to their MIDI value relative to C
NOTE_MAP = {'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3, 'E': 4,
            'F': 5, 'F#': 6, 'Gb': 6, 'G': 7, 'G#': 8, 'Ab': 8, 'A': 9,
            'A#': 10, 'Bb': 10, 'B': 11}

# Map chord qualities to their intervals in semitones from the root
# (R, 3rd, 5th, 7th, etc.)
# For simplicity, we define the most common chord types found in the data.
INTERVAL_MAP = {
    # Major chords
    '': [0, 4, 7],  # Major triad (e.g., "G")
    'maj7': [0, 4, 7, 11], # Major seventh (e.g., "Gmaj7")
    '6': [0, 4, 7, 9], # Major sixth (e.g., "A6")
    # Minor chords
    'm': [0, 3, 7], # Minor triad (e.g., "F#m")
    'm7': [0, 3, 7, 10], # Minor seventh (e.g., "F#m7")
    # Dominant chords
    '7': [0, 4, 7, 10], # Dominant seventh (e.g., "Ab7")
}
# Special handling for G6 in the data
INTERVAL_MAP['G6'] = INTERVAL_MAP['6']


def parse_chord(chord_symbol, base_octave=4):
    """
    Parses a chord symbol string into a list of MIDI note numbers.
    
    Args:
        chord_symbol (str): The chord symbol, e.g., "F#m7", "G", "N".
        base_octave (int): The octave for the root note (Middle C is octave 4).

    Returns:
        list: A list of integer MIDI note numbers for the chord.
              Returns an empty list for "N" (no chord) or unrecognized chords.
    """
    if chord_symbol == 'N':
        return []

    # Use regex to separate the root note from the quality
    match = re.match(r'([A-G][b#]?)(\S*)', chord_symbol)
    if not match:
        print(f"Warning: Could not parse chord symbol '{chord_symbol}'")
        return []

    root_name, quality = match.groups()

    # Special case for G6 which doesn't fit the regex pattern perfectly
    if chord_symbol == 'G6':
        root_name = 'G'
        quality = '6'

    # Look up the root note's base value
    if root_name not in NOTE_MAP:
        print(f"Warning: Root note '{root_name}' not found.")
        return []
    
    # Calculate the MIDI value of the root note
    # C4 (Middle C) is MIDI note 60
    root_midi = NOTE_MAP[root_name] + 12 * (base_octave + 1)
    
    # Look up the intervals for the chord quality
    if quality not in INTERVAL_MAP:
        print(f"Warning: Chord quality '{quality}' not recognized for '{chord_symbol}'.")
        # Default to a major triad if quality is unknown but root is valid
        intervals = INTERVAL_MAP['']
    else:
        intervals = INTERVAL_MAP[quality]
        
    # Calculate the absolute MIDI notes for the chord
    return [root_midi + i for i in intervals]


# --- 2. MIDI Creator ---

def create_midi_from_chords(chord_data, output_filename="chords.mid"):
    """
    Creates a MIDI file from a dictionary of chord symbols and timestamps.

    Args:
        chord_data (dict): A dictionary with "symbol" and "timestamp" lists.
        output_filename (str): The name of the MIDI file to save.
    """
    # MIDI constants
    TICKS_PER_BEAT = 480  # Standard resolution
    DEFAULT_VELOCITY = 64 # How hard the note is played (0-127)
    DEFAULT_DURATION_S = 2.0 # Duration for the very last chord in seconds

    # Create a new MIDI file (type 1 for multiple tracks)
    mid = mido.MidiFile(type=1, ticks_per_beat=TICKS_PER_BEAT)
    track = mido.MidiTrack()
    mid.tracks.append(track)
    
    # To correctly calculate delta times, we need a sorted list of all events
    # (note_on and note_off) with their absolute times in seconds.
    events = []
    
    symbols = chord_data['symbol']
    timestamps = chord_data['timestamp']

    for i in range(len(symbols)):
        start_time = timestamps[i]
        
        # Determine the end time of the current chord
        if i + 1 < len(timestamps):
            end_time = timestamps[i+1]
        else:
            # For the last chord, give it a default duration
            end_time = start_time + DEFAULT_DURATION_S
            
        chord_symbol = symbols[i]
        midi_notes = parse_chord(chord_symbol)
        
        # If it's a valid chord, create note_on and note_off events
        if midi_notes:
            for note in midi_notes:
                # Add a (time, event_type, note) tuple
                events.append((start_time, 'note_on', note))
                events.append((end_time, 'note_off', note))

    # Sort all events chronologically
    events.sort()

    # Now, convert the timed events into MIDI messages with delta times
    last_event_time_s = 0.0
    for time_s, event_type, note in events:
        # Calculate the time difference (delta) from the last event
        delta_time_s = time_s - last_event_time_s
        
        # Convert the delta time from seconds to MIDI ticks
        # Tempo is 120 bpm by default (500000 microseconds per beat)
        delta_ticks = int(mido.second2tick(delta_time_s, TICKS_PER_BEAT, mido.bpm2tempo(120)))
        
        # Create the MIDI message and add it to the track
        track.append(mido.Message(
            event_type,
            note=note,
            velocity=DEFAULT_VELOCITY if event_type == 'note_on' else 0,
            time=delta_ticks
        ))
        
        # Update the time of the last event
        last_event_time_s = time_s

    # Save the MIDI file
    try:
        mid.save(output_filename)
        print(f"Successfully created MIDI file: {output_filename}")
    except Exception as e:
        print(f"Error saving MIDI file: {e}")


# --- 3. Main Execution Block ---

if __name__ == "__main__":
    create_midi_from_chords(CHORD_DATA, output_filename="my_chord_progression.mid")