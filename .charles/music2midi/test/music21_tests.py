import argparse
from music21 import converter, midi
import tempfile
import os

# NOTE: For MIDI to ABC conversion, this script requires the `midi2abc` command-line tool.
# Install it on Debian/Ubuntu with: sudo apt-get install abcmidi

# def midi_to_abc(midi_path, abc_path=None):
#     midi_score = converter.parse(midi_path)
    
#     # Get ABC notation as string
#     if abc_path:
#         midi_score.write('abc', fp=abc_path)
#         with open(abc_path, "r") as f:
#             abc_string = f.read()
#     else:
#         with tempfile.NamedTemporaryFile(mode='w', suffix=".abc", delete=False) as tmp:
#             tmp_path = tmp.name
#         midi_score.write('abc', fp=tmp_path)
#         with open(tmp_path, "r") as f:
#             abc_string = f.read()
#         os.remove(tmp_path)
    
#     print("ABC Notation:\n", abc_string)
#     if abc_path:
#         print(f"ABC saved to: {abc_path}")
import subprocess
import os

def midi_to_abc_external(midi_path, abc_path=None):
    """Use external midi2abc tool for conversion.
    Requires the `midi2abc` command-line tool (from the abcmidi package).
    Install with: sudo apt-get install abcmidi
    """
    if not abc_path:
        abc_path = midi_path.replace('.mid', '.abc').replace('.midi', '.abc')
    
    try:
        # Use midi2abc command
        result = subprocess.run(['midi2abc', midi_path, '-o', abc_path], 
                              capture_output=True, text=True, check=True)
        
        with open(abc_path, 'r') as f:
            abc_string = f.read()
            
        print("ABC Notation:\n", abc_string)
        print(f"ABC saved to: {abc_path}")
        return abc_string
        
    except subprocess.CalledProcessError as e:
        print(f"Error converting MIDI to ABC: {e}")
        return None
    except FileNotFoundError:
        print("midi2abc tool not found. Install with: sudo apt-get install abcmidi")
        return None
    
def play_abc(abc_path):
    abc_score = converter.parse(abc_path)
    abc_score.show('midi')

def midi_info(midi_path):
    mf = midi.MidiFile()
    mf.open(midi_path)
    mf.read()
    mf.close()
    print(f"File: {midi_path}")
    print(f"Format: {mf.format}")
    print(f"Ticks per quarter: {mf.ticksPerQuarterNote}")
    print(f"Tracks: {len(mf.tracks)}")
    for i, track in enumerate(mf.tracks):
        print(f" Track {i}: {len(track.events)} events")
        for event in track.events[:5]:
            print(f"   {event}")
        if len(track.events) > 5:
            print("   ...")



# """
# It ends up with 4 (channels) files, not < 10 seconds as expected.
# """
# def cut_midi(midi_path, cut_length):

# score = converter.parse(midi_path)

# bpm = score.metronomeMarkBoundaries()[0][2].number  # first MetronomeMark
# step_ql = cut_length * (bpm / 60.0)

# offsets = [i * step_ql for i in range(int(score.highestTime // step_ql) + 1)]
# for idx, part in enumerate(score.sliceAtOffsets(offsets)):
#     part.write('midi', fp=f"__cut__{idx}.mid")


# # def cut_midi(midi_path, cut_length):
    # score = converter.parse(midi_path)
    
    # # Get tempo - handle case where no tempo mark exists
    # try:
    #     bpm = score.metronomeMarkBoundaries()[0][2].number
    # except (IndexError, AttributeError):
    #     # Default to 120 BPM if no tempo marking found
    #     bpm = 120.0
    
    # # Calculate quarter length for the cut duration
    # cut_ql = cut_length * (bpm / 60.0)
    
    # # Extract only the first segment of the specified length
    # cut_score = score.extractByQuarterLength(0, cut_ql)
    
    # # Write to a single output file
    # output_path = f"__cut__{cut_length}s.mid"
    # cut_score.write('midi', fp=output_path)
    # print(f"Cut MIDI saved to: {output_path}")


import copy
from music21 import converter, tempo, stream, note

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


def cut_midi(midi_path, cut_length_in_seconds):
    """
    Cuts a MIDI file to a specified length in seconds from the beginning.
    """
    print(f"Loading MIDI file: {midi_path}")
    score = converter.parse(midi_path)
    
    end_offset_ql = get_offset_for_seconds(score, float(cut_length_in_seconds))
    
    print(f"{cut_length_in_seconds} seconds corresponds to an end offset of {end_offset_ql:.2f} quarter notes.")

    # --- THIS IS THE CORRECTED AND FINAL LOGIC ---
    
    new_score = stream.Score()
    if score.metadata:
        new_score.metadata = copy.deepcopy(score.metadata)
        title = score.metadata.title or "MIDI Cut"
        new_score.metadata.title = f"{title} ({cut_length_in_seconds}s)"

    for part in score.parts:
        new_part = stream.Part(id=part.id)
        if part.getInstrument(returnDefault=False):
            new_part.insert(0, copy.deepcopy(part.getInstrument()))
        
        # 1. FLATTEN the part. This recalculates all note offsets to be
        #    global (from the beginning of the part). THIS IS THE KEY.
        flat_part = part.flatten().notesAndRests
        
        # 2. Now, filter this flattened stream by the GLOBAL offset.
        elements_in_range = flat_part.getElementsByOffset(
            offsetStart=0,
            offsetEnd=end_offset_ql
        )

        # 3. Append deep copies of the selected elements to our new part.
        #    This prevents any object identity or site issues.
        for el in elements_in_range:
            new_part.append(copy.deepcopy(el))
            
        new_score.insert(0, new_part)

    # ---------------------------------------------------------

    output_filename = f"cut_{cut_length_in_seconds}s.mid"
    new_score.write('midi', fp=output_filename)
    
    print(f"Successfully created '{output_filename}' with a duration of ~{cut_length_in_seconds} seconds.")


# working but same size
# import copy
# from music21 import converter, tempo, stream, note

# def get_offset_for_seconds(score, seconds_target):
#     """
#     Calculates the quarter-length offset that corresponds to a given
#     time in seconds, correctly handling all tempo changes.
#     """
#     mm_boundaries = score.metronomeMarkBoundaries()
#     if not mm_boundaries:
#         bpm = 120.0
#         seconds_per_quarter = 60.0 / bpm
#         target_offset = seconds_target / seconds_per_quarter
#         return target_offset

#     elapsed_seconds = 0.0
#     for i, (start_offset, end_offset, mm_obj) in enumerate(mm_boundaries):
#         bpm = mm_obj.number
#         if end_offset is None:
#             end_offset = score.duration.quarterLength

#         segment_duration_ql = end_offset - start_offset
#         seconds_per_ql_in_segment = 60.0 / bpm
#         segment_duration_seconds = segment_duration_ql * seconds_per_ql_in_segment

#         if elapsed_seconds + segment_duration_seconds >= seconds_target:
#             seconds_into_segment = seconds_target - elapsed_seconds
#             ql_into_segment = seconds_into_segment / seconds_per_ql_in_segment
#             final_offset = start_offset + ql_into_segment
#             return final_offset
        
#         elapsed_seconds += segment_duration_seconds

#     return score.duration.quarterLength


# def cut_midi(midi_path, cut_length_in_seconds):
#     """
#     Cuts a MIDI file to a specified length in seconds from the beginning.
#     """
#     print(f"Loading MIDI file: {midi_path}")
#     score = converter.parse(midi_path)
    
#     end_offset_ql = get_offset_for_seconds(score, float(cut_length_in_seconds))
    
#     print(f"{cut_length_in_seconds} seconds corresponds to an end offset of {end_offset_ql:.2f} quarter notes.")

#     # -------------------------------------------------------------------
#     # --- THIS IS THE CORRECTED SECTION ---
#     # -------------------------------------------------------------------
#     # 1. Get an iterator for all elements within the desired offset range.
#     cut_iterator = score.getElementsByOffset(offsetStart=0, offsetEnd=end_offset_ql)

#     # 2. Convert the iterator back into a new Stream object.
#     cut_score = cut_iterator.stream()
#     # -------------------------------------------------------------------

#     # Handle metadata
#     if score.metadata:
#         cut_score.metadata = copy.deepcopy(score.metadata)
#         title = score.metadata.title or "MIDI Cut"
#         cut_score.metadata.title = f"{title} ({cut_length_in_seconds}s)"

#     # Write the output file
#     output_filename = f"cut_{cut_length_in_seconds}s.mid"
#     cut_score.write('midi', fp=output_filename)
    
#     print(f"Successfully created '{output_filename}' with a duration of ~{cut_length_in_seconds} seconds.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--2abc", metavar="MIDI", help="Convert MIDI to ABC notation and print/save")
    parser.add_argument("--playabc", metavar="ABC", help="Play ABC file as MIDI")
    parser.add_argument("--info", metavar="MIDI", help="Show MIDI file metadata")
    #parser.add_argument("--out", metavar="ABC", help="Output ABC file path (used with --2abc)")
    parser.add_argument("--cut", nargs=2, metavar=("MIDI_FILE", "SECONDS"), help="Cut the MIDI file into segments of SECONDS length")
                        
    args = parser.parse_args()

    if args.info:
        midi_info(args.info)
    if args.__dict__["2abc"]:
        midi_to_abc_external(args.__dict__["2abc"], getattr(args, "out", None))
    if args.playabc:
        play_abc(args.playabc)
    if args.cut:
        midi_file, seconds = args.cut
        try:
            cut_length = float(seconds)
            cut_midi(midi_file, cut_length)
        except ValueError:
            print("Error: SECONDS must be a number.")


# from music21 import converter

# # Load your MIDI file
# midi_score = converter.parse("../samples/alan_walker_-_faded.mid")

# # Show the ABC notation as text
# abc_string = midi_score.write('abc')
# print("ABC Notation:\n", abc_string)


# # If you want the ABC as a string (not a file), you can use abcExporter:
# # from music21 import abcFormat

# # abc_handler = abcFormat.ABCHandler()
# # abc_string = abc_handler.fromStream(midi_score)
# # print(abc_string)


# ## Play

# # abc_score = converter.parse('your_file.abc')
# # abc_score.show('midi')


# from music21 import converter

# # Write ABC string to a temporary file and play it
# with open("../samples/__alan_walker_-_faded.abc", "w") as f:
#     f.write(abc_string)
#     f.write(abc_string)
