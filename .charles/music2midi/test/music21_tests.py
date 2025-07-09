from music21 import converter

# Load your MIDI file
midi_score = converter.parse("../samples/alan_walker_-_faded.mid")

# Show the ABC notation as text
abc_string = midi_score.write('abc')
print("ABC Notation:\n", abc_string)


# If you want the ABC as a string (not a file), you can use abcExporter:
# from music21 import abcFormat

# abc_handler = abcFormat.ABCHandler()
# abc_string = abc_handler.fromStream(midi_score)
# print(abc_string)


## Play

# abc_score = converter.parse('your_file.abc')
# abc_score.show('midi')


from music21 import converter

# Write ABC string to a temporary file and play it
with open("../samples/__alan_walker_-_faded.abc", "w") as f:
    f.write(abc_string)

abc_score = converter.parse("temp.abc")
abc_score.show('midi')


midi_fp = abc_score.write('midi', fp="../samples/__alan_walker_-_faded.mid")
print(f"MIDI saved to: {midi_fp}")