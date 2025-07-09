from midi2audio import FluidSynth
fs = FluidSynth() # Replace with your SoundFont
fs.midi_to_audio('./samples/alan_walker_-_faded.mid', './samples/alan_walker_-_faded_output.wav') # Output to WAV or other formats
