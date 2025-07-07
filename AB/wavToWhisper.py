import whisper
import os
from pathlib import Path


def transcribe_audio(input_path):
    input_path = Path(input_path)
    output_path = input_path.with_suffix(".txt")
    model = whisper.load_model("base")  # tiny, small are fast but worse. Turbo is too long. Base is fine.

    print(f"Processing {input_path}...")
    result = model.transcribe(str(input_path), language="en")
    transcription = result["text"].strip()

    with open(output_path, "w") as f:
        f.write(f"{input_path.name}: {transcription}\n")
    print(f"Transcription saved to {output_path}")

#        for audio_file in audio_dir.glob("*.wav"):
#            print(f"Processing {audio_file.name}...")
#            transcription = result["text"].strip()
#            result = model.transcribe(str(audio_file), language="en")
#            f.write(f"{audio_file.name}: {transcription}\n")
#            print(f"Transcription saved for {audio_file.name}")


