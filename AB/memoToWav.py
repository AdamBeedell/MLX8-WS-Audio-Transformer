### Goal here is to take voice memo-record m4a files and convert them to wav files for whisper to use
import os
import ffmpeg
import sys
from pathlib import Path


print(os.getcwd())


def convert_m4a_to_wav(input_path):
    input_path = Path(input_path)
    output_path = input_path.with_suffix(".wav")
    

    stream = (
        ffmpeg
        .input(str(input_path))
        .output(str(output_path), ar=16000, ac=1, format='wav', acodec='pcm_s16le')
        .overwrite_output()
    )
    stream.run()
    print(f"Saved: {output_path}")

## convert_m4a_to_wav("Data/Memos/Dingley Place 2.m4a")