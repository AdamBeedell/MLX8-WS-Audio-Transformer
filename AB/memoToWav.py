### Goal here is to take voice memo-record m4a files and convert them to wav files for whisper to use
import os
import ffmpeg
import sys
from pathlib import Path


print(os.getcwd())


def convert_m4a_to_wav(input_path):
    input_path = Path(input_path)
    output_path = input_path.with_suffix(".wav")
    
    try:
        (
            ffmpeg
            .input(str(input_path))
            .output(str(output_path), ar=16000, ac=1, format='wav', acodec='pcm_s16le')
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )
        print(f"✅ Converted: {input_path} → {output_path}")
    except ffmpeg.Error as e:
        print(f"❌ ffmpeg failed on {input_path}")
        print(e.stderr.decode())



## convert_m4a_to_wav("Data/Memos/Dingley Place 2.m4a")

m4a_files = Path("Data/Memos/").glob("*.m4a")
for m4a in m4a_files:
    print(m4a)
    convert_m4a_to_wav(m4a)