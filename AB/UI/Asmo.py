import streamlit as st
import whisper
import os
from pathlib import Path


st.title("Fine Tuning Demo")

from audio_recorder_streamlit import audio_recorder

st.session_state["audiorecorded"] = False

audio_bytes = audio_recorder()
if audio_bytes:
    st.audio(audio_bytes, format="audio/wav")
    st.session_state["audiorecorded"] = True


if st.session_state["audiorecorded"] == True:
    st.balloons()
    st.session_state["audiorecorded"] = False



def transcribe_audio(input_path):
    input_path = Path(input_path)
    output_path = input_path.with_suffix(".txt")
    model = whisper.load_model("small")  # Goes tiny<base<small<medium<large<largev2<largev3??Turbo. Turbo is too long. Base is fine.

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


#wav_files = Path("Data/Memos/").glob("*.wav")
#for wav in wav_files:
#    print(wav)
#    transcribe_audio(wav)
#    with open()



#from transformers import WhisperProcessor, WhisperForConditionalGeneration
#import torchaudio

#model = WhisperForConditionalGeneration.from_pretrained("./whisper-small-hi")
#processor = WhisperProcessor.from_pretrained("openai/whisper-small", language="en", task="transcribe")



def transcribe_audio_FT(input_path, results):
    input_path = Path(input_path)
    output_path = input_path.with_suffix(".text")
    model = WhisperForConditionalGeneration.from_pretrained("./whisper-small-hi")

    print(f"Processing {input_path}...")

    # Load .wav file
    waveform, sr = torchaudio.load(str(input_path))

    # Use processor to prepare input
    inputs = processor(waveform.squeeze().numpy(), sampling_rate=16000, return_tensors="pt")

    # Generate tokens
    with torch.no_grad():
        generated_ids = model.generate(inputs["input_features"])

    # Decode tokens to text
    transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()


    with open(output_path, "w") as f:
        f.write(f"{input_path.name}: {transcription}\n")
    print(f"Transcription saved to {output_path}")

    line = {"Path": input_path, "Transcription": transcription, "Actual": "Asmoranomardicadaistinaculdacar"}
    results.append(line)

#        for audio_file in audio_dir.glob("*.wav"):
#            print(f"Processing {audio_file.name}...")
#            transcription = result["text"].strip()
#            result = model.transcribe(str(audio_file), language="en")
#            f.write(f"{audio_file.name}: {transcription}\n")
#            print(f"Transcription saved for {audio_file.name}")

#wav_files = Path("Data/Memos/").glob("*.wav")
#results = []
#for wav in wav_files:
#    print(wav)
#    transcribe_audio_FT(wav, results)

#import pandas as pd

#df = pd.DataFrame(results)
#df.to_csv("transcriptions.csv", index=False)


#for row in results:
#    txt_path = Path("Data/Memos") / Path(row["Path"]).with_suffix(".txt").name
#    if txt_path.exists():
#        with open(txt_path, "r") as f:
#            row["Previous"] = f.read().strip()
#    else:
#        row["Previous"] = ""



#df = pd.DataFrame(results)
#df["Path"] = df["Path"].astype(str)
#df.to_csv("transcriptions2.csv", index=False)



col1, col2, col3, col4, col5, col6, col7 = st.columns(7)

with col1:
    st.button("Evaluate Whisper")

with col2:
    st.button("Evaluate Finetune")

with col3: 
    st.button("Evaluate Audience")

with col4:
    st.button("Add to finetune dataset")

with col5:
    st.button("Finetune")

with col6:
    st.button("Swap in model")

with col7:
    st.button("Delete Current Audio")