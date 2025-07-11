import streamlit as st
import whisper
import os
from pathlib import Path
#from streamlit_webrtc import webrtc_streamer
#import av
from audio_recorder_streamlit import audio_recorder
import tempfile


st.markdown(
    """
    <style>
    @keyframes gradientShift {
        0% {
            background-position: 0% 50%;
        }
        50% {
            background-position: 100% 50%;
        }
        100% {
            background-position: 0% 50%;
        }
    }

    .stApp {
        background: linear-gradient(-45deg, #ff9a9e, #de5833, #d95db6);
        background-size: 400% 400%;
        animation: gradientShift 30s ease infinite;
        color: white;
    }
    .stApp h1 {
        color: black !important;
        text-align: center;
        font-size: 6em;
        font-weight: bold;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.2);
    }

    </style>
    """,
    unsafe_allow_html=True
)

    OGmodel = whisper.load_model("small")


st.title("Fine Tuning Demo")

st.session_state["audiorecorded"] = False

audio_bytes = audio_recorder()
if audio_bytes:
    st.audio(audio_bytes, format="audio/wav")
    st.session_state["audiorecorded"] = True


if st.session_state["audiorecorded"] == True:
    st.balloons()
    st.session_state["audiorecorded"] = False



def OGtranscribe_audio(audio_bytes):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
        tmpfile.write(audio_bytes)
        tmpfile_path = tmpfile.name

    result = OGmodel.transcribe(tmpfile_path)

    st.markdown("### Transcription:")
    st.write(result["text"])



def FTtranscribe_audio(audio_bytes):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
        tmpfile.write(audio_bytes)
        tmpfile_path = tmpfile.name

    result = OGmodel.transcribe(tmpfile_path)

    st.markdown("### Transcription:")
    st.write(result["text"])


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



col1, col2, col3 = st.columns(3)

with col1:
    st.button("Evaluate Whisper")
    st.button("Evaluate Finetune")
    st.button("Evaluate Audience")

with col2: 
    st.button("Delete Current Audio")

with col3:
    st.button("Add to finetune dataset")
    st.button("Finetune")
    st.button("Swap in model")


    
    