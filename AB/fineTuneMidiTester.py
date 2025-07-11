#fineTuneMidiTester.py

import whisper
import os
from pathlib import Path
import csv
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch
import torchaudio
import pandas as pd



### read dataset csv

df = pd.read_csv("mididataset.csv")

## load model

model = WhisperForConditionalGeneration.from_pretrained("./whisper-small-piano")
processor = WhisperProcessor.from_pretrained("openai/whisper-small", language="en", task="transcribe")

## for each row in dataframe evaluate with whisper, write results or store for later

midi_results = []
for _, row in df.iterrows():
    wav_path = Path(row["WavPath"])
    if not wav_path.exists():
        print(f"Missing file: {wav_path}")
        continue

    waveform, sr = torchaudio.load(str(wav_path))
    inputs = processor(waveform.squeeze().numpy(), sampling_rate=16000, return_tensors="pt")
    with torch.no_grad():
        generated_ids = model.generate(inputs["input_features"])
        transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

## write new csv

#df.to_csv("midiDatasetResults.csv")


    midi_results.append({
        "WavPath": str(wav_path),
        "Predicted": transcription,
        "Actual": row["Labels"]
    })

pd.DataFrame(midi_results).to_csv("midiDatasetResults.csv", index=False)