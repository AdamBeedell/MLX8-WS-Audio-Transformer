import transformers
from transformers import WhisperFeatureExtractor, WhisperProcessor, WhisperTokenizer

feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")
tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-small", language="en", task="transcribe")

input_str = "Asmorandamardicadaistinaculdacar"
AsmoTokens = tokenizer(input_str).input_ids
decoded_with_special = tokenizer.decode(AsmoTokens, skip_special_tokens=False)
decoded_str = tokenizer.decode(AsmoTokens, skip_special_tokens=True)

print(f"Input:                 {input_str}")
print(f"Decoded w/ special:    {decoded_with_special}")
print(f"Decoded w/out special: {decoded_str}")
print(f"Are equal:             {input_str == decoded_str}")
