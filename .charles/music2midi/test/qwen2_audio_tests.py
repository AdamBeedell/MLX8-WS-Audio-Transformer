from io import BytesIO
from urllib.request import urlopen
import librosa
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration
import sys
import torch

"""
Code from HF: https://huggingface.co/Qwen/Qwen2-Audio-7B
Downloaded original URL locally: url = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Audio/glass-breaking-151256.mp3"
"""

# device = "cuda" if torch.cuda.is_available() else "cpu"
# print(f"Using device: {device}")

# try:
#     model = Qwen2AudioForConditionalGeneration.from_pretrained("Qwen/Qwen2-Audio-7B", trust_remote_code=True)
#     model = model.to(device)
# except RuntimeError as e:
#     if "out of memory" in str(e).lower() or "cuda" in str(e).lower():
#         print("CUDA out of memory. Falling back to CPU.")
#         device = "cpu"
#         torch.cuda.empty_cache()
#         model = Qwen2AudioForConditionalGeneration.from_pretrained("Qwen/Qwen2-Audio-7B", trust_remote_code=True)
#         model = model.to(device)
#     else:
#         raise

"""
 CPU ONLY CPU ONLY - GPU will KILL Homelat :( not sure why as 7B = 14GB VRAM needed
"""

model = Qwen2AudioForConditionalGeneration.from_pretrained("Qwen/Qwen2-Audio-7B", trust_remote_code=True)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B" ,trust_remote_code=True)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        # Use hardcoded defaults if no arguments provided
        prompt = "<|audio_bos|><|AUDIO|><|audio_eos|>Generate the caption in English:"
        url = "./glass-breaking.mp3"
        
        print("Usage: uv run qwen2_audio_tests.py <prompt> <url>")
    else:
        prompt = sys.argv[1]
        url = sys.argv[2]

    print(f"Prompt: {prompt}:")
    print(f"Sound path: {url}")

    try:
        audio, sr = librosa.load(url, sr=processor.feature_extractor.sampling_rate)
        inputs = processor(text=prompt, audio=audio, return_tensors="pt", sampling_rate=sr, padding=True, truncation=True)
        # # Move input tensors to the same device as the model
        # for k in inputs:
        #     if hasattr(inputs[k], "to"):
        #         inputs[k] = inputs[k].to(device)

        generated_ids = model.generate(**inputs, max_length=256)
        generated_ids = generated_ids[:, inputs.input_ids.size(1):]
        response = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        
        print(f"Generated caption: {response}")
    except Exception as e:
        print(f"Error during audio captioning: {e}")



"""
uv run inference.py "<|audio_bos|><|AUDIO|><|audio_eos|>Generate ABC Notation Transcript: " "../samples/alan_walker_-_the_spectre_fluidsynth_01.wav"
uv run inference.py "<|audio_bos|><|AUDIO|><|audio_eos|>Generate ABC Notation Transcript: " "../samples/alan_walker_-_the_spectre_fluidsynth_00.wav"   
 uv run qwen2_audio_tests.py "<|audio_bos|><|AUDIO|><|audio_eos|>Generate ABC Notation Transcript: " "../../samples/v2/alan_walker_-_the_spectre_fluidsynth_00_15s.wav"
"""
