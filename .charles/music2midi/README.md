## Planning and AI Coding 

Based on ai.studio Learning and Co-working:

https://aistudio.google.com/prompts/1623p9JpFEWFrU0KP-6rqN62D-YLVY5ZO (Under caidong@gmail.com)


## ABC Notation Tokens

### Preprocessing (Run Once): 

You run python abc_tokens.py. This creates the ./qwen_with_abc_tokenizer directory. This directory is the permanent, final definition of your vocabulary.


### Training: 
In your train.py, you load the tokenizer from this directory and the base Qwen model. You resize the embeddings and then train the model. You save the final model weights as a checkpoint (e.g., to ./my_music_model_checkpoint).

### Inference: 

In your inference.py, you do the following:

- Load the tokenizer from the exact same directory: AutoTokenizer.from_pretrained("./qwen_with_abc_tokenizer").
- Load the fine-tuned model weights you saved during training: AutoModelForCausalLM.from_pretrained("./my_music_model_checkpoint").
The resize_token_embeddings step is implicitly handled when you load a fine-tuned checkpoint, as its architecture (including the em- bedding size) was saved with it. However, it's good practice and harmless to call it again to ensure consistency.

The tokenizer and the model checkpoint are a matched pair. The model weights for token ID 50000 (for example) were learned specifically for the token that your custom tokenizer maps to that ID. You cannot use a different tokenizer with your fine-tuned model.

## Sample MIDI files

Source Folder: check if we have "0" folder (out of 0-9, a-f) MIDI files:

```bash
find /workspace/_github/AdamBeede/MLX8-WS-Audio-Transformer/.charles/.data/hf/projectlosangeles/Monster-MIDI-Dataset/__tmp/MIDIs/0 \
    -maxdepth 1 \ 
    -type f \
    -name "*.mid" \
    | head
```

Target Folder: 

```bash
mkdir -p /workspace/_github/AdamBeede/MLX8-WS-Audio-Transformer/.charles/.data/midis
```

Copy:

```bash
find /workspace/_github/AdamBeede/MLX8-WS-Audio-Transformer/.charles/.data/hf/projectlosangeles/Monster-MIDI-Dataset/__tmp/MIDIs/0 -maxdepth 1 -type f -name "*.mid" | head -n 10000 | while read file; do cp "$file" /workspace/_github/AdamBeede/MLX8-WS-Audio-Transformer/.charles/.data/midis/; done
```



> New

The following steps were used to organize and select MIDI files based on their file size:

- All MIDI files in the source directory were listed along with their sizes and sorted in ascending order.
- The sorted list was split into 10 approximately equal-sized bins, each representing a range of file sizes.
- Individual bins (e.g., `size_bin_af`) were used to select and copy a subset of files to the target directory for further processing or analysis. This allows for sampling MIDI files by size, which can be useful for balanced dataset creation or targeted experiments.

```bash
# List all MIDI files with size, sort by size (ascending)
find /workspace/_github/AdamBeede/MLX8-WS-Audio-Transformer/.charles/.data/hf/projectlosangeles/Monster-MIDI-Dataset/__tmp/MIDIs/0   -type f -name "*.mid" -printf "%s %p\n" | sort -n > files_by_size.txt
# Count total files
total=$(wc -l < files_by_size.txt)
lines_per_bin=$(( (total + 9) / 10 ))
# Split into 10 bins
split -l $lines_per_bin files_by_size.txt size_bin_
# Each size_bin_* file now contains ~1/10th of the files, sorted by size

# Example: Copy files from a specific bin to the target directory
awk '{print $2}' size_bin_af | while read file; do cp "$file" /workspace/_github/AdamBeede/MLX8-WS-Audio-Transformer/.charles/.data/midis/; done
```

__`size_bin_ag` has MIDI files between a few Ks to 10K in size.__ using size_bin_af, which file sizes are < 10K.

To keep only the 10,000 smallest MIDI files in a folder and move the larger ones to a `./t` folder:

```bash
# Make sure you are in the directory containing the MIDI files
mkdir -p ./t
# List all MIDI files, sort by size, keep the 10,000 smallest, move the rest
find . -maxdepth 1 -type f -name "*.mid" -printf "%s %p\n" | sort -n | awk 'NR>10000 {print $2}' | xargs -I{} mv {} ./t/
```

This will:
- Sort all `.mid` files by size (ascending).
- Keep the 10,000 smallest files in the current folder.
- Move all larger files to the `./t` directory.


## MIDI 2 WAV

Download sound file from here: https://member.keymusician.com/Member/FluidR3_GM/index.html
Direct download link is: https://keymusician01.s3.amazonaws.com/FluidR3_GM.zip

```bash
 fluidsynth -ni /Users/charles/Downloads/FluidR3_GM/FluidR3_GM.sf2 /Users/charles/Downloads/alan_walker_-_the_spectre.mid -F alan_walker_-_the_spectre_fluidsynth.wav
 ```


 ## FFMpeg to split WAV file:

DO NOT USE: copy codec but not resampling with 16000Hz as our trained model:
 ```bash
ffmpeg -i alan_walker_-_the_spectre_fluidsynth.wav -f segment -segment_time 10 -c copy alan_walker_-_the_spectre_fluidsynth_%02d.wav
```

DO THIS INSTEAD: resampling to 16000 Hz
```bash
ffmpeg -i alan_walker_-_the_spectre_fluidsynth.wav -ar 16000 -f segment -segment_time 10 alan_walker_-_the_spectre_fluidsynth_%02d.wav
```
