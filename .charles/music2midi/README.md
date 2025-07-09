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

