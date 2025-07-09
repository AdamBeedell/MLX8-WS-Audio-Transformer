# model.py

import os
import torch
import torch.nn as nn
from dotenv import load_dotenv
from transformers import WhisperProcessor, WhisperModel, AutoTokenizer, AutoModelForCausalLM

load_dotenv()

WHISPER_MODEL_NAME = os.getenv("WHISPER_MODEL", "openai/whisper-base")
QWEN_MODEL_NAME = os.getenv("QWEN_MODEL", "Qwen/Qwen3-0.6B-Base")
TOKENIZER_DIR = os.path.join(os.getenv("PREPROCESSED_DATA_DIR", "../.data/preprocessed"), "qwen_with_abc_tokenizer")
TOP_K_QWEN_LAYERS = int(os.getenv("TOP_K_QWEN_LAYERS", 4))

class WhisperAudioEncoder(nn.Module):
    """
    A frozen Whisper Encoder to extract rich audio features.
    This is Tower 1 of our model.
    """
    def __init__(self, model_name=WHISPER_MODEL_NAME, freeze_encoder=True):
        super().__init__()
        print(f"Loading Whisper model: {model_name}")
        self.processor = WhisperProcessor.from_pretrained(model_name)
        whisper_model = WhisperModel.from_pretrained(model_name)
        self.encoder = whisper_model.get_encoder()

        if freeze_encoder:
            print("Freezing Whisper encoder weights.")
            for param in self.encoder.parameters():
                param.requires_grad = False
        
        self.encoder.eval() # Set encoder to evaluation mode

    def forward(self, waveform: torch.Tensor, sampling_rate: int) -> torch.Tensor:
        """
        Takes a raw audio waveform and returns a sequence of hidden states.
        """
        # The processor handles spectrogram conversion, padding, and normalization.
        inputs = self.processor(
            waveform, 
            sampling_rate=sampling_rate, 
            return_tensors="pt"
        )
        input_features = inputs.input_features.to(self.encoder.device)
        
        # We don't need gradients for the frozen encoder.
        with torch.no_grad():
            encoder_outputs = self.encoder(input_features)
            
        return encoder_outputs.last_hidden_state

class MusicTranscriptionModel(nn.Module):
    """
    The main Encoder-Decoder model for music transcription.
    """
    def __init__(self):
        super().__init__()
        
        # Tower 1: The Audio Encoder
        self.audio_encoder = WhisperAudioEncoder()
        
        # Tower 2: The Text Decoder
        print(f"Loading custom ABC tokenizer from: {TOKENIZER_DIR}")
        self.tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR, trust_remote_code=True)
        
        print(f"Loading Qwen model: {QWEN_MODEL_NAME}")
        self.text_decoder = AutoModelForCausalLM.from_pretrained(
            QWEN_MODEL_NAME,
            trust_remote_code=True,
            torch_dtype="auto" # Use auto for best performance (e.g., bfloat16 on Ampere)
        )
        
        # ** CRITICAL STEP **
        # Resize token embeddings to match our extended ABC vocabulary
        self.text_decoder.resize_token_embeddings(len(self.tokenizer))
        print(f"Resized Qwen token embeddings to: {len(self.tokenizer)}")

        # Freeze all Qwen parameters initially
        for param in self.text_decoder.parameters():
            param.requires_grad = False
            
        # Unfreeze the cross-attention layers and top K decoder layers
        print(f"Unfreezing the top {TOP_K_QWEN_LAYERS} Qwen decoder layers for fine-tuning.")
        for i, layer in enumerate(self.text_decoder.model.layers):
            if i >= (len(self.text_decoder.model.layers) - TOP_K_QWEN_LAYERS):
                for param in layer.parameters():
                    param.requires_grad = True
        
        # Also unfreeze the final LayerNorm and LM head
        for param in self.text_decoder.model.norm.parameters():
            param.requires_grad = True
        for param in self.text_decoder.lm_head.parameters():
            param.requires_grad = True

    def forward(self, waveform: List[np.ndarray], sampling_rate: int, 
                target_token_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        The main forward pass for training.
        """
        encoder_hidden_states = self.audio_encoder(waveform, sampling_rate)
        
        # The Qwen model becomes a decoder when `encoder_hidden_states` is provided.
        outputs = self.text_decoder(
            input_ids=target_token_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            labels=target_token_ids, # The model handles shifting labels for loss calculation
            return_dict=True
        )
        
        return outputs.loss

    @torch.no_grad()
    def generate(self, waveform: np.ndarray, sampling_rate: int, max_new_tokens=256) -> str:
        """
        Inference method to generate ABC notation from an audio waveform.
        """
        self.eval() # Set model to evaluation mode
        
        # The audio encoder expects a list of waveforms, even for a single sample.
        encoder_hidden_states = self.audio_encoder([waveform], sampling_rate)
        
        # Generate token IDs autoregressively.
        # We start generation from the BOS token.
        decoder_start_token_id = self.tokenizer.bos_token_id
        if decoder_start_token_id is None:
             decoder_start_token_id = self.tokenizer.eos_token_id

        input_ids = torch.tensor([[decoder_start_token_id]], dtype=torch.long, device=self.text_decoder.device)

        generated_ids = self.text_decoder.generate(
            input_ids=input_ids,
            encoder_hidden_states=encoder_hidden_states,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_p=0.9,
            temperature=0.7,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        )
        
        # Decode the generated token IDs into a string.
        return self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)