# model.py

import os
import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional, Union
from transformers import WhisperProcessor, WhisperModel, AutoTokenizer, AutoModelForCausalLM

from logger_utils import setup_logger
logger = setup_logger(__name__)

from dotenv import load_dotenv
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
        logger.info(f"Loading Whisper model: {model_name}")
        self.processor = WhisperProcessor.from_pretrained(model_name)
        whisper_model = WhisperModel.from_pretrained(model_name)
        self.encoder = whisper_model.get_encoder()

        if freeze_encoder:
            logger.info("Freezing Whisper encoder weights.")
            for param in self.encoder.parameters():
                param.requires_grad = False
        
        self.encoder.eval() # Set encoder to evaluation mode

    def forward(self, waveforms: Union[torch.Tensor, List[np.ndarray]], sampling_rate: int) -> torch.Tensor:
        """
        Takes raw audio waveforms and returns a sequence of hidden states.
        Args:
            waveforms: Either a batch tensor or list of numpy arrays
            sampling_rate: Audio sampling rate
        """
        # Handle different input formats
        if isinstance(waveforms, list):
            # Convert list of numpy arrays to batch tensor
            batch_waveforms = []
            for wf in waveforms:
                if isinstance(wf, np.ndarray):
                    wf_tensor = torch.from_numpy(wf).float()
                else:
                    wf_tensor = wf.float()
                
                # Ensure waveform is 1D (mono)
                if len(wf_tensor.shape) > 1:
                    # If stereo or multi-channel, convert to mono by averaging
                    wf_tensor = wf_tensor.mean(dim=0)
                
                batch_waveforms.append(wf_tensor)
            
            # Find the maximum length for padding
            max_len = max(w.shape[0] for w in batch_waveforms)
            
            # Pad all waveforms to the same length
            padded_waveforms = []
            for w in batch_waveforms:
                if w.shape[0] < max_len:
                    padding_length = max_len - w.shape[0]
                    padding = torch.zeros(padding_length, dtype=w.dtype)
                    w = torch.cat([w, padding], dim=0)
                padded_waveforms.append(w)
            
            # Stack into batch tensor [batch_size, max_length]
            waveforms = torch.stack(padded_waveforms, dim=0)
        
        # Ensure waveforms is 2D: [batch_size, sequence_length]
        if len(waveforms.shape) == 1:
            waveforms = waveforms.unsqueeze(0)  # Add batch dimension
        elif len(waveforms.shape) > 2:
            # If multi-channel, convert to mono
            waveforms = waveforms.mean(dim=-1)
        
        # Convert to numpy for Whisper processor (it expects numpy arrays)
        if isinstance(waveforms, torch.Tensor):
            waveforms_np = waveforms.detach().cpu().numpy()
        else:
            waveforms_np = waveforms
        
        # Process each waveform in the batch separately to avoid shape issues
        batch_features = []
        for i in range(waveforms_np.shape[0]):
            single_waveform = waveforms_np[i]
            
            # Process with Whisper processor
            inputs = self.processor(
                single_waveform, 
                sampling_rate=sampling_rate, 
                return_tensors="pt"
            )
            input_features = inputs.input_features.to(self.encoder.device)
            
            # Extract features with frozen encoder
            with torch.no_grad() if self.training else torch.enable_grad():
                encoder_outputs = self.encoder(input_features)
                batch_features.append(encoder_outputs.last_hidden_state)
        
        # Concatenate all features along batch dimension
        if len(batch_features) == 1:
            audio_features = batch_features[0]
        else:
            audio_features = torch.cat(batch_features, dim=0)
        
        # Convert to the same dtype as the encoder (likely bfloat16 for efficiency)
        # Get the dtype from the encoder's first parameter
        target_dtype = next(self.encoder.parameters()).dtype
        audio_features = audio_features.to(dtype=target_dtype)
        
        return audio_features

class CrossAttentionAdapter(nn.Module):
    """
    Cross-attention adapter to connect Whisper encoder output to Qwen decoder.
    This enables the Qwen model to attend to audio features.
    """
    def __init__(self, audio_dim: int, text_dim: int, num_heads: int = 8, dtype=None):
        super().__init__()
        self.audio_dim = audio_dim
        self.text_dim = text_dim
        self.num_heads = num_heads
        self.dtype = dtype or torch.float32
        
        # Project audio features to text dimension
        self.audio_projection = nn.Linear(audio_dim, text_dim, dtype=self.dtype)
        
        # Cross-attention layer
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=text_dim,
            num_heads=num_heads,
            batch_first=True,
            dtype=self.dtype
        )
        
        # Layer norm and feedforward
        self.layer_norm1 = nn.LayerNorm(text_dim, dtype=self.dtype)
        self.layer_norm2 = nn.LayerNorm(text_dim, dtype=self.dtype)
        self.feedforward = nn.Sequential(
            nn.Linear(text_dim, text_dim * 4, dtype=self.dtype),
            nn.GELU(),
            nn.Linear(text_dim * 4, text_dim, dtype=self.dtype)
        )
        
    def forward(self, text_embeddings: torch.Tensor, audio_features: torch.Tensor) -> torch.Tensor:
        """
        Apply cross-attention between text embeddings and audio features.
        Args:
            text_embeddings: [batch, seq_len, text_dim] from Qwen
            audio_features: [batch, audio_seq_len, audio_dim] from Whisper
        """
        # Ensure both inputs have the same dtype
        target_dtype = text_embeddings.dtype
        audio_features = audio_features.to(dtype=target_dtype)
        
        # Project audio features to text dimension
        audio_projected = self.audio_projection(audio_features)  # [batch, audio_seq_len, text_dim]
        
        # Ensure projected audio has same dtype as text embeddings
        audio_projected = audio_projected.to(dtype=target_dtype)
        
        # Cross-attention: text queries attend to audio keys/values
        attended_text, _ = self.cross_attention(
            query=text_embeddings,
            key=audio_projected,
            value=audio_projected
        )
        
        # Residual connection and layer norm
        text_embeddings = self.layer_norm1(text_embeddings + attended_text)
        
        # Feedforward with residual
        ff_output = self.feedforward(text_embeddings)
        text_embeddings = self.layer_norm2(text_embeddings + ff_output)
        
        return text_embeddings

class MusicTranscriptionModel(nn.Module):
    """
    The main Encoder-Decoder model for music transcription.
    """
    def __init__(self, tokenizer=None):
        super().__init__()
        
        # Tower 1: The Audio Encoder
        self.audio_encoder = WhisperAudioEncoder()
        
        # Tower 2: The Text Decoder - use provided tokenizer or load custom one
        if tokenizer is not None:
            logger.info("Using provided tokenizer")
            self.tokenizer = tokenizer
        else:
            logger.info(f"Loading custom ABC tokenizer from: {TOKENIZER_DIR}")
            self.tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR, trust_remote_code=True)
        
        logger.info(f"Loading Qwen model: {QWEN_MODEL_NAME}")
        self.text_decoder = AutoModelForCausalLM.from_pretrained(
            QWEN_MODEL_NAME,
            trust_remote_code=True,
            torch_dtype="auto" # Use auto for best performance (e.g., bfloat16 on Ampere)
        )
        
        # ** CRITICAL STEP **
        # Resize token embeddings to match our extended ABC vocabulary
        original_vocab_size = self.text_decoder.config.vocab_size
        new_vocab_size = len(self.tokenizer)
        
        if new_vocab_size != original_vocab_size:
            self.text_decoder.resize_token_embeddings(new_vocab_size)
            logger.info(f"Resized Qwen token embeddings: {original_vocab_size} → {new_vocab_size}")
        else:
            logger.info(f"Qwen vocabulary size matches tokenizer: {new_vocab_size}")

        # Get dimensions and dtype for cross-attention adapter
        audio_dim = self.audio_encoder.encoder.config.d_model  # Whisper hidden size
        text_dim = self.text_decoder.config.hidden_size  # Qwen hidden size
        
        # Get the dtype from the text decoder to ensure consistency
        text_decoder_dtype = next(self.text_decoder.parameters()).dtype
        logger.info(f"Text decoder using dtype: {text_decoder_dtype}")
        
        # Cross-attention adapter to connect audio and text with matching dtype
        self.cross_attention_adapter = CrossAttentionAdapter(
            audio_dim=audio_dim,
            text_dim=text_dim,
            num_heads=8,
            dtype=text_decoder_dtype
        )

        # Freeze all Qwen parameters initially
        for param in self.text_decoder.parameters():
            param.requires_grad = False
            
        # Unfreeze the cross-attention layers and top K decoder layers
        logger.info(f"Unfreezing the top {TOP_K_QWEN_LAYERS} Qwen decoder layers for fine-tuning.")
        for i, layer in enumerate(self.text_decoder.model.layers):
            if i >= (len(self.text_decoder.model.layers) - TOP_K_QWEN_LAYERS):
                for param in layer.parameters():
                    param.requires_grad = True
        
        # Also unfreeze the final LayerNorm and LM head
        for param in self.text_decoder.model.norm.parameters():
            param.requires_grad = True
        for param in self.text_decoder.lm_head.parameters():
            param.requires_grad = True
            
        # Cross-attention adapter is trainable
        for param in self.cross_attention_adapter.parameters():
            param.requires_grad = True

    def forward(self, waveforms: Union[List[np.ndarray], torch.Tensor], sampling_rate: int, 
                target_token_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        The main forward pass for training.
        """
        # Get audio features from Whisper encoder
        audio_features = self.audio_encoder(waveforms, sampling_rate)
        
        # Get text embeddings from Qwen
        text_embeddings = self.text_decoder.model.embed_tokens(target_token_ids)
        
        # Ensure audio features match text embeddings dtype
        audio_features = audio_features.to(dtype=text_embeddings.dtype)
        
        # Apply cross-attention to fuse audio and text
        fused_embeddings = self.cross_attention_adapter(text_embeddings, audio_features)
        
        # Forward through Qwen with fused embeddings
        outputs = self.text_decoder(
            inputs_embeds=fused_embeddings,
            attention_mask=attention_mask,
            labels=target_token_ids, # The model handles shifting labels for loss calculation
            return_dict=True
        )
        
        return outputs.loss

    """
    Original generate method for autoregressive generation.
    """
    @torch.no_grad()
    def generate(self, waveform: np.ndarray, sampling_rate: int, max_new_tokens=256) -> str:
        """
        Inference method to generate ABC notation from an audio waveform.
        """
        self.eval() # Set model to evaluation mode
        
        # Get audio features
        audio_features = self.audio_encoder([waveform], sampling_rate)
        
        # Start generation from the BOS token
        decoder_start_token_id = self.tokenizer.bos_token_id
        if decoder_start_token_id is None:
             decoder_start_token_id = self.tokenizer.eos_token_id

        input_ids = torch.tensor([[decoder_start_token_id]], dtype=torch.long, device=self.text_decoder.device)
        
        # Generate autoregressively with cross-attention
        generated_tokens = []
        current_input_ids = input_ids
        
        for _ in range(max_new_tokens):
            # Get text embeddings
            text_embeddings = self.text_decoder.model.embed_tokens(current_input_ids)
            
            # Ensure audio features match text embeddings dtype
            audio_features = audio_features.to(dtype=text_embeddings.dtype)
            
            # Apply cross-attention
            fused_embeddings = self.cross_attention_adapter(text_embeddings, audio_features)
            
            # Forward pass
            outputs = self.text_decoder(
                inputs_embeds=fused_embeddings,
                return_dict=True
            )
            
            # Get next token
            next_token_logits = outputs.logits[:, -1, :]
            next_token_id = torch.multinomial(torch.softmax(next_token_logits / 0.7, dim=-1), 1)
            
            # Check for EOS
            if next_token_id.item() == self.tokenizer.eos_token_id:
                break
                
            generated_tokens.append(next_token_id.item())
            
            # Update input for next iteration
            current_input_ids = torch.cat([current_input_ids, next_token_id], dim=1)
        
        # Decode the generated tokens
        return self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

    # """
    # Updated generate method to prioritize ABC tokens during generation.
    # This method uses a custom token set to ensure we generate valid ABC notation - but due to no generated ABC Notation tokens, it is not working.
    # """

    # @torch.no_grad()
    # def generate(self, waveform: np.ndarray, sampling_rate: int, max_new_tokens=256) -> str:
    #     """
    #     Inference method to generate ABC notation from an audio waveform.
    #     Prioritizes ABC tokens during generation.
    #     """
    #     self.eval()
    #     device = self.text_decoder.device

    #     # Get audio features
    #     audio_features = self.audio_encoder([waveform], sampling_rate)

    #     # Start with a helpful ABC notation prompt
    #     abc_prefix = "X:1\nT:Transcription\nM:4/4\nL:1/8\nK:C\n"
    #     prefix_tokens = self.tokenizer(abc_prefix, return_tensors="pt").input_ids.to(device)
    #     current_input_ids = prefix_tokens

    #     # --- Build ABC token id set ---
    #     # Option 1: Use added_tokens.json if available
    #     abc_token_ids = set()
    #     added_tokens_path = os.path.join(TOKENIZER_DIR, "added_tokens.json")
    #     if os.path.exists(added_tokens_path):
    #         import json
    #         with open(added_tokens_path, "r") as f:
    #             added_tokens = json.load(f)
    #         abc_token_ids = set(added_tokens.values())
    #     else:
    #         # Option 2: Fallback to pattern matching (less precise)
    #         vocab = self.tokenizer.get_vocab()
    #         abc_token_ids = set(
    #             idx for token, idx in vocab.items()
    #             if any(x in token for x in ["X:", "T:", "M:", "L:", "K:", "|", "[", "]", ":", "/", "'", ",", "^", "_", "=", "A", "B", "C", "D", "E", "F", "G", "a", "b", "c", "d", "e", "f", "g", "z", "1", "2", "3", "4", "5", "6", "7", "8", "9", "0"])
    #         )

    #     eos_token_id = self.tokenizer.eos_token_id

    #     generated_tokens = []
    #     temperature = 0.7

    #     for _ in range(max_new_tokens):
    #         text_embeddings = self.text_decoder.model.embed_tokens(current_input_ids)
    #         audio_features = audio_features.to(dtype=text_embeddings.dtype)
    #         fused_embeddings = self.cross_attention_adapter(text_embeddings, audio_features)
    #         outputs = self.text_decoder(inputs_embeds=fused_embeddings, return_dict=True)
    #         next_token_logits = outputs.logits[:, -1, :]

    #         # Mask out non-ABC tokens
    #         mask = torch.full_like(next_token_logits, float('-inf'))
    #         mask[:, list(abc_token_ids)] = 0
    #         next_token_logits = next_token_logits + mask

    #         # Sample next token
    #         next_token_id = torch.multinomial(torch.softmax(next_token_logits / temperature, dim=-1), 1)

    #         if next_token_id.item() == eos_token_id:
    #             break
    #         generated_tokens.append(next_token_id.item())
    #         current_input_ids = torch.cat([current_input_ids, next_token_id], dim=1)

    #         # Optionally, trim context if too long
    #         if current_input_ids.shape[1] > 128:
    #             current_input_ids = torch.cat([
    #                 prefix_tokens,
    #                 current_input_ids[:, -64:][:, -(128-prefix_tokens.shape[1]):]
    #             ], dim=1)

    #     return abc_prefix + self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

    # """ 
    # Tried to fix things using prompts, but not working
    # """
    # @torch.no_grad()
    # def generate(self, waveform: np.ndarray, sampling_rate: int, max_new_tokens=256) -> str:
    #     """
    #     Inference method to generate ABC notation from an audio waveform.
    #     """
    #     self.eval() # Set model to evaluation mode
        
    #     # Get audio features
    #     audio_features = self.audio_encoder([waveform], sampling_rate)
        
    #     # Start with a helpful ABC notation prompt to guide the generation
    #     abc_prefix = "X:1\nT:Transcription\nM:4/4\nL:1/8\nK:C\n"
    #     prefix_tokens = self.tokenizer(abc_prefix, return_tensors="pt").input_ids.to(self.text_decoder.device)
    #     current_input_ids = prefix_tokens
        
    #     # Generate autoregressively with cross-attention
    #     generated_tokens = []
    #     temperature = 0.5  # Lower temperature for more focused generation
        
    #     for _ in range(max_new_tokens):
    #         # Get text embeddings
    #         text_embeddings = self.text_decoder.model.embed_tokens(current_input_ids)
            
    #         # Ensure audio features match text embeddings dtype
    #         audio_features = audio_features.to(dtype=text_embeddings.dtype)
            
    #         # Apply cross-attention
    #         fused_embeddings = self.cross_attention_adapter(text_embeddings, audio_features)
            
    #         # Forward pass
    #         outputs = self.text_decoder(
    #             inputs_embeds=fused_embeddings,
    #             return_dict=True
    #         )
            
    #         # Get next token with lower temperature
    #         next_token_logits = outputs.logits[:, -1, :]
            
    #         # Filter logits to favor common ABC notation characters
    #         abc_chars = "ABCDEFGabcdefg|:[]1234567890/^_=,'\""
    #         for i, char in enumerate(self.tokenizer.get_vocab()):
    #             if len(char) == 1 and char not in abc_chars:
    #                 next_token_logits[0, i] = -float('inf')
            
    #         # Sample with temperature
    #         next_token_id = torch.multinomial(torch.softmax(next_token_logits / temperature, dim=-1), 1)
            
    #         # Check for EOS
    #         if next_token_id.item() == self.tokenizer.eos_token_id:
    #             break
                
    #         generated_tokens.append(next_token_id.item())
            
    #         # Update input for next iteration - use only the latest token
    #         current_input_ids = torch.cat([current_input_ids, next_token_id], dim=1)
            
    #         # Trim context if it gets too long (keep the prefix and recent tokens)
    #         if current_input_ids.shape[1] > 128:
    #             current_input_ids = torch.cat([
    #                 prefix_tokens, 
    #                 current_input_ids[:, -64:][:, -(128-prefix_tokens.shape[1]):]], 
    #                 dim=1
    #             )
        
    #     # Combine the prefix with the generated tokens for the final output
    #     return abc_prefix + self.tokenizer.decode(generated_tokens, skip_special_tokens=True)