# inference.py

import os
import soundfile as sf
import torch
from transformers import AutoTokenizer

from model import MusicTranscriptionModel

from dotenv import load_dotenv
load_dotenv()

from logger_utils import setup_logger
logger = setup_logger(__name__)

# Get checkpoint and audio file from .env or use defaults
MODELS_DIR = os.getenv("MODELS_OUTPUT_DIR", "../.data/models/music2midi")
CUSTOM_TOKENIZER_DIR = os.getenv("PREPROCESSED_DATA_DIR", "../.data/preprocessed") + "/qwen_with_abc_tokenizer"

# Default to best_model.pt if no specific checkpoint specified
DEFAULT_CKPT = ""
if os.path.exists(MODELS_DIR):
    # First try best_model.pt
    best_model_path = os.path.join(MODELS_DIR, "best_model.pt")
    if os.path.exists(best_model_path):
        DEFAULT_CKPT = best_model_path
    else:
        # Fall back to latest epoch checkpoint
        checkpoints = [f for f in os.listdir(MODELS_DIR) if f.endswith(".pt")]
        if checkpoints:
            DEFAULT_CKPT = os.path.join(MODELS_DIR, sorted(checkpoints)[-1]) # Get latest

INFERENCE_CHECKPOINT = os.getenv("INFERENCE_CHECKPOINT_PATH", DEFAULT_CKPT)
INFERENCE_AUDIO = os.getenv("INFERENCE_AUDIO_FILE")
SAMPLE_RATE = int(os.getenv("SAMPLE_RATE", 16000))

def load_custom_tokenizer():
    """Load the custom ABC-extended tokenizer."""
    fallback_model = os.getenv("QWEN_MODEL", "Qwen/Qwen3-0.6B-Base")
    
    if os.path.exists(CUSTOM_TOKENIZER_DIR) and os.path.exists(os.path.join(CUSTOM_TOKENIZER_DIR, "tokenizer.json")):
        logger.info(f"Loading custom ABC tokenizer from: {CUSTOM_TOKENIZER_DIR}")
        tokenizer = AutoTokenizer.from_pretrained(CUSTOM_TOKENIZER_DIR, trust_remote_code=True)
        logger.success(f"Custom tokenizer loaded. Vocab size: {len(tokenizer)}")
        return tokenizer
    else:
        logger.warning(f"Custom tokenizer not found at: {CUSTOM_TOKENIZER_DIR}")
        logger.info(f"Falling back to base tokenizer: {fallback_model}")
        tokenizer = AutoTokenizer.from_pretrained(fallback_model, trust_remote_code=True)
        logger.info(f"Base tokenizer loaded. Vocab size: {len(tokenizer)}")
        return tokenizer

def load_checkpoint_safely(model, checkpoint_path, device):
    """Load checkpoint with proper error handling for our checkpoint format."""
    logger.info(f"Loading weights from checkpoint: {checkpoint_path}")
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Handle our custom checkpoint format from train.py
        if 'trainable_state_dict' in checkpoint:
            logger.info("Loading from custom training checkpoint format...")
            
            # Load trainable weights
            model.load_state_dict(checkpoint['trainable_state_dict'], strict=False)
            
            # Load critical frozen components if they exist
            if 'critical_frozen_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['critical_frozen_state_dict'], strict=False)
            
            # Log checkpoint info
            if 'epoch' in checkpoint:
                logger.info(f"Checkpoint from epoch: {checkpoint['epoch']}")
            if 'loss' in checkpoint:
                logger.info(f"Checkpoint validation loss: {checkpoint['loss']:.4f}")
            if 'config' in checkpoint and 'vocab_size' in checkpoint['config']:
                logger.info(f"Model vocab size: {checkpoint['config']['vocab_size']}")
                
        else:
            # Standard state dict format
            logger.info("Loading from standard state dict format...")
            model.load_state_dict(checkpoint, strict=False)
            
        logger.success("Model weights loaded successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Failed to load checkpoint: {e}")
        return False
    

# More comprehensive model analysis function than train.py, nowe have loaded last trained model checkpoint
def analyze_model(model):
    """
    Analyze model architecture, parameters, memory usage, and training status.
    Similar to train.py but with enhanced metrics for inference analysis.
    """
    logger.info("🔍 Analyzing model architecture and parameters...")
    
    # Component breakdown
    whisper_params = 0
    qwen_params = 0
    adapter_params = 0
    trainable_params = 0
    frozen_params = 0
    
    component_breakdown = {
        'whisper_encoder': 0,
        'qwen_embeddings': 0,
        'qwen_layers': 0,
        'qwen_norm_head': 0,
        'cross_attention_adapter': 0
    }
    
    # Get actual model dtype for accurate memory calculations
    model_dtypes = {}
    memory_multiplier = 4  # Default float32
    
    # Track parameters per layer
    layer_params = {}
    
    for name, param in model.named_parameters():
        param_count = param.numel()
        
        # Track dtype for memory calculations
        if param.dtype not in model_dtypes:
            model_dtypes[param.dtype] = 0
        model_dtypes[param.dtype] += param_count
        
        # Get layer number for transformer layers
        if 'text_decoder.model.layers' in name:
            layer_num = int(name.split('text_decoder.model.layers.')[1].split('.')[0])
            if f'qwen_layer_{layer_num}' not in layer_params:
                layer_params[f'qwen_layer_{layer_num}'] = 0
            layer_params[f'qwen_layer_{layer_num}'] += param_count
        
        # Component categorization
        if 'audio_encoder' in name:
            whisper_params += param_count
            component_breakdown['whisper_encoder'] += param_count
        elif 'text_decoder.model.embed_tokens' in name or 'text_decoder.lm_head' in name:
            qwen_params += param_count
            component_breakdown['qwen_embeddings'] += param_count
        elif 'text_decoder.model.layers' in name:
            qwen_params += param_count
            component_breakdown['qwen_layers'] += param_count
        elif 'text_decoder.model.norm' in name:
            qwen_params += param_count
            component_breakdown['qwen_norm_head'] += param_count
        elif 'cross_attention_adapter' in name:
            adapter_params += param_count
            component_breakdown['cross_attention_adapter'] += param_count
        
        # Training status
        if param.requires_grad:
            trainable_params += param_count
        else:
            frozen_params += param_count
    
    total_params = trainable_params + frozen_params
    
    # Determine dominant dtype for memory calculations
    dominant_dtype = max(model_dtypes.items(), key=lambda x: x[1])[0]
    if dominant_dtype == torch.float16 or dominant_dtype == torch.half:
        memory_multiplier = 2
    elif dominant_dtype == torch.bfloat16:
        memory_multiplier = 2
    elif dominant_dtype == torch.float32 or dominant_dtype == torch.float:
        memory_multiplier = 4
    elif dominant_dtype == torch.int8:
        memory_multiplier = 1
    
    # Memory calculations with actual dtype
    param_memory_gb = total_params * memory_multiplier / (1024**3)
    trainable_memory_gb = trainable_params * memory_multiplier / (1024**3)
    
    # Estimate inference memory (parameters + activations)
    batch_size = 1  # Inference batch size
    seq_len = 512  # Approximate sequence length
    
    # Rough estimate of activation memory based on model size and sequence length
    if hasattr(model.text_decoder, 'config'):
        hidden_size = model.text_decoder.config.hidden_size
        num_layers = model.text_decoder.config.num_hidden_layers
        # KV cache for attention + intermediate activations
        activation_memory_gb = (batch_size * seq_len * hidden_size * num_layers * 4 * memory_multiplier) / (1024**3)
    else:
        # Rough estimate without config
        activation_memory_gb = param_memory_gb * 0.3
    
    # Display results
    logger.info("=" * 70)
    logger.info("🔬 MODEL ARCHITECTURE ANALYSIS (INFERENCE)")
    logger.info("=" * 70)
    logger.info(
"""
    +---------------------+         +--------------------------+         +-----------------------------+
    |                     |         |                          |         |                             |
    |  WhisperAudioEncoder|         |   CrossAttentionAdapter   |         |   Qwen Text Decoder (LM)    |
    |   (Frozen Whisper)  |         | (Audio-to-Text Bridge)   |         |   (AutoModelForCausalLM)    |
    |                     |         |                          |         |                             |
    +----------+----------+         +-----------+--------------+         +-------------+---------------+
            |                                |                                      |
            |  audio_features                | fused_embeddings                      |
            +------------------------------->+                                      |
                                                |                                      |
                                                +------------------------------------->+
                                                                                    |
                                                                                    v
                                                                            +---------------------+
                                                                            |  ABC Notation Tokens|
                                                                            +---------------------+
                                                                            
"""
    )
    
    # Model type info
    logger.info("📋 Model Configuration:")
    logger.info(f"   Model Type:             Whisper-Qwen Audio-to-Text Transformer")
    logger.info(f"   Whisper Encoder:        {os.getenv('WHISPER_MODEL', 'openai/whisper-base')}")
    logger.info(f"   Text Decoder:           {os.getenv('QWEN_MODEL', 'Qwen/Qwen3-0.6B-Base')}")
    logger.info(f"   Dominant Data Type:     {dominant_dtype}")
    
    logger.info("-" * 70)
    
    # Component breakdown
    logger.info("📊 Component Parameter Breakdown:")
    logger.info(f"   Whisper Encoder:        {whisper_params:>12,} params ({whisper_params/1e6:.1f}M)")
    logger.info(f"   Qwen Embeddings/Head:   {component_breakdown['qwen_embeddings']:>12,} params ({component_breakdown['qwen_embeddings']/1e6:.1f}M)")
    logger.info(f"   Qwen Transformer Layers:{component_breakdown['qwen_layers']:>12,} params ({component_breakdown['qwen_layers']/1e6:.1f}M)")
    logger.info(f"   Qwen Norm/Head:         {component_breakdown['qwen_norm_head']:>12,} params ({component_breakdown['qwen_norm_head']/1e6:.1f}M)")
    logger.info(f"   Cross-Attention Adapter:{adapter_params:>12,} params ({adapter_params/1e6:.1f}M)")
    
    logger.info("-" * 70)
    
    # Per-layer breakdown (top 5)
    if layer_params:
        logger.info("📊 Per-Layer Parameter Distribution (top layers):")
        sorted_layers = sorted(layer_params.items(), key=lambda x: int(x[0].split('_')[-1]))
        for name, param_count in sorted_layers[-5:]:  # Show last 5 layers
            logger.info(f"   {name:<20} {param_count:>12,} params ({param_count/1e6:.1f}M)")
    
    logger.info("-" * 70)
    
    # Training status breakdown
    logger.info("🎓 Training Status:")
    logger.info(f"   Trainable Parameters:   {trainable_params:>12,} params ({trainable_params/1e6:.1f}M)")
    logger.info(f"   Frozen Parameters:      {frozen_params:>12,} params ({frozen_params/1e6:.1f}M)")
    logger.info(f"   Total Parameters:       {total_params:>12,} params ({total_params/1e6:.1f}M)")
    if total_params > 0:
        logger.info(f"   Trainable Percentage:   {(trainable_params/total_params)*100:>7.1f}% of model was fine-tuned")
    
    logger.info("-" * 70)
    
    # Memory usage
    logger.info("💾 Memory Usage Analysis:")
    logger.info(f"   Parameter Memory:       {param_memory_gb:>8.2f} GB")
    logger.info(f"   Est. Activation Memory: {activation_memory_gb:>8.2f} GB")
    logger.info(f"   Total Inference Memory: {param_memory_gb + activation_memory_gb:>8.2f} GB")
    
    # Model architecture info
    if hasattr(model.text_decoder, 'config'):
        config = model.text_decoder.config
        logger.info("-" * 70)
        logger.info("🏗️  Architecture Details:")
        logger.info(f"   Qwen Hidden Size:       {config.hidden_size:>12,}")
        logger.info(f"   Qwen Vocab Size:        {config.vocab_size:>12,}")
        logger.info(f"   Qwen Layers:            {config.num_hidden_layers:>12,}")
        logger.info(f"   Qwen Attention Heads:   {config.num_attention_heads:>12,}")
        if hasattr(config, 'kv_channels'):
            logger.info(f"   Qwen KV Channels:       {config.kv_channels:>12,}")
        if hasattr(config, 'max_position_embeddings'):
            logger.info(f"   Qwen Max Seq Length:    {config.max_position_embeddings:>12,}")
    
    if hasattr(model.audio_encoder.encoder, 'config'):
        whisper_config = model.audio_encoder.encoder.config
        logger.info(f"   Whisper Hidden Size:    {whisper_config.d_model:>12,}")
        logger.info(f"   Whisper Layers:         {whisper_config.encoder_layers:>12,}")
        logger.info(f"   Whisper Attention Heads:{whisper_config.encoder_attention_heads:>12,}")
        if hasattr(whisper_config, 'max_source_positions'):
            logger.info(f"   Whisper Max Audio Len:  {whisper_config.max_source_positions:>12,}")
    
    logger.info("=" * 70)
    
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'frozen_params': frozen_params,
        'whisper_params': whisper_params,
        'qwen_params': qwen_params,
        'adapter_params': adapter_params,
        'param_memory_gb': param_memory_gb,
        'activation_memory_gb': activation_memory_gb,
        'total_inference_memory_gb': param_memory_gb + activation_memory_gb,
        'component_breakdown': component_breakdown,
        'dominant_dtype': str(dominant_dtype),
        'layer_params': layer_params
    }

def main():
    """Main inference function."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"🚀 Starting music transcription inference")
    logger.info(f"Using device: {device}")
    
    if not INFERENCE_CHECKPOINT:
        logger.error("No model checkpoint found or specified in .env (INFERENCE_CHECKPOINT_PATH).")
        logger.error(f"Checked directory: {MODELS_DIR}")
        return
        
    if not INFERENCE_AUDIO or not os.path.exists(INFERENCE_AUDIO):
        logger.error("Audio file not found or specified in .env (INFERENCE_AUDIO_FILE).")
        logger.error(f"Checked path: {INFERENCE_AUDIO}")
        return
    
    # Load custom tokenizer with ABC notation support
    logger.info("🔤 Loading custom tokenizer...")
    tokenizer = load_custom_tokenizer()
    
    # Instantiate the full model architecture with custom tokenizer
    logger.info("🏗️ Loading model architecture...")
    model = MusicTranscriptionModel(tokenizer=tokenizer)
    
    # Load the fine-tuned weights
    if not load_checkpoint_safely(model, INFERENCE_CHECKPOINT, device):
        logger.error("Failed to load model checkpoint. Exiting.")
        return
        
    model.to(device)
    model.eval()
    logger.success("Model ready for inference!")

    # Perform comprehensive model analysis
    analyze_model(model)

    # Load and preprocess the audio file
    logger.info(f"🎵 Loading audio file: {os.path.basename(INFERENCE_AUDIO)}")
    try:
        waveform, sr = sf.read(INFERENCE_AUDIO, dtype='float32')
        logger.info(f"Audio loaded: {len(waveform)/sr:.2f}s duration, {sr}Hz sample rate")
        
        if sr != SAMPLE_RATE:
            logger.warning(f"Audio sample rate ({sr}) differs from model's expected rate ({SAMPLE_RATE})")
            logger.warning("Consider resampling the audio for best results")
            # In a real app, you would resample here using torchaudio or librosa.
            # For now, we'll assume it matches or is close enough.
            
    except Exception as e:
        logger.error(f"Failed to load audio file: {e}")
        return
        
    # Generate the transcription
    logger.info("🎼 Generating ABC notation transcription...")
    try:
        with torch.no_grad():
            generated_abc = model.generate(waveform, sampling_rate=sr)
        
        # Display results
        logger.success("Transcription completed!")
        logger.info("=" * 60)
        logger.info(f"📁 Audio File: {os.path.basename(INFERENCE_AUDIO)}")
        logger.info("🎵 Predicted ABC Notation:")
        logger.info("=" * 60)
        print(generated_abc)  # Use print for clean output of the ABC notation
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        return


if __name__ == "__main__":
    main()