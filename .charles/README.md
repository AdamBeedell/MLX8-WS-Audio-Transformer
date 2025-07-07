# MLX8-WS-Audio-Transformer

Audio Classification using Transformer Architecture on UrbanSound8K Dataset

## Task 1: Part 1 - UrbanSound8K Dataset Study

This project implements a comprehensive study and preprocessing pipeline for the UrbanSound8K dataset, focusing on audio classification using log-mel spectrograms and transformer architectures.

### Dataset Overview

**UrbanSound8K** is a dataset containing 8,732 labeled sound excerpts (≤4s) of urban sounds from 10 classes:
- Air Conditioner (0)
- Car Horn (1) 
- Children Playing (2)
- Dog Bark (3)
- Drilling (4)
- Engine Idling (5)
- Gun Shot (6)
- Jackhammer (7)
- Siren (8)
- Street Music (9)

The dataset is pre-divided into 10 folds for cross-validation, with audio files already pre-sliced according to the metadata timestamps.

### Features

#### 🎵 Audio Preprocessing Pipeline
- **Format Standardization**: Converts all audio to mono, 16kHz sampling rate
- **Duration Normalization**: Pads/trims audio to fixed 4-second duration
- **Mel-Spectrogram Generation**: Converts audio to log-mel spectrograms (64 mel bins)
- **Efficient Storage**: Saves preprocessed data as Parquet format for fast loading

#### 📊 Visualization & Analysis
- **Sample Generation**: Exports spectrogram images for each class (2 samples per class)
- **Visual Inspection**: Creates publication-ready spectrogram plots
- **Class Distribution**: Supports analysis across all 10 urban sound categories

#### ⚡ Performance Optimizations
- **GPU Acceleration**: Utilizes CUDA when available for faster processing
- **Batch Processing**: Efficient processing of entire dataset
- **Memory Management**: Optimized for large-scale audio processing

### Setup

#### Requirements
```bash
pip install torch torchaudio pandas numpy matplotlib python-dotenv colorlog tqdm
```

#### Environment Configuration
Create a `.env` file in the `.charles/` directory:

```env
DATA_ROOT=./.data/UrbanSound8K
METADATA_CSV=./.data/UrbanSound8K/metadata/UrbanSound8K.csv
PARQUET_PATH=./.data/UrbanSound8K/processed/urbansound8k.parquet
SAMPLE_RATE=16000
N_MELS=64
N_FFT=1024
HOP_LENGTH=512
FMIN=0
FMAX=8000
MONO=True
DURATION=4.0
```

### Usage

#### Preprocess Dataset
Convert raw audio files to log-mel spectrograms and save as Parquet:
```bash
cd .charles
python spectrogram.py --preprocess
```

#### Generate Sample Visualizations
Export spectrogram images for visual inspection:
```bash
python spectrogram.py --sampledata
```

#### Combined Processing
```bash
python spectrogram.py --preprocess --sampledata
```

### Technical Details

#### Audio Processing Parameters
- **Sample Rate**: 16,000 Hz
- **Duration**: 4.0 seconds (64,000 samples)
- **FFT Size**: 1,024 samples
- **Hop Length**: 512 samples
- **Mel Bins**: 64
- **Frequency Range**: 0-8,000 Hz

#### Spectrogram Specifications
- **Input Shape**: [1, 64,000] (mono audio)
- **Output Shape**: [64, time_frames] (log-mel spectrogram)
- **Time Frames**: ~126 frames for 4-second audio
- **Log Transform**: `log(mel + 1e-6)` for numerical stability

#### Dataset Structure
```
UrbanSound8K/
├── audio/
│   ├── fold1/
│   ├── fold2/
│   └── ...fold10/
├── metadata/
│   └── UrbanSound8K.csv
└── processed/
    ├── urbansound8k.parquet
    └── sample-<class_id>-(<class_name>)-{1,2}.png
```

### PyTorch Dataset Integration

The `UrbanSoundDataSet` class provides seamless integration with PyTorch workflows:

```python
from spectrogram import UrbanSoundDataSet

# Load training folds (1-8)
train_dataset = UrbanSoundDataSet(folds=[1,2,3,4,5,6,7,8])

# Load validation fold (9)
val_dataset = UrbanSoundDataSet(folds=[9])

# Load test fold (10)
test_dataset = UrbanSoundDataSet(folds=[10])
```

### Output Files

#### Processed Data
- `urbansound8k.parquet`: Complete preprocessed dataset with log-mel spectrograms

#### Sample Visualizations
- `sample-<0>-(air_conditioner)-1.png`
- `sample-<1>-(car_horn)-1.png`
- `sample-<2>-(children_playing)-1.png`
- ... (2 samples per class)

### Logging & Monitoring

The script includes comprehensive logging with color-coded output:
- ✅ **SUCCESS**: Green - successful operations
- 🖼️ **INFO**: White - progress information  
- ⚠️ **WARNING**: Yellow - non-critical issues
- ❌ **ERROR**: Red - processing errors

### Key Implementation Notes

1. **Pre-sliced Audio**: UrbanSound8K files are already extracted according to metadata timestamps - no additional slicing needed
2. **Deterministic Sampling**: Uses fixed random seed (42) for reproducible sample selection
3. **Error Handling**: Robust error handling for corrupted or missing audio files
4. **Memory Efficiency**: Processes files individually to manage memory usage
5. **Cross-platform**: Compatible with both CUDA and CPU environments

### Next Steps

This preprocessing pipeline prepares the UrbanSound8K dataset for:
- Transformer-based audio classification models
- Convolutional neural network training
- Feature analysis and visualization
- Cross-validation experiments using the 10-fold structure

The standardized log-mel spectrograms provide optimal input features for modern deep learning architectures while maintaining