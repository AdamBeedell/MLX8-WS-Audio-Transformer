# DuckDB SQL Queries for UrbanSound8K Dataset

## Reference Links
- 86% CNN 1D: https://www.kaggle.com/code/prabhavsingh/urbansound8k-classification
- 80%: https://www.kaggle.com/code/nandana19nxny/speech-casestudy-audioclassification
- 98.05%: https://github.com/gitmehrdad/face 
        Paper: https://arxiv.org/pdf/2303.03666
- 75.4% vs 84.9%: 1D vs 2D: https://github.com/CVxTz/audio_classification
        Medium: https://medium.com/@CVxTz/audio-classification-a-convolutional-neural-network-approach-b0a4fce8f6c

## Comparison to OpenAI Whiper 

Refer to ../course-materials/mlx_week5_catchup_shapes.pdf

**OpenAI Audio Hyperparameters**

https://github.com/openai/whisper/blob/main/whisper/audio.py#L26

```python
# hard-coded audio hyperparameters
SAMPLE_RATE = 16000
N_FFT = 400
HOP_LENGTH = 160
CHUNK_LENGTH = 30
N_SAMPLES = CHUNK_LENGTH * SAMPLE_RATE  # 480000 samples in a 30-second chunk
N_FRAMES = exact_div(N_SAMPLES, HOP_LENGTH)  # 3000 frames in a mel spectrogram input

N_SAMPLES_PER_TOKEN = HOP_LENGTH * 2  # the initial convolutions has stride 2
FRAMES_PER_SECOND = exact_div(SAMPLE_RATE, HOP_LENGTH)  # 10ms per audio frame
TOKENS_PER_SECOND = exact_div(SAMPLE_RATE, N_SAMPLES_PER_TOKEN)  # 20ms per audio token
```

OpenAI Whisper Model.py **AudioEncoder** Class
https://github.com/openai/whisper/blob/main/whisper/model.py#L174

```python
class AudioEncoder(nn.Module):
    def __init__(
        self, n_mels: int, n_ctx: int, n_state: int, n_head: int, n_layer: int
    ):
        super().__init__()
        self.conv1 = Conv1d(n_mels, n_state, kernel_size=3, padding=1)
        self.conv2 = Conv1d(n_state, n_state, kernel_size=3, stride=2, padding=1)
        self.register_buffer("positional_embedding", sinusoids(n_ctx, n_state))

        self.blocks: Iterable[ResidualAttentionBlock] = nn.ModuleList(
            [ResidualAttentionBlock(n_state, n_head) for _ in range(n_layer)]
        )
        self.ln_post = LayerNorm(n_state)

    def forward(self, x: Tensor):
        """
        x : torch.Tensor, shape = (batch_size, n_mels, n_ctx)
            the mel spectrogram of the audio
        """
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        x = x.permute(0, 2, 1)

        assert x.shape[1:] == self.positional_embedding.shape, "incorrect audio shape"
        x = (x + self.positional_embedding).to(x.dtype)

        for block in self.blocks:
            x = block(x)

        x = self.ln_post(x)
        return x
```


## Data Preprocessing & Schema V1 (Old)

**Uniformity Results**  
After preprocessing, all audio files are standardized as follows:
- **Sample rate:** 16,000 Hz
- **Duration:** Exactly 4.0 seconds
- **Samples:** Exactly 64,000 samples per file
- **Channels:** Mono (1 channel)

**Spectrogram Dimensions**
- **Mel bins:** 64 (`N_MELS`)
- **Time frames:** 126 frames (with `hop_length=512`)
- **Final shape:** `[64, 126]` (flattened to `[8064]` for storage)

**Handling Variable Lengths**
- Short audio (< 4s): Zero-padded to 4 seconds
- Long audio (> 4s): Truncated to first 4 seconds
- Exact 4s audio: No modification needed

This ensures the CNN receives consistent input dimensions `[batch_size, 64, 126]` for all samples, regardless of the original audio length variations in the dataset.

## Data Preprocessing & Schema V2 (New)

> Overall performance is similar: v1 **64%** vs V2 **68%**

**Overview**  
The new preprocessing pipeline stores each audio file as a *flattened* log-mel spectrogram array, along with its original shape, for efficient storage and fast loading. This enables flexible reshaping for model input and supports variable spectrogram shapes if hyperparameters change.

**Key Features**
- **Sample rate:** 16,000 Hz (same as V1)
- **Duration:** 4.0 seconds (same as V1)
- **Channels:** Mono (1 channel)
- **Spectrogram:** Log-mel, computed with configurable hyperparameters (see `.env`)
- **Shape:** `[N_MELS, N_FRAMES]` (default: `[128, 501]` for `N_MELS=128`, `HOP_LENGTH=128`)
- **Storage:** Flattened 1D array (`log_mel_flat`) and shape (`log_mel_shape`) columns in Parquet

**Differences from V1**
- V1 used `[64, 126]` shape (with `N_MELS=64`, `HOP_LENGTH=512`).
- V2 supports arbitrary `N_MELS` and `HOP_LENGTH` (see `.env`), defaulting to `[128, 501]`.
- V2 stores the *flattened* spectrogram and its shape, not a 2D array, for better compatibility and flexibility.

## SQL Scripts

**Processed Parquet Schema**
```sql
DESCRIBE './.data/UrbanSound8K/processed/urbansound8k.parquet';

-- ┌───────────────┬─────────────┬─────────┬─────────┬─────────┬─────────┐  
-- │  column_name  │ column_type │  null   │   key   │ default │  extra  │  
-- │    varchar    │   varchar   │ varchar │ varchar │ varchar │ varchar │  
-- ├───────────────┼─────────────┼─────────┼─────────┼─────────┼─────────┤  
-- │ rel_path      │ VARCHAR     │ YES     │ NULL    │ NULL    │ NULL    │  
-- │ fold          │ BIGINT      │ YES     │ NULL    │ NULL    │ NULL    │  
-- │ class_id      │ BIGINT      │ YES     │ NULL    │ NULL    │ NULL    │  
-- │ class_name    │ VARCHAR     │ YES     │ NULL    │ NULL    │ NULL    │  
-- │ log_mel_flat  │ FLOAT[]     │ YES     │ NULL    │ NULL    │ NULL    │  
-- │ log_mel_shape │ BIGINT[]    │ YES     │ NULL    │ NULL    │ NULL    │  
-- └───────────────┴─────────────┴─────────┴─────────┴─────────┴─────────┘
```

**Typical log-mel shape:**  
- For `N_MELS=128`, `HOP_LENGTH=128`, `DURATION=4.0s`, `SAMPLE_RATE=16000`:
  - `n_mels = 128`
  - `n_frames = 1 + int((SAMPLE_RATE * DURATION - N_FFT) / HOP_LENGTH)`  
    (with `N_FFT=1024` by default, gives `n_frames ≈ 501`)
  - **Final shape:** `[128, 501]` (flattened to `[64128]`)

**Handling Variable Lengths**
- All audio is padded or truncated to exactly 4.0 seconds before spectrogram computation, ensuring consistent input size.

**Usage**
- To reconstruct the spectrogram:  
  `log_mel = np.array(log_mel_flat).reshape(log_mel_shape)`

**Example Query: Preview Data with Spectrogram Shape**
```sql
SELECT 
    rel_path, 
    class_name, 
    log_mel_shape,
    array_length(log_mel_flat, 1) as flat_length
FROM read_parquet('./.data/UrbanSound8K/processed/urbansound8k.parquet')
LIMIT 5;
```

## Setup
```sql
-- Install and load parquet extension if needed
INSTALL parquet;
LOAD parquet;
```

## Basic Dataset Exploration

### 1. Dataset Overview
```sql
-- Get basic dataset info
SELECT 
    COUNT(*) as total_samples,
    COUNT(DISTINCT class_id) as num_classes,
    COUNT(DISTINCT fold) as num_folds
FROM read_parquet('./.data/UrbanSound8K/processed/urbansound8k.parquet');
```

### 2. Column Information
```sql
-- Examine column structure
DESCRIBE './.data/UrbanSound8K/processed/urbansound8k.parquet';

-- Schema:
-- ┌───────────────┬─────────────┬─────────┬─────────┬─────────┬─────────┐
-- │  column_name  │ column_type │  null   │   key   │ default │  extra  │
-- │    varchar    │   varchar   │ varchar │ varchar │ varchar │ varchar │
-- ├───────────────┼─────────────┼─────────┼─────────┼─────────┼─────────┤
-- │ rel_path      │ VARCHAR     │ YES     │ NULL    │ NULL    │ NULL    │
-- │ fold          │ BIGINT      │ YES     │ NULL    │ NULL    │ NULL    │
-- │ class_id      │ BIGINT      │ YES     │ NULL    │ NULL    │ NULL    │
-- │ class_name    │ VARCHAR     │ YES     │ NULL    │ NULL    │ NULL    │
-- │ log_mel_flat  │ FLOAT[]     │ YES     │ NULL    │ NULL    │ NULL    │
-- │ log_mel_shape │ BIGINT[]    │ YES     │ NULL    │ NULL    │ NULL    │
-- └───────────────┴─────────────┴─────────┴─────────┴─────────┴─────────┘

-- OLD Version
-- ┌─────────────┬─────────────┬─────────┬─────────┬─────────┬─────────┐
-- │ column_name │ column_type │  null   │   key   │ default │  extra  │
-- │   varchar   │   varchar   │ varchar │ varchar │ varchar │ varchar │
-- ├─────────────┼─────────────┼─────────┼─────────┼─────────┼─────────┤
-- │ rel_path    │ VARCHAR     │ YES     │ NULL    │ NULL    │ NULL    │
-- │ fold        │ BIGINT      │ YES     │ NULL    │ NULL    │ NULL    │
-- │ class_id    │ BIGINT      │ YES     │ NULL    │ NULL    │ NULL    │
-- │ class_name  │ VARCHAR     │ YES     │ NULL    │ NULL    │ NULL    │
-- │ log_mel     │ DOUBLE[][]  │ YES     │ NULL    │ NULL    │ NULL    │
-- └─────────────┴─────────────┴─────────┴─────────┴─────────┴─────────┘
```



### 3. First Few Records
```sql
-- Preview the data (excluding log_mel arrays for readability)
SELECT rel_path, fold, class_id, class_name
FROM read_parquet('./.data/UrbanSound8K/processed/urbansound8k.parquet')
LIMIT 10;
```

## Class Distribution Analysis

### 4. Samples per Class
```sql
-- Count samples per class
SELECT 
    class_id,
    class_name,
    COUNT(*) as sample_count,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 2) as percentage
FROM read_parquet('./.data/UrbanSound8K/processed/urbansound8k.parquet')
GROUP BY class_id, class_name
ORDER BY class_id;
```

### 5. Class Distribution by Fold
```sql
-- Cross-tabulation of classes and folds
SELECT 
    class_name,
    SUM(CASE WHEN fold = 1 THEN 1 ELSE 0 END) as fold_1,
    SUM(CASE WHEN fold = 2 THEN 1 ELSE 0 END) as fold_2,
    SUM(CASE WHEN fold = 3 THEN 1 ELSE 0 END) as fold_3,
    SUM(CASE WHEN fold = 4 THEN 1 ELSE 0 END) as fold_4,
    SUM(CASE WHEN fold = 5 THEN 1 ELSE 0 END) as fold_5,
    SUM(CASE WHEN fold = 6 THEN 1 ELSE 0 END) as fold_6,
    SUM(CASE WHEN fold = 7 THEN 1 ELSE 0 END) as fold_7,
    SUM(CASE WHEN fold = 8 THEN 1 ELSE 0 END) as fold_8,
    SUM(CASE WHEN fold = 9 THEN 1 ELSE 0 END) as fold_9,
    SUM(CASE WHEN fold = 10 THEN 1 ELSE 0 END) as fold_10,
    COUNT(*) as total
FROM read_parquet('./.data/UrbanSound8K/processed/urbansound8k.parquet')
GROUP BY class_name
ORDER BY class_name;
```

## Fold-based Analysis

### 6. Samples per Fold
```sql
-- Distribution across folds
SELECT 
    fold,
    COUNT(*) as sample_count,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 2) as percentage
FROM read_parquet('./.data/UrbanSound8K/processed/urbansound8k.parquet')
GROUP BY fold
ORDER BY fold;
```

### 7. Training/Validation Split Analysis
```sql
-- Typical train/test split (fold 10 for testing)
SELECT 
    CASE WHEN fold = 10 THEN 'Test' ELSE 'Train' END as split,
    COUNT(*) as sample_count,
    COUNT(DISTINCT class_id) as unique_classes
FROM read_parquet('./.data/UrbanSound8K/processed/urbansound8k.parquet')
GROUP BY CASE WHEN fold = 10 THEN 'Test' ELSE 'Train' END;
```

## Spectrogram Data Analysis

### 8. Log-Mel Spectrogram Array Information
```sql
-- Analyze the shape and properties of log_mel_flat arrays
SELECT 
    class_name,
    fold,
    rel_path,
    array_length(log_mel_flat, 1) as flat_array_length,
    log_mel_shape as shape
FROM read_parquet('./.data/UrbanSound8K/processed/urbansound8k.parquet')
LIMIT 5;
```

### Log-Mel Spectrogram Shape Analysis
```sql
-- Get the shape of the log_mel spectrogram for each row
SELECT
    rel_path,
    class_name,
    log_mel_shape[1] AS n_mels,
    log_mel_shape[2] AS n_frames
FROM read_parquet('./.data/UrbanSound8K/processed/urbansound8k.parquet')
LIMIT 10;
```

```sql
SELECT
    class_name,
    MAX(log_mel_shape[1]) AS n_mels,
    MAX(log_mel_shape[2]) AS n_frames
FROM read_parquet('./.data/UrbanSound8K/processed/urbansound8k.parquet')
GROUP BY class_name;
-- All classes should have n_mels=64, n_frames=126
```

-- To get summary statistics for all shapes:
```sql
SELECT
    log_mel_shape[1] AS n_mels,
    log_mel_shape[2] AS n_frames,
    COUNT(*) AS count
FROM read_parquet('./.data/UrbanSound8K/processed/urbansound8k.parquet')
GROUP BY n_mels, n_frames
ORDER BY count DESC;
```

### 9. Array Statistics per Class
```sql
-- Basic statistics on log_mel_flat array lengths per class
SELECT 
    class_name,
    COUNT(*) as sample_count,
    MIN(array_length(log_mel_flat, 1)) as min_flat_length,
    MAX(array_length(log_mel_flat, 1)) as max_flat_length,
    ROUND(AVG(array_length(log_mel_flat, 1)), 2) as avg_flat_length
FROM read_parquet('./.data/UrbanSound8K/processed/urbansound8k.parquet')
GROUP BY class_name
ORDER BY class_name;
```

## File Path Analysis

### 10. File Extensions and Patterns
```sql
-- Analyze file patterns
SELECT 
    RIGHT(rel_path, 4) as file_extension,
    COUNT(*) as count
FROM read_parquet('./.data/UrbanSound8K/processed/urbansound8k.parquet')
GROUP BY RIGHT(rel_path, 4)
ORDER BY count DESC;
```

### 11. Files by Fold Directory
```sql
-- Verify fold directory structure
SELECT 
    fold,
    SUBSTRING(rel_path, 1, 15) as path_prefix,
    COUNT(*) as file_count
FROM read_parquet('./.data/UrbanSound8K/processed/urbansound8k.parquet')
GROUP BY fold, SUBSTRING(rel_path, 1, 15)
ORDER BY fold;
```

## Quality Checks

### 12. Data Completeness Check
```sql
-- Check for missing or null values
SELECT 
    COUNT(*) as total_rows,
    COUNT(rel_path) as non_null_paths,
    COUNT(fold) as non_null_folds,
    COUNT(class_id) as non_null_class_ids,
    COUNT(class_name) as non_null_class_names,
    COUNT(log_mel_flat) as non_null_spectrograms
FROM read_parquet('./.data/UrbanSound8K/processed/urbansound8k.parquet');
```

### 13. Duplicate Detection
```sql
-- Check for potential duplicates
SELECT 
    rel_path,
    COUNT(*) as occurrence_count
FROM read_parquet('./.data/UrbanSound8K/processed/urbansound8k.parquet')
GROUP BY rel_path
HAVING COUNT(*) > 1;
```

## Advanced Queries

### 14. Class Balance Analysis
```sql
-- Calculate class imbalance metrics
WITH class_counts AS (
    SELECT 
        class_name,
        COUNT(*) as count
    FROM read_parquet('./.data/UrbanSound8K/processed/urbansound8k.parquet')
    GROUP BY class_name
),
stats AS (
    SELECT 
        MIN(count) as min_count,
        MAX(count) as max_count,
        AVG(count) as avg_count
    FROM class_counts
)
SELECT 
    cc.class_name,
    cc.count,
    ROUND(cc.count / s.avg_count, 2) as balance_ratio,
    CASE 
        WHEN cc.count < s.avg_count * 0.8 THEN 'Underrepresented'
        WHEN cc.count > s.avg_count * 1.2 THEN 'Overrepresented'
        ELSE 'Balanced'
    END as balance_status
FROM class_counts cc
CROSS JOIN stats s
ORDER BY cc.count DESC;
```

### 15. Sample Selection for Analysis
```sql
-- Select specific samples for detailed analysis
SELECT 
    rel_path,
    class_name,
    fold,
    array_length(log_mel_flat, 1) as spectrogram_flat_size,
    log_mel_shape as spectrogram_shape
FROM read_parquet('./.data/UrbanSound8K/processed/urbansound8k.parquet')
WHERE class_name IN ('dog_bark', 'car_horn', 'siren')
    AND fold IN (1, 2)
ORDER BY class_name, fold
LIMIT 20;
```

## Export Queries

### 16. Export Class Summary
```sql
-- Create a summary table for reporting
COPY (
    SELECT 
        class_id,
        class_name,
        COUNT(*) as total_samples,
        COUNT(DISTINCT fold) as folds_present,
        MIN(fold) as min_fold,
        MAX(fold) as max_fold
    FROM read_parquet('./.data/UrbanSound8K/processed/urbansound8k.parquet')
    GROUP BY class_id, class_name
    ORDER BY class_id
) TO './.data/UrbanSound8K/processed/class_summary.csv' (HEADER);
```

### 17. Export Training Set Metadata
```sql
-- Export training set metadata (excluding fold 10)
COPY (
    SELECT rel_path, fold, class_id, class_name
    FROM read_parquet('./.data/UrbanSound8K/processed/urbansound8k.parquet')
    WHERE fold != 10
    ORDER BY fold, class_id
) TO './.data/UrbanSound8K/processed/train_metadata.csv' (HEADER);
```

## Usage Examples

### Connect to DuckDB and run queries:
```bash
# Start DuckDB CLI
duckdb

# Or run specific query from command line
duckdb -c "SELECT COUNT(*) FROM read_parquet('./.data/UrbanSound8K/processed/urbansound8k.parquet')"
```

### Python integration:
```python
import duckdb

# Connect and query
conn = duckdb.connect()
result = conn.execute("""
    SELECT class_name, COUNT(*) as count 
    FROM read_parquet('./.data/UrbanSound8K/processed/urbansound8k.parquet') 
    GROUP BY class_name
""").fetchall()
print(result)
```
