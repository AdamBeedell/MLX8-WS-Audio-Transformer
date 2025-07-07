# DuckDB SQL Queries for UrbanSound8K Dataset

This file contains sample DuckDB SQL queries to explore the processed UrbanSound8K dataset stored in parquet format.

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
-- Analyze the shape and properties of log_mel arrays
SELECT 
    class_name,
    fold,
    rel_path,
    len(log_mel) as array_length,
    -- First few elements for inspection
    log_mel[1:5] as first_elements
FROM read_parquet('./.data/UrbanSound8K/processed/urbansound8k.parquet')
LIMIT 5;
```

### Log-Mel Spectrogram Shape Analysis
```sql
-- Get the shape of the log_mel array for each row
SELECT
    rel_path,
    class_name,
    array_length(log_mel, 1) AS n_mels,
    array_length(log_mel[1], 1) AS n_frames
FROM read_parquet('./.data/UrbanSound8K/processed/urbansound8k.parquet')
LIMIT 10;
```

```sql
SELECT
      class_name,
      max(array_length(log_mel, 1)) AS n_mels,
      MAX(array_length(log_mel[1], 1)) AS n_frames
  FROM read_parquet('./.data/UrbanSound8K/processed/urbansound8k.parquet') GROUP by class_name;
-- ┌──────────────────┬────────┬──────────┐
-- │    class_name    │ n_mels │ n_frames │
-- │     varchar      │ int64  │  int64   │
-- ├──────────────────┼────────┼──────────┤
-- │ air_conditioner  │     64 │      126 │
-- │ street_music     │     64 │      126 │
-- │ children_playing │     64 │      126 │
-- │ drilling         │     64 │      126 │
-- │ jackhammer       │     64 │      126 │
-- │ gun_shot         │     64 │      126 │
-- │ dog_bark         │     64 │      126 │
-- │ siren            │     64 │      126 │
-- │ car_horn         │     64 │      126 │
-- │ engine_idling    │     64 │      126 │
-- ├──────────────────┴────────┴──────────┤
-- │ 10 rows                    3 columns │
-- └──────────────────────────────────────┘
```

-- To get summary statistics for all shapes:
```sql
SELECT
    array_length(log_mel, 1) AS n_mels,
    array_length(log_mel[1], 1) AS n_frames,
    COUNT(*) AS count
FROM read_parquet('./.data/UrbanSound8K/processed/urbansound8k.parquet')
GROUP BY n_mels, n_frames
ORDER BY count DESC;
```
### 9. Array Statistics per Class
```sql
-- Basic statistics on log_mel array lengths per class
SELECT 
    class_name,
    COUNT(*) as sample_count,
    MIN(len(log_mel)) as min_array_length,
    MAX(len(log_mel)) as max_array_length,
    ROUND(AVG(len(log_mel)), 2) as avg_array_length
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
    COUNT(log_mel) as non_null_spectrograms
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
    len(log_mel) as spectrogram_size
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
