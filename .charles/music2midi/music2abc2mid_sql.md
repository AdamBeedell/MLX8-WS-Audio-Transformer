# Music2MIDI Dataset Exploration with DuckDB

This document contains SQL queries to explore and analyze the preprocessed music dataset stored in Parquet format.

## Setup

First, install DuckDB and connect to explore the dataset:

```bash
pip install duckdb
# Or: conda install -c conda-forge duckdb
```

```sql
-- Load the dataset
CREATE VIEW music_dataset AS 
SELECT * FROM read_parquet('../.data/preprocessed/music_dataset.parquet');
```

## Basic Dataset Overview

### 1. Dataset Statistics
```sql
-- Basic counts and success rate
SELECT 
    COUNT(*) as total_records,
    SUM(CASE WHEN processing_success THEN 1 ELSE 0 END) as successful_records,
    SUM(CASE WHEN processing_success THEN 0 ELSE 1 END) as failed_records,
    ROUND(AVG(CASE WHEN processing_success THEN 1.0 ELSE 0.0 END) * 100, 2) as success_rate_percent
FROM music_dataset;
```

### 2. Schema Information
```sql
-- Show dataset structure
DESCRIBE music_dataset;
```

### 3. Sample Data Preview
```sql
-- Preview first 5 successful records
SELECT 
    filename,
    duration_seconds,
    sampling_rate,
    tempo_bpm,
    key_signature,
    time_signature,
    token_count,
    processing_success
FROM music_dataset 
WHERE processing_success = true
LIMIT 5;
```

## Audio Data Analysis

### 4. Duration Statistics
```sql
-- Audio duration analysis
SELECT 
    ROUND(MIN(duration_seconds), 2) as min_duration,
    ROUND(MAX(duration_seconds), 2) as max_duration,
    ROUND(AVG(duration_seconds), 2) as avg_duration,
    ROUND(STDDEV(duration_seconds), 2) as stddev_duration,
    ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY duration_seconds), 2) as median_duration
FROM music_dataset 
WHERE processing_success = true;
```

### 5. Duration Distribution
```sql
-- Duration histogram (10 bins)
WITH duration_bins AS (
    SELECT 
        FLOOR(duration_seconds) as duration_bin,
        COUNT(*) as count
    FROM music_dataset 
    WHERE processing_success = true
    GROUP BY FLOOR(duration_seconds)
    ORDER BY duration_bin
)
SELECT 
    duration_bin || 's-' || (duration_bin + 1) || 's' as duration_range,
    count,
    REPEAT('█', CAST(count * 50.0 / MAX(count) OVER () AS INTEGER)) as histogram
FROM duration_bins;
```

### 6. Sampling Rate Analysis
```sql
-- Check sampling rate consistency
SELECT 
    sampling_rate,
    COUNT(*) as count,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) as percentage
FROM music_dataset 
WHERE processing_success = true
GROUP BY sampling_rate
ORDER BY count DESC;
```

## Musical Content Analysis

### 7. Tempo Analysis
```sql
-- Tempo statistics (excluding nulls)
SELECT 
    COUNT(*) as records_with_tempo,
    ROUND(MIN(tempo_bpm), 1) as min_tempo,
    ROUND(MAX(tempo_bpm), 1) as max_tempo,
    ROUND(AVG(tempo_bpm), 1) as avg_tempo,
    ROUND(STDDEV(tempo_bpm), 1) as stddev_tempo,
    ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY tempo_bpm), 1) as median_tempo
FROM music_dataset 
WHERE processing_success = true AND tempo_bpm IS NOT NULL;
```

### 8. Tempo Distribution
```sql
-- Tempo ranges
WITH tempo_ranges AS (
    SELECT 
        CASE 
            WHEN tempo_bpm < 60 THEN 'Very Slow (<60)'
            WHEN tempo_bpm < 80 THEN 'Slow (60-79)'
            WHEN tempo_bpm < 100 THEN 'Moderate (80-99)'
            WHEN tempo_bpm < 120 THEN 'Medium (100-119)'
            WHEN tempo_bpm < 140 THEN 'Fast (120-139)'
            WHEN tempo_bpm < 160 THEN 'Very Fast (140-159)'
            ELSE 'Extremely Fast (160+)'
        END as tempo_range,
        COUNT(*) as count
    FROM music_dataset 
    WHERE processing_success = true AND tempo_bpm IS NOT NULL
    GROUP BY 1
)
SELECT 
    tempo_range,
    count,
    ROUND(count * 100.0 / SUM(count) OVER (), 2) as percentage,
    REPEAT('█', CAST(count * 30.0 / MAX(count) OVER () AS INTEGER)) as bar_chart
FROM tempo_ranges
ORDER BY count DESC;
```

### 9. Key Signature Analysis
```sql
-- Most common keys
SELECT 
    key_signature,
    COUNT(*) as count,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) as percentage
FROM music_dataset 
WHERE processing_success = true AND key_signature IS NOT NULL
GROUP BY key_signature
ORDER BY count DESC
LIMIT 15;
```

### 10. Time Signature Analysis
```sql
-- Time signature distribution
SELECT 
    time_signature,
    COUNT(*) as count,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) as percentage
FROM music_dataset 
WHERE processing_success = true AND time_signature IS NOT NULL
GROUP BY time_signature
ORDER BY count DESC;
```

## Token Analysis

### 11. Token Count Statistics
```sql
-- ABC token statistics
SELECT 
    COUNT(*) as records_with_tokens,
    MIN(token_count) as min_tokens,
    MAX(token_count) as max_tokens,
    ROUND(AVG(token_count), 1) as avg_tokens,
    ROUND(STDDEV(token_count), 1) as stddev_tokens,
    ROUND(PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY token_count), 1) as q1_tokens,
    ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY token_count), 1) as median_tokens,
    ROUND(PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY token_count), 1) as q3_tokens
FROM music_dataset 
WHERE processing_success = true;
```

### 12. Token Count Distribution
```sql
-- Token count histogram
WITH token_bins AS (
    SELECT 
        CASE 
            WHEN token_count < 50 THEN '0-49'
            WHEN token_count < 100 THEN '50-99'
            WHEN token_count < 200 THEN '100-199'
            WHEN token_count < 300 THEN '200-299'
            WHEN token_count < 500 THEN '300-499'
            WHEN token_count < 1000 THEN '500-999'
            ELSE '1000+'
        END as token_range,
        COUNT(*) as count
    FROM music_dataset 
    WHERE processing_success = true
    GROUP BY 1
)
SELECT 
    token_range,
    count,
    ROUND(count * 100.0 / SUM(count) OVER (), 2) as percentage,
    REPEAT('█', CAST(count * 40.0 / MAX(count) OVER () AS INTEGER)) as visualization
FROM token_bins
ORDER BY count DESC;
```

## Quality Checks

### 13. Data Completeness
```sql
-- Check for missing or null values
SELECT 
    'filename' as field,
    SUM(CASE WHEN filename IS NULL THEN 1 ELSE 0 END) as null_count,
    SUM(CASE WHEN filename = '' THEN 1 ELSE 0 END) as empty_count
FROM music_dataset WHERE processing_success = true
UNION ALL
SELECT 
    'abc_string',
    SUM(CASE WHEN abc_string IS NULL THEN 1 ELSE 0 END),
    SUM(CASE WHEN abc_string = '' THEN 1 ELSE 0 END)
FROM music_dataset WHERE processing_success = true
UNION ALL
SELECT 
    'tempo_bpm',
    SUM(CASE WHEN tempo_bpm IS NULL THEN 1 ELSE 0 END),
    0
FROM music_dataset WHERE processing_success = true
UNION ALL
SELECT 
    'key_signature',
    SUM(CASE WHEN key_signature IS NULL THEN 1 ELSE 0 END),
    SUM(CASE WHEN key_signature = '' THEN 1 ELSE 0 END)
FROM music_dataset WHERE processing_success = true;
```

### 14. Outlier Detection
```sql
-- Find potential outliers
WITH stats AS (
    SELECT 
        AVG(duration_seconds) as avg_duration,
        STDDEV(duration_seconds) as std_duration,
        AVG(token_count) as avg_tokens,
        STDDEV(token_count) as std_tokens
    FROM music_dataset WHERE processing_success = true
)
SELECT 
    filename,
    duration_seconds,
    token_count,
    tempo_bpm,
    CASE 
        WHEN ABS(duration_seconds - stats.avg_duration) > 3 * stats.std_duration THEN 'Duration Outlier'
        WHEN ABS(token_count - stats.avg_tokens) > 3 * stats.std_tokens THEN 'Token Outlier'
        ELSE 'Normal'
    END as outlier_type
FROM music_dataset, stats
WHERE processing_success = true 
    AND (ABS(duration_seconds - stats.avg_duration) > 3 * stats.std_duration 
         OR ABS(token_count - stats.avg_tokens) > 3 * stats.std_tokens)
ORDER BY duration_seconds DESC;
```

## Content Exploration

### 15. ABC Notation Samples
```sql
-- Show some ABC content examples
SELECT 
    filename,
    LEFT(abc_string, 200) || '...' as abc_preview,
    token_count,
    duration_seconds
FROM music_dataset 
WHERE processing_success = true 
    AND abc_string IS NOT NULL
ORDER BY token_count DESC
LIMIT 5;
```

### 16. Musical Complexity Analysis
```sql
-- Correlation between duration and token count
SELECT 
    ROUND(CORR(duration_seconds, token_count), 3) as duration_token_correlation,
    ROUND(CORR(tempo_bpm, token_count), 3) as tempo_token_correlation,
    ROUND(CORR(duration_seconds, tempo_bpm), 3) as duration_tempo_correlation
FROM music_dataset 
WHERE processing_success = true 
    AND tempo_bpm IS NOT NULL;
```

### 17. Chunk Duration Validation
```sql
-- Validate chunk duration targets vs actual
SELECT 
    chunk_duration_target,
    COUNT(*) as count,
    ROUND(AVG(duration_seconds), 2) as avg_actual_duration,
    ROUND(MIN(duration_seconds), 2) as min_actual_duration,
    ROUND(MAX(duration_seconds), 2) as max_actual_duration,
    SUM(CASE WHEN ABS(duration_seconds - chunk_duration_target) > 1.0 THEN 1 ELSE 0 END) as off_target_count
FROM music_dataset 
WHERE processing_success = true
GROUP BY chunk_duration_target
ORDER BY chunk_duration_target;
```

## Advanced Queries

### 18. Dataset Summary Report
```sql
-- Comprehensive dataset summary
WITH summary AS (
    SELECT 
        COUNT(*) as total_records,
        SUM(CASE WHEN processing_success THEN 1 ELSE 0 END) as successful_records,
        ROUND(AVG(duration_seconds), 2) as avg_duration,
        ROUND(AVG(token_count), 1) as avg_tokens,
        COUNT(DISTINCT key_signature) as unique_keys,
        COUNT(DISTINCT time_signature) as unique_time_sigs,
        ROUND(AVG(tempo_bpm), 1) as avg_tempo,
        SUM(duration_seconds) / 3600.0 as total_hours
    FROM music_dataset 
    WHERE processing_success = true
)
SELECT 
    'Dataset Summary' as metric_group,
    'Total Records: ' || total_records || 
    ' | Success Rate: ' || ROUND(successful_records * 100.0 / total_records, 1) || '%' ||
    ' | Avg Duration: ' || avg_duration || 's' ||
    ' | Avg Tokens: ' || avg_tokens ||
    ' | Total Audio: ' || ROUND(total_hours, 1) || ' hours' ||
    ' | Keys: ' || unique_keys ||
    ' | Time Signatures: ' || unique_time_sigs ||
    ' | Avg Tempo: ' || COALESCE(CAST(avg_tempo AS VARCHAR), 'N/A') || ' BPM'
    as summary
FROM summary;
```

### 19. Export Sample for Manual Review
```sql
-- Export a sample for manual inspection
COPY (
    SELECT 
        filename,
        duration_seconds,
        tempo_bpm,
        key_signature,
        time_signature,
        token_count,
        LEFT(abc_string, 500) as abc_sample
    FROM music_dataset 
    WHERE processing_success = true
    ORDER BY RANDOM()
    LIMIT 10
) TO '../.data/preprocessed/sample_review.csv' (HEADER);
```

## Usage Example

To run these queries in Python:

```python
import duckdb

# Connect and run queries
conn = duckdb.connect()

# Basic dataset info
result = conn.execute("""
    SELECT COUNT(*) as total, 
           SUM(CASE WHEN processing_success THEN 1 ELSE 0 END) as successful
    FROM read_parquet('../.data/preprocessed/music_dataset.parquet')
""").fetchone()

print(f"Total records: {result[0]}, Successful: {result[1]}")
```
