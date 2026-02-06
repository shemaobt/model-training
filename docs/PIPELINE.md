# Pipeline Execution Guide

This document explains how to run each phase of the acoustic tokenization and vocoder training pipeline.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Phase 0: Data Preparation](#phase-0-data-preparation)
3. [Phase 1: Acoustic Tokenization](#phase-1-acoustic-tokenization)
4. [Phase 2: BPE Motif Discovery](#phase-2-bpe-motif-discovery)
5. [Phase 3: Vocoder Training](#phase-3-vocoder-training)
6. [Testing & Validation](#testing--validation)
7. [Monitoring & Troubleshooting](#monitoring--troubleshooting)

---

## Prerequisites

### Local Setup

```bash
# Install dependencies
pip install modal torch torchaudio transformers scikit-learn sentencepiece tqdm librosa soundfile pydub

# Authenticate with Modal
python3 -m modal token set --token-id <your-id> --token-secret <your-secret>
```

### Modal Volume Structure

All data is stored in a Modal volume named `bible-audio-data`:

```
/mnt/audio_data/
├── raw_audio/                    # Original MP3 files
├── converted_audio/              # WAV files (16kHz mono)
├── segmented_audio/              # Portuguese segments
├── segmented_audio_satere/       # Sateré segments
├── portuguese_units/             # Phase 1 outputs (Portuguese)
├── satere_units/                 # Phase 1 outputs (Sateré)
├── phase2_output/                # Phase 2 outputs (Portuguese)
├── phase2_satere_output/         # Phase 2 outputs (Sateré)
├── vocoder_checkpoints/          # V1 vocoder (Portuguese)
├── vocoder_v2_checkpoints/       # V2 vocoder (Portuguese)
├── vocoder_v2_satere_checkpoints/# V2 vocoder (Sateré)
├── vocoder_test_output/          # V1 test results
└── vocoder_v2_test_output/       # V2 test results
```

---

## Phase 0: Data Preparation

### Step 0.1: Segment Audio Locally

Audio segmentation is CPU-intensive and runs better on your local machine.

```bash
# Portuguese (default)
python3 scripts/segment_audio.py --language portuguese

# Sateré-Mawé
python3 scripts/segment_audio.py --language satere
```

**Processing Steps**:

| Step | Description | Output |
|------|-------------|--------|
| 1 | Load MP3 files | Raw audio |
| 2 | Convert to WAV (16kHz mono) | Normalized audio |
| 3 | Detect silence (-40dB threshold) | Silence regions |
| 4 | Split at pauses (min 0.5s silence) | Segments |
| 5 | Filter by length (2-30 seconds) | Valid segments |

**Typical Output**:
- Portuguese: ~18,000 segments, average ~17 seconds
- Sateré: Varies based on source material

**Output Directory**: `local_segments_<language>/`

### Step 0.2: Upload to Modal

```bash
# Portuguese
python3 -m modal run scripts/upload_to_modal.py --language portuguese

# Sateré
python3 -m modal run scripts/upload_to_modal.py --language satere
```

**Features**:
- Resume capability (skips already uploaded files)
- Progress bar with ETA
- Batch processing (100 files at a time)
- Automatic directory creation

---

## Phase 1: Acoustic Tokenization

### What It Does

| Step | Process | Output |
|------|---------|--------|
| 1 | Load Model (MMS-300M default) | Feature extractor |
| 2 | Extract Layer 14 features | 1024-dim vectors per 20ms |
| 3 | Fit K-Means (k=100) | Cluster centroids |
| 4 | Predict cluster IDs | Unit sequences |
| 5 | Save with timestamps | Aligned corpus |

### Run Command

```bash
# Portuguese (default model: MMS-300M)
python3 -m modal run --detach src/training/phase1_acoustic.py::main_skip_segmentation

# Use legacy XLSR-53
python3 -m modal run --detach src/training/phase1_acoustic.py::main_skip_segmentation --model xlsr-53

# Sateré (with default MMS-300M)
python3 -m modal run --detach src/training/phase1_acoustic.py::main_skip_segmentation --language satere
```

### Output Files

| File | Description | Size |
|------|-------------|------|
| `<lang>_kmeans.pkl` | Trained K-Means model | ~400 KB |
| `<lang>_corpus_timestamped.json` | Unit sequences with timestamps | ~50 MB |
| `all_units_for_bpe.txt` | All units as text (for Phase 2) | ~20 MB |

### Checkpointing

The script saves checkpoints every 5 minutes:
- `checkpoint.json` - Current state
- `processed_files.txt` - Completed files

**Auto-resume**: If interrupted, restart with the same command.

### Resource Usage

| Resource | Specification |
|----------|---------------|
| GPU | A10G (24GB) |
| Memory | ~16 GB |
| Time | 2-4 hours (depends on data size) |
| Storage | ~100 MB outputs |

---

## Phase 2: BPE Motif Discovery

### What It Does

| Step | Process | Output |
|------|---------|--------|
| 1 | Load unit sequences | Text corpus |
| 2 | Train SentencePiece BPE | Merge rules |
| 3 | Build vocabulary | Token → motif mapping |
| 4 | Analyze frequencies | Motif statistics |

### Run Command

```bash
# Portuguese (default)
python3 -m modal run --detach src/training/phase2_bpe.py::main

# Sateré
python3 -m modal run --detach src/training/phase2_bpe.py::main --language satere

# Custom vocabulary size
python3 -m modal run --detach src/training/phase2_bpe.py::main --vocab-size 300 --min-frequency 10
```

### Output Files

| File | Description |
|------|-------------|
| `<lang>_bpe.model` | Trained SentencePiece model |
| `<lang>_bpe.vocab` | Vocabulary (unit sequences → tokens) |
| `motif_analysis.json` | Top motifs with frequencies |

### Vocabulary Size Auto-Adjustment

If the requested vocabulary size is too large for the data:

```python
# Automatic fallback
try:
    train(vocab_size=500)
except RuntimeError as e:
    if "Vocabulary size too high" in str(e):
        max_size = extract_max_from_error(e)
        train(vocab_size=max_size - 10)
```

### Resource Usage

| Resource | Specification |
|----------|---------------|
| GPU | Not required (CPU only) |
| Memory | ~8 GB |
| Time | 15-30 minutes |
| Storage | ~10 MB outputs |

---

## Phase 3: Vocoder Training

### V1 vs V2 Comparison

| Aspect | V1 | V2 |
|--------|----|----|
| Architecture | Simple TransConv | HiFi-GAN + MRF |
| Pitch | Not supported | 32-bin conditioning |
| Quality | Robotic | Natural prosody |
| Training Time | 2-4 hours | 4-8 hours |
| Recommended | No | Yes |

### V2 Training (Recommended)

```bash
# Portuguese (default)
python3 -m modal run --detach src/training/phase3_vocoder_v2.py::main

# Sateré
python3 -m modal run --detach src/training/phase3_vocoder_v2.py::main --language satere

# Custom parameters
python3 -m modal run --detach src/training/phase3_vocoder_v2.py::main \
    --language satere \
    --epochs 1000 \
    --batch-size 12 \
    --segment-length 32000 \
    --patience 100

# Resume from checkpoint
python3 -m modal run --detach src/training/phase3_vocoder_v2.py::main --resume v2_latest.pt
```

### V1 Training (Legacy)

```bash
# Portuguese only (V1 doesn't support --language)
python3 -m modal run --detach src/training/phase3_vocoder.py::main

# Custom parameters
python3 -m modal run --detach src/training/phase3_vocoder.py::main \
    --epochs 500 \
    --batch-size 16
```

### Training Parameters

| Parameter | V1 Default | V2 Default | Description |
|-----------|------------|------------|-------------|
| `epochs` | 500 | 1000 | Maximum training epochs |
| `batch_size` | 16 | 12 | Samples per batch |
| `lr` / `lr_g` / `lr_d` | 0.0002 | 0.0002 | Learning rates |
| `segment_length` | 16000 | 32000 | Audio samples per segment |
| `patience` | 50 | 100 | Early stopping patience |
| `save_every` | 20 | 25 | Checkpoint frequency |

### Output Files

| File | Description |
|------|-------------|
| `v2_best.pt` | Best model (lowest mel loss) |
| `v2_latest.pt` | Latest checkpoint |
| `v2_epoch_XXXX.pt` | Periodic checkpoints |
| `sample_v2_XXXX.wav` | Generated samples |
| `training_log_v2.json` | Loss history |

### Resource Usage

| Resource | V1 | V2 |
|----------|----|----|
| GPU | A10G (24GB) | A10G (24GB) |
| GPU Memory | ~12 GB | ~20 GB |
| Time | 2-4 hours | 4-8 hours |
| Storage | ~500 MB | ~800 MB |

---

## Testing & Validation

### V2 Vocoder Test

```bash
# Portuguese
python3 -m modal run src/training/vocoder_test_v2.py::main --num-samples 50

# Sateré
python3 -m modal run src/training/vocoder_test_v2.py::main --language satere --num-samples 50

# Specific checkpoint
python3 -m modal run src/training/vocoder_test_v2.py::main --checkpoint v2_epoch_0500.pt
```

### V1 Vocoder Test

```bash
python3 -m modal run src/training/vocoder_test.py::main --num-samples 20
```

### Metrics Computed

| Metric | Description | Good Value |
|--------|-------------|------------|
| **SNR** | Signal-to-noise ratio | > 5 dB |
| **MCD** | Mel Cepstral Distortion | < 8 |
| **F0 RMSE** | Pitch accuracy (V2 only) | < 20 Hz |
| **Length Match** | Synth/Original ratio | 0.95 - 1.05 |

### Download Results

```bash
# V2 test audio
modal volume get bible-audio-data vocoder_v2_test_output/ ./results/

# V1 test audio
modal volume get bible-audio-data vocoder_test_output/ ./results/

# Sateré V2 test audio
modal volume get bible-audio-data vocoder_v2_satere_test_output/ ./results/
```

**Output Files**:
- `v2_orig_XXXX.wav` - Original audio
- `v2_synth_XXXX.wav` - Synthesized audio
- `test_results_v2.json` - Metrics summary

---

## Monitoring & Troubleshooting

### Modal Dashboard

Monitor all jobs at: `https://modal.com/apps/<org>/main/`

### Check Volume Contents

```bash
# List all directories
modal volume ls bible-audio-data

# List specific directory
modal volume ls bible-audio-data vocoder_v2_checkpoints/

# Check file sizes
modal volume ls bible-audio-data portuguese_units/
```

### Download Outputs

```bash
# Phase 1 outputs
modal volume get bible-audio-data portuguese_units/ ./downloads/phase1/

# Phase 2 outputs
modal volume get bible-audio-data phase2_output/ ./downloads/phase2/

# V2 Vocoder checkpoints
modal volume get bible-audio-data vocoder_v2_checkpoints/ ./downloads/vocoder_v2/

# Training logs
modal volume get bible-audio-data vocoder_v2_checkpoints/training_log_v2.json ./
```

### Common Issues & Solutions

#### CUDA Out of Memory

**Symptom**: Training crashes with OOM error

**Solutions**:
```bash
# Reduce batch size
python3 -m modal run --detach src/training/phase3_vocoder_v2.py::main --batch-size 8

# Reduce segment length
python3 -m modal run --detach src/training/phase3_vocoder_v2.py::main --segment-length 16000
```

#### Discriminator Collapse (D Loss = 0)

**Symptom**: D loss drops to 0, generated audio becomes noise

**Solutions**:
1. Add gradient penalty (modify code)
2. Reduce generator learning rate
3. Train D more steps per G step

#### Early Stopping Too Soon

**Symptom**: Training stops before quality improves

**Solution**:
```bash
python3 -m modal run --detach src/training/phase3_vocoder_v2.py::main --patience 150
```

#### Job Timeout

**Symptom**: Modal job terminates unexpectedly

**Solution**: Increase timeout in script (default is 12 hours for vocoder)
```python
@app.function(timeout=43200)  # 12 hours
```

#### Phase 1 Stalls

**Symptom**: Progress stops during feature extraction

**Cause**: Usually memory pressure from XLSR-53

**Solution**: Reduce batch processing, restart (will resume from checkpoint)

---

## Full Pipeline Commands

### Portuguese (Complete)

```bash
# Step 1: Segment locally
python3 scripts/segment_audio.py --language portuguese

# Step 2: Upload to Modal
python3 -m modal run scripts/upload_to_modal.py --language portuguese

# Step 3: Run Phase 1
python3 -m modal run --detach src/training/phase1_acoustic.py::main_skip_segmentation --language portuguese

# Step 4: Run Phase 2
python3 -m modal run --detach src/training/phase2_bpe.py::main --language portuguese

# Step 5: Run Phase 3 V2
python3 -m modal run --detach src/training/phase3_vocoder_v2.py::main --language portuguese

# Step 6: Test
python3 -m modal run src/training/vocoder_test_v2.py::main --language portuguese
```

### Sateré (Complete)

```bash
# Step 1: Segment locally
python3 scripts/segment_audio.py --language satere

# Step 2: Upload to Modal
python3 -m modal run scripts/upload_to_modal.py --language satere

# Step 3: Run Phase 1
python3 -m modal run --detach src/training/phase1_acoustic.py::main_skip_segmentation --language satere

# Step 4: Run Phase 2
python3 -m modal run --detach src/training/phase2_bpe.py::main --language satere

# Step 5: Run Phase 3 V2
python3 -m modal run --detach src/training/phase3_vocoder_v2.py::main --language satere

# Step 6: Test
python3 -m modal run src/training/vocoder_test_v2.py::main --language satere
```

### All-in-One (Using run_full_pipeline.py)

```bash
# Portuguese with V2 vocoder
python3 -m modal run src/training/run_full_pipeline.py::main --language portuguese

# Sateré with V2 vocoder
python3 -m modal run src/training/run_full_pipeline.py::main --language satere

# Skip already completed phases
python3 -m modal run src/training/run_full_pipeline.py::main --language satere --phases 3
```
