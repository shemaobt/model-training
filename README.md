# Bible Audio Acoustic Tokenization

Acoustic tokenization pipeline for Portuguese Bible audio using XLSR-53 and BPE.

## Overview

This project processes Portuguese Bible audio recordings to:
1. **Extract acoustic features** using Facebook's XLSR-53 multilingual speech model
2. **Discover acoustic units** via K-Means clustering (100 units)
3. **Train BPE tokenizer** to find recurring acoustic motifs/patterns
4. **Generate analysis** and validation of discovered patterns

## Pipeline Phases

### Phase 1: Acoustic Tokenization (`modal_phase1.py`)
- Converts MP3 to WAV (16kHz mono)
- Segments audio by silence detection
- Extracts XLSR-53 features (layer 14)
- Clusters features into 100 acoustic units

### Phase 2: BPE Motif Discovery (`modal_phase2.py`)
- Trains SentencePiece BPE on unit sequences
- Discovers acoustic motifs (recurring patterns)
- Analyzes motif frequencies

### Validation (`modal_checking.py`)
- Validates unit sequences
- Computes audio quality metrics
- Tests pipeline outputs

## Dataset

- **Source**: Portuguese Bible (Old and New Testament)
- **Total segments**: 18,072 audio segments
- **Total duration**: ~5,057 minutes (~84 hours)
- **Acoustic units**: 100 clusters
- **BPE vocabulary**: 100 motifs

## Results

Key findings (see `analyze_results.ipynb` for full analysis):
- All 100 acoustic units are used across the corpus
- Top 10 motifs cover ~20% of all tokens
- Average sequence length: 778.8 units per segment
- Unit frame rate: ~50ms per unit

## Running on Modal

### Prerequisites
```bash
pip install modal tqdm
python3 -m modal token set --token-id <your-id> --token-secret <your-secret>
```

### Upload Audio
```bash
# Local segmentation (recommended)
python3 segment_local.py

# Upload segments to Modal
python3 -m modal run upload_segments_to_modal.py
```

### Run Pipeline
```bash
# Phase 1: Acoustic tokenization (skip segmentation if already done)
python3 -m modal run --detach modal_phase1.py::main_skip_segmentation

# Phase 2: BPE training
python3 -m modal run --detach modal_phase2.py::main

# Validation
python3 -m modal run modal_checking.py::main
```

### Download Results
```bash
python3 -m modal volume get bible-audio-data phase2_output/ ./modal_downloads/phase2_outputs/
python3 -m modal volume get bible-audio-data checking_output/ ./modal_downloads/checking_outputs/
```

## Analysis Notebook

Open `analyze_results.ipynb` for interactive analysis:
- Motif frequency visualization
- Unit sequence analysis
- Audio playback and quality metrics
- Cluster statistics

## Files

### Scripts
- `modal_phase1.py` - Phase 1: Audio processing and acoustic tokenization
- `modal_phase2.py` - Phase 2: BPE tokenizer training
- `modal_checking.py` - Validation and audio synthesis testing
- `segment_local.py` - Local audio segmentation
- `upload_segments_to_modal.py` - Upload segments to Modal volume

### Notebooks
- `analyze_results.ipynb` - Results analysis with visualizations
- `Phase_1.ipynb` - Original Phase 1 notebook (reference)
- `Phase_2.ipynb` - Original Phase 2 notebook (reference)

### Data (not in repo - stored on Modal volume)
- `portuguese_kmeans.pkl` - K-Means model (100 clusters)
- `portuguese_corpus_timestamped.json` - Unit sequences with timestamps
- `portuguese_bpe.model` - Trained BPE model
- `motif_analysis.json` - Motif frequency analysis

## License

Private - shemaobt organization only.
