# Modal Deployment for Phase 1 Notebook

This directory contains scripts to run the Phase 1 notebook on Modal servers.

## Setup

1. **Install Modal CLI and dependencies** (if not already installed):
   ```bash
   python3 -m pip install modal tqdm
   ```

2. **Set Modal credentials** (already done):
   ```bash
   python3 -m modal token set --token-id ak-Nt2Hce8cPk2ERcyeOqWvki --token-secret as-IG0LMImu6tzd4id1X4X721
   ```
   
   **Note:** Use `python3 -m modal` if the `modal` command is not in your PATH.

## Files

- `modal_phase1.py` - Phase 1: MP3→WAV conversion, silence-based segmentation, acoustic tokenization
- `modal_phase2.py` - Phase 2: BPE tokenizer training on acoustic units (discovers acoustic motifs)
- `upload_audio_to_modal.py` - Script to upload Bible audio files to Modal volume
- `upload_audio_streaming.py` - Streaming upload with progress (recommended)
- `Phase_1.ipynb` - Updated notebook that uses local Bible audio directories
- `Phase_2.ipynb` - Phase 2 notebook (BPE training)

## Usage

**Note:** If `modal` command is not found, use `python3 -m modal` instead.

### Step 1: Upload Audio Files to Modal ⚠️ REQUIRED FIRST

**You must upload files before running the processing pipeline!**

**Recommended:** Use the streaming upload with progress (checks existing files, only uploads missing):

```bash
cd /Users/joao/Desktop/work/shema/shemaobt/model-training

# Use system python3 (not venv) - checks existing files, only uploads what's missing
# Memory-efficient to prevent crashes
/usr/local/bin/python3 -m modal run upload_audio_streaming.py
```

This script will:
- ✅ Check what files are already in Modal volume
- ✅ Only upload missing files (resumes from where it stopped)
- ✅ Upload in small batches (2 files) to prevent memory issues
- ✅ Show detailed progress

**Alternative:** Use the mount-based upload (slower first time, but caches files):

```bash
/usr/local/bin/python3 -m modal run upload_audio_to_modal.py
```

Or if you have modal in your PATH:
```bash
python3 -m modal run upload_audio_to_modal.py
```

**Note:** If you get "No module named modal", you're in a venv. Either:
- Use system python: `/usr/local/bin/python3 -m modal run upload_audio_to_modal.py`
- Or install modal in venv: `pip install modal tqdm`

This will:
- Read MP3 files from:
  - `/Users/joao/Desktop/work/shema/shemaobt/scripts/Antigo_Testamento_COMPLETO`
  - `/Users/joao/Desktop/work/shema/shemaobt/scripts/NOVO_TESTAMENTO_COMPLETO`
- Upload them to Modal volume `bible-audio-data`

### Step 2: Run the Processing Pipeline

Run the complete pipeline on Modal:

```bash
python3 -m modal run modal_phase1.py
```

Or if `modal` is in your PATH:
```bash
modal run modal_phase1.py
```

This will:
1. **Phase 1**: Convert all MP3 files to 16kHz mono WAV format
2. **Phase 2**: Extract features using XLSR-53 and perform acoustic tokenization

### Running Individual Phases

You can also run phases separately:

```python
import modal

app = modal.App.lookup("bible-audio-training")

# Phase 1 functions
app.convert_mp3_to_wav.remote()           # MP3 to WAV conversion
app.segment_by_silence.remote()          # Segment by silence
app.acoustic_tokenization.remote()       # Acoustic tokenization

# Phase 2 functions
app.train_bpe_tokenizer.remote(vocab_size=500)  # Train BPE tokenizer
app.analyze_motifs.remote()              # Analyze discovered motifs
```

## Local Notebook Usage

The notebook `Phase_1.ipynb` has been updated to use local Bible audio directories:

- **Antigo Testamento**: `/Users/joao/Desktop/work/shema/shemaobt/scripts/Antigo_Testamento_COMPLETO`
- **Novo Testamento**: `/Users/joao/Desktop/work/shema/shemaobt/scripts/NOVO_TESTAMENTO_COMPLETO`

The notebook will:
1. Copy MP3 files from these directories to a local project directory
2. Convert them to WAV format
3. Perform acoustic tokenization

## Output

Results are stored in Modal volume `bible-audio-data`:

**Phase 1 Outputs:**
- `/mnt/audio_data/raw_audio/` - Original MP3 files
- `/mnt/audio_data/converted_audio/` - Converted WAV files
- `/mnt/audio_data/segmented_audio/` - Audio segments (by silence)
- `/mnt/audio_data/portuguese_units/` - Acoustic tokenization outputs
  - `portuguese_kmeans.pkl` - Trained K-Means model
  - `portuguese_corpus_timestamped.json` - Full corpus with timestamps
  - `all_units_for_bpe.txt` - Unit sequences for BPE training
  - Individual `.units.txt` files for each segment

**Phase 2 Outputs:**
- `/mnt/audio_data/phase2_output/` - BPE tokenizer outputs
  - `portuguese_bpe.model` - Trained BPE tokenizer model
  - `bpe_train.txt` - Training data used
  - `motif_analysis.json` - Motif frequency analysis

## Notes

- The Modal deployment uses GPU (A10G for Phase 2, CPU for Phase 1) for faster processing
- Timeouts are set to 1 hour for Phase 1 and 2 hours for Phase 2
- All data is persisted in Modal volumes, so you can resume if interrupted
- The notebook can also be run locally if you have the required dependencies installed
