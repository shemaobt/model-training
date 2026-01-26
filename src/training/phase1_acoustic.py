# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.20.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Phase 1: Acoustic Tokenization
#
# This script performs the first phase of the acoustic tokenization pipeline:
#
# 1. **Convert MP3 to WAV**: Standardize audio to 16kHz mono
# 2. **Segment by Silence**: Split long files at natural pauses
# 3. **Extract XLSR-53 Features**: Layer 14 hidden states (1024-dim)
# 4. **K-Means Clustering**: Discover 100 acoustic units
#
# ## Architecture
#
# ```
# MP3 Files → WAV (16kHz) → Segments → XLSR-53 → K-Means → Unit Sequences
# ```
#
# ## Run on Modal
#
# ```bash
# # Full pipeline
# python3 -m modal run --detach src/training/phase1_acoustic.py::main
#
# # Skip segmentation (use existing segments)
# python3 -m modal run --detach src/training/phase1_acoustic.py::main_skip_segmentation
# ```

# %% [markdown]
# ## Modal Configuration

# %%
import modal
import os
from pathlib import Path

# %%
# Create Modal app
app = modal.App("bible-audio-training")

# Create persistent volume for audio files and outputs
audio_volume = modal.Volume.from_name(
    "bible-audio-data",
    create_if_missing=True
)

# %%
# Create image with dependencies
image = (
    modal.Image.debian_slim()
    .apt_install("ffmpeg", "libsndfile1")
    .pip_install(
        "torch>=2.0.0",
        "torchaudio>=2.0.0",
        "transformers>=4.40.0",
        "sentencepiece",
        "scikit-learn",
        "joblib",
        "soundfile>=0.12.0",
        "tqdm",
        "numpy<2",
        "librosa>=0.10.0",
    )
)

# %%
# Mount points
AUDIO_MOUNT = "/mnt/audio_data"
RAW_AUDIO_DIR = f"{AUDIO_MOUNT}/raw_audio"
CONVERTED_DIR = f"{AUDIO_MOUNT}/converted_audio"
SEGMENTED_DIR = f"{AUDIO_MOUNT}/segmented_audio"
OUTPUT_DIR = f"{AUDIO_MOUNT}/portuguese_units"

# %% [markdown]
# ## Step 1: MP3 to WAV Conversion
#
# Convert all MP3 files to 16kHz mono WAV format using ffmpeg.

# %%
@app.function(
    image=image,
    volumes={AUDIO_MOUNT: audio_volume},
    timeout=3600,
)
def convert_mp3_to_wav():
    """Convert MP3 files to 16kHz mono WAV."""
    import subprocess
    from tqdm import tqdm
    
    os.makedirs(CONVERTED_DIR, exist_ok=True)
    
    # Find all MP3 files
    mp3_files = []
    for root, dirs, files in os.walk(RAW_AUDIO_DIR):
        for f in files:
            if f.endswith('.mp3'):
                mp3_files.append(os.path.join(root, f))
    
    print(f"Found {len(mp3_files)} MP3 files")
    
    # Check what's already converted
    existing = set()
    if os.path.exists(CONVERTED_DIR):
        existing = {f.replace('.wav', '.mp3') for f in os.listdir(CONVERTED_DIR) if f.endswith('.wav')}
    
    to_convert = [f for f in mp3_files if os.path.basename(f) not in existing]
    print(f"Already converted: {len(existing)}")
    print(f"To convert: {len(to_convert)}")
    
    for mp3_path in tqdm(to_convert, desc="Converting to WAV"):
        mp3_file = os.path.basename(mp3_path)
        wav_path = os.path.join(CONVERTED_DIR, mp3_file.replace('.mp3', '.wav'))
        
        cmd = ["ffmpeg", "-y", "-i", mp3_path, "-ar", "16000", "-ac", "1", wav_path]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"Error converting {mp3_path}: {result.stderr}")
    
    audio_volume.commit()
    print(f"\n✓ Done! Converted files are in: {CONVERTED_DIR}")
    return len(to_convert)

# %% [markdown]
# ## Step 2: Silence-Based Segmentation
#
# Split audio files at natural pauses (silence > 0.5s).
# This creates segments of 2-120 seconds for efficient processing.

# %%
@app.function(
    image=image,
    volumes={AUDIO_MOUNT: audio_volume},
    timeout=7200,
)
def segment_by_silence():
    """Segment audio files by silence/pauses."""
    import soundfile as sf
    import librosa
    import numpy as np
    from tqdm import tqdm
    import subprocess
    
    os.makedirs(SEGMENTED_DIR, exist_ok=True)
    
    wav_files = sorted([
        os.path.join(CONVERTED_DIR, f) 
        for f in os.listdir(CONVERTED_DIR) 
        if f.endswith('.wav')
    ])
    print(f"Found {len(wav_files)} WAV files to segment")
    
    # Check what's already segmented
    existing_segments = set()
    if os.path.exists(SEGMENTED_DIR):
        for f in os.listdir(SEGMENTED_DIR):
            if f.endswith('.wav'):
                base = f.rsplit('_seg_', 1)[0] if '_seg_' in f else f.replace('.wav', '')
                existing_segments.add(base)
    
    # Parameters
    SILENCE_THRESHOLD = -40  # dB
    MIN_SILENCE_DURATION = 0.5  # seconds
    MIN_SEGMENT_DURATION = 2.0  # seconds
    MAX_SEGMENT_DURATION = 120.0  # seconds
    
    total_segments = 0
    files_to_segment = [
        f for f in wav_files 
        if os.path.splitext(os.path.basename(f))[0] not in existing_segments
    ]
    
    print(f"Already segmented: {len(existing_segments)} files")
    print(f"To segment: {len(files_to_segment)} files")
    
    for wav_path in tqdm(files_to_segment, desc="Segmenting by silence"):
        try:
            audio, sr = librosa.load(wav_path, sr=16000, mono=True)
            duration = len(audio) / sr
            
            # Compute RMS energy
            frame_length = int(0.025 * sr)
            hop_length = int(0.010 * sr)
            rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
            rms_db = librosa.power_to_db(rms**2, ref=np.max)
            
            # Find silence regions
            silence_mask = rms_db < SILENCE_THRESHOLD
            segment_boundaries = [0]
            
            in_silence = False
            silence_start = None
            
            for i, is_silent in enumerate(silence_mask):
                frame_time = i * hop_length / sr
                
                if is_silent and not in_silence:
                    in_silence = True
                    silence_start = frame_time
                elif not is_silent and in_silence:
                    if silence_start is not None and (frame_time - silence_start) >= MIN_SILENCE_DURATION:
                        boundary = (silence_start + frame_time) / 2
                        segment_boundaries.append(boundary)
                    in_silence = False
                    silence_start = None
            
            segment_boundaries.append(duration)
            
            # Create segments
            base_name = os.path.splitext(os.path.basename(wav_path))[0]
            segment_num = 0
            
            for i in range(len(segment_boundaries) - 1):
                start_time = segment_boundaries[i]
                end_time = segment_boundaries[i + 1]
                segment_duration = end_time - start_time
                
                if segment_duration < MIN_SEGMENT_DURATION:
                    continue
                
                if segment_duration > MAX_SEGMENT_DURATION:
                    num_splits = int(np.ceil(segment_duration / MAX_SEGMENT_DURATION))
                    split_duration = segment_duration / num_splits
                    
                    for split_idx in range(num_splits):
                        split_start = start_time + (split_idx * split_duration)
                        split_end = min(start_time + ((split_idx + 1) * split_duration), end_time)
                        
                        segment_filename = f"{base_name}_seg_{segment_num:04d}.wav"
                        segment_path = os.path.join(SEGMENTED_DIR, segment_filename)
                        
                        cmd = [
                            "ffmpeg", "-y", "-i", wav_path,
                            "-ss", str(split_start),
                            "-t", str(split_end - split_start),
                            "-ar", "16000", "-ac", "1",
                            segment_path
                        ]
                        subprocess.run(cmd, capture_output=True, text=True)
                        segment_num += 1
                        total_segments += 1
                else:
                    segment_filename = f"{base_name}_seg_{segment_num:04d}.wav"
                    segment_path = os.path.join(SEGMENTED_DIR, segment_filename)
                    
                    cmd = [
                        "ffmpeg", "-y", "-i", wav_path,
                        "-ss", str(start_time),
                        "-t", str(segment_duration),
                        "-ar", "16000", "-ac", "1",
                        segment_path
                    ]
                    subprocess.run(cmd, capture_output=True, text=True)
                    segment_num += 1
                    total_segments += 1
                    
        except Exception as e:
            print(f"Error segmenting {wav_path}: {e}")
            continue
    
    audio_volume.commit()
    print(f"\n✓ Segmentation complete! Total segments: {total_segments}")
    return total_segments

# %% [markdown]
# ## Step 3: Acoustic Tokenization
#
# Extract XLSR-53 features and cluster with K-Means to discover acoustic units.
#
# **Key parameters:**
# - Model: `facebook/wav2vec2-large-xlsr-53`
# - Layer: 14 (optimal for phonetic information)
# - Clusters: 100 (approximates phoneme count)

# %%
@app.function(
    image=image,
    volumes={AUDIO_MOUNT: audio_volume},
    timeout=14400,
    gpu="A10G",
)
def acoustic_tokenization():
    """Extract XLSR-53 features and train K-Means clustering."""
    import torch
    import torchaudio
    import numpy as np
    import joblib
    import json
    from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor
    from sklearn.cluster import MiniBatchKMeans
    from tqdm import tqdm
    import traceback
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    MODEL_NAME = "facebook/wav2vec2-large-xlsr-53"
    LAYER = 14
    NUM_CLUSTERS = 100
    
    CHECKPOINT_FILE = f"{OUTPUT_DIR}/checkpoint_kmeans.pkl"
    PROCESSED_FILES_LOG = f"{OUTPUT_DIR}/processed_files_step1.txt"
    
    # Load model
    print("Loading XLSR-53 model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using: {device}")
    
    extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)
    model = Wav2Vec2Model.from_pretrained(MODEL_NAME).to(device)
    model.eval()
    print("✓ Model loaded!")
    
    def get_features(path):
        """Extract features from audio segment."""
        try:
            import soundfile as sf
            
            waveform_data, rate = sf.read(path, dtype='float32')
            if waveform_data.ndim > 1:
                waveform_data = waveform_data.mean(axis=1)
            
            if rate != 16000:
                waveform_tensor = torch.from_numpy(waveform_data).unsqueeze(0).float()
                resampler = torchaudio.transforms.Resample(rate, 16000)
                waveform_tensor = resampler(waveform_tensor)
                waveform_data = waveform_tensor.squeeze().numpy()
            
            duration = len(waveform_data) / 16000
            
            inputs = extractor(waveform_data, return_tensors="pt", sampling_rate=16000)
            inputs = inputs.input_values.to(device)
            
            with torch.no_grad():
                outputs = model(inputs, output_hidden_states=True)
            
            feats = outputs.hidden_states[LAYER].squeeze(0).cpu().numpy()
            torch.cuda.empty_cache()
            del inputs, outputs
            
            timestamps = np.linspace(0, duration, feats.shape[0])
            return feats, timestamps, duration
        except Exception as e:
            print(f"⚠️  Error processing {os.path.basename(path)}: {str(e)[:100]}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return None, None, None
    
    # Get segmented files
    files = sorted([
        os.path.join(SEGMENTED_DIR, f) 
        for f in os.listdir(SEGMENTED_DIR) 
        if f.endswith('.wav')
    ])
    print(f"Found {len(files)} audio segments to process")
    
    if len(files) == 0:
        print("⚠️  No segmented audio files found!")
        return 0
    
    # Step 1: Train K-Means
    print("\n--- Learning Acoustic Alphabet ---")
    
    processed_files = set()
    kmeans = None
    if os.path.exists(CHECKPOINT_FILE) and os.path.exists(PROCESSED_FILES_LOG):
        try:
            print("📂 Loading checkpoint...")
            kmeans = joblib.load(CHECKPOINT_FILE)
            with open(PROCESSED_FILES_LOG, 'r') as f:
                processed_files = set(line.strip() for line in f if line.strip())
            print(f"✓ Resuming: {len(processed_files)} files already processed")
        except Exception as e:
            print(f"⚠️  Could not load checkpoint: {e}")
            kmeans = None
            processed_files = set()
    
    if kmeans is None:
        kmeans = MiniBatchKMeans(n_clusters=NUM_CLUSTERS, batch_size=1024, random_state=42, n_init=3)
    
    files_to_process = [f for f in files if f not in processed_files]
    print(f"Files remaining: {len(files_to_process)}/{len(files)}")
    
    buffer = []
    buffer_limit = 20000
    checkpoint_interval = 1000
    
    for idx, f in enumerate(tqdm(files_to_process, desc="Step 1/2: Extracting features"), start=len(processed_files)):
        try:
            feats, _, _ = get_features(f)
            if feats is not None:
                buffer.append(feats)
                
                total_frames = sum(b.shape[0] for b in buffer)
                if total_frames >= buffer_limit:
                    stacked = np.vstack(buffer)
                    kmeans.partial_fit(stacked)
                    buffer = []
                    torch.cuda.empty_cache()
            
            if (idx + 1) % checkpoint_interval == 0:
                processed_files.add(f)
                with open(PROCESSED_FILES_LOG, 'a') as log:
                    log.write(f"{f}\n")
                joblib.dump(kmeans, CHECKPOINT_FILE)
                audio_volume.commit()
                print(f"\n💾 Checkpoint saved at {idx + 1}")
        except Exception as e:
            print(f"⚠️  Error: {e}")
            continue
    
    if buffer:
        stacked = np.vstack(buffer)
        kmeans.partial_fit(stacked)
    
    joblib.dump(kmeans, f"{OUTPUT_DIR}/portuguese_kmeans.pkl")
    print("✓ Acoustic alphabet saved!")
    
    # Step 2: Convert to units
    print("\n--- Converting Audio to Units ---")
    corpus = {}
    step2_checkpoint = f"{OUTPUT_DIR}/checkpoint_step2.json"
    
    if os.path.exists(step2_checkpoint):
        try:
            with open(step2_checkpoint, 'r') as f:
                corpus = json.load(f)
            print(f"✓ Resuming Step 2: {len(corpus)} segments processed")
        except:
            pass
    
    processed_step2 = set(corpus.keys())
    files_to_process_step2 = [
        f for f in files 
        if os.path.splitext(os.path.basename(f))[0] not in processed_step2
    ]
    
    for idx, f in enumerate(tqdm(files_to_process_step2, desc="Step 2/2: Tokenizing")):
        try:
            feats, timestamps, duration = get_features(f)
            if feats is None:
                continue
            
            units = kmeans.predict(feats)
            name = os.path.splitext(os.path.basename(f))[0]
            
            corpus[name] = {
                "units": units.tolist(),
                "timestamps": timestamps.tolist(),
                "duration_sec": duration,
                "num_frames": len(units)
            }
            
            with open(f"{OUTPUT_DIR}/{name}.units.txt", "w") as txt:
                txt.write(" ".join(map(str, units)))
            
            if (idx + 1) % 500 == 0:
                with open(step2_checkpoint, 'w') as cp:
                    json.dump(corpus, cp)
                audio_volume.commit()
                print(f"\n💾 Step 2 checkpoint: {idx + 1}")
            
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"⚠️  Error tokenizing: {e}")
            continue
    
    # Save final outputs
    with open(f"{OUTPUT_DIR}/portuguese_corpus_timestamped.json", "w") as f:
        json.dump(corpus, f)
    
    with open(f"{OUTPUT_DIR}/all_units_for_bpe.txt", "w") as f:
        for data in corpus.values():
            f.write(" ".join(map(str, data["units"])) + "\n")
    
    # Clean up checkpoints
    for cp_file in [CHECKPOINT_FILE, PROCESSED_FILES_LOG, step2_checkpoint]:
        if os.path.exists(cp_file):
            try:
                os.remove(cp_file)
            except:
                pass
    
    audio_volume.commit()
    
    total_mins = sum(d["duration_sec"] for d in corpus.values()) / 60
    print(f"\n✓ Phase 1 Complete!")
    print(f"  Segments: {len(corpus)}")
    print(f"  Duration: {total_mins:.1f} min")
    print(f"  Units: {NUM_CLUSTERS}")
    
    return len(corpus)

# %% [markdown]
# ## Entry Points

# %%
@app.local_entrypoint()
def main():
    """Run the complete pipeline."""
    print("🚀 Starting Bible Audio Processing Pipeline")
    print("=" * 50)
    
    print("\n📁 Phase 1: Converting MP3 to WAV...")
    converted = convert_mp3_to_wav.remote()
    print(f"✓ Converted {converted} files")
    
    print("\n✂️  Phase 1.5: Segmenting by silence...")
    segments = segment_by_silence.remote()
    print(f"✓ Created {segments} segments")
    
    print("\n🎵 Phase 1: Acoustic Tokenization...")
    processed = acoustic_tokenization.remote()
    print(f"✓ Processed {processed} segments")
    
    print("\n✅ Pipeline complete!")


# %%
@app.local_entrypoint()
def main_skip_segmentation():
    """Run pipeline using existing segments."""
    print("🚀 Starting Pipeline (skip segmentation)")
    print("=" * 50)
    
    print("\n📁 Converting MP3 to WAV...")
    converted = convert_mp3_to_wav.remote()
    print(f"✓ Converted {converted} files")
    
    print("\n✂️  SKIPPED: Using existing segments")
    
    print("\n🎵 Acoustic Tokenization...")
    processed = acoustic_tokenization.remote()
    print(f"✓ Processed {processed} segments")
    
    print("\n✅ Pipeline complete!")
