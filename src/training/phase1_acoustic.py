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
app = modal.App("bible-audio-training")

audio_volume = modal.Volume.from_name(
    "bible-audio-data",
    create_if_missing=True
)

# %%
image = (
    modal.Image.debian_slim()
    .apt_install("ffmpeg", "libsndfile1")
    .pip_install(
        "torch>=2.0.0",
        "torchaudio>=2.0.0",
        "transformers>=4.40.0",
        "sentencepiece>=0.1.99",
        "scikit-learn>=1.3.0",
        "joblib>=1.3.0",
        "soundfile>=0.12.0",
        "tqdm>=4.66.0",
        "numpy<2",
        "librosa>=0.10.0",
    )
)

# %%
AUDIO_MOUNT = "/mnt/audio_data"

LANGUAGE_CONFIGS = {
    "portuguese": {
        "segmented_dir": f"{AUDIO_MOUNT}/segmented_audio",
        "output_dir": f"{AUDIO_MOUNT}/portuguese_units",
        "raw_audio_dir": f"{AUDIO_MOUNT}/raw_audio",
        "converted_dir": f"{AUDIO_MOUNT}/converted_audio",
    },
    "satere": {
        "segmented_dir": f"{AUDIO_MOUNT}/segmented_audio_satere",
        "output_dir": f"{AUDIO_MOUNT}/satere_units",
        "raw_audio_dir": f"{AUDIO_MOUNT}/raw_audio_satere",
        "converted_dir": f"{AUDIO_MOUNT}/converted_audio_satere",
    },
}

import os as _os
_LANGUAGE = _os.environ.get("TRAINING_LANGUAGE", "portuguese")
_config = LANGUAGE_CONFIGS.get(_LANGUAGE, LANGUAGE_CONFIGS["portuguese"])

RAW_AUDIO_DIR = _config["raw_audio_dir"]
CONVERTED_DIR = _config["converted_dir"]
SEGMENTED_DIR = _config["segmented_dir"]
OUTPUT_DIR = _config["output_dir"]

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
def convert_mp3_to_wav(language: str = "portuguese"):
    import subprocess
    from tqdm import tqdm
    
    config = LANGUAGE_CONFIGS.get(language, LANGUAGE_CONFIGS["portuguese"])
    raw_audio_dir = config["raw_audio_dir"]
    converted_dir = config["converted_dir"]
    
    os.makedirs(converted_dir, exist_ok=True)
    
    mp3_files = []
    for root, dirs, files in os.walk(raw_audio_dir):
        for f in files:
            if f.endswith('.mp3'):
                mp3_files.append(os.path.join(root, f))
    
    print(f"Found {len(mp3_files)} MP3 files")
    
    existing = set()
    if os.path.exists(converted_dir):
        existing = {f.replace('.wav', '.mp3') for f in os.listdir(converted_dir) if f.endswith('.wav')}
    
    to_convert = [f for f in mp3_files if os.path.basename(f) not in existing]
    print(f"Already converted: {len(existing)}")
    print(f"To convert: {len(to_convert)}")
    
    for mp3_path in tqdm(to_convert, desc="Converting to WAV"):
        mp3_file = os.path.basename(mp3_path)
        wav_path = os.path.join(converted_dir, mp3_file.replace('.mp3', '.wav'))
        
        cmd = ["ffmpeg", "-y", "-i", mp3_path, "-ar", "16000", "-ac", "1", wav_path]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"Error converting {mp3_path}: {result.stderr}")
    
    audio_volume.commit()
    print(f"\n✓ Done! Converted files are in: {converted_dir}")
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
def segment_by_silence(
    language: str = "portuguese",
    silence_threshold_db: float = -40.0,
    min_silence_duration: float = 0.5,
    min_segment_duration: float = 2.0,
    max_segment_duration: float = 120.0,
):
    import soundfile as sf
    import librosa
    import numpy as np
    from tqdm import tqdm
    import subprocess
    
    config = LANGUAGE_CONFIGS.get(language, LANGUAGE_CONFIGS["portuguese"])
    converted_dir = config["converted_dir"]
    segmented_dir = config["segmented_dir"]
    
    os.makedirs(segmented_dir, exist_ok=True)
    
    wav_files = sorted([
        os.path.join(converted_dir, f) 
        for f in os.listdir(converted_dir) 
        if f.endswith('.wav')
    ])
    print(f"Found {len(wav_files)} WAV files to segment")
    
    existing_segments = set()
    if os.path.exists(segmented_dir):
        for f in os.listdir(segmented_dir):
            if f.endswith('.wav'):
                base = f.rsplit('_seg_', 1)[0] if '_seg_' in f else f.replace('.wav', '')
                existing_segments.add(base)
    
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
            
            frame_length = int(0.025 * sr)
            hop_length = int(0.010 * sr)
            rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
            rms_db = librosa.power_to_db(rms**2, ref=np.max)
            
            silence_mask = rms_db < silence_threshold_db
            segment_boundaries = [0]
            
            in_silence = False
            silence_start = None
            
            for i, is_silent in enumerate(silence_mask):
                frame_time = i * hop_length / sr
                
                if is_silent and not in_silence:
                    in_silence = True
                    silence_start = frame_time
                elif not is_silent and in_silence:
                    if silence_start is not None and (frame_time - silence_start) >= min_silence_duration:
                        boundary = (silence_start + frame_time) / 2
                        segment_boundaries.append(boundary)
                    in_silence = False
                    silence_start = None
            
            segment_boundaries.append(duration)
            
            base_name = os.path.splitext(os.path.basename(wav_path))[0]
            segment_num = 0
            
            for i in range(len(segment_boundaries) - 1):
                start_time = segment_boundaries[i]
                end_time = segment_boundaries[i + 1]
                seg_duration = end_time - start_time
                
                if seg_duration < min_segment_duration:
                    continue
                
                if seg_duration > max_segment_duration:
                    num_splits = int(np.ceil(seg_duration / max_segment_duration))
                    split_duration = seg_duration / num_splits
                    
                    for split_idx in range(num_splits):
                        split_start = start_time + (split_idx * split_duration)
                        split_end = min(start_time + ((split_idx + 1) * split_duration), end_time)
                        
                        segment_filename = f"{base_name}_seg_{segment_num:04d}.wav"
                        segment_path = os.path.join(segmented_dir, segment_filename)
                        
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
                    segment_path = os.path.join(segmented_dir, segment_filename)
                    
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
def acoustic_tokenization(
    language: str = "portuguese",
    model_name: str = "facebook/wav2vec2-large-xlsr-53",
    layer: int = 14,
    num_clusters: int = 100,
    buffer_limit: int = 20000,
    checkpoint_interval: int = 1000,
):
    import torch
    import torchaudio
    import numpy as np
    import joblib
    import json
    from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor
    from sklearn.cluster import MiniBatchKMeans
    from tqdm import tqdm
    import traceback
    
    config = LANGUAGE_CONFIGS.get(language, LANGUAGE_CONFIGS["portuguese"])
    segmented_dir = config["segmented_dir"]
    output_dir = config["output_dir"]
    
    os.makedirs(output_dir, exist_ok=True)
    
    checkpoint_file = f"{output_dir}/checkpoint_kmeans.pkl"
    processed_files_log = f"{output_dir}/processed_files_step1.txt"
    
    print(f"Loading {model_name}...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using: {device}")
    
    extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
    w2v_model = Wav2Vec2Model.from_pretrained(model_name).to(device)
    w2v_model.eval()
    print("✓ Model loaded!")
    
    def get_features(path):
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
                outputs = w2v_model(inputs, output_hidden_states=True)
            
            feats = outputs.hidden_states[layer].squeeze(0).cpu().numpy()
            torch.cuda.empty_cache()
            del inputs, outputs
            
            timestamps = np.linspace(0, duration, feats.shape[0])
            return feats, timestamps, duration
        except Exception as e:
            print(f"⚠️  Error processing {os.path.basename(path)}: {str(e)[:100]}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return None, None, None
    
    files = sorted([
        os.path.join(segmented_dir, f) 
        for f in os.listdir(segmented_dir) 
        if f.endswith('.wav')
    ])
    print(f"Found {len(files)} audio segments to process")
    
    if len(files) == 0:
        print("⚠️  No segmented audio files found!")
        return 0
    
    print("\n--- Learning Acoustic Alphabet ---")
    
    processed_files_set = set()
    kmeans = None
    if os.path.exists(checkpoint_file) and os.path.exists(processed_files_log):
        try:
            print("📂 Loading checkpoint...")
            kmeans = joblib.load(checkpoint_file)
            with open(processed_files_log, 'r') as f:
                processed_files_set = set(line.strip() for line in f if line.strip())
            print(f"✓ Resuming: {len(processed_files_set)} files already processed")
        except Exception as e:
            print(f"⚠️  Could not load checkpoint: {e}")
            kmeans = None
            processed_files_set = set()
    
    if kmeans is None:
        kmeans = MiniBatchKMeans(n_clusters=num_clusters, batch_size=1024, random_state=42, n_init=3)
    
    files_to_process = [f for f in files if f not in processed_files_set]
    print(f"Files remaining: {len(files_to_process)}/{len(files)}")
    
    buffer = []
    
    for idx, f in enumerate(tqdm(files_to_process, desc="Step 1/2: Extracting features"), start=len(processed_files_set)):
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
                processed_files_set.add(f)
                with open(processed_files_log, 'a') as log:
                    log.write(f"{f}\n")
                joblib.dump(kmeans, checkpoint_file)
                audio_volume.commit()
                print(f"\n💾 Checkpoint saved at {idx + 1}")
        except Exception as e:
            print(f"⚠️  Error: {e}")
            continue
    
    if buffer:
        stacked = np.vstack(buffer)
        kmeans.partial_fit(stacked)
    
    kmeans_output_file = f"{language}_kmeans.pkl"
    joblib.dump(kmeans, f"{output_dir}/{kmeans_output_file}")
    print("✓ Acoustic alphabet saved!")
    
    print("\n--- Converting Audio to Units ---")
    corpus = {}
    step2_checkpoint = f"{output_dir}/checkpoint_step2.json"
    
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
            
            with open(f"{output_dir}/{name}.units.txt", "w") as txt:
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
    
    corpus_output_file = f"{language}_corpus_timestamped.json"
    with open(f"{output_dir}/{corpus_output_file}", "w") as f:
        json.dump(corpus, f)
    
    with open(f"{output_dir}/all_units_for_bpe.txt", "w") as f:
        for data in corpus.values():
            f.write(" ".join(map(str, data["units"])) + "\n")
    
    for cp_file in [checkpoint_file, processed_files_log, step2_checkpoint]:
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
    print(f"  Units: {num_clusters}")
    
    return len(corpus)

# %% [markdown]
# ## Entry Points

# %%
@app.local_entrypoint()
def main(language: str = "portuguese"):
    config = LANGUAGE_CONFIGS.get(language, LANGUAGE_CONFIGS["portuguese"])
    
    print("🚀 Starting Bible Audio Processing Pipeline")
    print("=" * 50)
    print(f"  Language: {language}")
    print(f"  Output: {config['output_dir']}")
    print("=" * 50)
    
    print("\n📁 Phase 1: Converting MP3 to WAV...")
    converted = convert_mp3_to_wav.remote(language=language)
    print(f"✓ Converted {converted} files")
    
    print("\n✂️  Phase 1.5: Segmenting by silence...")
    segments = segment_by_silence.remote(language=language)
    print(f"✓ Created {segments} segments")
    
    print("\n🎵 Phase 1: Acoustic Tokenization...")
    processed = acoustic_tokenization.remote(language=language)
    print(f"✓ Processed {processed} segments")
    
    print("\n✅ Pipeline complete!")


# %%
@app.local_entrypoint()
def main_skip_segmentation(language: str = "portuguese"):
    config = LANGUAGE_CONFIGS.get(language, LANGUAGE_CONFIGS["portuguese"])
    
    print("🚀 Starting Pipeline (skip segmentation)")
    print("=" * 50)
    print(f"  Language: {language}")
    print(f"  Segments: {config['segmented_dir']}")
    print(f"  Output: {config['output_dir']}")
    print("=" * 50)
    
    print("\n📁 Converting MP3 to WAV...")
    converted = convert_mp3_to_wav.remote(language=language)
    print(f"✓ Converted {converted} files")
    
    print("\n✂️  SKIPPED: Using existing segments")
    
    print("\n🎵 Acoustic Tokenization...")
    processed = acoustic_tokenization.remote(language=language)
    print(f"✓ Processed {processed} segments")
    
    print("\n✅ Pipeline complete!")
