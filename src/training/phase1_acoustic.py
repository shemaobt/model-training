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
from src.constants import (
    SAMPLE_RATE,
    SAMPLE_RATE,
    ACOUSTIC_MODELS,
    DEFAULT_MODEL,
    NUM_ACOUSTIC_UNITS,
    KMEANS_BATCH_SIZE,
    KMEANS_RANDOM_STATE,
    KMEANS_N_INIT,
    SILENCE_THRESHOLD_DB,
    MIN_SILENCE_DURATION,
    MIN_SEGMENT_DURATION,
    MAX_SEGMENT_DURATION,
    FRAME_DURATION,
    HOP_DURATION,
    DEFAULT_BUFFER_LIMIT,
    DEFAULT_CHECKPOINT_INTERVAL,
    PROGRESS_LOG_INTERVAL,
)

# %%
app = modal.App("bible-audio-training")

audio_volume = modal.Volume.from_name(
    "bible-audio-data",
    create_if_missing=True
)

# %%
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("ffmpeg", "libsndfile1", "git")
    .pip_install(
        "torch>=2.0.0",
        "torchaudio>=2.0.0",
        "transformers>=4.40.0",
        "huggingface_hub>=0.23.0",
        "espnet @ git+https://github.com/wanchichen/espnet.git@ssl",
        "sentencepiece==0.1.97",
        "scikit-learn>=1.3.0",
        "joblib>=1.3.0",
        "soundfile>=0.12.0",
        "tqdm>=4.66.0",
        "numpy<2",
        "librosa==0.9.2",
    )
    .add_local_dir("src", remote_path="/root/src")
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
    "english": {
        "segmented_dir": f"{AUDIO_MOUNT}/segmented_audio_english",
        "output_dir": f"{AUDIO_MOUNT}/english_units",
        "raw_audio_dir": f"{AUDIO_MOUNT}/raw_audio_english",
        "converted_dir": f"{AUDIO_MOUNT}/converted_audio_english",
    },
    "portuguese_fleurs": {
        "segmented_dir": f"{AUDIO_MOUNT}/parallel_pt_en/segmented_audio_portuguese_fleurs",
        "output_dir": f"{AUDIO_MOUNT}/portuguese_fleurs_units",
        "raw_audio_dir": f"{AUDIO_MOUNT}/raw_audio_portuguese_fleurs",
        "converted_dir": f"{AUDIO_MOUNT}/converted_audio_portuguese_fleurs",
    },
    "english_fleurs": {
        "segmented_dir": f"{AUDIO_MOUNT}/parallel_pt_en/segmented_audio_english_fleurs",
        "output_dir": f"{AUDIO_MOUNT}/english_fleurs_units",
        "raw_audio_dir": f"{AUDIO_MOUNT}/raw_audio_english_fleurs",
        "converted_dir": f"{AUDIO_MOUNT}/converted_audio_english_fleurs",
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
        
        cmd = ["ffmpeg", "-y", "-i", mp3_path, "-ar", str(SAMPLE_RATE), "-ac", "1", wav_path]
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
    silence_threshold_db: float = SILENCE_THRESHOLD_DB,
    min_silence_duration: float = MIN_SILENCE_DURATION,
    min_segment_duration: float = MIN_SEGMENT_DURATION,
    max_segment_duration: float = MAX_SEGMENT_DURATION,
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
            audio, sr = librosa.load(wav_path, sr=SAMPLE_RATE, mono=True)
            duration = len(audio) / sr
            
            frame_length = int(FRAME_DURATION * sr)
            hop_length = int(HOP_DURATION * sr)
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
                            "-ar", str(SAMPLE_RATE), "-ac", "1",
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
                        "-t", str(seg_duration),
                        "-ar", str(SAMPLE_RATE), "-ac", "1",
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
    timeout=21600,  # 6 hours (increased from 4h for large datasets)
    gpu="A10G",
)
def acoustic_tokenization(
    language: str = "portuguese",
    model_key: str = DEFAULT_MODEL,
    layer: int = None, # Optional override
    num_clusters: int = NUM_ACOUSTIC_UNITS,
    buffer_limit: int = DEFAULT_BUFFER_LIMIT,
    checkpoint_interval: int = DEFAULT_CHECKPOINT_INTERVAL,
):
    import torch
    import torchaudio
    import numpy as np
    import joblib
    import json
    from huggingface_hub import snapshot_download
    from torch.nn.utils.rnn import pad_sequence
    from sklearn.cluster import MiniBatchKMeans
    from tqdm import tqdm
    import traceback
    
    config = LANGUAGE_CONFIGS.get(language, LANGUAGE_CONFIGS["portuguese"])
    segmented_dir = config["segmented_dir"]
    output_dir = config["output_dir"]
    
    if model_key not in ACOUSTIC_MODELS:
        raise ValueError(f"Unknown model: {model_key}. Available: {list(ACOUSTIC_MODELS.keys())}")
    
    model_config = ACOUSTIC_MODELS[model_key]
    model_name = model_config["model_name"]
    target_layer = layer if layer is not None else model_config["layer"]
    
    print(f"Configuration:")
    print(f"  Model: {model_name} ({model_key})")
    print(f"  Layer: {target_layer}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    checkpoint_file = f"{output_dir}/checkpoint_kmeans_{model_key}.pkl"
    processed_files_log = f"{output_dir}/processed_files_step1_{model_key}.txt"
    
    print(f"Loading {model_name}...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using: {device}")
    
    extractor = None
    w2v_model = None
    xeus_model = None
    xeus_mode = model_key == "xeus"
    if xeus_mode:
        from espnet2.tasks.ssl import SSLTask
        import yaml

        xeus_repo_path = snapshot_download(repo_id=model_name)
        config_path = os.path.join(xeus_repo_path, "model", "config.yaml")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"XEUS config not found at {config_path}")

        with open(config_path, "r") as file:
            xeus_config = yaml.safe_load(file)
        if isinstance(xeus_config.get("frontend_conf"), dict):
            xeus_config["frontend_conf"].pop("normalize_output", None)
        if isinstance(xeus_config.get("loss"), list) and xeus_config["loss"]:
            first_loss = xeus_config["loss"][0]
            if isinstance(first_loss, dict):
                xeus_config["loss"] = first_loss.get("name")
                xeus_config["loss_conf"] = first_loss.get("conf", {})
        xeus_config.setdefault("masker", "hubert")
        xeus_config.setdefault(
            "masker_conf",
            {
                "mask_prob": 0.8,
                "mask_selection": "static",
                "mask_other": 0.0,
                "mask_length": 10,
                "no_mask_overlap": False,
                "mask_min_space": 1,
                "mask_channel_prob": 0.0,
                "mask_channel_selection": "static",
                "mask_channel_other": 0.0,
                "mask_channel_length": 10,
                "no_mask_channel_overlap": False,
                "mask_channel_min_space": 1,
            },
        )
        compat_config_path = os.path.join(xeus_repo_path, "model", "config_compat.yaml")
        with open(compat_config_path, "w") as file:
            yaml.safe_dump(xeus_config, file)

        candidate_checkpoints = [
            os.path.join(xeus_repo_path, "model", "xeus_checkpoint_old.pth"),
            os.path.join(xeus_repo_path, "model", "xeus_checkpoint_new.pth"),
        ]
        selected_checkpoint = None
        last_error = None
        for checkpoint_path in candidate_checkpoints:
            if not os.path.exists(checkpoint_path):
                continue
            try:
                import argparse

                args = argparse.Namespace(**xeus_config)
                xeus_model = SSLTask.build_model(args)
                checkpoint = torch.load(checkpoint_path, map_location=str(device))
                state_dict = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
                xeus_model.load_state_dict(state_dict, strict=False)
                xeus_model = xeus_model.to(device)
                selected_checkpoint = checkpoint_path
                break
            except Exception as exc:
                import traceback as _traceback

                print(f"Failed XEUS checkpoint load: {checkpoint_path}")
                _traceback.print_exc()
                last_error = exc
        if selected_checkpoint is None:
            raise RuntimeError(f"Unable to load XEUS checkpoint. Last error: {last_error}")
        xeus_model.eval()
        print(f"Using XEUS checkpoint: {selected_checkpoint}")
    else:
        from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor

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
            
            if rate != SAMPLE_RATE:
                waveform_tensor = torch.from_numpy(waveform_data).unsqueeze(0).float()
                resampler = torchaudio.transforms.Resample(rate, SAMPLE_RATE)
                waveform_tensor = resampler(waveform_tensor)
                waveform_data = waveform_tensor.squeeze().numpy()
            
            duration = len(waveform_data) / SAMPLE_RATE
            
            if xeus_mode:
                wav_tensor = torch.tensor(waveform_data, dtype=torch.float32, device=device)
                wav_lengths = torch.LongTensor([wav_tensor.numel()]).to(device)
                wavs = pad_sequence([wav_tensor], batch_first=True)
                with torch.no_grad():
                    encoded = xeus_model.encode(
                        wavs,
                        wav_lengths,
                        use_mask=False,
                        use_final_output=False,
                    )
                hidden_states = encoded[0]
                layer_idx = target_layer if target_layer >= 0 else (len(hidden_states) + target_layer)
                layer_idx = min(max(layer_idx, 0), len(hidden_states) - 1)
                feats = hidden_states[layer_idx].squeeze(0).detach().cpu().numpy()
            else:
                inputs = extractor(waveform_data, return_tensors="pt", sampling_rate=SAMPLE_RATE)
                inputs = inputs.input_values.to(device)

                with torch.no_grad():
                    outputs = w2v_model(inputs, output_hidden_states=True)

                feats = outputs.hidden_states[target_layer].squeeze(0).cpu().numpy()
            torch.cuda.empty_cache()
            if xeus_mode:
                del wav_tensor, wav_lengths, wavs, encoded
            else:
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
        kmeans = MiniBatchKMeans(
            n_clusters=num_clusters,
            batch_size=KMEANS_BATCH_SIZE,
            random_state=KMEANS_RANDOM_STATE,
            n_init=KMEANS_N_INIT,
        )
    
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
    
        kmeans.partial_fit(stacked)
    
    kmeans_output_file = f"{language}_kmeans_{model_key}.pkl"
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
            
            with open(f"{output_dir}/{name}.units_{model_key}.txt", "w") as txt:
                txt.write(" ".join(map(str, units)))
            
            if (idx + 1) % PROGRESS_LOG_INTERVAL == 0:
                with open(step2_checkpoint, 'w') as cp:
                    json.dump(corpus, cp)
                audio_volume.commit()
                print(f"\n💾 Step 2 checkpoint: {idx + 1}")
            
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"⚠️  Error tokenizing: {e}")
            continue
    
    corpus_output_file = f"{language}_corpus_timestamped_{model_key}.json"
    with open(f"{output_dir}/{corpus_output_file}", "w") as f:
        json.dump(corpus, f)
    
    with open(f"{output_dir}/all_units_for_bpe_{model_key}.txt", "w") as f:
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
def main(language: str = "portuguese", model: str = DEFAULT_MODEL):
    config = LANGUAGE_CONFIGS.get(language, LANGUAGE_CONFIGS["portuguese"])
    
    print("🚀 Starting Bible Audio Processing Pipeline")
    print("=" * 50)
    print(f"  Language: {language}")
    print(f"  Model:    {model}")
    print(f"  Output:   {config['output_dir']}")
    print("=" * 50)
    
    print("\n📁 Phase 1: Converting MP3 to WAV...")
    converted = convert_mp3_to_wav.remote(language=language)
    print(f"✓ Converted {converted} files")
    
    print("\n✂️  Phase 1.5: Segmenting by silence...")
    segments = segment_by_silence.remote(language=language)
    print(f"✓ Created {segments} segments")
    
    print("\n🎵 Phase 1: Acoustic Tokenization...")
    processed = acoustic_tokenization.remote(language=language, model_key=model)
    print(f"✓ Processed {processed} segments")
    
    print("\n✅ Pipeline complete!")


# %%
@app.local_entrypoint()
def main_skip_segmentation(language: str = "portuguese", model: str = DEFAULT_MODEL):
    config = LANGUAGE_CONFIGS.get(language, LANGUAGE_CONFIGS["portuguese"])
    
    print("🚀 Starting Pipeline (skip segmentation)")
    print("=" * 50)
    print(f"  Language: {language}")
    print(f"  Model:    {model}")
    print(f"  Segments: {config['segmented_dir']}")
    print(f"  Output:   {config['output_dir']}")
    print("=" * 50)
    
    print("\n📁 Converting MP3 to WAV...")
    converted = convert_mp3_to_wav.remote(language=language)
    print(f"✓ Converted {converted} files")
    
    print("\n✂️  SKIPPED: Using existing segments")
    
    print("\n🎵 Acoustic Tokenization...")
    processed = acoustic_tokenization.remote(language=language, model_key=model)
    print(f"✓ Processed {processed} segments")
    
    print("\n✅ Pipeline complete!")
