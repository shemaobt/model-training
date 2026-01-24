"""
Modal deployment for Phase 1 notebook - Bible Audio Processing
Converts MP3 to WAV, segments by silence/pauses, then performs acoustic tokenization
"""

import modal
import os
from pathlib import Path

# Create Modal app
app = modal.App("bible-audio-training")

# Create persistent volume for audio files and outputs
audio_volume = modal.Volume.from_name(
    "bible-audio-data",
    create_if_missing=True
)

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
        "librosa>=0.10.0",  # For silence detection
    )
)

# Mount points
AUDIO_MOUNT = "/mnt/audio_data"
RAW_AUDIO_DIR = f"{AUDIO_MOUNT}/raw_audio"
CONVERTED_DIR = f"{AUDIO_MOUNT}/converted_audio"
SEGMENTED_DIR = f"{AUDIO_MOUNT}/segmented_audio"  # New: store segmented audio
OUTPUT_DIR = f"{AUDIO_MOUNT}/portuguese_units"


@app.function(
    image=image,
    volumes={AUDIO_MOUNT: audio_volume},
    timeout=3600,  # 1 hour timeout
    # No GPU needed - ffmpeg conversion is CPU-based
)
def convert_mp3_to_wav():
    """Phase 1: Convert MP3 files to 16kHz mono WAV"""
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
    
    # Convert each file
    for mp3_path in tqdm(to_convert, desc="Converting to WAV"):
        mp3_file = os.path.basename(mp3_path)
        wav_path = os.path.join(CONVERTED_DIR, mp3_file.replace('.mp3', '.wav'))
        
        cmd = ["ffmpeg", "-y", "-i", mp3_path, "-ar", "16000", "-ac", "1", wav_path]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"Error converting {mp3_path}: {result.stderr}")
    
    # Commit volume changes
    audio_volume.commit()
    
    print(f"\n✓ Done! Converted files are in: {CONVERTED_DIR}")
    return len(to_convert)


@app.function(
    image=image,
    volumes={AUDIO_MOUNT: audio_volume},
    timeout=7200,  # 2 hours for segmentation
    # No GPU needed for silence detection
)
def segment_by_silence():
    """Phase 1.5: Segment audio files by silence/pauses (acoustic boundaries)"""
    import soundfile as sf
    import librosa
    import numpy as np
    from tqdm import tqdm
    import subprocess
    
    os.makedirs(SEGMENTED_DIR, exist_ok=True)
    
    # Get all WAV files
    wav_files = sorted([os.path.join(CONVERTED_DIR, f) for f in os.listdir(CONVERTED_DIR) if f.endswith('.wav')])
    print(f"Found {len(wav_files)} WAV files to segment")
    
    # Check what's already segmented
    existing_segments = set()
    if os.path.exists(SEGMENTED_DIR):
        # Get base filenames (without segment numbers)
        for f in os.listdir(SEGMENTED_DIR):
            if f.endswith('.wav'):
                # Remove segment suffix (e.g., "_seg_001.wav" -> original filename)
                base = f.rsplit('_seg_', 1)[0] if '_seg_' in f else f.replace('.wav', '')
                existing_segments.add(base)
    
    # Parameters for silence detection
    SILENCE_THRESHOLD = -40  # dB threshold for silence (adjust based on your audio)
    MIN_SILENCE_DURATION = 0.5  # Minimum silence duration in seconds to create a segment
    MIN_SEGMENT_DURATION = 2.0  # Minimum segment duration in seconds (avoid too short segments)
    MAX_SEGMENT_DURATION = 120.0  # Maximum segment duration in seconds (2 minutes max)
    
    total_segments = 0
    files_to_segment = [f for f in wav_files if os.path.splitext(os.path.basename(f))[0] not in existing_segments]
    
    print(f"Already segmented: {len(existing_segments)} files")
    print(f"To segment: {len(files_to_segment)} files")
    
    for wav_path in tqdm(files_to_segment, desc="Segmenting by silence"):
        try:
            # Load audio
            audio, sr = librosa.load(wav_path, sr=16000, mono=True)
            duration = len(audio) / sr
            
            # Detect silence using energy-based approach
            # Compute short-time energy
            frame_length = int(0.025 * sr)  # 25ms frames
            hop_length = int(0.010 * sr)    # 10ms hop
            
            # Calculate RMS energy per frame
            rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
            
            # Convert to dB
            rms_db = librosa.power_to_db(rms**2, ref=np.max)
            
            # Find silence regions (below threshold)
            silence_mask = rms_db < SILENCE_THRESHOLD
            
            # Find segment boundaries (transitions from speech to silence)
            segment_boundaries = [0]  # Start with beginning
            
            in_silence = False
            silence_start = None
            
            for i, is_silent in enumerate(silence_mask):
                frame_time = i * hop_length / sr
                
                if is_silent and not in_silence:
                    # Entering silence
                    in_silence = True
                    silence_start = frame_time
                elif not is_silent and in_silence:
                    # Exiting silence - check if silence was long enough
                    if silence_start is not None and (frame_time - silence_start) >= MIN_SILENCE_DURATION:
                        # Use middle of silence as boundary
                        boundary = (silence_start + frame_time) / 2
                        segment_boundaries.append(boundary)
                    in_silence = False
                    silence_start = None
            
            # Add end boundary
            segment_boundaries.append(duration)
            
            # Create segments
            base_name = os.path.splitext(os.path.basename(wav_path))[0]
            segment_num = 0
            
            for i in range(len(segment_boundaries) - 1):
                start_time = segment_boundaries[i]
                end_time = segment_boundaries[i + 1]
                segment_duration = end_time - start_time
                
                # Skip segments that are too short or too long
                if segment_duration < MIN_SEGMENT_DURATION:
                    continue
                
                # If segment is too long, split it at max duration
                if segment_duration > MAX_SEGMENT_DURATION:
                    num_splits = int(np.ceil(segment_duration / MAX_SEGMENT_DURATION))
                    split_duration = segment_duration / num_splits
                    
                    for split_idx in range(num_splits):
                        split_start = start_time + (split_idx * split_duration)
                        split_end = min(start_time + ((split_idx + 1) * split_duration), end_time)
                        
                        segment_filename = f"{base_name}_seg_{segment_num:04d}.wav"
                        segment_path = os.path.join(SEGMENTED_DIR, segment_filename)
                        
                        # Extract segment using ffmpeg (more reliable than librosa slicing)
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
                    # Segment is within acceptable range
                    segment_filename = f"{base_name}_seg_{segment_num:04d}.wav"
                    segment_path = os.path.join(SEGMENTED_DIR, segment_filename)
                    
                    # Extract segment using ffmpeg
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
    
    # Commit volume changes
    audio_volume.commit()
    
    print(f"\n✓ Segmentation complete!")
    print(f"  Total segments created: {total_segments}")
    print(f"  Segments stored in: {SEGMENTED_DIR}")
    return total_segments


@app.function(
    image=image,
    volumes={AUDIO_MOUNT: audio_volume},
    timeout=14400,  # 4 hour timeout for model processing (increased)
    gpu="A10G",  # Use A10G for intensive training/inference (matches train_adapter pattern)
)
def acoustic_tokenization():
    """Phase 2: Acoustic Tokenization using XLSR-53 + K-Means
    Now processes pre-segmented audio files (from silence-based segmentation)
    """
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
    
    # Checkpoint file to resume from
    CHECKPOINT_FILE = f"{OUTPUT_DIR}/checkpoint_kmeans.pkl"
    PROCESSED_FILES_LOG = f"{OUTPUT_DIR}/processed_files_step1.txt"
    
    # Load model
    print("Loading XLSR-53 model (this takes a minute)...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using: {device}")
    
    try:
        extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)
        model = Wav2Vec2Model.from_pretrained(MODEL_NAME).to(device)
        model.eval()
        print("✓ Model loaded!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        traceback.print_exc()
        raise
    
    def get_features(path):
        """Extract features from audio segment using soundfile.
        Segments are already small, so no need for further chunking.
        """
        try:
            import soundfile as sf
            
            # Load audio segment (already small from silence-based segmentation)
            waveform_data, rate = sf.read(path, dtype='float32')
            
            # Convert to mono if stereo
            if waveform_data.ndim > 1:
                waveform_data = waveform_data.mean(axis=1)
            
            # Resample if needed
            if rate != 16000:
                waveform_tensor = torch.from_numpy(waveform_data).unsqueeze(0).float()
                resampler = torchaudio.transforms.Resample(rate, 16000)
                waveform_tensor = resampler(waveform_tensor)
                waveform_data = waveform_tensor.squeeze().numpy()
            
            duration = len(waveform_data) / 16000
            
            # Extract features (segments are small, safe to process at once)
            inputs = extractor(waveform_data, return_tensors="pt", sampling_rate=16000)
            inputs = inputs.input_values.to(device)
            
            with torch.no_grad():
                outputs = model(inputs, output_hidden_states=True)
            
            feats = outputs.hidden_states[LAYER].squeeze(0).cpu().numpy()
            
            # Clear GPU cache
            torch.cuda.empty_cache()
            del inputs, outputs
            
            timestamps = np.linspace(0, duration, feats.shape[0])
            return feats, timestamps, duration
        except Exception as e:
            print(f"⚠️  Error processing {os.path.basename(path)}: {str(e)[:100]}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return None, None, None
    
    # Get segmented audio files (instead of full files)
    files = sorted([os.path.join(SEGMENTED_DIR, f) for f in os.listdir(SEGMENTED_DIR) if f.endswith('.wav')])
    print(f"Found {len(files)} audio segments to process")
    
    if len(files) == 0:
        print("⚠️  No segmented audio files found. Run segment_by_silence() first!")
        return 0
    
    # Step 1: Train K-Means with checkpointing
    print("\n--- Learning Acoustic Alphabet ---")
    
    # Try to load checkpoint
    processed_files = set()
    kmeans = None
    if os.path.exists(CHECKPOINT_FILE) and os.path.exists(PROCESSED_FILES_LOG):
        try:
            print("📂 Loading checkpoint...")
            kmeans = joblib.load(CHECKPOINT_FILE)
            with open(PROCESSED_FILES_LOG, 'r') as f:
                processed_files = set(line.strip() for line in f if line.strip())
            print(f"✓ Resuming from checkpoint: {len(processed_files)} files already processed")
        except Exception as e:
            print(f"⚠️  Could not load checkpoint: {e}, starting fresh")
            kmeans = None
            processed_files = set()
    
    if kmeans is None:
        kmeans = MiniBatchKMeans(n_clusters=NUM_CLUSTERS, batch_size=1024, random_state=42, n_init=3)
    
    # Filter out already processed files
    files_to_process = [f for f in files if f not in processed_files]
    print(f"Files remaining: {len(files_to_process)}/{len(files)}")
    
    buffer = []
    buffer_limit = 20000  # Reduced buffer size to avoid memory issues
    checkpoint_interval = 1000  # Save checkpoint every N files
    
    try:
        for idx, f in enumerate(tqdm(files_to_process, desc="Step 1/2: Extracting features"), start=len(processed_files)):
            try:
                feats, _, _ = get_features(f)
                if feats is not None:
                    buffer.append(feats)
                    
                    # Check buffer size more efficiently
                    total_frames = sum(b.shape[0] for b in buffer)
                    if total_frames >= buffer_limit:
                        try:
                            stacked = np.vstack(buffer)
                            kmeans.partial_fit(stacked)
                            buffer = []
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                        except Exception as e:
                            print(f"⚠️  Error in partial_fit: {e}")
                            traceback.print_exc()
                            # Clear buffer and continue
                            buffer = []
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                
                # Save checkpoint periodically
                if (idx + 1) % checkpoint_interval == 0:
                    try:
                        processed_files.add(f)
                        with open(PROCESSED_FILES_LOG, 'a') as log:
                            log.write(f"{f}\n")
                        joblib.dump(kmeans, CHECKPOINT_FILE)
                        audio_volume.commit()
                        print(f"\n💾 Checkpoint saved at file {idx + 1}/{len(files_to_process)}")
                    except Exception as e:
                        print(f"⚠️  Error saving checkpoint: {e}")
            except Exception as e:
                print(f"⚠️  Error processing file {f}: {e}")
                traceback.print_exc()
                continue
        
        # Final buffer flush
        if buffer:
            try:
                stacked = np.vstack(buffer)
                kmeans.partial_fit(stacked)
            except Exception as e:
                print(f"⚠️  Error in final partial_fit: {e}")
                traceback.print_exc()
        
        # Save final K-Means model
        joblib.dump(kmeans, f"{OUTPUT_DIR}/portuguese_kmeans.pkl")
        print("✓ Acoustic alphabet saved!")
        
    except Exception as e:
        print(f"❌ Critical error in Step 1: {e}")
        traceback.print_exc()
        # Try to save checkpoint anyway
        try:
            joblib.dump(kmeans, CHECKPOINT_FILE)
            audio_volume.commit()
            print("💾 Emergency checkpoint saved")
        except:
            pass
        raise
    
    # Step 2: Convert all segments to unit sequences
    print("\n--- Converting Audio Segments to Units ---")
    corpus = {}
    processed_step2 = set()
    step2_checkpoint = f"{OUTPUT_DIR}/checkpoint_step2.json"
    
    # Try to load Step 2 checkpoint
    if os.path.exists(step2_checkpoint):
        try:
            with open(step2_checkpoint, 'r') as f:
                corpus = json.load(f)
            processed_step2 = set(corpus.keys())
            print(f"✓ Resuming Step 2: {len(processed_step2)} segments already processed")
        except Exception as e:
            print(f"⚠️  Could not load Step 2 checkpoint: {e}")
    
    files_to_process_step2 = [f for f in files if os.path.splitext(os.path.basename(f))[0] not in processed_step2]
    
    try:
        for idx, f in enumerate(tqdm(files_to_process_step2, desc="Step 2/2: Tokenizing"), start=len(processed_step2)):
            try:
                feats, timestamps, duration = get_features(f)
                if feats is None:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
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
                
                # Save checkpoint every 500 files
                if (idx + 1) % 500 == 0:
                    try:
                        with open(step2_checkpoint, 'w') as cp:
                            json.dump(corpus, cp)
                        audio_volume.commit()
                        print(f"\n💾 Step 2 checkpoint saved at {idx + 1}/{len(files_to_process_step2)}")
                    except Exception as e:
                        print(f"⚠️  Error saving Step 2 checkpoint: {e}")
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception as e:
                print(f"⚠️  Error tokenizing {os.path.basename(f)}: {e}")
                traceback.print_exc()
                continue
        
        # Save everything
        with open(f"{OUTPUT_DIR}/portuguese_corpus_timestamped.json", "w") as f:
            json.dump(corpus, f)
        
        with open(f"{OUTPUT_DIR}/all_units_for_bpe.txt", "w") as f:
            for data in corpus.values():
                f.write(" ".join(map(str, data["units"])) + "\n")
        
        # Clean up checkpoint files
        for cp_file in [CHECKPOINT_FILE, PROCESSED_FILES_LOG, step2_checkpoint]:
            if os.path.exists(cp_file):
                try:
                    os.remove(cp_file)
                except:
                    pass
        
        # Commit volume changes
        audio_volume.commit()
        
    except Exception as e:
        print(f"❌ Critical error in Step 2: {e}")
        traceback.print_exc()
        # Try to save checkpoint
        try:
            with open(step2_checkpoint, 'w') as cp:
                json.dump(corpus, cp)
            audio_volume.commit()
            print("💾 Emergency Step 2 checkpoint saved")
        except:
            pass
        raise
    
    # Summary
    total_mins = sum(d["duration_sec"] for d in corpus.values()) / 60
    print(f"\n✓ Phase 2 Complete!")
    print(f"  Segments processed: {len(corpus)}")
    print(f"  Total audio: {total_mins:.1f} minutes")
    print(f"  Acoustic units: {NUM_CLUSTERS}")
    
    return len(corpus)


@app.local_entrypoint()
def main():
    """Run the complete pipeline"""
    print("🚀 Starting Bible Audio Processing Pipeline")
    print("=" * 50)
    
    print("\n📁 Phase 1: Converting MP3 to WAV...")
    converted = convert_mp3_to_wav.remote()
    print(f"✓ Converted {converted} files")
    
    # Check if segments already exist - skip segmentation if they do
    print("\n✂️  Phase 1.5: Checking for existing segments...")
    # We'll check in the function itself, but let's skip if segments exist
    segments = segment_by_silence.remote()
    if segments == 0:
        print("✓ Using existing segments (skipped segmentation)")
    else:
        print(f"✓ Created {segments} segments")
    
    print("\n🎵 Phase 2: Acoustic Tokenization...")
    processed = acoustic_tokenization.remote()
    print(f"✓ Processed {processed} segments")
    
    print("\n✅ Pipeline complete!")
    print(f"Results are stored in Modal volume: {AUDIO_MOUNT}")
    print(f"  - Segmented audio: {SEGMENTED_DIR}")
    print(f"  - Training outputs: {OUTPUT_DIR}")


@app.local_entrypoint()
def main_skip_segmentation():
    """Run pipeline skipping segmentation (use existing segments)"""
    print("🚀 Starting Bible Audio Processing Pipeline")
    print("=" * 50)
    print("⚠️  Skipping segmentation - using existing segments")
    
    print("\n📁 Phase 1: Converting MP3 to WAV...")
    converted = convert_mp3_to_wav.remote()
    print(f"✓ Converted {converted} files")
    
    print("\n✂️  Phase 1.5: SKIPPED (using existing segments)")
    
    print("\n🎵 Phase 2: Acoustic Tokenization...")
    processed = acoustic_tokenization.remote()
    print(f"✓ Processed {processed} segments")
    
    print("\n✅ Pipeline complete!")
    print(f"Results are stored in Modal volume: {AUDIO_MOUNT}")
    print(f"  - Segmented audio: {SEGMENTED_DIR}")
    print(f"  - Training outputs: {OUTPUT_DIR}")
