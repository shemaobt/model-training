"""
Modal deployment with optional cloud storage (S3/GCS) integration
This version can read from and write to cloud storage if configured
"""

import modal
import os

# Create Modal app
app = modal.App("bible-audio-training")

# Create persistent volume for audio files and outputs
audio_volume = modal.Volume.from_name(
    "bible-audio-data",
    create_if_missing=True
)

# Optional: Add cloud storage secret if you want to use S3/GCS
# Uncomment and configure if needed:
# cloud_secret = modal.Secret.from_name("aws-credentials")  # or "gcs-credentials"

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
        # Uncomment if using cloud storage:
        # "boto3",  # For AWS S3
        # "google-cloud-storage",  # For GCS
    )
)

# Mount points
AUDIO_MOUNT = "/mnt/audio_data"
RAW_AUDIO_DIR = f"{AUDIO_MOUNT}/raw_audio"
CONVERTED_DIR = f"{AUDIO_MOUNT}/converted_audio"
OUTPUT_DIR = f"{AUDIO_MOUNT}/satere_units"


@app.function(
    image=image,
    volumes={AUDIO_MOUNT: audio_volume},
    timeout=3600,
    gpu="T4",
    # secrets=[cloud_secret],  # Uncomment if using cloud storage
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
    timeout=7200,
    gpu="T4",
    # secrets=[cloud_secret],  # Uncomment if using cloud storage
)
def acoustic_tokenization():
    """Phase 2: Acoustic Tokenization using XLSR-53 + K-Means"""
    import torch
    import torchaudio
    import numpy as np
    import joblib
    import json
    from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor
    from sklearn.cluster import MiniBatchKMeans
    from tqdm import tqdm
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    MODEL_NAME = "facebook/wav2vec2-large-xlsr-53"
    LAYER = 14
    NUM_CLUSTERS = 100
    
    # Load model
    print("Loading XLSR-53 model (this takes a minute)...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using: {device}")
    
    extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)
    model = Wav2Vec2Model.from_pretrained(MODEL_NAME).to(device)
    model.eval()
    print("✓ Model loaded!")
    
    def get_features(path):
        """Extract features from audio file."""
        try:
            waveform, rate = torchaudio.load(path)
            if rate != 16000:
                waveform = torchaudio.transforms.Resample(rate, 16000)(waveform)
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            
            duration = waveform.shape[1] / 16000
            inputs = extractor(waveform.squeeze().numpy(), return_tensors="pt", sampling_rate=16000)
            inputs = inputs.input_values.to(device)
            
            with torch.no_grad():
                outputs = model(inputs, output_hidden_states=True)
            
            feats = outputs.hidden_states[LAYER].squeeze(0).cpu().numpy()
            timestamps = np.linspace(0, duration, feats.shape[0])
            return feats, timestamps, duration
        except Exception as e:
            print(f"Error: {path}: {e}")
            return None, None, None
    
    # Get file list
    files = sorted([os.path.join(CONVERTED_DIR, f) for f in os.listdir(CONVERTED_DIR) if f.endswith('.wav')])
    print(f"Found {len(files)} WAV files")
    
    # Step 1: Train K-Means
    print("\n--- Learning Acoustic Alphabet ---")
    kmeans = MiniBatchKMeans(n_clusters=NUM_CLUSTERS, batch_size=1024, random_state=42, n_init=3)
    
    buffer = []
    buffer_limit = 50000
    
    for f in tqdm(files, desc="Step 1/2: Extracting features"):
        feats, _, _ = get_features(f)
        if feats is not None:
            buffer.append(feats)
            if sum(b.shape[0] for b in buffer) >= buffer_limit:
                kmeans.partial_fit(np.vstack(buffer))
                buffer = []
    
    if buffer:
        kmeans.partial_fit(np.vstack(buffer))
    
    joblib.dump(kmeans, f"{OUTPUT_DIR}/satere_kmeans.pkl")
    print("✓ Acoustic alphabet saved!")
    
    # Step 2: Convert all audio to unit sequences
    print("\n--- Converting Audio to Units ---")
    corpus = {}
    
    for f in tqdm(files, desc="Step 2/2: Tokenizing"):
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
    
    # Save everything
    with open(f"{OUTPUT_DIR}/satere_corpus_timestamped.json", "w") as f:
        json.dump(corpus, f)
    
    with open(f"{OUTPUT_DIR}/all_units_for_bpe.txt", "w") as f:
        for data in corpus.values():
            f.write(" ".join(map(str, data["units"])) + "\n")
    
    # Commit volume changes
    audio_volume.commit()
    
    # Summary
    total_mins = sum(d["duration_sec"] for d in corpus.values()) / 60
    print(f"\n✓ Phase 2 Complete!")
    print(f"  Files processed: {len(corpus)}")
    print(f"  Total audio: {total_mins:.1f} minutes")
    print(f"  Acoustic units: {NUM_CLUSTERS}")
    
    return len(corpus)


@app.function(
    image=image,
    volumes={AUDIO_MOUNT: audio_volume},
    timeout=1800,
    # secrets=[cloud_secret],  # Uncomment if using cloud storage
)
def sync_to_cloud_storage(bucket_name: str, storage_type: str = "s3"):
    """
    Optional: Sync results to cloud storage (S3 or GCS)
    
    Args:
        bucket_name: Name of the S3 bucket or GCS bucket
        storage_type: "s3" or "gcs"
    """
    if storage_type == "s3":
        import boto3
        s3 = boto3.client('s3')
        
        # Upload all files from output directory
        for root, dirs, files in os.walk(OUTPUT_DIR):
            for file in files:
                local_path = os.path.join(root, file)
                relative_path = os.path.relpath(local_path, OUTPUT_DIR)
                s3_key = f"bible-audio-training/{relative_path}"
                
                print(f"Uploading {relative_path} to s3://{bucket_name}/{s3_key}")
                s3.upload_file(local_path, bucket_name, s3_key)
        
        print(f"✓ Synced to S3 bucket: {bucket_name}")
        
    elif storage_type == "gcs":
        from google.cloud import storage
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        
        # Upload all files from output directory
        for root, dirs, files in os.walk(OUTPUT_DIR):
            for file in files:
                local_path = os.path.join(root, file)
                relative_path = os.path.relpath(local_path, OUTPUT_DIR)
                blob_name = f"bible-audio-training/{relative_path}"
                
                print(f"Uploading {relative_path} to gs://{bucket_name}/{blob_name}")
                blob = bucket.blob(blob_name)
                blob.upload_from_filename(local_path)
        
        print(f"✓ Synced to GCS bucket: {bucket_name}")
    
    else:
        raise ValueError(f"Unknown storage type: {storage_type}")


@app.local_entrypoint()
def main(sync_cloud: bool = False, bucket_name: str = None, storage_type: str = "s3"):
    """Run the complete pipeline"""
    print("🚀 Starting Bible Audio Processing Pipeline")
    print("=" * 50)
    
    print("\n📁 Phase 1: Converting MP3 to WAV...")
    converted = convert_mp3_to_wav.remote()
    print(f"✓ Converted {converted} files")
    
    print("\n🎵 Phase 2: Acoustic Tokenization...")
    processed = acoustic_tokenization.remote()
    print(f"✓ Processed {processed} files")
    
    if sync_cloud and bucket_name:
        print(f"\n☁️ Syncing to {storage_type.upper()}...")
        sync_to_cloud_storage.remote(bucket_name, storage_type)
        print("✓ Cloud sync complete!")
    
    print("\n✅ Pipeline complete!")
    print(f"Results are stored in Modal volume: {AUDIO_MOUNT}")
