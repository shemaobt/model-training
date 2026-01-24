"""
Upload Bible audio files to Modal volume
Run with: python3 -m modal run upload_audio_to_modal.py

Note: Make sure you're using system python3 (not venv) or have modal installed in your venv
"""

import modal
import os

app = modal.App("bible-audio-training")
audio_volume = modal.Volume.from_name("bible-audio-data", create_if_missing=True)

AUDIO_MOUNT = "/mnt/audio_data"
RAW_AUDIO_DIR = f"{AUDIO_MOUNT}/raw_audio"

# Local directories
LOCAL_ANTIGO_TESTAMENTO = "/Users/joao/Desktop/work/shema/shemaobt/scripts/Antigo_Testamento_COMPLETO"
LOCAL_NOVO_TESTAMENTO = "/Users/joao/Desktop/work/shema/shemaobt/scripts/NOVO_TESTAMENTO_COMPLETO"

# Create image with local directories mounted
image = (
    modal.Image.debian_slim()
    .pip_install("tqdm")
)

# Add local directories to image if they exist
if os.path.exists(LOCAL_ANTIGO_TESTAMENTO):
    image = image.add_local_dir(LOCAL_ANTIGO_TESTAMENTO, remote_path="/local/antigo")
if os.path.exists(LOCAL_NOVO_TESTAMENTO):
    image = image.add_local_dir(LOCAL_NOVO_TESTAMENTO, remote_path="/local/novo")


@app.function(
    image=image,
    volumes={AUDIO_MOUNT: audio_volume},
    timeout=3600,
)
def upload_from_mounts():
    """Upload MP3 files from mounted local directories to Modal volume"""
    import shutil
    from tqdm import tqdm
    
    os.makedirs(RAW_AUDIO_DIR, exist_ok=True)
    
    mp3_files = []
    
    # Collect from mounted directories
    for mount_path in ["/local/antigo", "/local/novo"]:
        if os.path.exists(mount_path):
            for root, dirs, files in os.walk(mount_path):
                for f in files:
                    if f.endswith('.mp3'):
                        mp3_files.append(os.path.join(root, f))
    
    print(f"Found {len(mp3_files)} MP3 files in mounted directories")
    
    if not mp3_files:
        print("⚠️  No MP3 files found. Check that directories are mounted correctly.")
        return 0
    
    uploaded = 0
    skipped = 0
    for mp3_path in tqdm(mp3_files, desc="Uploading files"):
        filename = os.path.basename(mp3_path)
        dest_path = os.path.join(RAW_AUDIO_DIR, filename)
        
        # Skip if already exists
        if os.path.exists(dest_path):
            skipped += 1
            continue
        
        try:
            shutil.copy2(mp3_path, dest_path)
            uploaded += 1
        except Exception as e:
            print(f"Error copying {filename}: {e}")
    
    audio_volume.commit()
    print(f"\n✓ Uploaded {uploaded} new files, skipped {skipped} existing files")
    return uploaded


@app.local_entrypoint()
def main():
    """Upload audio files to Modal"""
    print("📤 Uploading Bible audio files to Modal...")
    print(f"  - Antigo Testamento: {LOCAL_ANTIGO_TESTAMENTO}")
    print(f"  - Novo Testamento: {LOCAL_NOVO_TESTAMENTO}")
    result = upload_from_mounts.remote()
    print(f"✅ Upload complete! {result} files uploaded")
