"""
Upload locally segmented audio files to Modal volume - Generic for any language

Supports multiple languages with separate Modal directories.

Usage:
    # Upload Portuguese segments
    python upload_to_modal.py --language portuguese
    
    # Upload Sateré segments
    python upload_to_modal.py --language satere
    
    # Custom directories
    python upload_to_modal.py --local /path/to/segments --remote /mnt/audio_data/custom_segments
"""

import modal
import os
import argparse
from tqdm import tqdm


# ============================================================================
# Configuration
# ============================================================================

# Base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Predefined language configurations
LANGUAGE_CONFIGS = {
    "portuguese": {
        "local_dir": os.path.join(BASE_DIR, "local_segments"),
        "remote_dir": "/mnt/audio_data/segmented_audio",
        "description": "Portuguese Bible (Old + New Testament)",
    },
    "satere": {
        "local_dir": os.path.join(BASE_DIR, "local_segments_satere"),
        "remote_dir": "/mnt/audio_data/segmented_audio_satere",
        "description": "Sateré-Mawé Bible",
    },
}

# Modal configuration
AUDIO_MOUNT = "/mnt/audio_data"


# ============================================================================
# Modal Setup
# ============================================================================

app = modal.App("bible-audio-training")

audio_volume = modal.Volume.from_name(
    "bible-audio-data",
    create_if_missing=True
)

image = modal.Image.debian_slim().pip_install("tqdm")


# ============================================================================
# Modal Functions
# ============================================================================

@app.function(
    image=image,
    volumes={AUDIO_MOUNT: audio_volume},
    timeout=3600,
)
def list_existing_segments(remote_dir: str):
    """List segments already in Modal volume."""
    existing = set()
    if os.path.exists(remote_dir):
        for f in os.listdir(remote_dir):
            if f.endswith('.wav'):
                existing.add(f)
    return list(existing)


@app.function(
    image=image,
    volumes={AUDIO_MOUNT: audio_volume},
    timeout=3600,
)
def upload_segment_batch(file_data_list: list, remote_dir: str, batch_num: int, total_batches: int):
    """Upload a batch of segment files to specified remote directory."""
    os.makedirs(remote_dir, exist_ok=True)
    
    uploaded = 0
    errors = []
    
    for filename, file_data in file_data_list:
        try:
            file_path = os.path.join(remote_dir, filename)
            
            # Write file in chunks to avoid memory issues
            with open(file_path, 'wb') as f:
                chunk_size = 10 * 1024 * 1024  # 10MB chunks
                for i in range(0, len(file_data), chunk_size):
                    chunk = file_data[i:i+chunk_size]
                    f.write(chunk)
            
            uploaded += 1
        except Exception as e:
            errors.append(f"{filename}: {str(e)}")
    
    # Commit after each batch
    audio_volume.commit()
    
    return uploaded, errors


# ============================================================================
# Local Entrypoint
# ============================================================================

@app.local_entrypoint()
def main(
    language: str = "portuguese",
    local: str = None,
    remote: str = None,
    batch_size: int = 5,
):
    """
    Upload locally segmented files to Modal.
    
    Args:
        language: Language preset (portuguese, satere)
        local: Custom local directory (overrides language preset)
        remote: Custom remote directory (overrides language preset)
        batch_size: Number of files per upload batch
    """
    # Determine directories
    if local:
        local_dir = local
        remote_dir = remote or f"{AUDIO_MOUNT}/segmented_audio_custom"
        description = "Custom audio"
    elif language in LANGUAGE_CONFIGS:
        config = LANGUAGE_CONFIGS[language]
        local_dir = config["local_dir"]
        remote_dir = remote or config["remote_dir"]
        description = config["description"]
    else:
        print(f"❌ Unknown language: {language}")
        print(f"   Available: {', '.join(LANGUAGE_CONFIGS.keys())}")
        print("   Or use --local and --remote for custom directories")
        return
    
    print("=" * 60)
    print(f"📤 Upload Segments to Modal: {language.upper()}")
    print("=" * 60)
    print(f"\n📝 {description}")
    print(f"📂 Local: {local_dir}")
    print(f"☁️  Remote: {remote_dir}")
    
    # Check local directory
    if not os.path.exists(local_dir):
        print(f"\n❌ Local segments directory not found: {local_dir}")
        print(f"   Please run first: python scripts/segment_audio.py --language {language}")
        return
    
    # Collect local segment files
    print("\n📁 Collecting local segment files...")
    segment_files = []
    for f in os.listdir(local_dir):
        if f.endswith('.wav') and not f.startswith('.'):
            segment_files.append(os.path.join(local_dir, f))
    
    print(f"✓ Found {len(segment_files)} segment files locally")
    
    if len(segment_files) == 0:
        print("❌ No segment files found!")
        return
    
    # Check existing files in Modal
    print("\n🔍 Checking existing files in Modal volume...")
    existing_files = set(list_existing_segments.remote(remote_dir))
    print(f"✓ Found {len(existing_files)} files already in volume")
    
    # Filter out already uploaded files
    files_to_upload = []
    file_sizes = []
    
    for seg_path in segment_files:
        filename = os.path.basename(seg_path)
        if filename not in existing_files:
            files_to_upload.append(seg_path)
            try:
                size = os.path.getsize(seg_path)
                file_sizes.append(size)
            except:
                file_sizes.append(0)
    
    already_uploaded = len(segment_files) - len(files_to_upload)
    total_size = sum(file_sizes)
    total_size_mb = total_size / (1024 * 1024)
    
    print(f"\n📊 Upload Status:")
    print(f"   Already uploaded: {already_uploaded}")
    print(f"   To upload: {len(files_to_upload)}")
    
    if len(files_to_upload) == 0:
        print("\n✅ All segments already uploaded!")
        return
    
    print(f"   Size to upload: {total_size_mb:.1f} MB ({total_size / (1024**3):.2f} GB)")
    
    # Upload in batches
    print(f"\n📤 Uploading {len(files_to_upload)} files (batch size: {batch_size})...")
    print("-" * 60)
    
    total_batches = (len(files_to_upload) + batch_size - 1) // batch_size
    total_uploaded = 0
    total_errors = []
    
    with tqdm(total=len(files_to_upload), desc="Uploading", unit="file") as pbar:
        for i in range(0, len(files_to_upload), batch_size):
            batch = files_to_upload[i:i+batch_size]
            batch_num = (i // batch_size) + 1
            file_data_list = []
            
            batch_size_mb = 0
            for seg_path in batch:
                filename = os.path.basename(seg_path)
                try:
                    with open(seg_path, "rb") as f:
                        file_data = f.read()
                    
                    file_data_list.append((filename, file_data))
                    batch_size_mb += len(file_data) / (1024 * 1024)
                except Exception as e:
                    print(f"\n❌ Error reading {filename}: {e}")
                    total_errors.append(f"{filename}: {str(e)}")
            
            if file_data_list:
                try:
                    uploaded, errors = upload_segment_batch.remote(
                        file_data_list, remote_dir, batch_num, total_batches
                    )
                    total_uploaded += uploaded
                    total_errors.extend(errors)
                    pbar.update(uploaded)
                    pbar.set_postfix({'batch': f"{batch_num}/{total_batches}", 'size': f"{batch_size_mb:.1f}MB"})
                except Exception as e:
                    print(f"\n❌ Error uploading batch {batch_num}: {e}")
                    total_errors.append(f"Batch {batch_num}: {str(e)}")
    
    # Summary
    print("\n" + "=" * 60)
    print("✅ Upload Complete!")
    print("=" * 60)
    print(f"Language: {language}")
    print(f"Uploaded: {total_uploaded} files")
    if total_errors:
        print(f"Errors: {len(total_errors)}")
        for error in total_errors[:10]:
            print(f"  - {error}")
    print(f"\nSegments in Modal: {remote_dir}")
    print(f"\n🎵 Next: Run Phase 1 (acoustic tokenization) for this language")
