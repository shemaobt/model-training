"""
Upload locally segmented audio files to Modal volume
Run this after segment_local.py completes
"""

import modal
import os
from tqdm import tqdm

# Create Modal app
app = modal.App("bible-audio-training")

# Create persistent volume (same as Phase 1)
audio_volume = modal.Volume.from_name(
    "bible-audio-data",
    create_if_missing=True
)

# Local segmented directory
LOCAL_SEGMENTED_DIR = "/Users/joao/Desktop/work/shema/shemaobt/model-training/local_segments"

# Modal paths
AUDIO_MOUNT = "/mnt/audio_data"
SEGMENTED_DIR = f"{AUDIO_MOUNT}/segmented_audio"

# Create minimal image (just for uploading)
image = (
    modal.Image.debian_slim()
    .pip_install("tqdm")
)


@app.function(
    image=image,
    volumes={AUDIO_MOUNT: audio_volume},
    timeout=3600,
)
def list_existing_segments():
    """List segments already in Modal volume"""
    existing = set()
    if os.path.exists(SEGMENTED_DIR):
        for f in os.listdir(SEGMENTED_DIR):
            if f.endswith('.wav'):
                existing.add(f)
    return list(existing)


@app.function(
    image=image,
    volumes={AUDIO_MOUNT: audio_volume},
    timeout=3600,
)
def upload_segment_batch(file_data_list, batch_num, total_batches):
    """Upload a batch of segment files"""
    os.makedirs(SEGMENTED_DIR, exist_ok=True)
    
    uploaded = 0
    errors = []
    
    for filename, file_data in file_data_list:
        try:
            file_path = os.path.join(SEGMENTED_DIR, filename)
            
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


@app.local_entrypoint()
def main():
    """Upload locally segmented files to Modal"""
    print("=" * 60)
    print("📤 Uploading Segmented Audio Files to Modal")
    print("=" * 60)
    
    # Check local directory
    if not os.path.exists(LOCAL_SEGMENTED_DIR):
        print(f"❌ Local segments directory not found: {LOCAL_SEGMENTED_DIR}")
        print("   Please run segment_local.py first!")
        return
    
    # Collect local segment files
    print("\n📁 Collecting local segment files...")
    segment_files = []
    for f in os.listdir(LOCAL_SEGMENTED_DIR):
        if f.endswith('.wav') and not f.startswith('.'):
            segment_files.append(os.path.join(LOCAL_SEGMENTED_DIR, f))
    
    print(f"✓ Found {len(segment_files)} segment files locally")
    
    if len(segment_files) == 0:
        print("❌ No segment files found!")
        return
    
    # Check existing files in Modal
    print("\n🔍 Checking existing files in Modal volume...")
    existing_files = set(list_existing_segments.remote())
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
    print(f"\n📤 Uploading {len(files_to_upload)} files...")
    print("-" * 60)
    
    BATCH_SIZE = 5  # Upload 5 files at a time
    total_batches = (len(files_to_upload) + BATCH_SIZE - 1) // BATCH_SIZE
    total_uploaded = 0
    total_errors = []
    
    with tqdm(total=len(files_to_upload), desc="Uploading", unit="file") as pbar:
        for i in range(0, len(files_to_upload), BATCH_SIZE):
            batch = files_to_upload[i:i+BATCH_SIZE]
            batch_num = (i // BATCH_SIZE) + 1
            file_data_list = []
            
            batch_size_mb = 0
            for seg_path in batch:
                filename = os.path.basename(seg_path)
                try:
                    # Read file in chunks to avoid memory issues
                    file_data = b""
                    with open(seg_path, "rb") as f:
                        while True:
                            chunk = f.read(10 * 1024 * 1024)  # 10MB chunks
                            if not chunk:
                                break
                            file_data += chunk
                    
                    file_data_list.append((filename, file_data))
                    batch_size_mb += len(file_data) / (1024 * 1024)
                except Exception as e:
                    print(f"\n❌ Error reading {filename}: {e}")
                    total_errors.append(f"{filename}: {str(e)}")
            
            if file_data_list:
                try:
                    uploaded, errors = upload_segment_batch.remote(file_data_list, batch_num, total_batches)
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
    print(f"Uploaded: {total_uploaded} files")
    if total_errors:
        print(f"Errors: {len(total_errors)}")
        for error in total_errors[:10]:  # Show first 10 errors
            print(f"  - {error}")
    print(f"\nSegments are now in Modal volume: {SEGMENTED_DIR}")
    print("You can now run Phase 2 (acoustic tokenization) on Modal!")
