"""
Streaming upload with progress - uploads files directly to volume in batches
Checks existing files first, only uploads what's missing.
Memory-efficient to prevent crashes.

Run with: /usr/local/bin/python3 -m modal run upload_audio_streaming.py
"""

import modal
import os
from pathlib import Path

app = modal.App("bible-audio-training")
audio_volume = modal.Volume.from_name("bible-audio-data", create_if_missing=True)

AUDIO_MOUNT = "/mnt/audio_data"
RAW_AUDIO_DIR = f"{AUDIO_MOUNT}/raw_audio"

# Local directories
LOCAL_ANTIGO_TESTAMENTO = "/Users/joao/Desktop/work/shema/shemaobt/scripts/Antigo_Testamento_COMPLETO"
LOCAL_NOVO_TESTAMENTO = "/Users/joao/Desktop/work/shema/shemaobt/scripts/NOVO_TESTAMENTO_COMPLETO"

image = modal.Image.debian_slim().pip_install("tqdm")


@app.function(
    image=image,
    volumes={AUDIO_MOUNT: audio_volume},
    timeout=300,  # 5 minutes
)
def list_existing_files():
    """Check what files already exist in the volume"""
    existing_files = set()
    
    if os.path.exists(RAW_AUDIO_DIR):
        for filename in os.listdir(RAW_AUDIO_DIR):
            file_path = os.path.join(RAW_AUDIO_DIR, filename)
            if os.path.isfile(file_path) and filename.endswith('.mp3'):
                existing_files.add(filename)
    
    return list(existing_files)


@app.function(
    image=image,
    volumes={AUDIO_MOUNT: audio_volume},
    timeout=7200,  # 2 hours for large uploads
)
def upload_file_batch(file_data_list: list[tuple[str, bytes]], batch_num: int, total_batches: int):
    """Upload a batch of files to the volume (memory-efficient)"""
    import os
    
    os.makedirs(RAW_AUDIO_DIR, exist_ok=True)
    
    uploaded = 0
    errors = []
    
    for filename, file_data in file_data_list:
        dest_path = os.path.join(RAW_AUDIO_DIR, filename)
        
        try:
            # Write file in chunks to avoid memory issues
            with open(dest_path, "wb") as f:
                # Write in 10MB chunks
                chunk_size = 10 * 1024 * 1024
                for i in range(0, len(file_data), chunk_size):
                    chunk = file_data[i:i+chunk_size]
                    f.write(chunk)
            uploaded += 1
        except Exception as e:
            errors.append(f"{filename}: {str(e)}")
    
    # Commit after each batch
    if uploaded > 0:
        audio_volume.commit()
    
    result_msg = f"  ✓ Batch {batch_num}/{total_batches}: Uploaded {uploaded} files"
    if errors:
        result_msg += f" ({len(errors)} errors)"
    print(result_msg)
    
    if errors:
        print(f"    Errors: {', '.join(errors[:3])}")  # Show first 3 errors
    
    return uploaded, errors


@app.local_entrypoint()
def main():
    """Collect files locally and upload in batches with progress"""
    from tqdm import tqdm
    
    print("=" * 60)
    print("📤 Bible Audio Upload to Modal")
    print("=" * 60)
    
    # Step 1: Check existing files
    print("\n🔍 Step 1: Checking existing files in volume...")
    existing_files = set(list_existing_files.remote())
    print(f"✓ Found {len(existing_files)} files already in volume")
    
    # Step 2: Collect local files
    print("\n📁 Step 2: Collecting MP3 files from local directories...")
    mp3_files = []
    for directory in [LOCAL_ANTIGO_TESTAMENTO, LOCAL_NOVO_TESTAMENTO]:
        if os.path.exists(directory):
            for root, dirs, files in os.walk(directory):
                for f in files:
                    if f.endswith('.mp3'):
                        mp3_files.append(os.path.join(root, f))
    
    total_files = len(mp3_files)
    print(f"✓ Found {total_files} MP3 files locally")
    
    if total_files == 0:
        print("❌ No MP3 files found. Check your directory paths.")
        return
    
    # Step 3: Filter out already uploaded files
    print("\n🔍 Step 3: Comparing local vs uploaded files...")
    files_to_upload = []
    file_sizes = []
    
    for mp3_path in mp3_files:
        filename = os.path.basename(mp3_path)
        if filename not in existing_files:
            files_to_upload.append(mp3_path)
            try:
                size = os.path.getsize(mp3_path)
                file_sizes.append(size)
            except:
                file_sizes.append(0)
    
    already_uploaded = total_files - len(files_to_upload)
    total_size = sum(file_sizes)
    total_size_mb = total_size / (1024 * 1024)
    
    print(f"✓ Files already uploaded: {already_uploaded}")
    print(f"✓ Files to upload: {len(files_to_upload)}")
    
    if len(files_to_upload) == 0:
        print("\n✅ All files already uploaded! Nothing to do.")
        return
    
    print(f"📊 Size to upload: {total_size_mb:.1f} MB ({total_size / (1024**3):.2f} GB)")
    
    # Step 4: Upload in small batches (memory-efficient)
    print(f"\n📤 Step 4: Uploading {len(files_to_upload)} missing files...")
    print("-" * 60)
    
    # Smaller batches to prevent memory issues (2 files at a time)
    BATCH_SIZE = 2
    total_batches = (len(files_to_upload) + BATCH_SIZE - 1) // BATCH_SIZE
    total_uploaded = 0
    total_errors = []
    
    # Progress bar
    with tqdm(total=len(files_to_upload), desc="Uploading", unit="file", unit_scale=False) as pbar:
        for i in range(0, len(files_to_upload), BATCH_SIZE):
            batch = files_to_upload[i:i+BATCH_SIZE]
            batch_num = (i // BATCH_SIZE) + 1
            file_data_list = []
            
            # Read files in batch (one at a time to save memory)
            batch_size_mb = 0
            for mp3_path in batch:
                filename = os.path.basename(mp3_path)
                try:
                    # Read file in chunks to avoid loading entire file in memory
                    file_data = b""
                    with open(mp3_path, "rb") as f:
                        while True:
                            chunk = f.read(10 * 1024 * 1024)  # 10MB chunks
                            if not chunk:
                                break
                            file_data += chunk
                    
                    file_data_list.append((filename, file_data))
                    batch_size_mb += len(file_data) / (1024 * 1024)
                    
                    # Clear memory hint
                    del chunk
                except Exception as e:
                    print(f"\n❌ Error reading {filename}: {e}")
                    total_errors.append(f"{filename}: {str(e)}")
            
            # Upload batch
            if file_data_list:
                try:
                    uploaded, errors = upload_file_batch.remote(file_data_list, batch_num, total_batches)
                    total_uploaded += uploaded
                    total_errors.extend(errors)
                    pbar.update(uploaded)
                    pbar.set_postfix({
                        'batch': f"{batch_num}/{total_batches}",
                        'size': f"{batch_size_mb:.1f}MB"
                    })
                    
                    # Clear memory
                    del file_data_list
                except Exception as e:
                    print(f"\n❌ Error uploading batch {batch_num}: {e}")
                    total_errors.append(f"Batch {batch_num}: {str(e)}")
    
    # Final summary
    print("\n" + "=" * 60)
    print(f"✅ Upload complete!")
    print(f"   Files already in volume: {already_uploaded}")
    print(f"   Files uploaded this run: {total_uploaded}/{len(files_to_upload)}")
    print(f"   Total files in volume now: {already_uploaded + total_uploaded}")
    print(f"   Total size uploaded: {total_size_mb:.1f} MB")
    
    if total_errors:
        print(f"\n⚠️  Errors encountered: {len(total_errors)}")
        print("   First few errors:")
        for error in total_errors[:5]:
            print(f"   - {error}")
    
    print(f"   Location: {RAW_AUDIO_DIR}")
    print("=" * 60)
