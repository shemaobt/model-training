import modal
import os

# Create Modal app
app = modal.App("bible-audio-training")

# Create persistent volume (same as Phase 1)
audio_volume = modal.Volume.from_name(
    "bible-audio-data",
    create_if_missing=True
)

# Modal paths
AUDIO_MOUNT = "/mnt/audio_data"
SEGMENTED_DIR = f"{AUDIO_MOUNT}/segmented_audio"

# Create minimal image
image = (
    modal.Image.debian_slim()
)


@app.function(
    image=image,
    volumes={AUDIO_MOUNT: audio_volume},
    timeout=600,  # 10 minutes
)
def delete_segmented_audio():
    deleted_count = 0
    total_size = 0
    
    if not os.path.exists(SEGMENTED_DIR):
        print(f"❌ Segmented directory does not exist: {SEGMENTED_DIR}")
        return {"deleted": 0, "size_mb": 0}
    
    print(f"📁 Scanning directory: {SEGMENTED_DIR}")
    
    # List all files
    files_to_delete = []
    for filename in os.listdir(SEGMENTED_DIR):
        file_path = os.path.join(SEGMENTED_DIR, filename)
        if os.path.isfile(file_path):
            files_to_delete.append(file_path)
            try:
                total_size += os.path.getsize(file_path)
            except:
                pass
    
    print(f"Found {len(files_to_delete)} files to delete")
    print(f"Total size: {total_size / (1024 * 1024):.1f} MB ({total_size / (1024**3):.2f} GB)")
    
    if len(files_to_delete) == 0:
        print("✓ No files to delete")
        return {"deleted": 0, "size_mb": 0}
    
    # Delete files
    print(f"\n🗑️  Deleting {len(files_to_delete)} files...")
    for file_path in files_to_delete:
        try:
            os.remove(file_path)
            deleted_count += 1
        except Exception as e:
            print(f"⚠️  Error deleting {os.path.basename(file_path)}: {e}")
    
    # Commit volume changes
    audio_volume.commit()
    
    print(f"\n✅ Deletion complete!")
    print(f"   Deleted: {deleted_count} files")
    print(f"   Freed: {total_size / (1024 * 1024):.1f} MB ({total_size / (1024**3):.2f} GB)")
    
    return {
        "deleted": deleted_count,
        "size_mb": total_size / (1024 * 1024)
    }


@app.local_entrypoint()
def main():
    print("=" * 60)
    print("🗑️  Delete Segmented Audio Files from Modal Volume")
    print("=" * 60)
    print("\n⚠️  WARNING: This will delete ALL segmented audio files!")
    print("   This action cannot be undone.")
    print()
    
    result = delete_segmented_audio.remote()
    
    if result:
        print(f"\n✅ Successfully deleted {result['deleted']} files")
        print(f"   Freed {result['size_mb']:.1f} MB of storage")
    else:
        print("\n❌ Deletion failed or no files found")
