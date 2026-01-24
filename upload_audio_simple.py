"""
Simple script to upload audio files to Modal volume
This uses Modal's volume API to write files directly.

Run with: python3 upload_audio_simple.py
"""

import modal
import os
from pathlib import Path
from tqdm import tqdm

# Connect to volume
audio_volume = modal.Volume.from_name("bible-audio-data", create_if_missing=True)

# Local directories
LOCAL_ANTIGO_TESTAMENTO = "/Users/joao/Desktop/work/shema/shemaobt/scripts/Antigo_Testamento_COMPLETO"
LOCAL_NOVO_TESTAMENTO = "/Users/joao/Desktop/work/shema/shemaobt/scripts/NOVO_TESTAMENTO_COMPLETO"

def collect_mp3_files():
    """Collect all MP3 files from local directories"""
    mp3_files = []
    
    for directory in [LOCAL_ANTIGO_TESTAMENTO, LOCAL_NOVO_TESTAMENTO]:
        if os.path.exists(directory):
            for root, dirs, files in os.walk(directory):
                for f in files:
                    if f.endswith('.mp3'):
                        mp3_files.append(os.path.join(root, f))
    
    return mp3_files

def main():
    print("📤 Collecting Bible audio files...")
    mp3_files = collect_mp3_files()
    print(f"Found {len(mp3_files)} MP3 files")
    
    if not mp3_files:
        print("❌ No MP3 files found. Check your directory paths.")
        return
    
    print("\n📤 Uploading to Modal volume...")
    print("Note: This will upload files one by one. Large files may take time.")
    
    uploaded = 0
    for mp3_path in tqdm(mp3_files, desc="Uploading"):
        filename = os.path.basename(mp3_path)
        volume_path = f"raw_audio/{filename}"
        
        # Check if file already exists
        try:
            if audio_volume.path(volume_path).exists():
                continue
        except:
            pass
        
        # Read and write file
        with open(mp3_path, "rb") as f:
            file_data = f.read()
        
        # Write to volume
        with audio_volume.open(volume_path, "wb") as vf:
            vf.write(file_data)
        
        uploaded += 1
    
    # Commit changes
    audio_volume.commit()
    
    print(f"\n✅ Upload complete! {uploaded} files uploaded to Modal volume")
    print(f"Files are now available at: /mnt/audio_data/raw_audio/")

if __name__ == "__main__":
    main()
