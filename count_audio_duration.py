"""
Simple script to count Bible audio files and calculate total duration
Run with: python3 count_audio_duration.py
"""

import os
from pathlib import Path
import subprocess
from tqdm import tqdm

# Local directories
LOCAL_ANTIGO_TESTAMENTO = "/Users/joao/Desktop/work/shema/shemaobt/scripts/Antigo_Testamento_COMPLETO"
LOCAL_NOVO_TESTAMENTO = "/Users/joao/Desktop/work/shema/shemaobt/scripts/NOVO_TESTAMENTO_COMPLETO"


def get_audio_duration(file_path):
    """Get duration of audio file using ffprobe"""
    try:
        cmd = [
            "ffprobe", "-v", "error", "-show_entries",
            "format=duration", "-of", "default=noprint_wrappers=1:nokey=1",
            file_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            return float(result.stdout.strip())
        return None
    except Exception as e:
        return None


def main():
    print("=" * 60)
    print("📊 Bible Audio File Counter & Duration Calculator")
    print("=" * 60)
    
    # Collect all MP3 files
    mp3_files = []
    
    print("\n📁 Scanning directories...")
    
    # Antigo Testamento
    if os.path.exists(LOCAL_ANTIGO_TESTAMENTO):
        for root, dirs, files in os.walk(LOCAL_ANTIGO_TESTAMENTO):
            for f in files:
                if f.endswith('.mp3'):
                    mp3_files.append(os.path.join(root, f))
        print(f"  ✓ Antigo Testamento: Found files")
    else:
        print(f"  ⚠️  Antigo Testamento directory not found")
    
    # Novo Testamento
    if os.path.exists(LOCAL_NOVO_TESTAMENTO):
        for root, dirs, files in os.walk(LOCAL_NOVO_TESTAMENTO):
            for f in files:
                if f.endswith('.mp3'):
                    mp3_files.append(os.path.join(root, f))
        print(f"  ✓ Novo Testamento: Found files")
    else:
        print(f"  ⚠️  Novo Testamento directory not found")
    
    total_files = len(mp3_files)
    print(f"\n📊 Total MP3 files found: {total_files}")
    
    if total_files == 0:
        print("❌ No MP3 files found!")
        return
    
    # Calculate total duration
    print(f"\n⏱️  Calculating total duration...")
    print("   (This may take a few minutes for large files)")
    
    total_seconds = 0
    successful = 0
    failed = 0
    
    for mp3_path in tqdm(mp3_files, desc="Processing files"):
        duration = get_audio_duration(mp3_path)
        if duration is not None:
            total_seconds += duration
            successful += 1
        else:
            failed += 1
    
    # Convert to hours and minutes
    total_hours = int(total_seconds // 3600)
    remaining_seconds = total_seconds % 3600
    total_minutes = int(remaining_seconds // 60)
    remaining_secs = int(remaining_seconds % 60)
    
    # Display results
    print("\n" + "=" * 60)
    print("📊 RESULTS")
    print("=" * 60)
    print(f"Total files: {total_files}")
    print(f"  ✓ Successfully processed: {successful}")
    if failed > 0:
        print(f"  ⚠️  Failed to read: {failed}")
    
    print(f"\n⏱️  Total Duration:")
    print(f"   {total_hours} hours, {total_minutes} minutes, {remaining_secs} seconds")
    print(f"   ({total_seconds:.1f} seconds total)")
    print(f"   ({total_seconds / 3600:.2f} hours)")
    
    # Calculate average
    if successful > 0:
        avg_seconds = total_seconds / successful
        avg_minutes = avg_seconds / 60
        print(f"\n📈 Average per file:")
        print(f"   {avg_minutes:.1f} minutes ({avg_seconds:.0f} seconds)")
    
    print("=" * 60)


if __name__ == "__main__":
    main()
