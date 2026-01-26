"""
Local audio segmentation script
Segments audio files by silence/pauses before uploading to Modal

Run this locally first, then upload the segments to Modal
"""

import os
import librosa
import numpy as np
import soundfile as sf
import subprocess
import shutil
from tqdm import tqdm

# Local directories
LOCAL_ANTIGO_TESTAMENTO = "/Users/joao/Desktop/work/shema/shemaobt/scripts/Antigo_Testamento_COMPLETO"
LOCAL_NOVO_TESTAMENTO = "/Users/joao/Desktop/work/shema/shemaobt/scripts/NOVO_TESTAMENTO_COMPLETO"

# Output directory for segments
LOCAL_SEGMENTED_DIR = "/Users/joao/Desktop/work/shema/shemaobt/model-training/local_segments"

# Parameters for silence detection
SILENCE_THRESHOLD = -40  # dB threshold for silence
MIN_SILENCE_DURATION = 0.5  # Minimum silence duration in seconds
MIN_SEGMENT_DURATION = 2.0  # Minimum segment duration in seconds
MAX_SEGMENT_DURATION = 120.0  # Maximum segment duration in seconds (2 minutes max)

# Sample rate
SAMPLE_RATE = 16000


def convert_mp3_to_wav(mp3_path, wav_path):
    """Convert MP3 to 16kHz mono WAV"""
    cmd = [
        "ffmpeg", "-y", "-i", mp3_path,
        "-ar", str(SAMPLE_RATE),
        "-ac", "1",  # mono
        wav_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Warning: ffmpeg conversion failed for {mp3_path}: {result.stderr}")
        return False
    return True


def segment_audio_file(wav_path, output_dir):
    """Segment a single audio file by silence"""
    try:
        # Load audio
        audio, sr = librosa.load(wav_path, sr=SAMPLE_RATE, mono=True)
        duration = len(audio) / sr
        
        # Detect silence using energy-based approach
        frame_length = int(0.025 * sr)  # 25ms frames
        hop_length = int(0.010 * sr)    # 10ms hop
        
        # Calculate RMS energy per frame
        rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
        
        # Convert to dB
        rms_db = librosa.power_to_db(rms**2, ref=np.max)
        
        # Find silence regions (below threshold)
        silence_mask = rms_db < SILENCE_THRESHOLD
        
        # Find segment boundaries
        segment_boundaries = [0]  # Start with beginning
        
        in_silence = False
        silence_start = None
        
        for i, is_silent in enumerate(silence_mask):
            frame_time = i * hop_length / sr
            
            if is_silent and not in_silence:
                in_silence = True
                silence_start = frame_time
            elif not is_silent and in_silence:
                if silence_start is not None and (frame_time - silence_start) >= MIN_SILENCE_DURATION:
                    boundary = (silence_start + frame_time) / 2
                    segment_boundaries.append(boundary)
                in_silence = False
                silence_start = None
        
        # Add end boundary
        segment_boundaries.append(duration)
        
        # Create segments
        base_name = os.path.splitext(os.path.basename(wav_path))[0]
        segment_num = 0
        segments_created = 0
        
        for i in range(len(segment_boundaries) - 1):
            start_time = segment_boundaries[i]
            end_time = segment_boundaries[i + 1]
            segment_duration = end_time - start_time
            
            # Skip segments that are too short
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
                    segment_path = os.path.join(output_dir, segment_filename)
                    
                    # Extract segment using ffmpeg
                    cmd = [
                        "ffmpeg", "-y", "-i", wav_path,
                        "-ss", str(split_start),
                        "-t", str(split_end - split_start),
                        "-ar", str(SAMPLE_RATE), "-ac", "1",
                        segment_path
                    ]
                    subprocess.run(cmd, capture_output=True, text=True)
                    segment_num += 1
                    segments_created += 1
            else:
                segment_filename = f"{base_name}_seg_{segment_num:04d}.wav"
                segment_path = os.path.join(output_dir, segment_filename)
                
                # Extract segment using ffmpeg
                cmd = [
                    "ffmpeg", "-y", "-i", wav_path,
                    "-ss", str(start_time),
                    "-t", str(segment_duration),
                    "-ar", str(SAMPLE_RATE), "-ac", "1",
                    segment_path
                ]
                subprocess.run(cmd, capture_output=True, text=True)
                segment_num += 1
                segments_created += 1
        
        return segments_created
        
    except Exception as e:
        print(f"Error segmenting {wav_path}: {e}")
        return 0


def main():
    print("=" * 60)
    print("Local Audio Segmentation by Silence")
    print("=" * 60)
    
    # Create output directory
    os.makedirs(LOCAL_SEGMENTED_DIR, exist_ok=True)
    
    # Collect all MP3 files
    print("\n📁 Scanning for MP3 files...")
    mp3_files = []
    
    for directory in [LOCAL_ANTIGO_TESTAMENTO, LOCAL_NOVO_TESTAMENTO]:
        if os.path.exists(directory):
            for root, dirs, files in os.walk(directory):
                for f in files:
                    if f.endswith('.mp3'):
                        mp3_files.append(os.path.join(root, f))
    
    print(f"✓ Found {len(mp3_files)} MP3 files")
    
    if len(mp3_files) == 0:
        print("❌ No MP3 files found!")
        return
    
    # Check what's already segmented
    existing_segments = set()
    if os.path.exists(LOCAL_SEGMENTED_DIR):
        for f in os.listdir(LOCAL_SEGMENTED_DIR):
            if f.endswith('.wav'):
                # Extract base filename (without segment number)
                base = f.rsplit('_seg_', 1)[0] if '_seg_' in f else f.replace('.wav', '')
                existing_segments.add(base)
    
    # Filter out already processed files
    files_to_process = []
    for mp3_path in mp3_files:
        base_name = os.path.splitext(os.path.basename(mp3_path))[0]
        if base_name not in existing_segments:
            files_to_process.append(mp3_path)
    
    print(f"\n📊 Status:")
    print(f"   Already segmented: {len(existing_segments)} files")
    print(f"   To process: {len(files_to_process)} files")
    
    if len(files_to_process) == 0:
        print("\n✅ All files already segmented!")
        return
    
    # Process files
    print(f"\n✂️  Segmenting {len(files_to_process)} files...")
    print("-" * 60)
    
    total_segments = 0
    temp_wav_dir = os.path.join(LOCAL_SEGMENTED_DIR, ".temp_wav")
    os.makedirs(temp_wav_dir, exist_ok=True)
    
    for mp3_path in tqdm(files_to_process, desc="Processing files"):
        base_name = os.path.splitext(os.path.basename(mp3_path))[0]
        temp_wav_path = os.path.join(temp_wav_dir, f"{base_name}.wav")
        
        # Convert MP3 to WAV first
        if not convert_mp3_to_wav(mp3_path, temp_wav_path):
            print(f"⚠️  Skipping {os.path.basename(mp3_path)} (conversion failed)")
            continue
        
        # Segment the WAV file
        segments = segment_audio_file(temp_wav_path, LOCAL_SEGMENTED_DIR)
        total_segments += segments
        
        # Clean up temp WAV file
        if os.path.exists(temp_wav_path):
            os.remove(temp_wav_path)
    
    # Clean up temp directory (remove all contents)
    if os.path.exists(temp_wav_dir):
        shutil.rmtree(temp_wav_dir)
    
    print("\n" + "=" * 60)
    print("✅ Segmentation Complete!")
    print("=" * 60)
    print(f"Total segments created: {total_segments}")
    print(f"Segments saved to: {LOCAL_SEGMENTED_DIR}")
    print(f"\nNext step: Upload segments to Modal using upload_segments_to_modal.py")


if __name__ == "__main__":
    main()
