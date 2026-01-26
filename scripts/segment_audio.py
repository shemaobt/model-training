"""
Local audio segmentation script - Generic for any language

Segments audio files by silence/pauses before uploading to Modal.
Supports multiple languages with separate output directories.

Usage:
    # Segment Portuguese Bible audio
    python segment_audio.py --language portuguese
    
    # Segment Sateré audio
    python segment_audio.py --language satere
    
    # Custom input/output directories
    python segment_audio.py --input /path/to/audio --output /path/to/segments --language custom
"""

import os
import argparse
import librosa
import numpy as np
import soundfile as sf
import subprocess
import shutil
from tqdm import tqdm


# ============================================================================
# Configuration
# ============================================================================

# Base directory for audio data
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Predefined language configurations
LANGUAGE_CONFIGS = {
    "portuguese": {
        "input_dirs": [
            "/Users/joao/Desktop/work/shema/shemaobt/scripts/Antigo_Testamento_COMPLETO",
            "/Users/joao/Desktop/work/shema/shemaobt/scripts/NOVO_TESTAMENTO_COMPLETO",
        ],
        "output_dir": os.path.join(BASE_DIR, "local_segments"),
    },
    "satere": {
        "input_dirs": [
            os.path.join(BASE_DIR, "audio_data", "raw_audio_satere"),
        ],
        "output_dir": os.path.join(BASE_DIR, "local_segments_satere"),
    },
}

# Segmentation parameters
SILENCE_THRESHOLD = -40  # dB threshold for silence
MIN_SILENCE_DURATION = 0.5  # Minimum silence duration in seconds
MIN_SEGMENT_DURATION = 2.0  # Minimum segment duration in seconds
MAX_SEGMENT_DURATION = 120.0  # Maximum segment duration in seconds
SAMPLE_RATE = 16000


# ============================================================================
# Audio Processing Functions
# ============================================================================

def convert_mp3_to_wav(mp3_path: str, wav_path: str) -> bool:
    """Convert MP3 to 16kHz mono WAV."""
    cmd = [
        "ffmpeg", "-y", "-i", mp3_path,
        "-ar", str(SAMPLE_RATE),
        "-ac", "1",  # mono
        wav_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Warning: ffmpeg conversion failed for {mp3_path}: {result.stderr[:100]}")
        return False
    return True


def segment_audio_file(wav_path: str, output_dir: str) -> int:
    """Segment a single audio file by silence detection."""
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
            
            # If segment is too long, split it
            if segment_duration > MAX_SEGMENT_DURATION:
                num_splits = int(np.ceil(segment_duration / MAX_SEGMENT_DURATION))
                split_duration = segment_duration / num_splits
                
                for split_idx in range(num_splits):
                    split_start = start_time + (split_idx * split_duration)
                    split_end = min(start_time + ((split_idx + 1) * split_duration), end_time)
                    
                    segment_filename = f"{base_name}_seg_{segment_num:04d}.wav"
                    segment_path = os.path.join(output_dir, segment_filename)
                    
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


def collect_audio_files(input_dirs: list, extensions: tuple = ('.mp3', '.wav', '.flac')) -> list:
    """Collect all audio files from input directories."""
    audio_files = []
    
    for directory in input_dirs:
        if not os.path.exists(directory):
            print(f"⚠️  Directory not found: {directory}")
            continue
        
        for root, dirs, files in os.walk(directory):
            for f in files:
                if f.lower().endswith(extensions):
                    audio_files.append(os.path.join(root, f))
    
    return sorted(audio_files)


def get_already_segmented(output_dir: str) -> set:
    """Get set of base filenames already segmented."""
    existing = set()
    if os.path.exists(output_dir):
        for f in os.listdir(output_dir):
            if f.endswith('.wav'):
                # Extract base filename (without segment number)
                base = f.rsplit('_seg_', 1)[0] if '_seg_' in f else f.replace('.wav', '')
                existing.add(base)
    return existing


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Segment audio files by silence for acoustic tokenization"
    )
    parser.add_argument(
        "--language", "-l",
        type=str,
        default="portuguese",
        help=f"Language preset: {', '.join(LANGUAGE_CONFIGS.keys())} (default: portuguese)"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        nargs="+",
        help="Custom input directory/directories (overrides language preset)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Custom output directory (overrides language preset)"
    )
    
    args = parser.parse_args()
    
    # Determine input/output directories
    if args.input:
        input_dirs = args.input
        output_dir = args.output or os.path.join(BASE_DIR, f"local_segments_{args.language}")
    elif args.language in LANGUAGE_CONFIGS:
        config = LANGUAGE_CONFIGS[args.language]
        input_dirs = config["input_dirs"]
        output_dir = args.output or config["output_dir"]
    else:
        print(f"❌ Unknown language: {args.language}")
        print(f"   Available: {', '.join(LANGUAGE_CONFIGS.keys())}")
        print("   Or use --input to specify custom directories")
        return
    
    print("=" * 60)
    print(f"🔊 Audio Segmentation: {args.language.upper()}")
    print("=" * 60)
    print(f"\n📂 Input directories:")
    for d in input_dirs:
        exists = "✓" if os.path.exists(d) else "✗"
        print(f"   {exists} {d}")
    print(f"\n📁 Output directory: {output_dir}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Collect audio files
    print("\n🔍 Scanning for audio files...")
    audio_files = collect_audio_files(input_dirs)
    print(f"✓ Found {len(audio_files)} audio files")
    
    if len(audio_files) == 0:
        print("❌ No audio files found!")
        return
    
    # Check what's already segmented
    existing_segments = get_already_segmented(output_dir)
    
    # Filter out already processed files
    files_to_process = []
    for audio_path in audio_files:
        base_name = os.path.splitext(os.path.basename(audio_path))[0]
        if base_name not in existing_segments:
            files_to_process.append(audio_path)
    
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
    temp_wav_dir = os.path.join(output_dir, ".temp_wav")
    os.makedirs(temp_wav_dir, exist_ok=True)
    
    for audio_path in tqdm(files_to_process, desc="Processing"):
        base_name = os.path.splitext(os.path.basename(audio_path))[0]
        ext = os.path.splitext(audio_path)[1].lower()
        
        # Convert to WAV if needed
        if ext == '.wav':
            wav_path = audio_path
            needs_cleanup = False
        else:
            wav_path = os.path.join(temp_wav_dir, f"{base_name}.wav")
            if not convert_mp3_to_wav(audio_path, wav_path):
                print(f"⚠️  Skipping {os.path.basename(audio_path)} (conversion failed)")
                continue
            needs_cleanup = True
        
        # Segment the file
        segments = segment_audio_file(wav_path, output_dir)
        total_segments += segments
        
        # Clean up temp WAV file
        if needs_cleanup and os.path.exists(wav_path):
            os.remove(wav_path)
    
    # Clean up temp directory
    if os.path.exists(temp_wav_dir):
        shutil.rmtree(temp_wav_dir)
    
    print("\n" + "=" * 60)
    print("✅ Segmentation Complete!")
    print("=" * 60)
    print(f"Language: {args.language}")
    print(f"Total segments created: {total_segments}")
    print(f"Segments saved to: {output_dir}")
    print(f"\n📤 Next step: Upload segments to Modal:")
    print(f"   python scripts/upload_to_modal.py --language {args.language}")


if __name__ == "__main__":
    main()
