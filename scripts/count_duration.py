import os
import argparse
from pathlib import Path
import subprocess
from tqdm import tqdm


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_audio_duration(file_path):
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


LANGUAGE_CONFIGS = {
    "portuguese": {
        "input_dirs": [
            os.path.join(BASE_DIR, "audio_data", "raw_audio"),
        ],
    },
    "satere": {
        "input_dirs": [
            os.path.join(BASE_DIR, "audio_data", "raw_audio_satere"),
        ],
    },
}


def collect_audio_files(input_dirs, extensions=('.mp3', '.wav', '.flac')):
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


def main():
    parser = argparse.ArgumentParser(
        description="Count audio files and calculate total duration"
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
    
    args = parser.parse_args()
    
    if args.input:
        input_dirs = args.input
    elif args.language in LANGUAGE_CONFIGS:
        input_dirs = LANGUAGE_CONFIGS[args.language]["input_dirs"]
    else:
        print(f"❌ Unknown language: {args.language}")
        print(f"   Available: {', '.join(LANGUAGE_CONFIGS.keys())}")
        return
    
    print("=" * 60)
    print(f"📊 Audio File Counter & Duration Calculator: {args.language.upper()}")
    print("=" * 60)
    
    print("\n📁 Scanning directories...")
    for d in input_dirs:
        exists = "✓" if os.path.exists(d) else "✗"
        print(f"   {exists} {d}")
    
    audio_files = collect_audio_files(input_dirs)
    total_files = len(audio_files)
    print(f"\n📊 Total audio files found: {total_files}")
    
    if total_files == 0:
        print("❌ No audio files found!")
        return
    
    print(f"\n⏱️  Calculating total duration...")
    print("   (This may take a few minutes for large files)")
    
    total_seconds = 0
    successful = 0
    failed = 0
    
    for audio_path in tqdm(audio_files, desc="Processing files"):
        duration = get_audio_duration(audio_path)
        if duration is not None:
            total_seconds += duration
            successful += 1
        else:
            failed += 1
    
    total_hours = int(total_seconds // 3600)
    remaining_seconds = total_seconds % 3600
    total_minutes = int(remaining_seconds // 60)
    remaining_secs = int(remaining_seconds % 60)
    
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
    
    if successful > 0:
        avg_seconds = total_seconds / successful
        avg_minutes = avg_seconds / 60
        print(f"\n📈 Average per file:")
        print(f"   {avg_minutes:.1f} minutes ({avg_seconds:.0f} seconds)")
    
    print("=" * 60)


if __name__ == "__main__":
    main()
