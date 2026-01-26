#!/usr/bin/env python3
"""
Helper script to download files from Modal volume using Modal CLI
Run the commands shown below, or use: python3 download_modal_results.py
"""

import subprocess
import os

# Local download directory
DOWNLOAD_DIR = "./modal_downloads"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

# Volume name
VOLUME_NAME = "bible-audio-data"

print("📥 Downloading files from Modal volume...")
print("=" * 60)

# Commands to download specific directories/files
commands = [
    # Phase 1 outputs
    f'modal volume download {VOLUME_NAME} portuguese_units/portuguese_kmeans.pkl {DOWNLOAD_DIR}/phase1_outputs/portuguese_kmeans.pkl',
    f'modal volume download {VOLUME_NAME} portuguese_units/portuguese_corpus_timestamped.json {DOWNLOAD_DIR}/phase1_outputs/portuguese_corpus_timestamped.json',
    f'modal volume download {VOLUME_NAME} portuguese_units/all_units_for_bpe.txt {DOWNLOAD_DIR}/phase1_outputs/all_units_for_bpe.txt',
    
    # Phase 2 outputs
    f'modal volume download {VOLUME_NAME} phase2_output/portuguese_bpe.model {DOWNLOAD_DIR}/phase2_outputs/portuguese_bpe.model',
    f'modal volume download {VOLUME_NAME} phase2_output/portuguese_bpe.vocab {DOWNLOAD_DIR}/phase2_outputs/portuguese_bpe.vocab',
    f'modal volume download {VOLUME_NAME} phase2_output/motif_analysis.json {DOWNLOAD_DIR}/phase2_outputs/motif_analysis.json',
    
    # Checking outputs
    f'modal volume download {VOLUME_NAME} checking_output/validation_results.json {DOWNLOAD_DIR}/checking_outputs/validation_results.json',
]

# Create directories
os.makedirs(f"{DOWNLOAD_DIR}/phase1_outputs", exist_ok=True)
os.makedirs(f"{DOWNLOAD_DIR}/phase2_outputs", exist_ok=True)
os.makedirs(f"{DOWNLOAD_DIR}/checking_outputs", exist_ok=True)
os.makedirs(f"{DOWNLOAD_DIR}/sample_units", exist_ok=True)
os.makedirs(f"{DOWNLOAD_DIR}/sample_audio", exist_ok=True)

print("\n📋 Run these Modal CLI commands:\n")
print("=" * 60)

for i, cmd in enumerate(commands, 1):
    print(f"{i}. {cmd}")

print("\n" + "=" * 60)
print("\n💡 Or run all at once:")
print("\n".join(commands))

print("\n" + "=" * 60)
print("\n📝 To download sample files, use:")
print(f"modal volume download {VOLUME_NAME} portuguese_units/ <first_5_files> {DOWNLOAD_DIR}/sample_units/")
print(f"modal volume download {VOLUME_NAME} segmented_audio/ <first_5_files> {DOWNLOAD_DIR}/sample_audio/")

print("\n" + "=" * 60)
print("\n🚀 Executing downloads...\n")

# Execute commands
for cmd in commands:
    parts = cmd.split()
    try:
        # Use python3 -m modal instead of direct modal command
        modal_cmd = ["/usr/local/bin/python3", "-m", "modal"] + parts[1:]
        result = subprocess.run(modal_cmd, capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            print(f"✓ {parts[-1]}")
        else:
            print(f"⚠️  {parts[-1]}: {result.stderr[:100]}")
    except Exception as e:
        print(f"❌ Error running {cmd}: {e}")

print("\n✅ Download complete!")
print(f"\n📁 Files saved to: {DOWNLOAD_DIR}/")
print("\n📊 Key files to analyze:")
print("   - phase2_outputs/motif_analysis.json (discovered acoustic patterns)")
print("   - phase2_outputs/portuguese_bpe.vocab (BPE vocabulary)")
print("   - phase1_outputs/portuguese_corpus_timestamped.json (unit sequences with timestamps)")
print("   - checking_outputs/validation_results.json (validation stats)")
