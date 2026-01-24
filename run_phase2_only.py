"""
Run only Phase 2 (Acoustic Tokenization) on Modal
Assumes segments are already uploaded to Modal volume

Run with: /usr/local/bin/python3 -m modal run run_phase2_only.py
"""

import modal

# Import the app and function from modal_phase1
# We'll use Modal's app lookup to get the function
app = modal.App("bible-audio-training")

# We need to redeclare the function or import it
# Actually, simpler: just import and call the function directly
import sys
sys.path.insert(0, '/Users/joao/Desktop/work/shema/shemaobt/model-training')

# Import the acoustic_tokenization function from modal_phase1
from modal_phase1 import acoustic_tokenization

@app.local_entrypoint()
def main():
    """Run only Phase 2: Acoustic Tokenization"""
    print("🎵 Running Phase 2: Acoustic Tokenization")
    print("=" * 60)
    print("(Skipping Phase 1 and 1.5 - segments already uploaded)")
    print()
    
    print("🎵 Phase 2: Acoustic Tokenization...")
    processed = acoustic_tokenization.remote()
    print(f"✓ Processed {processed} segments")
    
    print("\n✅ Phase 2 complete!")
    print("Results stored in Modal volume: /mnt/audio_data/portuguese_units")
