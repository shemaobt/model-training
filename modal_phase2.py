"""
Modal deployment for Phase 2 notebook - BPE Tokenizer Training
Trains SentencePiece BPE tokenizer on acoustic units from Phase 1
to discover acoustic motifs/patterns

Run with: /usr/local/bin/python3 -m modal run modal_phase2.py
"""

import modal
import os

# Create Modal app
app = modal.App("bible-audio-training")

# Create persistent volume (same as Phase 1)
audio_volume = modal.Volume.from_name(
    "bible-audio-data",
    create_if_missing=True
)

# Create image with dependencies
image = (
    modal.Image.debian_slim()
    .pip_install(
        "sentencepiece>=0.1.99",
        "tqdm",
        "numpy<2",
    )
)

# Mount points (using Phase 1 outputs)
AUDIO_MOUNT = "/mnt/audio_data"
OUTPUT_DIR = f"{AUDIO_MOUNT}/portuguese_units"
PHASE2_OUTPUT_DIR = f"{AUDIO_MOUNT}/phase2_output"  # Phase 2 specific outputs


@app.function(
    image=image,
    volumes={AUDIO_MOUNT: audio_volume},
    timeout=3600,  # 1 hour timeout
    # No GPU needed - BPE training is CPU-based
)
def train_bpe_tokenizer(vocab_size: int = 500, min_frequency: int = 5):
    """Phase 2: Train BPE tokenizer on acoustic units from Phase 1
    
    Args:
        vocab_size: Number of BPE tokens/motifs to discover
        min_frequency: Minimum frequency for a pattern to be included
    """
    import sentencepiece as spm
    from tqdm import tqdm
    
    os.makedirs(PHASE2_OUTPUT_DIR, exist_ok=True)
    
    # Check if Phase 1 output exists
    units_file = os.path.join(OUTPUT_DIR, "all_units_for_bpe.txt")
    
    if not os.path.exists(units_file):
        print(f"❌ Error: Phase 1 output not found: {units_file}")
        print("   Please run Phase 1 first to generate acoustic units!")
        return None
    
    print(f"✓ Found Phase 1 output: {units_file}")
    
    # Read unit sequences from Phase 1
    print("\n📖 Reading acoustic units from Phase 1...")
    unit_sequences = []
    
    with open(units_file, 'r') as f:
        for line in tqdm(f, desc="Reading unit sequences"):
            line = line.strip()
            if line:
                unit_sequences.append(line)
    
    print(f"✓ Loaded {len(unit_sequences)} unit sequences")
    
    # Prepare training data for SentencePiece
    # Each line should be a "sentence" of units
    train_path = os.path.join(PHASE2_OUTPUT_DIR, "bpe_train.txt")
    
    print(f"\n📝 Preparing BPE training data...")
    with open(train_path, 'w') as f:
        for seq in tqdm(unit_sequences, desc="Writing training data"):
            f.write(seq + "\n")
    
    print(f"✓ Training data written to: {train_path}")
    
    # Train BPE model
    model_prefix = os.path.join(PHASE2_OUTPUT_DIR, "portuguese_bpe")
    
    print(f"\n🔤 Training BPE tokenizer...")
    print(f"   Requested vocabulary size: {vocab_size}")
    print(f"   Min frequency: {min_frequency}")
    print(f"   Training on {len(unit_sequences)} sequences")
    
    # Try training with requested vocab_size, but auto-adjust if too high
    current_vocab_size = vocab_size
    max_attempts = 3
    
    for attempt in range(max_attempts):
        try:
            print(f"\n   Attempt {attempt + 1}: Training with vocab_size={current_vocab_size}...")
            
            spm.SentencePieceTrainer.train(
                input=train_path,
                model_prefix=model_prefix,
                vocab_size=current_vocab_size,
                model_type='bpe',
                character_coverage=1.0,
                max_sentence_length=10000,
                num_threads=4,
                input_sentence_size=len(unit_sequences),  # Use all sequences
                shuffle_input_sentence=True,
                # BPE-specific parameters
                byte_fallback=True,
                split_by_unicode_script=True,
                split_by_whitespace=True,
                treat_whitespace_as_suffix=False,
                allow_whitespace_only_pieces=False,
                normalization_rule_name='nmt_nfkc_cf',
            )
            
            # Load and verify the model
            sp = spm.SentencePieceProcessor()
            sp.load(f"{model_prefix}.model")
            
            vocab_size_actual = sp.get_piece_size()
            
            print(f"\n✓ BPE training complete!")
            print(f"   Model saved to: {model_prefix}.model")
            print(f"   Vocabulary size: {vocab_size_actual}")
            
            # Commit volume changes
            audio_volume.commit()
            
            return {
                "model_path": f"{model_prefix}.model",
                "vocab_size": vocab_size_actual,
                "requested_vocab_size": vocab_size,
                "training_sequences": len(unit_sequences)
            }
            
        except RuntimeError as e:
            error_msg = str(e)
            # Check if it's a vocab size error
            if "Vocabulary size too high" in error_msg or "vocab_size" in error_msg.lower():
                # Try to extract the max allowed size from error message
                import re
                match = re.search(r'<= (\d+)', error_msg)
                if match:
                    max_allowed = int(match.group(1))
                    print(f"⚠️  Vocabulary size {current_vocab_size} too high. Max allowed: {max_allowed}")
                    current_vocab_size = max_allowed - 10  # Use slightly less to be safe
                    if attempt < max_attempts - 1:
                        print(f"   Retrying with vocab_size={current_vocab_size}...")
                        continue
                else:
                    # Fallback: reduce by 20% each attempt
                    current_vocab_size = int(current_vocab_size * 0.8)
                    if attempt < max_attempts - 1:
                        print(f"   Retrying with reduced vocab_size={current_vocab_size}...")
                        continue
            # If it's not a vocab size error or we've exhausted attempts, raise
            print(f"❌ Error training BPE: {e}")
            import traceback
            traceback.print_exc()
            return None
        except Exception as e:
            print(f"❌ Unexpected error training BPE: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    print(f"❌ Failed to train BPE after {max_attempts} attempts")
    return None


@app.function(
    image=image,
    volumes={AUDIO_MOUNT: audio_volume},
    timeout=1800,  # 30 minutes
)
def analyze_motifs():
    """Analyze discovered motifs from BPE tokenizer"""
    import sentencepiece as spm
    from collections import Counter
    import json
    
    # Load BPE model
    model_path = os.path.join(PHASE2_OUTPUT_DIR, "portuguese_bpe.model")
    
    if not os.path.exists(model_path):
        print(f"❌ BPE model not found: {model_path}")
        print("   Please run train_bpe_tokenizer() first!")
        return None
    
    print("📊 Loading BPE model...")
    sp = spm.SentencePieceProcessor()
    sp.load(model_path)
    
    print(f"✓ Model loaded (vocab size: {sp.get_piece_size()})")
    
    # Load unit sequences
    units_file = os.path.join(OUTPUT_DIR, "all_units_for_bpe.txt")
    
    if not os.path.exists(units_file):
        print(f"❌ Units file not found: {units_file}")
        return None
    
    print("\n🔍 Analyzing motifs...")
    
    # Read and encode all sequences
    all_encoded = []
    total_tokens = 0
    
    with open(units_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                encoded = sp.encode_as_pieces(line)
                all_encoded.extend(encoded)
                total_tokens += len(encoded)
    
    # Count motif frequencies
    motif_counts = Counter(all_encoded)
    
    # Analyze motifs
    print(f"\n📊 Motif Analysis Results:")
    print(f"   Total tokens: {total_tokens:,}")
    print(f"   Unique motifs: {len(motif_counts)}")
    
    # Get top motifs
    top_motifs = motif_counts.most_common(50)
    
    # Save analysis
    analysis_path = os.path.join(PHASE2_OUTPUT_DIR, "motif_analysis.json")
    with open(analysis_path, 'w') as f:
        json.dump({
            "total_tokens": total_tokens,
            "unique_motifs": len(motif_counts),
            "top_motifs": [{"motif": m, "count": c} for m, c in top_motifs],
            "all_motifs": [{"motif": m, "count": c} for m, c in motif_counts.most_common()]
        }, f, indent=2)
    
    print(f"\n✓ Analysis saved to: {analysis_path}")
    
    # Print top motifs
    print(f"\n🏆 Top 30 Motifs by Frequency:")
    print("-" * 60)
    for i, (motif, count) in enumerate(top_motifs[:30], 1):
        # Count units in motif
        units_in_motif = motif.replace("▁", "").strip().split()
        n_units = len([u for u in units_in_motif if u])
        percentage = (count / total_tokens) * 100
        print(f"  {i:2d}. {motif:30s} count={count:6,} ({percentage:5.2f}%)  units={n_units}")
    
    # Commit volume changes
    audio_volume.commit()
    
    return {
        "total_tokens": total_tokens,
        "unique_motifs": len(motif_counts),
        "top_motifs_count": len(top_motifs)
    }


@app.local_entrypoint()
def main(vocab_size: int = 500, min_frequency: int = 5):
    """Run Phase 2: BPE Tokenizer Training"""
    print("🚀 Starting Phase 2: BPE Tokenizer Training")
    print("=" * 60)
    
    print("\n🔤 Step 1: Training BPE tokenizer...")
    result = train_bpe_tokenizer.remote(vocab_size, min_frequency)
    
    if result:
        print(f"✓ BPE model trained successfully!")
        print(f"   Model: {result['model_path']}")
        print(f"   Vocabulary: {result['vocab_size']} tokens")
        
        print("\n📊 Step 2: Analyzing discovered motifs...")
        analysis = analyze_motifs.remote()
        
        if analysis:
            print(f"✓ Analysis complete!")
            print(f"   Total tokens analyzed: {analysis['total_tokens']:,}")
            print(f"   Unique motifs: {analysis['unique_motifs']}")
        
        print("\n✅ Phase 2 complete!")
        print(f"Results stored in: {PHASE2_OUTPUT_DIR}")
    else:
        print("\n❌ Phase 2 failed. Check logs above for details.")
