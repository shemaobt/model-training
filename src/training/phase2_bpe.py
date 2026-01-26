# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.20.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Phase 2: BPE Motif Discovery
#
# This script trains a BPE (Byte Pair Encoding) tokenizer on acoustic units
# to discover recurring acoustic patterns (motifs).
#
# ## What BPE Discovers
#
# BPE iteratively merges the most frequent pairs of tokens:
#
# ```
# Initial:   31 31 87 43 43 17
#            ↓ merge (31, 31) → 100
# Step 1:    100 87 43 43 17
#            ↓ merge (43, 43) → 101
# Step 2:    100 87 101 17
# ```
#
# Discovered patterns may correspond to:
# - Sustained sounds (e.g., vowels)
# - Consonant clusters
# - Syllable fragments
#
# ## Run on Modal
#
# ```bash
# python3 -m modal run --detach src/training/phase2_bpe.py::main
# ```

# %% [markdown]
# ## Modal Configuration

# %%
import modal
import os

# %%
app = modal.App("bible-audio-training")

audio_volume = modal.Volume.from_name(
    "bible-audio-data",
    create_if_missing=True
)

# %%
image = (
    modal.Image.debian_slim()
    .pip_install(
        "sentencepiece>=0.1.99",
        "tqdm",
        "numpy<2",
    )
)

# %%
AUDIO_MOUNT = "/mnt/audio_data"

# Language configuration
LANGUAGE_CONFIGS = {
    "portuguese": {
        "output_dir": f"{AUDIO_MOUNT}/portuguese_units",
        "phase2_output_dir": f"{AUDIO_MOUNT}/phase2_output",
        "bpe_prefix": "portuguese_bpe",
    },
    "satere": {
        "output_dir": f"{AUDIO_MOUNT}/satere_units",
        "phase2_output_dir": f"{AUDIO_MOUNT}/phase2_satere_output",
        "bpe_prefix": "satere_bpe",
    },
}

# Default paths (can be overridden via CLI)
import os as _os
_LANGUAGE = _os.environ.get("TRAINING_LANGUAGE", "portuguese")
_config = LANGUAGE_CONFIGS.get(_LANGUAGE, LANGUAGE_CONFIGS["portuguese"])

OUTPUT_DIR = _config["output_dir"]
PHASE2_OUTPUT_DIR = _config["phase2_output_dir"]

# %% [markdown]
# ## BPE Tokenizer Training
#
# Train SentencePiece BPE model on acoustic unit sequences.
# Automatically adjusts vocabulary size if requested size is too large.

# %%
@app.function(
    image=image,
    volumes={AUDIO_MOUNT: audio_volume},
    timeout=3600,
)
def train_bpe_tokenizer(vocab_size: int = 500, min_frequency: int = 5):
    """Train BPE tokenizer on acoustic units.
    
    Args:
        vocab_size: Number of BPE tokens/motifs to discover
        min_frequency: Minimum frequency for a pattern
    """
    import sentencepiece as spm
    from tqdm import tqdm
    
    os.makedirs(PHASE2_OUTPUT_DIR, exist_ok=True)
    
    units_file = os.path.join(OUTPUT_DIR, "all_units_for_bpe.txt")
    
    if not os.path.exists(units_file):
        print(f"❌ Phase 1 output not found: {units_file}")
        return None
    
    print(f"✓ Found Phase 1 output: {units_file}")
    
    # Read unit sequences
    print("\n📖 Reading acoustic units...")
    unit_sequences = []
    
    with open(units_file, 'r') as f:
        for line in tqdm(f, desc="Reading"):
            line = line.strip()
            if line:
                unit_sequences.append(line)
    
    print(f"✓ Loaded {len(unit_sequences)} sequences")
    
    # Prepare training data
    train_path = os.path.join(PHASE2_OUTPUT_DIR, "bpe_train.txt")
    
    with open(train_path, 'w') as f:
        for seq in tqdm(unit_sequences, desc="Writing"):
            f.write(seq + "\n")
    
    # Train BPE with auto-adjustment
    model_prefix = os.path.join(PHASE2_OUTPUT_DIR, "portuguese_bpe")
    current_vocab_size = vocab_size
    max_attempts = 3
    
    for attempt in range(max_attempts):
        try:
            print(f"\n🔤 Training BPE (vocab_size={current_vocab_size})...")
            
            spm.SentencePieceTrainer.train(
                input=train_path,
                model_prefix=model_prefix,
                vocab_size=current_vocab_size,
                model_type='bpe',
                character_coverage=1.0,
                max_sentence_length=10000,
                num_threads=4,
                input_sentence_size=len(unit_sequences),
                shuffle_input_sentence=True,
                byte_fallback=True,
                split_by_unicode_script=True,
                split_by_whitespace=True,
                treat_whitespace_as_suffix=False,
                allow_whitespace_only_pieces=False,
                normalization_rule_name='nmt_nfkc_cf',
            )
            
            # Verify model
            sp = spm.SentencePieceProcessor()
            sp.load(f"{model_prefix}.model")
            
            vocab_size_actual = sp.get_piece_size()
            
            print(f"\n✓ BPE training complete!")
            print(f"   Model: {model_prefix}.model")
            print(f"   Vocabulary: {vocab_size_actual}")
            
            audio_volume.commit()
            
            return {
                "model_path": f"{model_prefix}.model",
                "vocab_size": vocab_size_actual,
                "requested_vocab_size": vocab_size,
                "training_sequences": len(unit_sequences)
            }
            
        except RuntimeError as e:
            error_msg = str(e)
            if "Vocabulary size too high" in error_msg or "vocab_size" in error_msg.lower():
                import re
                match = re.search(r'<= (\d+)', error_msg)
                if match:
                    max_allowed = int(match.group(1))
                    print(f"⚠️  Vocab size too high. Max: {max_allowed}")
                    current_vocab_size = max_allowed - 10
                else:
                    current_vocab_size = int(current_vocab_size * 0.8)
                
                if attempt < max_attempts - 1:
                    print(f"   Retrying with {current_vocab_size}...")
                    continue
            
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    return None

# %% [markdown]
# ## Motif Analysis
#
# Analyze the discovered motifs: frequencies, patterns, and statistics.

# %%
@app.function(
    image=image,
    volumes={AUDIO_MOUNT: audio_volume},
    timeout=1800,
)
def analyze_motifs():
    """Analyze discovered motifs from BPE tokenizer."""
    import sentencepiece as spm
    from collections import Counter
    import json
    
    model_path = os.path.join(PHASE2_OUTPUT_DIR, "portuguese_bpe.model")
    
    if not os.path.exists(model_path):
        print(f"❌ BPE model not found. Run training first.")
        return None
    
    print("📊 Loading BPE model...")
    sp = spm.SentencePieceProcessor()
    sp.load(model_path)
    print(f"✓ Model loaded (vocab: {sp.get_piece_size()})")
    
    # Load and encode sequences
    units_file = os.path.join(OUTPUT_DIR, "all_units_for_bpe.txt")
    
    print("\n🔍 Analyzing motifs...")
    all_encoded = []
    total_tokens = 0
    
    with open(units_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                encoded = sp.encode_as_pieces(line)
                all_encoded.extend(encoded)
                total_tokens += len(encoded)
    
    # Count frequencies
    motif_counts = Counter(all_encoded)
    
    print(f"\n📊 Results:")
    print(f"   Total tokens: {total_tokens:,}")
    print(f"   Unique motifs: {len(motif_counts)}")
    
    # Save analysis
    top_motifs = motif_counts.most_common(50)
    
    analysis_path = os.path.join(PHASE2_OUTPUT_DIR, "motif_analysis.json")
    with open(analysis_path, 'w') as f:
        json.dump({
            "total_tokens": total_tokens,
            "unique_motifs": len(motif_counts),
            "top_motifs": [{"motif": m, "count": c} for m, c in top_motifs],
            "all_motifs": [{"motif": m, "count": c} for m, c in motif_counts.most_common()]
        }, f, indent=2)
    
    print(f"\n✓ Saved: {analysis_path}")
    
    # Print top motifs
    print(f"\n🏆 Top 30 Motifs:")
    print("-" * 60)
    for i, (motif, count) in enumerate(top_motifs[:30], 1):
        units_in_motif = motif.replace("▁", "").strip().split()
        n_units = len([u for u in units_in_motif if u])
        percentage = (count / total_tokens) * 100
        print(f"  {i:2d}. {motif:30s} {count:6,} ({percentage:5.2f}%)  units={n_units}")
    
    audio_volume.commit()
    
    return {
        "total_tokens": total_tokens,
        "unique_motifs": len(motif_counts),
        "top_motifs_count": len(top_motifs)
    }

# %% [markdown]
# ## Entry Point

# %%
@app.local_entrypoint()
def main(language: str = "portuguese", vocab_size: int = 500, min_frequency: int = 5):
    """Run Phase 2: BPE Tokenizer Training.
    
    Args:
        language: Language to process ('portuguese' or 'satere')
        vocab_size: Target vocabulary size
        min_frequency: Minimum token frequency
    """
    # Set environment variable for language selection
    import os
    os.environ["TRAINING_LANGUAGE"] = language
    
    config = LANGUAGE_CONFIGS.get(language, LANGUAGE_CONFIGS["portuguese"])
    
    print("🚀 Starting Phase 2: BPE Tokenizer Training")
    print("=" * 60)
    print(f"  Language: {language}")
    print(f"  Input: {config['output_dir']}")
    print(f"  Output: {config['phase2_output_dir']}")
    print("=" * 60)
    
    print("\n🔤 Step 1: Training BPE tokenizer...")
    result = train_bpe_tokenizer.remote(vocab_size, min_frequency)
    
    if result:
        print(f"✓ BPE model trained!")
        print(f"   Model: {result['model_path']}")
        print(f"   Vocabulary: {result['vocab_size']} tokens")
        
        print("\n📊 Step 2: Analyzing motifs...")
        analysis = analyze_motifs.remote()
        
        if analysis:
            print(f"✓ Analysis complete!")
            print(f"   Total tokens: {analysis['total_tokens']:,}")
            print(f"   Unique motifs: {analysis['unique_motifs']}")
        
        print("\n✅ Phase 2 complete!")
        print(f"Results in: {config['phase2_output_dir']}")
    else:
        print("\n❌ Phase 2 failed.")
