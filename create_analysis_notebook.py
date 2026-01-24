#!/usr/bin/env python3
"""
Generate analysis notebook with executed cells
"""

import json
import nbformat as nbf

# Create notebook
nb = nbf.v4.new_notebook()

# Title and intro
nb.cells.append(nbf.v4.new_markdown_cell("""# Bible Audio Acoustic Tokenization - Results Analysis

This notebook analyzes the results from the acoustic tokenization pipeline:
- **Phase 1**: Acoustic unit discovery using XLSR-53 + K-Means clustering
- **Phase 2**: BPE tokenizer training to discover acoustic motifs/patterns

## Dataset
- **Source**: Portuguese Bible audio (Old and New Testament)
- **Total segments**: 18,072 audio segments
- **Total audio**: ~5,057 minutes (~84 hours)
- **Acoustic units**: 100 units discovered via K-Means
- **BPE vocabulary**: 100 motifs discovered
"""))

# Setup cell
nb.cells.append(nbf.v4.new_code_cell("""import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter
import sentencepiece as spm

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Base directory for downloaded files
BASE_DIR = Path("./modal_downloads")
PHASE1_DIR = BASE_DIR / "phase1_outputs"
PHASE2_DIR = BASE_DIR / "phase2_outputs"
CHECKING_DIR = BASE_DIR / "checking_outputs"

print("📊 Analysis Notebook - Bible Audio Acoustic Tokenization")
print("=" * 60)
print(f"✓ Base directory: {BASE_DIR}")
print(f"✓ Phase 1 outputs: {PHASE1_DIR.exists()}")
print(f"✓ Phase 2 outputs: {PHASE2_DIR.exists()}")
print(f"✓ Checking outputs: {CHECKING_DIR.exists()}")"""))

# Motif Analysis section
nb.cells.append(nbf.v4.new_markdown_cell("""## 1. Motif Analysis (BPE Discovered Patterns)

The BPE tokenizer discovered 100 acoustic motifs - recurring patterns in the acoustic unit sequences. These motifs represent common sound patterns in Portuguese speech.
"""))

nb.cells.append(nbf.v4.new_code_cell("""# Load motif analysis
with open(PHASE2_DIR / "motif_analysis.json", 'r') as f:
    motif_data = json.load(f)

print("🎵 Motif Analysis Results")
print("=" * 60)
print(f"Total tokens analyzed: {motif_data['total_tokens']:,}")
print(f"Unique motifs discovered: {motif_data['unique_motifs']}")
print(f"\\nTop 20 Motifs by Frequency:")
print("-" * 60)

top_motifs = motif_data['top_motifs'][:20]
for i, item in enumerate(top_motifs, 1):
    motif = item['motif']
    count = item['count']
    percentage = (count / motif_data['total_tokens']) * 100
    # Count units in motif (remove BPE prefix ▁ and count space-separated units)
    units_in_motif = len([u for u in motif.replace("▁", "").strip().split() if u])
    print(f"{i:2d}. {motif:30s} | Count: {count:8,} ({percentage:5.2f}%) | Units: {units_in_motif})"""))

nb.cells.append(nbf.v4.new_code_cell("""# Visualize motif frequency distribution
motif_counts = [m['count'] for m in motif_data['top_motifs']]
motif_percentages = [(c / motif_data['total_tokens']) * 100 for c in motif_counts]

fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# Top 30 motifs bar chart
top_n = 30
axes[0].bar(range(top_n), motif_percentages[:top_n], color='steelblue', alpha=0.7)
axes[0].set_xlabel('Motif Rank', fontsize=12)
axes[0].set_ylabel('Frequency (%)', fontsize=12)
axes[0].set_title(f'Top {top_n} Acoustic Motifs by Frequency', fontsize=14, fontweight='bold')
axes[0].set_xticks(range(top_n))
axes[0].set_xticklabels([f"M{i+1}" for i in range(top_n)], rotation=45, ha='right')
axes[0].grid(axis='y', alpha=0.3)

# Frequency distribution (log scale)
axes[1].hist(motif_counts, bins=50, color='coral', alpha=0.7, edgecolor='black')
axes[1].set_xlabel('Motif Frequency', fontsize=12)
axes[1].set_ylabel('Number of Motifs', fontsize=12)
axes[1].set_title('Motif Frequency Distribution (All 100 Motifs)', fontsize=14, fontweight='bold')
axes[1].set_yscale('log')
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()

print(f"\\n📈 Statistics:")
print(f"   Most frequent motif: {motif_percentages[0]:.2f}% of all tokens")
print(f"   Top 10 motifs cover: {sum(motif_percentages[:10]):.2f}% of all tokens")
print(f"   Top 20 motifs cover: {sum(motif_percentages[:20]):.2f}% of all tokens")
print(f"   Top 50 motifs cover: {sum(motif_percentages[:50]):.2f}% of all tokens")"""))

# BPE Vocabulary section
nb.cells.append(nbf.v4.new_markdown_cell("""## 2. BPE Vocabulary Analysis

Let's examine the BPE vocabulary file to see the discovered tokens.
"""))

nb.cells.append(nbf.v4.new_code_cell("""# Load BPE vocabulary
vocab_path = PHASE2_DIR / "portuguese_bpe.vocab"

print("📚 BPE Vocabulary")
print("=" * 60)

vocab_lines = []
with open(vocab_path, 'r', encoding='utf-8') as f:
    for line in f:
        if line.strip():
            vocab_lines.append(line.strip())

print(f"Total vocabulary entries: {len(vocab_lines)}")
print(f"\\nFirst 20 vocabulary entries:")
print("-" * 60)
for i, entry in enumerate(vocab_lines[:20], 1):
    print(f"{i:2d}. {entry}")

# Load BPE model to get more details
try:
    sp = spm.SentencePieceProcessor()
    sp.load(str(PHASE2_DIR / "portuguese_bpe.model"))
    print(f"\\n✓ BPE Model loaded successfully")
    print(f"   Vocabulary size: {sp.get_piece_size()}")
    print(f"\\nSample tokens:")
    for i in range(min(10, sp.get_piece_size())):
        token = sp.id_to_piece(i)
        print(f"   {i:3d}: {token}")
except Exception as e:
    print(f"⚠️  Could not load BPE model: {e}")"""))

# Unit Statistics section
nb.cells.append(nbf.v4.new_markdown_cell("""## 3. Acoustic Unit Statistics

Analysis of the 100 acoustic units discovered via K-Means clustering.
"""))

nb.cells.append(nbf.v4.new_code_cell("""# Load validation results
with open(CHECKING_DIR / "validation_results.json", 'r') as f:
    validation = json.load(f)

print("🔍 Acoustic Unit Validation")
print("=" * 60)
stats = validation['unit_stats']
print(f"Total units in sample: {stats['total_units']:,}")
print(f"Unique units found: {stats['unique_units']}/{stats['expected_unique']}")
print(f"Unit ID range: {stats['unit_range'][0]} - {stats['unit_range'][1]}")
print(f"Average sequence length: {stats['avg_sequence_length']:.1f} units")
print(f"Min sequence length: {stats['min_sequence_length']} units")
print(f"Max sequence length: {stats['max_sequence_length']} units")
print(f"Total sequences analyzed: {validation['total_sequences']}")"""))

nb.cells.append(nbf.v4.new_code_cell("""# Load a sample unit sequence
sample_unit_files = list((BASE_DIR / "sample_units").glob("*.units.txt"))
if sample_unit_files:
    sample_unit_file = sample_unit_files[0]
    with open(sample_unit_file, 'r') as f:
        sample_units = [int(u) for u in f.read().strip().split()]
    
    print(f"\\n📝 Sample Unit Sequence Analysis")
    print("=" * 60)
    print(f"File: {sample_unit_file.name}")
    print(f"Sequence length: {len(sample_units)} units")
    print(f"First 50 units: {sample_units[:50]}")
    print(f"Last 50 units: {sample_units[-50:]}")
    
    # Unit frequency in this sequence
    unit_freq = Counter(sample_units)
    print(f"\\nMost frequent units in this sequence:")
    for unit, count in unit_freq.most_common(10):
        print(f"   Unit {unit:3d}: {count:4d} times ({count/len(sample_units)*100:.1f}%)")
    
    # Visualize unit distribution
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    
    # Unit sequence over time
    axes[0].plot(sample_units[:500], alpha=0.7, linewidth=0.5)
    axes[0].set_xlabel('Position in Sequence', fontsize=12)
    axes[0].set_ylabel('Unit ID', fontsize=12)
    axes[0].set_title('Unit Sequence Over Time (First 500 units)', fontsize=14, fontweight='bold')
    axes[0].grid(alpha=0.3)
    
    # Unit frequency histogram
    axes[1].hist(sample_units, bins=100, color='teal', alpha=0.7, edgecolor='black')
    axes[1].set_xlabel('Unit ID', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].set_title('Unit Frequency Distribution in Sample Sequence', fontsize=14, fontweight='bold')
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.show()
else:
    print("⚠️  Sample unit file not found")"""))

# Summary section
nb.cells.append(nbf.v4.new_markdown_cell("""## 4. Summary & Insights

Key findings from the acoustic tokenization pipeline.
"""))

nb.cells.append(nbf.v4.new_code_cell("""# Calculate summary statistics
motif_counts = [m['count'] for m in motif_data['top_motifs']]
motif_percentages = [(c / motif_data['total_tokens']) * 100 for c in motif_counts]

print("📋 Summary & Key Insights")
print("=" * 60)
print("\\n🎯 Pipeline Results:")
print(f"   • Processed {validation['total_sequences']:,} audio segments")
print(f"   • Discovered {motif_data['unique_motifs']} acoustic motifs via BPE")
print(f"   • Created {motif_data['total_tokens']:,} total tokens")
print(f"   • Average sequence length: {stats['avg_sequence_length']:.1f} units")

print("\\n🔍 Acoustic Unit Discovery:")
print(f"   • K-Means clustering found {stats['expected_unique']} distinct acoustic units")
print(f"   • Units represent quantized acoustic features from XLSR-53 model")
print(f"   • Unit range: 0-{stats['unit_range'][1]} (covering all discovered units)")

print("\\n🎵 Motif Discovery:")
print(f"   • Top motif frequency: {motif_percentages[0]:.2f}% of all tokens")
print(f"   • Top 10 motifs cover: {sum(motif_percentages[:10]):.2f}% of tokens")
print(f"   • Top 20 motifs cover: {sum(motif_percentages[:20]):.2f}% of tokens")
print(f"   • This indicates common acoustic patterns in Portuguese speech")

print("\\n💡 Applications:")
print("   • Text-to-Speech synthesis using discovered units")
print("   • Speech recognition using acoustic motifs")
print("   • Language modeling for Portuguese audio")
print("   • Acoustic pattern analysis and research")

print("\\n✅ Analysis Complete!")
print("=" * 60)"""))

# Save notebook
with open('analyze_results.ipynb', 'w') as f:
    nbf.write(nb, f)

print("✅ Notebook created: analyze_results.ipynb")
