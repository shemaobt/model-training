# Bible Audio Acoustic Tokenization & Vocoder

A machine learning pipeline for **unsupervised speech representation learning**. This project converts continuous speech (specifically the Bible) into discrete acoustic tokens and learns to resynthesize speech from those tokens.

The goal is to enable NLP-like capabilities (translation, search, pattern discovery) directly on audio data for low-resource languages (like Sateré-Mawé).

## 🧠 Core Concept

Traditional speech processing relies on text transcripts (ASR). We don't.
We use **Self-Supervised Learning** to discover the "units" of a language purely from audio.

```mermaid
flowchart LR
    subgraph Input
        A[🗣️ Raw Audio]
    end

    subgraph Discretization
        B[XLSR-53<br/>Features] --> C[K-Means<br/>Clustering] --> D[Discrete Tokens<br/>31 87 14...]
    end

    subgraph Applications
        D --> E[Language Modeling<br/>(BPE, GPT)]
        D --> F[Vocoder<br/>(Synthesis)]
    end

    A --> B
    F --> G[🗣️ Resynthesized Audio]
    
    style D fill:#f9f,stroke:#333,stroke-width:2px
```

## 🏗️ Pipeline Architecture

The pipeline consists of three distinct training phases.

### Phase 1: Acoustic Tokenization
Extracts features from audio and learns a discrete vocabulary of 100 sounds.

```mermaid
flowchart LR
    A[Audio Segment] -->|XLSR-53 Layer 14| B[1024-dim Vectors]
    B -->|K-Means| C[Cluster Centroids]
    B -->|Nearest Neighbor| D[Unit Sequence]
```

### Phase 2: Pattern Discovery (BPE)
Analyzes the sequence of units to find recurring motifs (acoustic "words").

```mermaid
flowchart LR
    A[Unit Sequence<br/>31 31 87 43] -->|SentencePiece| B[BPE Training]
    B --> C[Vocabulary<br/>Motifs]
    C --> D[Tokenized Sequence<br/>105 43]
```

### Phase 3: Vocoder (Synthesis)
Trains a generative model to convert discrete units back into continuous audio waveforms.

```mermaid
flowchart TB
    A[Unit Sequence] -->|Generator| B[Fake Audio]
    C[Real Audio] -->|Discriminator| D[Real/Fake Score]
    B -->|Discriminator| D
    B -->|Mel Loss| C
```

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Modal account (for cloud GPU training)
- `ffmpeg` installed locally

### Installation
```bash
pip install -r requirements.txt
python3 -m modal token set --token-id <id> --token-secret <secret>
```

### Execution Flow

**1. Segment Audio Locally** (Generic support for multiple languages)
```bash
# For Portuguese
python scripts/segment_audio.py --language portuguese

# For Sateré-Mawé
python scripts/segment_audio.py --language satere
```

**2. Upload to Cloud Storage**
```bash
python3 -m modal run scripts/upload_to_modal.py --language satere
```

**3. Run Training Pipeline (on Modal)**
```bash
# Phase 1: Discover acoustic units
python3 -m modal run --detach src/training/phase1_acoustic.py

# Phase 2: Learn motifs (BPE)
python3 -m modal run --detach src/training/phase2_bpe.py

# Phase 3: Train Vocoder (Synthesizer)
python3 -m modal run --detach src/training/phase3_vocoder.py
```

## 📂 Project Structure

```
model-training/
├── src/
│   ├── models/                    # Neural Network Architectures (Jupytext)
│   │   ├── generator.py           # Vocoder (Upsamples units to audio)
│   │   └── discriminator.py       # Quality control (Multi-scale GAN)
│   ├── training/                  # Cloud Training Scripts
│   │   ├── phase1_acoustic.py     # Feature extraction & clustering
│   │   ├── phase3_vocoder.py      # GAN training loop
│   │   └── ...
├── scripts/                       # Local Utilities
│   ├── segment_audio.py           # Silence-based segmentation
│   └── upload_to_modal.py         # Data transfer
├── docs/
│   ├── ARCHITECTURE.md            # Deep dive into design decisions
│   └── PIPELINE.md                # Step-by-step manual
└── audio_data/                    # Raw input files (gitignored)
```

## 🔬 Design Decisions & Trade-offs

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for a comprehensive analysis.

**Key Highlights:**
- **Why 100 Units?** Balanced trade-off between phonetic granularity and model trainability.
- **Why Layer 14?** Best layer in XLSR-53 for phonetic content, filtering out speaker identity.
- **Why Robotic Audio?** We deliberately discarded pitch (F0) to focus on phonetic content. Future work involves pitch conditioning.

## 📊 Results

- **Compression:** ~775x reduction in bitrate (Raw Audio → Discrete Tokens).
- **Intelligibility:** High. The vocoder successfully reconstructs words from tokens.
- **Naturalness:** Low/Medium. Prosody is flat due to F0 loss.

## License
Private - shemaobt organization.
