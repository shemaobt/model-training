# Architecture Deep Dive

This document provides detailed explanations of the models and algorithms used in the acoustic tokenization pipeline.

## Table of Contents

1. [XLSR-53 Feature Extraction](#xlsr-53-feature-extraction)
2. [K-Means Acoustic Clustering](#k-means-acoustic-clustering)
3. [BPE Motif Discovery](#bpe-motif-discovery)
4. [Vocoder Architecture](#vocoder-architecture)
5. [Training Dynamics](#training-dynamics)

---

## XLSR-53 Feature Extraction

### What is XLSR-53?

XLSR-53 (Cross-Lingual Speech Representations) is a wav2vec 2.0 model pre-trained on 53 languages. It learns universal speech representations without any transcription.

```mermaid
flowchart TB
    subgraph Input
        A[Raw Waveform<br/>16kHz, mono]
    end
    
    subgraph CNN Feature Encoder
        B[7 Conv Layers<br/>stride 5,2,2,2,2,2,2]
        C[Temporal Reduction<br/>320x downsample]
    end
    
    subgraph Transformer
        D[24 Transformer Layers<br/>1024-dim hidden]
        E[Self-Attention<br/>16 heads]
    end
    
    subgraph Output
        F[Layer 14 Hidden States<br/>Best for phonetic info]
    end
    
    A --> B --> C --> D --> E --> F
    
    style F fill:#c8e6c9
```

### Why Layer 14?

Research shows different layers capture different information:

| Layers | Information Type |
|--------|------------------|
| 1-6 | Acoustic features (pitch, energy) |
| 7-12 | Phonetic features (consonants, vowels) |
| **13-18** | **Phonemic features (language-specific sounds)** |
| 19-24 | Semantic features (word-level) |

Layer 14 provides the best balance for discovering **language-independent phonetic units**.

### Frame Rate Calculation

```
Input: 1 second of audio = 16,000 samples
CNN stride: 5 × 2 × 2 × 2 × 2 × 2 × 2 = 320
Output: 16,000 / 320 = 50 frames per second
Frame duration: 1000ms / 50 = 20ms per frame
```

Each acoustic unit represents approximately **20ms of speech**.

---

## K-Means Acoustic Clustering

### Why K-Means?

K-Means provides a simple, deterministic mapping from continuous features to discrete units:

```mermaid
flowchart LR
    subgraph Continuous Space
        A[XLSR-53 Features<br/>1024 dimensions]
    end
    
    subgraph Clustering
        B[MiniBatch K-Means<br/>k=100 clusters]
        C[Cluster Centroids<br/>100 × 1024]
    end
    
    subgraph Discrete Space
        D[Unit ID<br/>0-99]
    end
    
    A --> B --> C --> D
```

### Choosing k=100

The number of clusters affects the trade-off between:

| k Value | Granularity | Training | Coverage |
|---------|-------------|----------|----------|
| 50 | Coarse | Easy | May merge distinct sounds |
| **100** | **Balanced** | **Good** | **~phoneme level** |
| 200 | Fine | Harder | May split similar sounds |
| 500 | Very fine | Difficult | Sparse data per cluster |

100 clusters approximates the number of phonemes across languages (~44 in English, ~33 in Portuguese), making it suitable for phonetic-level tokenization.

### MiniBatch K-Means

We use MiniBatch K-Means instead of standard K-Means for efficiency:

```python
MiniBatchKMeans(
    n_clusters=100,
    batch_size=10000,      # Process 10k samples at a time
    max_iter=100,          # 100 iterations
    random_state=42,       # Reproducibility
    n_init=3               # 3 random initializations
)
```

This allows processing ~15 million feature vectors efficiently.

---

## BPE Motif Discovery

### What is BPE?

Byte Pair Encoding iteratively merges the most frequent pairs of tokens:

```mermaid
flowchart TB
    subgraph Initial
        A["Sequence: 31 31 87 43 43 17"]
    end
    
    subgraph Iteration 1
        B["Most frequent pair: (31, 31)"]
        C["Merge into: 100"]
        D["Result: 100 87 43 43 17"]
    end
    
    subgraph Iteration 2
        E["Most frequent pair: (43, 43)"]
        F["Merge into: 101"]
        G["Result: 100 87 101 17"]
    end
    
    A --> B --> C --> D --> E --> F --> G
```

### Motif Interpretation

BPE discovers recurring patterns that may correspond to:

| Pattern Type | Example | Linguistic Meaning |
|--------------|---------|-------------------|
| Unit pairs | `31 31` | Sustained sound (vowel) |
| Triplets | `43 17 39` | Consonant cluster |
| Longer patterns | `31 87 43 17 17 39` | Syllable or word fragment |

### SentencePiece Configuration

```python
spm.SentencePieceTrainer.train(
    input='all_units.txt',
    model_prefix='portuguese_bpe',
    vocab_size=100,           # 100 motifs
    model_type='bpe',         # Byte Pair Encoding
    character_coverage=1.0,   # Cover all units
    num_threads=4
)
```

---

## Vocoder Architecture

### Generator (Unit → Audio)

The generator converts discrete unit sequences to continuous audio waveforms.

```mermaid
flowchart TB
    subgraph Input
        A[Units: T integers<br/>e.g., T=50]
    end
    
    subgraph Embedding
        B[nn.Embedding<br/>100 → 256]
        C[Shape: T × 256]
    end
    
    subgraph Pre-Processing
        D[Conv1d 7×1<br/>256 → 512]
        E[LeakyReLU 0.1]
    end
    
    subgraph Upsampling["Upsampling (320x total)"]
        F[ConvTranspose 5x<br/>512 → 256]
        G[ConvTranspose 4x<br/>256 → 128]
        H[ConvTranspose 4x<br/>128 → 64]
        I[ConvTranspose 4x<br/>64 → 32]
    end
    
    subgraph Refinement
        J[ResBlock × 3<br/>32 channels]
        K[Conv1d 7×1<br/>32 → 1]
        L[Tanh activation]
    end
    
    subgraph Output
        M[Audio: T×320 samples<br/>e.g., 16,000 samples]
    end
    
    A --> B --> C --> D --> E --> F --> G --> H --> I --> J --> K --> L --> M
```

### Upsampling Blocks

Each upsampling block:

```python
def _make_upsample_block(in_ch, out_ch, kernel, stride):
    return nn.Sequential(
        nn.ConvTranspose1d(in_ch, out_ch, kernel, stride, padding),
        nn.LeakyReLU(0.1),
        nn.Conv1d(out_ch, out_ch, 7, padding=3),  # Smooth artifacts
        nn.LeakyReLU(0.1),
    )
```

The conv1d after transposed conv helps reduce checkerboard artifacts.

### Discriminator (Multi-Scale)

The discriminator operates at multiple audio resolutions:

```mermaid
flowchart TB
    subgraph Input
        A[Audio Waveform]
    end
    
    subgraph Scale1[Scale 1: Original]
        B1[Conv Layers]
        C1[Score 1]
    end
    
    subgraph Scale2[Scale 2: 2x Downsampled]
        B2[AvgPool 4x2]
        C2[Conv Layers]
        D2[Score 2]
    end
    
    subgraph Scale3[Scale 3: 4x Downsampled]
        B3[AvgPool 4x2]
        C3[Conv Layers]
        D3[Score 3]
    end
    
    A --> B1 --> C1
    A --> B2 --> C2 --> D2
    B2 --> B3 --> C3 --> D3
    
    C1 --> E[Combined Loss]
    D2 --> E
    D3 --> E
```

Multi-scale discrimination helps capture both:
- Fine details (high frequency) at original scale
- Global structure (low frequency) at downsampled scales

---

## Training Dynamics

### Loss Functions

```mermaid
flowchart LR
    subgraph Generator Losses
        A[Mel Spectrogram L1]
        B[Adversarial Loss]
    end
    
    subgraph Discriminator Loss
        C[Real vs Fake<br/>LSGAN Loss]
    end
    
    A --> D[Total G Loss]
    B --> D
    
    style A fill:#c8e6c9
    style B fill:#fff3e0
```

#### Mel Spectrogram Loss

```python
mel_transform = MelSpectrogram(
    sample_rate=16000,
    n_fft=1024,
    hop_length=256,
    n_mels=80
)

mel_loss = L1Loss(mel_transform(fake), mel_transform(real))
```

This ensures spectral similarity between generated and real audio.

#### Adversarial Loss (LSGAN)

```python
# Discriminator
d_loss = MSE(D(real), 1) + MSE(D(fake), 0)

# Generator
g_adv_loss = MSE(D(fake), 1)  # Fool discriminator
```

### Training Issues Observed

#### Discriminator Collapse (D Loss → 0)

Our training showed D loss dropping to 0.0, indicating:

```mermaid
flowchart LR
    A[Generator improves] --> B[Discriminator can't distinguish]
    B --> C[D loss → 0]
    C --> D[No gradient to Generator]
    D --> E[Generator stops improving]
    E --> F[Mode collapse / Robotic audio]
```

**Solutions:**
1. Train D more often (e.g., 5 D steps per G step)
2. Add gradient penalty (WGAN-GP)
3. Use feature matching loss
4. Add multi-period discriminator (HiFi-GAN)

### Information Bottleneck

The quantization to 100 units creates severe information loss:

```mermaid
flowchart TB
    subgraph Original
        A[Continuous Audio<br/>16,000 values/sec<br/>∞ precision]
    end
    
    subgraph XLSR Features
        B[1024-dim vectors<br/>50 vectors/sec<br/>High precision]
    end
    
    subgraph Quantized
        C[100 discrete units<br/>50 units/sec<br/>log₂(100) ≈ 6.6 bits/unit]
    end
    
    subgraph Bitrate
        D[330 bits/sec<br/>vs ~256 kbps original]
    end
    
    A -->|"Feature extraction"| B -->|"K-Means"| C -->|"Calculation"| D
    
    style C fill:#ffcdd2
    style D fill:#ffcdd2
```

This 775x compression ratio makes perfect reconstruction impossible.

---

## Recommendations for Improvement

### 1. Pitch Conditioning

Add F0 (fundamental frequency) as additional input:

```python
class PitchConditionedGenerator(nn.Module):
    def __init__(self, num_units=100, embed_dim=256):
        self.unit_embed = nn.Embedding(num_units, embed_dim)
        self.pitch_embed = nn.Linear(1, embed_dim)  # Add pitch
        
    def forward(self, units, pitch):
        x = self.unit_embed(units) + self.pitch_embed(pitch)
        # ... rest of generator
```

### 2. HiFi-GAN Architecture

Key improvements from HiFi-GAN:

| Component | Current | HiFi-GAN |
|-----------|---------|----------|
| Generator | Simple ConvTranspose | Multi-Receptive Field Fusion |
| Discriminator | Single multi-scale | Multi-Period + Multi-Scale |
| Loss | Mel L1 + Adversarial | + Multi-resolution STFT + Feature Matching |

### 3. More Training Data

Current limitation: Single narrator's voice limits generalization.

Options:
- Focus on single-speaker (easier)
- Add speaker embeddings for multi-speaker
- Use more diverse training data
