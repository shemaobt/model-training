# Vocoder V2: Complete Technical Reference

This document provides a comprehensive technical specification of the V2 vocoder architecture, including exact layer dimensions, design rationale, and implementation details.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Generator V2: Complete Specification](#generator-v2-complete-specification)
3. [Discriminator V2: Complete Specification](#discriminator-v2-complete-specification)
4. [Pitch Extraction & Conditioning](#pitch-extraction--conditioning)
5. [Loss Functions: Complete Implementation](#loss-functions-complete-implementation)
6. [Training Configuration](#training-configuration)
7. [V1 vs V2 Comparison](#v1-vs-v2-comparison)

---

## Architecture Overview

The V2 vocoder solves the "robotic audio" problem through three key improvements:

1. **Pitch Conditioning**: Re-injects F0 (fundamental frequency) information
2. **HiFi-GAN Generator**: Multi-Receptive Field Fusion for better temporal patterns
3. **Combined Discriminator**: MPD + MSD for comprehensive quality assessment

### High-Level Data Flow

```
Inputs:
  Units [B, T]: Acoustic unit indices (0-99)
  Pitch [B, T]: Quantized pitch bins (0-32)
         ↓
  ┌─────────────────────────────────────┐
  │          Generator V2               │
  │  ┌─────────┐    ┌─────────┐        │
  │  │  Unit   │    │  Pitch  │        │
  │  │ Embed   │    │ Embed   │        │
  │  │ 100→256 │    │ 33→64   │        │
  │  └────┬────┘    └────┬────┘        │
  │       └──────┬───────┘             │
  │              ↓                      │
  │       Concatenate (320)             │
  │              ↓                      │
  │       Pre-Conv (320→512)            │
  │              ↓                      │
  │       4× Upsample + MRF             │
  │       (5x, 4x, 4x, 4x = 320x)       │
  │              ↓                      │
  │       Post-Conv (32→1)              │
  └──────────────┬──────────────────────┘
                 ↓
         Audio [B, T×320]
                 ↓
  ┌─────────────────────────────────────┐
  │       Discriminator V2              │
  │  ┌──────────────┬──────────────┐   │
  │  │     MPD      │     MSD      │   │
  │  │  5 periods   │   3 scales   │   │
  │  │  [2,3,5,7,11]│ [1x,2x,4x]   │   │
  │  └──────┬───────┴──────┬───────┘   │
  │         └──────┬───────┘           │
  │                ↓                    │
  │    8 Scores + 8 Feature Lists      │
  └─────────────────────────────────────┘
```

### Parameter Counts

| Component | Parameters | Description |
|-----------|------------|-------------|
| Generator V2 | ~4.5M | Unit + Pitch embedding, 4 upsample blocks, 12 MRF ResBlocks |
| MPD (5 periods) | ~10M | 5 period discriminators with 2D convolutions |
| MSD (3 scales) | ~5M | 3 scale discriminators with 1D convolutions |
| **Total** | **~19.5M** | |

---

## Generator V2: Complete Specification

### Input/Output Specification

| Property | Specification |
|----------|---------------|
| **Unit Input** | [B, T] integers in range [0, 99] |
| **Pitch Input** | [B, T] integers in range [0, 32] |
| **Audio Output** | [B, T×320] floats in range [-1, 1] |
| **Sample Rate** | 16,000 Hz |
| **Frame Rate** | 50 Hz (20ms per unit) |

### Layer-by-Layer Breakdown

#### 1. Embedding Layers

```python
# Unit Embedding
self.unit_embed = nn.Embedding(100, 256)
# Input:  [B, T] int64
# Output: [B, T, 256] float32

# Pitch Embedding  
self.pitch_embed = nn.Embedding(33, 64)  # 32 bins + 1 unvoiced
# Input:  [B, T] int64
# Output: [B, T, 64] float32

# Concatenation
x = torch.cat([unit_emb, pitch_emb], dim=-1)
# Output: [B, T, 320] float32
```

**Design Rationale**:
- 256 dims for units captures phonetic distinctions
- 64 dims for pitch is sufficient for 32 logarithmically-spaced bins
- Separate embeddings allow independent learning of phonetic vs prosodic features

#### 2. Pre-Convolution

```python
self.pre_conv = weight_norm(nn.Conv1d(320, 512, kernel_size=7, padding=3))
# Input:  [B, 320, T]
# Output: [B, 512, T]
```

**Design Rationale**:
- Large kernel (7) provides initial context mixing
- Weight normalization stabilizes training
- Expands to 512 channels for rich feature representation

#### 3. Upsample Blocks (4 total)

Each block consists of:
1. Transposed convolution for upsampling
2. Multi-Receptive Field Fusion for refinement

```python
# Block 1: 5x upsample
TransConv1d(512, 256, kernel=10, stride=5, padding=2)  # [B, 512, T] → [B, 256, T×5]
MRF(256, kernels=[3, 7, 11])

# Block 2: 4x upsample  
TransConv1d(256, 128, kernel=8, stride=4, padding=2)   # [B, 256, T×5] → [B, 128, T×20]
MRF(128, kernels=[3, 7, 11])

# Block 3: 4x upsample
TransConv1d(128, 64, kernel=8, stride=4, padding=2)    # [B, 128, T×20] → [B, 64, T×80]
MRF(64, kernels=[3, 7, 11])

# Block 4: 4x upsample
TransConv1d(64, 32, kernel=8, stride=4, padding=2)     # [B, 64, T×80] → [B, 32, T×320]
MRF(32, kernels=[3, 7, 11])
```

**Stride Selection**:
- 5 × 4 × 4 × 4 = 320 (matches XLSR-53 hop size)
- Kernel = 2 × stride for smooth upsampling

#### 4. Multi-Receptive Field Fusion (MRF)

```python
class MRF(nn.Module):
    def __init__(self, channels, kernels=[3, 7, 11]):
        self.resblocks = nn.ModuleList([
            ResBlock(channels, kernel=3, dilations=[[1,1], [3,1], [5,1]]),
            ResBlock(channels, kernel=7, dilations=[[1,1], [3,1], [5,1]]),
            ResBlock(channels, kernel=11, dilations=[[1,1], [3,1], [5,1]]),
        ])
    
    def forward(self, x):
        out = sum(rb(x) for rb in self.resblocks) / 3
        return out
```

**Receptive Field Analysis**:

| Kernel | Dilation | Receptive Field | Time at 16kHz |
|--------|----------|-----------------|---------------|
| 3 | 1 | 3 samples | 0.19 ms |
| 3 | 3 | 7 samples | 0.44 ms |
| 3 | 5 | 11 samples | 0.69 ms |
| 7 | 1 | 7 samples | 0.44 ms |
| 7 | 3 | 15 samples | 0.94 ms |
| 7 | 5 | 23 samples | 1.44 ms |
| 11 | 1 | 11 samples | 0.69 ms |
| 11 | 3 | 23 samples | 1.44 ms |
| 11 | 5 | 35 samples | 2.19 ms |

**Combined effective receptive field**: ~50ms (covers a full pitch period)

#### 5. ResBlock with Dilations

```python
class ResBlock(nn.Module):
    def __init__(self, channels, kernel_size, dilations=[[1,1], [3,1], [5,1]]):
        # 3 dilation pairs, each with 2 convolutions
        for d1, d2 in dilations:
            self.convs1.append(weight_norm(Conv1d(ch, ch, kernel, dilation=d1, padding=...)))
            self.convs2.append(weight_norm(Conv1d(ch, ch, kernel, dilation=d2, padding=...)))
    
    def forward(self, x):
        for conv1, conv2 in zip(self.convs1, self.convs2):
            residual = x
            x = leaky_relu(conv1(leaky_relu(x)))
            x = conv2(x)
            x = x + residual  # Skip connection
        return x
```

**Per ResBlock**: 6 convolutions (3 pairs × 2)
**Per MRF**: 3 ResBlocks × 6 = 18 convolutions
**Total MRF convolutions**: 4 blocks × 18 = 72 convolutions

#### 6. Post-Convolution

```python
x = F.leaky_relu(x, 0.1)
self.post_conv = weight_norm(nn.Conv1d(32, 1, kernel_size=7, padding=3))
x = torch.tanh(self.post_conv(x))
# Input:  [B, 32, T×320]
# Output: [B, 1, T×320] → squeeze → [B, T×320]
```

---

## Discriminator V2: Complete Specification

The V2 discriminator combines Multi-Period Discriminator (MPD) and Multi-Scale Discriminator (MSD).

### Multi-Period Discriminator (MPD)

**Concept**: Reshape 1D audio into 2D based on period, then apply 2D convolutions.

#### Period Selection: [2, 3, 5, 7, 11]

| Period | Frequency Pattern | What It Detects |
|--------|-------------------|-----------------|
| 2 | 8,000 Hz | Nyquist artifacts, aliasing |
| 3 | 5,333 Hz | High formant distortion |
| 5 | 3,200 Hz | F3 formant region |
| 7 | 2,286 Hz | F2 formant region |
| 11 | 1,455 Hz | F1/pitch region |

**Why prime numbers?** Ensures non-overlapping, diverse pattern detection.

#### Single Period Discriminator Architecture

```python
class PeriodDiscriminator(nn.Module):
    def __init__(self, period, use_spectral_norm=True):
        self.period = period
        norm = spectral_norm if use_spectral_norm else weight_norm
        
        self.convs = nn.ModuleList([
            norm(Conv2d(1, 32, (5,1), stride=(3,1), padding=(2,0))),
            norm(Conv2d(32, 128, (5,1), stride=(3,1), padding=(2,0))),
            norm(Conv2d(128, 512, (5,1), stride=(3,1), padding=(2,0))),
            norm(Conv2d(512, 1024, (5,1), stride=(3,1), padding=(2,0))),
            norm(Conv2d(1024, 1024, (5,1), stride=1, padding=(2,0))),
        ])
        self.conv_post = norm(Conv2d(1024, 1, (3,1), padding=(1,0)))
```

**Dimension Flow (for period=5, T=16000)**:

| Layer | Input Shape | Output Shape | Stride |
|-------|-------------|--------------|--------|
| Reshape | [B, 1, 16000] | [B, 1, 3200, 5] | - |
| Conv1 | [B, 1, 3200, 5] | [B, 32, 1067, 5] | (3,1) |
| Conv2 | [B, 32, 1067, 5] | [B, 128, 356, 5] | (3,1) |
| Conv3 | [B, 128, 356, 5] | [B, 512, 119, 5] | (3,1) |
| Conv4 | [B, 512, 119, 5] | [B, 1024, 40, 5] | (3,1) |
| Conv5 | [B, 1024, 40, 5] | [B, 1024, 40, 5] | 1 |
| Post | [B, 1024, 40, 5] | [B, 1, 40, 5] | 1 |
| Flatten | [B, 1, 40, 5] | [B, 200] | - |

### Multi-Scale Discriminator (MSD)

**Concept**: Apply 1D convolutions at multiple audio resolutions.

#### Scale Configuration

| Scale | Input Resolution | Downsampling | Purpose |
|-------|------------------|--------------|---------|
| 1 | 16,000 Hz | None | Fine details, high frequency |
| 2 | 8,000 Hz | AvgPool(4,2,2) | Medium details |
| 3 | 4,000 Hz | AvgPool(4,2,2) | Global structure |

#### Single Scale Discriminator Architecture

```python
class ScaleDiscriminator(nn.Module):
    def __init__(self, use_spectral_norm=True):
        norm = spectral_norm if use_spectral_norm else weight_norm
        
        self.convs = nn.ModuleList([
            norm(Conv1d(1, 128, 15, stride=1, padding=7)),
            norm(Conv1d(128, 128, 41, stride=2, padding=20, groups=4)),
            norm(Conv1d(128, 256, 41, stride=2, padding=20, groups=16)),
            norm(Conv1d(256, 512, 41, stride=4, padding=20, groups=16)),
            norm(Conv1d(512, 1024, 41, stride=4, padding=20, groups=16)),
            norm(Conv1d(1024, 1024, 41, stride=1, padding=20, groups=16)),
            norm(Conv1d(1024, 1024, 5, stride=1, padding=2)),
        ])
        self.conv_post = norm(Conv1d(1024, 1, 3, padding=1))
```

**Dimension Flow (T=16000)**:

| Layer | Input Shape | Output Shape | Stride | Groups |
|-------|-------------|--------------|--------|--------|
| Conv1 | [B, 1, 16000] | [B, 128, 16000] | 1 | 1 |
| Conv2 | [B, 128, 16000] | [B, 128, 8000] | 2 | 4 |
| Conv3 | [B, 128, 8000] | [B, 256, 4000] | 2 | 16 |
| Conv4 | [B, 256, 4000] | [B, 512, 1000] | 4 | 16 |
| Conv5 | [B, 512, 1000] | [B, 1024, 250] | 4 | 16 |
| Conv6 | [B, 1024, 250] | [B, 1024, 250] | 1 | 16 |
| Conv7 | [B, 1024, 250] | [B, 1024, 250] | 1 | 1 |
| Post | [B, 1024, 250] | [B, 1, 250] | 1 | 1 |

**Why grouped convolutions?** Reduces parameters while maintaining capacity (groups=16 → 16× fewer params per layer).

---

## Pitch Extraction & Conditioning

### The Problem

XLSR-53 Layer 14 is designed to be **pitch-invariant**. The same phoneme spoken at different pitches maps to the same unit. This is good for ASR but bad for synthesis.

### The Solution: Explicit Pitch Conditioning

Extract pitch from source audio and provide it as a separate input to the generator.

### Pitch Extraction (PYIN Algorithm)

```python
import librosa

def extract_pitch(audio, sr=16000, hop_length=320):
    f0, voiced_flag, voiced_probs = librosa.pyin(
        audio,
        fmin=50,      # Min frequency (deep male voice)
        fmax=400,     # Max frequency (children/high female)
        sr=sr,
        hop_length=hop_length
    )
    return f0, voiced_flag
```

### Pitch Quantization

We quantize continuous F0 to 32 logarithmically-spaced bins:

```python
def quantize_pitch(f0, num_bins=32, f0_min=50.0, f0_max=400.0):
    # Handle unvoiced frames
    voiced_mask = (f0 > 0) & ~np.isnan(f0)
    
    # Log-scale binning (perceptually uniform)
    log_f0 = np.log(np.clip(f0, f0_min, f0_max))
    log_min, log_max = np.log(f0_min), np.log(f0_max)
    
    # Normalize to [0, 1] then scale to [1, num_bins]
    normalized = (log_f0 - log_min) / (log_max - log_min)
    bins = (normalized * (num_bins - 1) + 1).astype(int)
    
    # Unvoiced frames get bin 0
    result = np.zeros_like(f0, dtype=int)
    result[voiced_mask] = bins[voiced_mask]
    
    return result
```

### Pitch Bin to Frequency Mapping

| Bin | Frequency Range | Typical Voice |
|-----|-----------------|---------------|
| 0 | Unvoiced | Silence, unvoiced consonants |
| 1-4 | 50-65 Hz | Very deep bass |
| 5-8 | 65-90 Hz | Bass male |
| 9-12 | 90-120 Hz | Average male |
| 13-16 | 120-160 Hz | High male / low female |
| 17-20 | 160-200 Hz | Average female |
| 21-24 | 200-260 Hz | High female |
| 25-28 | 260-330 Hz | Very high female |
| 29-32 | 330-400 Hz | Children |

---

## Loss Functions: Complete Implementation

### 1. Mel Spectrogram Loss

**Purpose**: Ensure phonetic content matches (what is being said).

```python
def mel_spectrogram_loss(real_audio, fake_audio):
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000,
        n_fft=1024,
        win_length=1024,
        hop_length=256,
        n_mels=80,
        f_min=0,
        f_max=8000,
        power=1.0  # Magnitude, not power
    )
    
    real_mel = mel_transform(real_audio)
    fake_mel = mel_transform(fake_audio)
    
    return F.l1_loss(fake_mel, real_mel)
```

**Weight**: 45.0 (highest priority)

### 2. Multi-Resolution STFT Loss

**Purpose**: Capture harmonic structure at multiple time-frequency resolutions.

```python
def stft_loss(real_audio, fake_audio, fft_sizes=[512, 1024, 2048]):
    sc_loss = 0  # Spectral Convergence
    mag_loss = 0  # Log Magnitude
    
    for n_fft in fft_sizes:
        hop = n_fft // 4
        
        real_stft = torch.stft(real_audio, n_fft, hop, n_fft, 
                               window=torch.hann_window(n_fft),
                               return_complex=True)
        fake_stft = torch.stft(fake_audio, n_fft, hop, n_fft,
                               window=torch.hann_window(n_fft),
                               return_complex=True)
        
        real_mag = torch.abs(real_stft)
        fake_mag = torch.abs(fake_stft)
        
        # Spectral Convergence: Frobenius norm ratio
        sc_loss += torch.norm(real_mag - fake_mag, p='fro') / torch.norm(real_mag, p='fro')
        
        # Log Magnitude: L1 in log domain
        mag_loss += F.l1_loss(torch.log(fake_mag + 1e-7), torch.log(real_mag + 1e-7))
    
    return (sc_loss + mag_loss) / len(fft_sizes)
```

**Weight**: 2.0

### 3. Feature Matching Loss

**Purpose**: Stabilize training by matching intermediate discriminator representations.

```python
def feature_matching_loss(real_features, fake_features):
    """
    Args:
        real_features: List[List[Tensor]] - 8 discriminators × N layers each
        fake_features: List[List[Tensor]] - same structure
    """
    loss = 0
    for real_feat_list, fake_feat_list in zip(real_features, fake_features):
        for real_feat, fake_feat in zip(real_feat_list, fake_feat_list):
            # Detach real to not backprop through D
            loss += F.l1_loss(fake_feat, real_feat.detach())
    return loss
```

**Weight**: 2.0

### 4. Adversarial Loss (LSGAN)

**Purpose**: Encourage realistic audio generation.

```python
def discriminator_loss(real_scores, fake_scores):
    """LSGAN discriminator loss."""
    real_loss = 0
    fake_loss = 0
    for real_out, fake_out in zip(real_scores, fake_scores):
        real_loss += torch.mean((1 - real_out) ** 2)  # Real should be 1
        fake_loss += torch.mean(fake_out ** 2)        # Fake should be 0
    return real_loss + fake_loss

def generator_adversarial_loss(fake_scores):
    """LSGAN generator loss."""
    loss = 0
    for fake_out in fake_scores:
        loss += torch.mean((1 - fake_out) ** 2)  # Fake should fool D (be 1)
    return loss
```

**Weight**: 1.0

### Total Generator Loss

```python
g_loss = (
    45.0 * mel_loss +      # Content preservation
    2.0 * stft_loss +      # Harmonic structure
    2.0 * fm_loss +        # Training stability
    1.0 * adv_loss         # Realism
)
```

---

## Training Configuration

### Hyperparameters

| Parameter | V2 Value | Rationale |
|-----------|----------|-----------|
| **Batch Size** | 12 | Limited by GPU memory with 2s segments |
| **Segment Length** | 32,000 (2s) | Longer context for prosody |
| **Learning Rate (G)** | 0.0002 | Standard for Adam |
| **Learning Rate (D)** | 0.0002 | Matched to G |
| **Adam Betas** | (0.8, 0.99) | More momentum than default |
| **Weight Decay** | 0.01 | AdamW regularization |
| **Gradient Clipping** | 5.0 (G only) | Prevents exploding gradients |
| **Early Stopping** | patience=100 | More epochs before giving up |
| **Checkpoint Frequency** | 25 epochs | Balance storage vs recovery |

### GPU Requirements

| Configuration | GPU Memory | Throughput |
|---------------|------------|------------|
| batch=12, seg=32000 | ~20 GB | ~50 samples/sec |
| batch=8, seg=32000 | ~14 GB | ~35 samples/sec |
| batch=12, seg=16000 | ~12 GB | ~80 samples/sec |

---

## V1 vs V2 Comparison

### Architecture Differences

| Component | V1 | V2 |
|-----------|----|----|
| **Unit Embedding** | 100 → 256 | 100 → 256 |
| **Pitch Embedding** | None | 33 → 64 |
| **Pre-Conv Input** | 256 | 320 (256+64) |
| **Residual Blocks** | 3 simple ResBlocks | 4× MRF (12 ResBlocks total) |
| **Dilation** | None | [[1,1], [3,1], [5,1]] |
| **Normalization (G)** | None | Weight Norm |
| **Discriminator** | MSD only (3 scales) | MPD (5) + MSD (3) |
| **Normalization (D)** | None | Spectral Norm |

### Loss Differences

| Loss | V1 | V2 |
|------|----|----|
| Mel Spectrogram | Yes (λ=45) | Yes (λ=45) |
| Multi-Res STFT | No | Yes (λ=2) |
| Feature Matching | No | Yes (λ=2) |
| Adversarial | Yes (λ=1) | Yes (λ=1) |

### Expected Quality Improvements

| Metric | V1 | V2 (Expected) |
|--------|----|----|
| **SNR** | ~0 dB | > 5 dB |
| **MCD** | ~80 | < 10 |
| **F0 RMSE** | N/A | < 20 Hz |
| **MOS (Naturalness)** | 1-2/5 | 3-4/5 |

---

## References

- **HiFi-GAN**: [Kong et al., 2020](https://arxiv.org/abs/2010.05646) - Generator architecture, MRF
- **MelGAN**: [Kumar et al., 2019](https://arxiv.org/abs/1910.06711) - Multi-scale discriminator
- **PYIN**: [Mauch & Dixon, 2014](https://www.eecs.qmul.ac.uk/~simondi/papers/2014-ICASSP-pyin.pdf) - Pitch extraction
- **Spectral Norm**: [Miyato et al., 2018](https://arxiv.org/abs/1802.05957) - Discriminator stability
- **Weight Norm**: [Salimans & Kingma, 2016](https://arxiv.org/abs/1602.07868) - Generator normalization
- **LSGAN**: [Mao et al., 2017](https://arxiv.org/abs/1611.04076) - Least squares adversarial loss
