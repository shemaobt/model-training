# Vocoder V2: Enhanced Architecture & Implementation

This document provides a comprehensive technical analysis of the V2 vocoder architecture, explaining the design decisions, implementation details, and improvements over V1.

## Table of Contents

1. [Overview](#overview)
2. [Generator V2 Architecture](#generator-v2-architecture)
3. [Discriminator V2 Architecture](#discriminator-v2-architecture)
4. [Pitch Extraction & Conditioning](#pitch-extraction--conditioning)
5. [Loss Functions](#loss-functions)
6. [Training Configuration](#training-configuration)
7. [Comparison: V1 vs V2](#comparison-v1-vs-v2)

---

## Overview

The V2 vocoder is a complete architectural upgrade designed to solve the "robotic audio" problem identified in V1. It introduces **pitch conditioning** and uses a **HiFi-GAN style** architecture.

### High-Level Architecture

```mermaid
flowchart TB
    subgraph Inputs [Input Processing]
        A[Unit Sequence<br/>0-99 integers] --> B[Unit Embedding<br/>100 → 256 dim]
        C[Pitch Sequence<br/>0-32 bins] --> D[Pitch Embedding<br/>33 → 64 dim]
    end
    
    subgraph Generator [Generator V2]
        E[Concatenate<br/>256 + 64 = 320] --> F[Pre-Conv<br/>320 → 512]
        F --> G[Upsample + MRF<br/>5x: 512 → 256]
        G --> H[Upsample + MRF<br/>4x: 256 → 128]
        H --> I[Upsample + MRF<br/>4x: 128 → 64]
        I --> J[Upsample + MRF<br/>4x: 64 → 32]
        J --> K[Post-Conv<br/>32 → 1]
    end
    
    subgraph Discriminator [Discriminator V2]
        L[Multi-Period<br/>Periods: 2,3,5,7,11] --> N[Scores + Features]
        M[Multi-Scale<br/>3 Scales] --> N
    end
    
    B --> E
    D --> E
    K --> O[Generated Audio<br/>T × 320 samples]
    O --> L
    O --> M
    
    style C fill:#e8f5e9
    style D fill:#e8f5e9
```

### Key Improvements Summary

| Feature | V1 | V2 |
|---------|----|----|
| **Pitch Conditioning** | ❌ None | ✅ 32-bin F0 embedding |
| **Generator Architecture** | Simple TransConv | HiFi-GAN with MRF |
| **Discriminator** | MSD only (3 scales) | MPD (5 periods) + MSD (3 scales) |
| **Normalization** | None | Weight Norm (G) + Spectral Norm (D) |
| **Losses** | Mel + Adversarial | Mel + STFT + FM + Adversarial |
| **Segment Length** | 1 second | 2 seconds |
| **Total Params (G)** | ~3.3M | ~4.5M |
| **Total Params (D)** | ~5.1M | ~15M |

---

## Generator V2 Architecture

The Generator converts discrete acoustic units (with pitch) into continuous audio waveforms.

### Architecture Diagram

```mermaid
flowchart TB
    subgraph Input [Input Layer]
        U[Units: B×T] --> UE[Unit Embedding<br/>nn.Embedding<br/>100 → 256]
        P[Pitch: B×T] --> PE[Pitch Embedding<br/>nn.Embedding<br/>33 → 64]
    end
    
    subgraph PreProcess [Pre-Processing]
        UE --> CAT[Concatenate]
        PE --> CAT
        CAT --> PRE[Pre-Conv<br/>Conv1d 320→512<br/>kernel=7, weight_norm]
    end
    
    subgraph Up1 [Upsample Block 1 - 5x]
        PRE --> TC1[TransConv<br/>512→256<br/>k=10, s=5]
        TC1 --> MRF1[MRF Block<br/>kernels: 3,7,11]
    end
    
    subgraph Up2 [Upsample Block 2 - 4x]
        MRF1 --> TC2[TransConv<br/>256→128<br/>k=8, s=4]
        TC2 --> MRF2[MRF Block<br/>kernels: 3,7,11]
    end
    
    subgraph Up3 [Upsample Block 3 - 4x]
        MRF2 --> TC3[TransConv<br/>128→64<br/>k=8, s=4]
        TC3 --> MRF3[MRF Block<br/>kernels: 3,7,11]
    end
    
    subgraph Up4 [Upsample Block 4 - 4x]
        MRF3 --> TC4[TransConv<br/>64→32<br/>k=8, s=4]
        TC4 --> MRF4[MRF Block<br/>kernels: 3,7,11]
    end
    
    subgraph Output [Output Layer]
        MRF4 --> POST[Post-Conv<br/>Conv1d 32→1<br/>kernel=7]
        POST --> TANH[Tanh]
        TANH --> OUT[Audio: B×T×320]
    end
```

### Multi-Receptive Field Fusion (MRF)

The MRF block is the core innovation from HiFi-GAN. It applies multiple residual blocks with different kernel sizes and sums their outputs.

```mermaid
flowchart LR
    subgraph MRF [MRF Block]
        IN[Input] --> RB1[ResBlock<br/>kernel=3]
        IN --> RB2[ResBlock<br/>kernel=7]
        IN --> RB3[ResBlock<br/>kernel=11]
        RB1 --> SUM[Sum / 3]
        RB2 --> SUM
        RB3 --> SUM
        SUM --> OUT[Output]
    end
```

**Why MRF?**
- **Kernel=3**: Captures fine-grained phonetic details (~60 samples = 3.75ms)
- **Kernel=7**: Captures sub-phonemic patterns (~140 samples = 8.75ms)
- **Kernel=11**: Captures phoneme-level patterns (~220 samples = 13.75ms)

### ResBlock with Dilations

Each ResBlock uses multiple dilation rates to increase receptive field without increasing parameters.

```mermaid
flowchart TB
    subgraph ResBlock [ResBlock - Dilated Convolutions]
        IN[Input x]
        
        subgraph Dilation1 [Dilation 1,1]
            D1C1[Conv d=1] --> D1A1[LeakyReLU]
            D1A1 --> D1C2[Conv d=1] --> D1O[Output]
        end
        
        subgraph Dilation2 [Dilation 3,1]
            D2C1[Conv d=3] --> D2A1[LeakyReLU]
            D2A1 --> D2C2[Conv d=1] --> D2O[Output]
        end
        
        subgraph Dilation3 [Dilation 5,1]
            D3C1[Conv d=5] --> D3A1[LeakyReLU]
            D3A1 --> D3C2[Conv d=1] --> D3O[Output]
        end
        
        IN --> D1C1
        D1O --> SKIP1[+ Skip]
        IN --> SKIP1
        
        SKIP1 --> D2C1
        D2O --> SKIP2[+ Skip]
        SKIP1 --> SKIP2
        
        SKIP2 --> D3C1
        D3O --> SKIP3[+ Skip]
        SKIP2 --> SKIP3
        
        SKIP3 --> OUT[Output]
    end
```

### Weight Normalization

All convolutional layers in the Generator use **Weight Normalization**:

```python
self.conv = weight_norm(nn.Conv1d(in_ch, out_ch, kernel_size))
```

**Why Weight Norm?**
- Decouples magnitude and direction of weights
- Faster convergence than Batch Norm
- No running statistics (better for variable-length audio)
- Can be removed at inference for efficiency

---

## Discriminator V2 Architecture

The V2 Discriminator combines two complementary architectures:

### Combined Architecture

```mermaid
flowchart TB
    subgraph Input
        A[Audio Waveform<br/>B × T]
    end
    
    subgraph MPD [Multi-Period Discriminator]
        A --> P2[Period 2<br/>8kHz patterns]
        A --> P3[Period 3<br/>5.3kHz patterns]
        A --> P5[Period 5<br/>3.2kHz patterns]
        A --> P7[Period 7<br/>2.3kHz patterns]
        A --> P11[Period 11<br/>1.5kHz patterns]
        
        P2 --> S1[Score + Features]
        P3 --> S2[Score + Features]
        P5 --> S3[Score + Features]
        P7 --> S4[Score + Features]
        P11 --> S5[Score + Features]
    end
    
    subgraph MSD [Multi-Scale Discriminator]
        A --> SC1[Scale 1<br/>Original]
        A --> POOL1[AvgPool 2x]
        POOL1 --> SC2[Scale 2<br/>Downsampled]
        POOL1 --> POOL2[AvgPool 2x]
        POOL2 --> SC3[Scale 3<br/>Further Down]
        
        SC1 --> S6[Score + Features]
        SC2 --> S7[Score + Features]
        SC3 --> S8[Score + Features]
    end
    
    subgraph Output
        S1 --> COMBINE[8 Total Scores<br/>8 Feature Lists]
        S2 --> COMBINE
        S3 --> COMBINE
        S4 --> COMBINE
        S5 --> COMBINE
        S6 --> COMBINE
        S7 --> COMBINE
        S8 --> COMBINE
    end
```

### Period Discriminator

The Period Discriminator reshapes 1D audio into 2D based on a period and applies 2D convolutions.

```mermaid
flowchart LR
    subgraph Reshape [Period Reshape - Example p=5]
        A1["Audio: [B, 1, 16000]"] --> PAD[Pad to divisible]
        PAD --> R["Reshape: [B, 1, 3200, 5]"]
    end
    
    subgraph Convs [2D Convolutions]
        R --> C1["Conv2d 1→32<br/>(5,1) stride (3,1)"]
        C1 --> C2["Conv2d 32→128<br/>(5,1) stride (3,1)"]
        C2 --> C3["Conv2d 128→512<br/>(5,1) stride (3,1)"]
        C3 --> C4["Conv2d 512→1024<br/>(5,1) stride (3,1)"]
        C4 --> C5["Conv2d 1024→1<br/>(3,1) stride 1"]
    end
    
    C5 --> OUT[Score + Features]
```

**Why Period Discriminator?**
- Voiced speech is quasi-periodic (pitch period = 1/F0)
- Period 2 at 16kHz → detects 8kHz artifacts
- Period 3 → detects 5.3kHz artifacts
- Prime numbers ensure diverse, non-overlapping patterns

### Scale Discriminator

Each Scale Discriminator uses 1D convolutions at a different audio resolution.

```mermaid
flowchart TB
    subgraph ScaleDisc [Scale Discriminator]
        A[Audio] --> C1[Conv1d 1→128<br/>k=15, s=1]
        C1 --> C2[Conv1d 128→128<br/>k=41, s=2, g=4]
        C2 --> C3[Conv1d 128→256<br/>k=41, s=2, g=16]
        C3 --> C4[Conv1d 256→512<br/>k=41, s=4, g=16]
        C4 --> C5[Conv1d 512→1024<br/>k=41, s=4, g=16]
        C5 --> C6[Conv1d 1024→1024<br/>k=41, s=1, g=16]
        C6 --> C7[Conv1d 1024→1024<br/>k=5, s=1]
        C7 --> OUT[Conv1d 1024→1<br/>k=3, s=1]
    end
```

**Why Multi-Scale?**
- Scale 1 (original): Fine details, high-frequency content
- Scale 2 (2x down): Speech rhythm, envelope
- Scale 3 (4x down): Global structure, long-term dependencies

### Spectral Normalization

All Discriminator layers use **Spectral Normalization**:

```python
self.conv = spectral_norm(nn.Conv1d(...))
```

**Why Spectral Norm?**
- Constrains Lipschitz constant of discriminator
- Prevents discriminator from becoming too powerful
- Stabilizes training (prevents mode collapse)
- Maintains D_loss > 0 throughout training

---

## Pitch Extraction & Conditioning

### The Problem

V1 units are **pitch-invariant** because XLSR-53 Layer 14 discards F0 information. The generator has no way to know what pitch to produce.

### The Solution

Extract pitch from source audio and provide it as a conditioning signal.

```mermaid
flowchart LR
    subgraph Extraction [Pitch Extraction]
        A[Audio] --> PYIN["librosa.pyin<br/>fmin=50Hz, fmax=400Hz"]
        PYIN --> F0[F0 Contour<br/>Hz values]
    end
    
    subgraph Quantization [Pitch Quantization]
        F0 --> LOG[Log Scale]
        LOG --> NORM["Normalize to [0,1]"]
        NORM --> BIN["Bin to 1-32"]
        BIN --> FINAL["Pitch Bins<br/>0=unvoiced, 1-32=voiced"]
    end
    
    subgraph Embedding [In Generator]
        FINAL --> EMB["nn.Embedding(33, 64)"]
        EMB --> CAT[Concatenate with Units]
    end
```

### Pitch Bin Calculation

```python
def quantize_pitch(f0, num_bins=32, f0_min=50.0, f0_max=400.0):
    # Voiced frames only
    voiced = f0 > 0
    
    # Log-scale for perceptual uniformity
    log_f0 = np.log(np.clip(f0[voiced], f0_min, f0_max))
    log_min, log_max = np.log(f0_min), np.log(f0_max)
    
    # Normalize to [0, 1] then to [1, num_bins]
    normalized = (log_f0 - log_min) / (log_max - log_min)
    bins = (normalized * (num_bins - 1) + 1).astype(int)
    
    # Unvoiced = 0
    result = np.zeros_like(f0, dtype=int)
    result[voiced] = bins
    return result
```

### Pitch Range

| Bin | Frequency Range | Typical Voice |
|-----|-----------------|---------------|
| 0 | Unvoiced | Silence, consonants |
| 1-8 | 50-80 Hz | Very deep male |
| 9-16 | 80-140 Hz | Male speech |
| 17-24 | 140-220 Hz | Female speech |
| 25-32 | 220-400 Hz | Children, high female |

---

## Loss Functions

The V2 training uses a composite loss function with four components.

### Loss Computation Flow

```mermaid
flowchart TB
    subgraph Inputs
        REAL[Real Audio]
        FAKE[Generated Audio]
    end
    
    subgraph Losses [Loss Functions]
        REAL --> MEL1[Mel Spectrogram]
        FAKE --> MEL2[Mel Spectrogram]
        MEL1 --> MELL["Mel Loss<br/>L1(mel_fake, mel_real)"]
        MEL2 --> MELL
        
        REAL --> STFT1[Multi-Res STFT]
        FAKE --> STFT2[Multi-Res STFT]
        STFT1 --> STFTL["STFT Loss<br/>Spectral Conv + Log Mag"]
        STFT2 --> STFTL
        
        FAKE --> D1[Discriminator]
        REAL --> D2[Discriminator]
        D1 --> ADVL["Adversarial Loss<br/>LSGAN"]
        D2 --> FML["Feature Matching<br/>L1(feats_fake, feats_real)"]
        D1 --> FML
    end
    
    subgraph Total [Total Generator Loss]
        MELL --> SUM["G_loss = 45×Mel + 2×STFT + 2×FM + Adv"]
        STFTL --> SUM
        ADVL --> SUM
        FML --> SUM
    end
```

### 1. Mel Spectrogram Loss

```python
def mel_spectrogram_loss(real, fake):
    mel_transform = T.MelSpectrogram(
        sample_rate=16000, n_fft=1024, 
        hop_length=256, n_mels=80
    )
    return F.l1_loss(mel_transform(fake), mel_transform(real))
```

**Purpose**: Ensures phonetic content matches (what is said).

### 2. Multi-Resolution STFT Loss

```python
def stft_loss(real, fake, fft_sizes=[512, 1024, 2048]):
    for fft_size in fft_sizes:
        real_stft = torch.stft(real, fft_size, ...)
        fake_stft = torch.stft(fake, fft_size, ...)
        
        # Spectral Convergence
        sc_loss += ||real_mag - fake_mag||_F / ||real_mag||_F
        
        # Log Magnitude
        mag_loss += L1(log(fake_mag), log(real_mag))
    
    return (sc_loss + mag_loss) / len(fft_sizes)
```

**Purpose**: Captures harmonic structure at multiple time-frequency resolutions.

### 3. Feature Matching Loss

```python
def feature_matching_loss(real_features, fake_features):
    loss = 0
    for real_feat_list, fake_feat_list in zip(real_features, fake_features):
        for real_feat, fake_feat in zip(real_feat_list, fake_feat_list):
            loss += F.l1_loss(fake_feat, real_feat.detach())
    return loss
```

**Purpose**: Stabilizes training by matching intermediate discriminator features.

### 4. Adversarial Loss (LSGAN)

```python
# Discriminator
d_loss = mean((1 - real_scores)^2) + mean(fake_scores^2)

# Generator
g_adv_loss = mean((1 - fake_scores)^2)
```

**Purpose**: Encourages generator to produce realistic audio.

### Loss Weights

| Loss | Weight | Rationale |
|------|--------|-----------|
| Mel | 45.0 | Primary content loss |
| STFT | 2.0 | Harmonic refinement |
| Feature Matching | 2.0 | Training stability |
| Adversarial | 1.0 | Realism |

---

## Training Configuration

### Hyperparameters

```python
# Segment length
segment_length = 32000  # 2 seconds (vs 16000 in V1)
hop_size = 320          # XLSR-53 frame rate
unit_length = 100       # 32000 / 320

# Batch & Learning
batch_size = 12         # Reduced due to larger segments
learning_rate_g = 0.0002
learning_rate_d = 0.0002
betas = (0.8, 0.99)     # Adam betas

# Early Stopping
patience = 100          # Epochs
min_delta = 0.1         # Loss improvement threshold

# Checkpointing
save_every = 25         # Epochs
```

### Training Loop

```mermaid
sequenceDiagram
    participant Data as DataLoader
    participant G as Generator
    participant D as Discriminator
    participant Opt as Optimizers
    
    loop Each Batch
        Data->>G: units, pitch, audio
        
        Note over D: Train Discriminator
        G->>D: fake_audio (detached)
        D->>D: real_scores, fake_scores
        D->>Opt: d_loss.backward()
        Opt->>D: optimizer_d.step()
        
        Note over G: Train Generator
        G->>D: fake_audio
        D->>G: fake_scores, features
        G->>G: Compute all losses
        G->>Opt: g_loss.backward()
        Opt->>G: optimizer_g.step()
    end
```

---

## Comparison: V1 vs V2

### Architecture Comparison

```mermaid
flowchart LR
    subgraph V1 [V1 Generator]
        V1A[Units] --> V1B[Embed 256]
        V1B --> V1C[4x TransConv]
        V1C --> V1D[3x ResBlock]
        V1D --> V1E[Audio]
    end
    
    subgraph V2 [V2 Generator]
        V2A[Units] --> V2B[Embed 256]
        V2P[Pitch] --> V2Q[Embed 64]
        V2B --> V2C[Concat 320]
        V2Q --> V2C
        V2C --> V2D[4x TransConv + MRF]
        V2D --> V2E[Audio]
    end
    
    style V2P fill:#c8e6c9
    style V2Q fill:#c8e6c9
```

### Expected Quality Improvements

| Metric | V1 (Baseline) | V2 (Expected) | Improvement |
|--------|---------------|---------------|-------------|
| **SNR** | ~0 dB | > 5 dB | Better signal quality |
| **MCD** | ~80+ | < 10 | Much better spectral match |
| **F0 RMSE** | N/A | < 20 Hz | Pitch accuracy |
| **Naturalness** | 1-2/5 | 3-4/5 | Less robotic |

### Run Commands

```bash
# V1 Training (original, simpler)
python3 -m modal run --detach src/training/phase3_vocoder.py::main

# V2 Training (enhanced, recommended)
python3 -m modal run --detach src/training/phase3_vocoder_v2.py::main

# V2 Testing
python3 -m modal run src/training/vocoder_test_v2.py::main --num-samples 50
```

---

## References

- **HiFi-GAN**: [Kong et al., 2020](https://arxiv.org/abs/2010.05646) - Generator architecture
- **MelGAN**: [Kumar et al., 2019](https://arxiv.org/abs/1910.06711) - Multi-scale discriminator
- **PYIN**: [Mauch & Dixon, 2014](https://www.eecs.qmul.ac.uk/~siMDi/papers/2014-ICASSP-pyin.pdf) - Pitch extraction
- **Spectral Norm**: [Miyato et al., 2018](https://arxiv.org/abs/1802.05957) - Discriminator stabilization
