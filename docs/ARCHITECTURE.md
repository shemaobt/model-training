# Architecture Deep Dive & Design Decisions

This document provides a comprehensive technical analysis of all models, algorithms, and architectural decisions in the acoustic tokenization and vocoder pipeline. It includes exact layer specifications, dimensions, and the reasoning behind each design choice.

## Table of Contents

1. [XLSR-53 Feature Extraction](#xlsr-53-feature-extraction)
2. [K-Means Acoustic Clustering](#k-means-acoustic-clustering)
3. [V1 Generator Architecture](#v1-generator-architecture)
4. [V1 Discriminator Architecture](#v1-discriminator-architecture)
5. [V2 Generator Architecture](#v2-generator-architecture)
6. [V2 Discriminator Architecture](#v2-discriminator-architecture)
7. [Loss Functions](#loss-functions)
8. [Training Dynamics](#training-dynamics)

---

## XLSR-53 Feature Extraction

### Model Overview

We use `facebook/wav2vec2-large-xlsr-53`, a self-supervised model pre-trained on 53 languages. Unlike ASR models, it learned speech structure from raw audio without text labels.

### Architecture Specification

| Component | Specification |
|-----------|---------------|
| **Model** | wav2vec 2.0 XLSR-53 |
| **Parameters** | 315 million |
| **Input** | Raw audio, 16kHz mono |
| **CNN Encoder** | 7 convolutional layers |
| **Transformer** | 24 layers, 1024-dim hidden, 16 attention heads |
| **Output** | 1024-dimensional vectors per frame |

### CNN Feature Encoder

The CNN encoder downsamples raw audio by a factor of 320:

| Layer | Kernel | Stride | Channels | Output Rate |
|-------|--------|--------|----------|-------------|
| Conv1 | 10 | 5 | 512 | 3,200 Hz |
| Conv2 | 3 | 2 | 512 | 1,600 Hz |
| Conv3 | 3 | 2 | 512 | 800 Hz |
| Conv4 | 3 | 2 | 512 | 400 Hz |
| Conv5 | 3 | 2 | 512 | 200 Hz |
| Conv6 | 2 | 2 | 512 | 100 Hz |
| Conv7 | 2 | 2 | 512 | 50 Hz |

**Total downsampling**: 5 × 2 × 2 × 2 × 2 × 2 × 2 = **320x**

### Frame Rate Calculation

```
1 second audio = 16,000 samples
16,000 / 320 = 50 frames per second
Frame duration = 20ms per frame
```

Each 1024-dimensional vector represents 20ms of speech context.

### Layer Selection: Why Layer 14?

We extract features from **Layer 14** of 24 transformer layers.

| Layer Range | Information Captured | Use Case |
|-------------|---------------------|----------|
| **1-6** | Acoustic (pitch, energy, timbre) | Speaker ID |
| **7-12** | Sub-phonetic (articulation) | Fine-grained ASR |
| **13-18** | Phonemic (what is said) | **Our choice** |
| **19-24** | Semantic (word context) | Language understanding |

**Why Layer 14?**
- Highest mutual information with phone boundaries across languages
- Abstracts away speaker identity to focus on phonetic content
- Robust to noise and speaker variation

**Trade-off**: Layer 14 discards prosodic information (pitch, rhythm), causing the "robotic" voice in V1.

---

## K-Means Acoustic Clustering

### Quantization Process

We transform continuous 1024-dim vectors into discrete integers (0-99).

| Stage | Dimensions | Bits/Frame |
|-------|------------|------------|
| **Input** | 1024 × float32 | 32,768 bits |
| **Output** | 1 integer (0-99) | ~7 bits |
| **Compression** | 4,600:1 | |

### K-Means Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **k (clusters)** | 100 | Covers ~phonemes + allophones |
| **Algorithm** | Mini-batch K-Means | Scalable to millions of frames |
| **Distance** | Euclidean | Standard for continuous features |
| **Initialization** | k-means++ | Faster convergence |
| **Batch size** | 10,000 | Memory-efficient |

### Why 100 Clusters?

Human languages typically have 30-50 phonemes. We use 100 to capture:
- Core phonemes (~40)
- Allophones (phoneme variants, ~30)
- Transitions (co-articulation, ~30)

**Trade-off**: Too few clusters → phoneme confusion. Too many → speaker-specific tokens.

---

## V1 Generator Architecture

The V1 generator converts discrete units to audio waveforms using transposed convolutions.

### Layer-by-Layer Specification

```
Input: Unit indices [B, T] where T = sequence length

Layer 1: Unit Embedding
  nn.Embedding(100, 256)
  Output: [B, T, 256]

Layer 2: Transpose
  [B, T, 256] → [B, 256, T]

Layer 3: Pre-Convolution
  nn.Conv1d(256, 512, kernel=7, padding=3)
  nn.LeakyReLU(0.1)
  Output: [B, 512, T]

Layer 4: Upsample Block 1 (5x)
  nn.ConvTranspose1d(512, 256, kernel=10, stride=5, padding=2)
  nn.LeakyReLU(0.1)
  nn.Conv1d(256, 256, kernel=7, padding=3)
  nn.LeakyReLU(0.1)
  Output: [B, 256, T×5]

Layer 5: Upsample Block 2 (4x)
  nn.ConvTranspose1d(256, 128, kernel=8, stride=4, padding=2)
  nn.LeakyReLU(0.1)
  nn.Conv1d(128, 128, kernel=7, padding=3)
  nn.LeakyReLU(0.1)
  Output: [B, 128, T×20]

Layer 6: Upsample Block 3 (4x)
  nn.ConvTranspose1d(128, 64, kernel=8, stride=4, padding=2)
  nn.LeakyReLU(0.1)
  nn.Conv1d(64, 64, kernel=7, padding=3)
  nn.LeakyReLU(0.1)
  Output: [B, 64, T×80]

Layer 7: Upsample Block 4 (4x)
  nn.ConvTranspose1d(64, 32, kernel=8, stride=4, padding=2)
  nn.LeakyReLU(0.1)
  nn.Conv1d(32, 32, kernel=7, padding=3)
  nn.LeakyReLU(0.1)
  Output: [B, 32, T×320]

Layer 8-10: Residual Blocks (×3)
  nn.Conv1d(32, 32, kernel=3, padding=1)
  nn.LeakyReLU(0.1)
  nn.Conv1d(32, 32, kernel=3, padding=1)
  + skip connection (scaled by 0.1)
  Output: [B, 32, T×320]

Layer 11: Post-Convolution
  nn.Conv1d(32, 1, kernel=7, padding=3)
  nn.Tanh()
  Output: [B, 1, T×320]

Layer 12: Squeeze
  [B, 1, T×320] → [B, T×320]

Output: Audio waveform [B, T×320]
```

### V1 Generator Summary

| Property | Value |
|----------|-------|
| **Total Parameters** | ~3.3 million |
| **Upsampling Factor** | 5 × 4 × 4 × 4 = 320 |
| **Input** | 50 units → 50 × 320 = 16,000 samples (1 sec) |
| **Activation** | LeakyReLU(0.1) |
| **Output Activation** | Tanh ([-1, 1]) |

---

## V1 Discriminator Architecture

The V1 discriminator uses multi-scale analysis to assess audio quality.

### Multi-Scale Discriminator (MSD)

Three identical discriminators operating at different resolutions:

```
Scale 1: Original audio (16kHz)
Scale 2: 2x downsampled (8kHz effective)
Scale 3: 4x downsampled (4kHz effective)
```

### Single Scale Discriminator Layers

```
Input: Audio [B, 1, T]

Layer 1:
  nn.Conv1d(1, 16, kernel=15, stride=1, padding=7)
  nn.LeakyReLU(0.1)
  Output: [B, 16, T]

Layer 2:
  nn.Conv1d(16, 64, kernel=41, stride=4, padding=20, groups=4)
  nn.LeakyReLU(0.1)
  Output: [B, 64, T/4]

Layer 3:
  nn.Conv1d(64, 256, kernel=41, stride=4, padding=20, groups=16)
  nn.LeakyReLU(0.1)
  Output: [B, 256, T/16]

Layer 4:
  nn.Conv1d(256, 512, kernel=41, stride=4, padding=20, groups=16)
  nn.LeakyReLU(0.1)
  Output: [B, 512, T/64]

Layer 5:
  nn.Conv1d(512, 512, kernel=5, stride=1, padding=2)
  nn.LeakyReLU(0.1)
  Output: [B, 512, T/64]

Layer 6:
  nn.Conv1d(512, 1, kernel=3, stride=1, padding=1)
  Output: [B, 1, T/64] → flatten → [B, T/64]
```

### Downsampling Between Scales

```python
nn.AvgPool1d(kernel_size=4, stride=2, padding=1)
```

### V1 Discriminator Summary

| Property | Value |
|----------|-------|
| **Total Parameters** | ~5.1 million |
| **Scales** | 3 (original, 2x, 4x down) |
| **Grouped Convolutions** | Yes (efficiency) |
| **Kernel Size** | 41 (large receptive field) |

---

## V2 Generator Architecture

The V2 generator is a HiFi-GAN style architecture with pitch conditioning.

### Key Improvements Over V1

| Feature | V1 | V2 |
|---------|----|----|
| Pitch Input | None | 32-bin F0 embedding |
| Normalization | None | Weight Normalization |
| Residual Blocks | Simple 2-conv | Multi-Receptive Field Fusion |
| Dilation | None | Multi-rate dilations |

### Layer-by-Layer Specification

```
Inputs:
  - Units: [B, T] integers in [0, 99]
  - Pitch: [B, T] integers in [0, 32] (0 = unvoiced)

Layer 1a: Unit Embedding
  nn.Embedding(100, 256)
  Output: [B, T, 256]

Layer 1b: Pitch Embedding
  nn.Embedding(33, 64)  # 33 = 32 bins + 1 unvoiced
  Output: [B, T, 64]

Layer 2: Concatenate
  torch.cat([unit_emb, pitch_emb], dim=-1)
  Output: [B, T, 320]

Layer 3: Transpose
  [B, T, 320] → [B, 320, T]

Layer 4: Pre-Convolution
  weight_norm(nn.Conv1d(320, 512, kernel=7, padding=3))
  Output: [B, 512, T]

Layer 5: Upsample Block 1 (5x) + MRF
  weight_norm(nn.ConvTranspose1d(512, 256, kernel=10, stride=5, padding=2))
  + MultiReceptiveFieldFusion(256, kernels=[3, 7, 11])
  Output: [B, 256, T×5]

Layer 6: Upsample Block 2 (4x) + MRF
  weight_norm(nn.ConvTranspose1d(256, 128, kernel=8, stride=4, padding=2))
  + MultiReceptiveFieldFusion(128, kernels=[3, 7, 11])
  Output: [B, 128, T×20]

Layer 7: Upsample Block 3 (4x) + MRF
  weight_norm(nn.ConvTranspose1d(128, 64, kernel=8, stride=4, padding=2))
  + MultiReceptiveFieldFusion(64, kernels=[3, 7, 11])
  Output: [B, 64, T×80]

Layer 8: Upsample Block 4 (4x) + MRF
  weight_norm(nn.ConvTranspose1d(64, 32, kernel=8, stride=4, padding=2))
  + MultiReceptiveFieldFusion(32, kernels=[3, 7, 11])
  Output: [B, 32, T×320]

Layer 9: Post-Convolution
  nn.LeakyReLU(0.1)
  weight_norm(nn.Conv1d(32, 1, kernel=7, padding=3))
  nn.Tanh()
  Output: [B, 1, T×320]

Layer 10: Squeeze
  [B, 1, T×320] → [B, T×320]

Output: Audio waveform [B, T×320]
```

### Multi-Receptive Field Fusion (MRF) Block

The MRF block applies multiple ResBlocks with different kernel sizes and averages their outputs:

```
Input: x [B, C, L]

Branch 1: ResBlock(kernel=3)  → out1
Branch 2: ResBlock(kernel=7)  → out2
Branch 3: ResBlock(kernel=11) → out3

Output: (out1 + out2 + out3) / 3
```

**Why different kernels?**
- Kernel 3: Captures fine phonetic details (~60 samples = 3.75ms)
- Kernel 7: Captures sub-phonemic patterns (~140 samples = 8.75ms)
- Kernel 11: Captures phoneme-level patterns (~220 samples = 13.75ms)

### ResBlock with Dilations

Each ResBlock uses dilated convolutions to increase receptive field:

```
Input: x [B, C, L]

Dilation Pair 1: d=[1, 1]
  x1 = LeakyReLU(Conv(x, d=1))
  x1 = Conv(x1, d=1)
  x = x + x1

Dilation Pair 2: d=[3, 1]
  x2 = LeakyReLU(Conv(x, d=3))
  x2 = Conv(x2, d=1)
  x = x + x2

Dilation Pair 3: d=[5, 1]
  x3 = LeakyReLU(Conv(x, d=5))
  x3 = Conv(x3, d=1)
  x = x + x3

Output: x
```

### V2 Generator Summary

| Property | Value |
|----------|-------|
| **Total Parameters** | ~4.5 million |
| **Upsampling Factor** | 5 × 4 × 4 × 4 = 320 |
| **Unit Embedding** | 100 → 256 |
| **Pitch Embedding** | 33 → 64 |
| **MRF Kernels** | [3, 7, 11] |
| **Dilations** | [[1,1], [3,1], [5,1]] |
| **Normalization** | Weight Norm |

---

## V2 Discriminator Architecture

The V2 discriminator combines Multi-Period (MPD) and Multi-Scale (MSD) discriminators.

### Combined Architecture

```
Input: Audio [B, T]
           ↓
    ┌──────┴──────┐
    ↓             ↓
   MPD           MSD
(5 periods)   (3 scales)
    ↓             ↓
5 scores     3 scores
5 feat lists 3 feat lists
    ↓             ↓
    └──────┬──────┘
           ↓
Combined: 8 scores, 8 feature lists
```

### Multi-Period Discriminator (MPD)

Uses 2D convolutions on audio reshaped by different periods:

**Periods**: [2, 3, 5, 7, 11] (prime numbers for diverse patterns)

| Period | Detects Frequency | Purpose |
|--------|-------------------|---------|
| 2 | 8 kHz patterns | High-frequency artifacts |
| 3 | 5.3 kHz patterns | Mid-high artifacts |
| 5 | 3.2 kHz patterns | Formant-level |
| 7 | 2.3 kHz patterns | Voice harmonics |
| 11 | 1.5 kHz patterns | Low harmonics |

### Single Period Discriminator Layers

```
Input: Audio [B, 1, T]

Step 1: Reshape by period p
  Pad T to be divisible by p
  Reshape: [B, 1, T] → [B, 1, T/p, p]

Layer 1:
  spectral_norm(nn.Conv2d(1, 32, (5,1), stride=(3,1), padding=(2,0)))
  nn.LeakyReLU(0.1)
  Output: [B, 32, T/(p×3), p]

Layer 2:
  spectral_norm(nn.Conv2d(32, 128, (5,1), stride=(3,1), padding=(2,0)))
  nn.LeakyReLU(0.1)
  Output: [B, 128, T/(p×9), p]

Layer 3:
  spectral_norm(nn.Conv2d(128, 512, (5,1), stride=(3,1), padding=(2,0)))
  nn.LeakyReLU(0.1)
  Output: [B, 512, T/(p×27), p]

Layer 4:
  spectral_norm(nn.Conv2d(512, 1024, (5,1), stride=(3,1), padding=(2,0)))
  nn.LeakyReLU(0.1)
  Output: [B, 1024, T/(p×81), p]

Layer 5:
  spectral_norm(nn.Conv2d(1024, 1024, (5,1), stride=1, padding=(2,0)))
  nn.LeakyReLU(0.1)
  Output: [B, 1024, T/(p×81), p]

Layer 6 (Output):
  spectral_norm(nn.Conv2d(1024, 1, (3,1), stride=1, padding=(1,0)))
  Flatten
  Output: score [B, N], features [list of 6 tensors]
```

### Multi-Scale Discriminator (MSD)

Uses 1D convolutions at 3 resolutions:

```
Scale 1: Original audio
Scale 2: 2x downsampled via AvgPool1d(4, 2, padding=2)
Scale 3: 4x downsampled via AvgPool1d(4, 2, padding=2)
```

### Single Scale Discriminator Layers

```
Input: Audio [B, 1, T]

Layer 1:
  spectral_norm(nn.Conv1d(1, 128, kernel=15, stride=1, padding=7))
  nn.LeakyReLU(0.1)
  Output: [B, 128, T]

Layer 2:
  spectral_norm(nn.Conv1d(128, 128, kernel=41, stride=2, padding=20, groups=4))
  nn.LeakyReLU(0.1)
  Output: [B, 128, T/2]

Layer 3:
  spectral_norm(nn.Conv1d(128, 256, kernel=41, stride=2, padding=20, groups=16))
  nn.LeakyReLU(0.1)
  Output: [B, 256, T/4]

Layer 4:
  spectral_norm(nn.Conv1d(256, 512, kernel=41, stride=4, padding=20, groups=16))
  nn.LeakyReLU(0.1)
  Output: [B, 512, T/16]

Layer 5:
  spectral_norm(nn.Conv1d(512, 1024, kernel=41, stride=4, padding=20, groups=16))
  nn.LeakyReLU(0.1)
  Output: [B, 1024, T/64]

Layer 6:
  spectral_norm(nn.Conv1d(1024, 1024, kernel=41, stride=1, padding=20, groups=16))
  nn.LeakyReLU(0.1)
  Output: [B, 1024, T/64]

Layer 7:
  spectral_norm(nn.Conv1d(1024, 1024, kernel=5, stride=1, padding=2))
  nn.LeakyReLU(0.1)
  Output: [B, 1024, T/64]

Layer 8 (Output):
  spectral_norm(nn.Conv1d(1024, 1, kernel=3, stride=1, padding=1))
  Flatten
  Output: score [B, N], features [list of 8 tensors]
```

### V2 Discriminator Summary

| Property | Value |
|----------|-------|
| **Total Parameters** | ~15 million |
| **MPD Periods** | [2, 3, 5, 7, 11] |
| **MSD Scales** | 3 |
| **Total Discriminators** | 8 (5 + 3) |
| **Normalization** | Spectral Norm |
| **Grouped Convolutions** | Yes (groups=4, 16) |

### Why Spectral Normalization?

Spectral normalization constrains the Lipschitz constant of the discriminator:
- Prevents discriminator from becoming too powerful
- Stabilizes GAN training
- Maintains D_loss > 0 throughout training
- Reduces mode collapse

---

## Loss Functions

### V1 Losses

| Loss | Formula | Weight | Purpose |
|------|---------|--------|---------|
| **Mel Spectrogram** | L1(mel_fake, mel_real) | 45.0 | Phonetic content |
| **Adversarial** | mean((1 - D(fake))²) | 1.0 | Realism |

### V2 Losses (Enhanced)

| Loss | Formula | Weight | Purpose |
|------|---------|--------|---------|
| **Mel Spectrogram** | L1(mel_fake, mel_real) | 45.0 | Phonetic content |
| **Multi-Res STFT** | SC + LogMag at [512, 1024, 2048] | 2.0 | Harmonic structure |
| **Feature Matching** | L1(D_feats_fake, D_feats_real) | 2.0 | Training stability |
| **Adversarial** | mean((1 - D(fake))²) | 1.0 | Realism |

### Mel Spectrogram Loss

```python
mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=16000,
    n_fft=1024,
    hop_length=256,
    n_mels=80,
    f_min=0,
    f_max=8000
)

mel_loss = F.l1_loss(mel_transform(fake), mel_transform(real))
```

### Multi-Resolution STFT Loss

Computed at three FFT sizes for different time-frequency trade-offs:

```python
for fft_size in [512, 1024, 2048]:
    hop = fft_size // 4
    win = fft_size
    
    real_stft = torch.stft(real, fft_size, hop, win, return_complex=True)
    fake_stft = torch.stft(fake, fft_size, hop, win, return_complex=True)
    
    real_mag = torch.abs(real_stft)
    fake_mag = torch.abs(fake_stft)
    
    # Spectral Convergence
    sc_loss += ||real_mag - fake_mag||_F / ||real_mag||_F
    
    # Log Magnitude
    log_loss += L1(log(fake_mag + eps), log(real_mag + eps))

total_stft_loss = (sc_loss + log_loss) / 3
```

### Feature Matching Loss

```python
def feature_matching_loss(real_features, fake_features):
    loss = 0
    for real_feat_list, fake_feat_list in zip(real_features, fake_features):
        for real_feat, fake_feat in zip(real_feat_list, fake_feat_list):
            loss += F.l1_loss(fake_feat, real_feat.detach())
    return loss
```

### Adversarial Loss (LSGAN)

```python
# Discriminator loss
d_loss = mean((1 - real_scores)²) + mean(fake_scores²)

# Generator loss
g_adv_loss = mean((1 - fake_scores)²)
```

---

## Training Dynamics

### V1 Training Configuration

| Parameter | Value |
|-----------|-------|
| Batch Size | 16 |
| Segment Length | 16,000 samples (1 sec) |
| Learning Rate | 0.0002 |
| Optimizer | Adam (β1=0.9, β2=0.999) |
| Epochs | 500 max |
| Early Stopping | patience=50 |

### V2 Training Configuration

| Parameter | Value |
|-----------|-------|
| Batch Size | 12 |
| Segment Length | 32,000 samples (2 sec) |
| Learning Rate (G) | 0.0002 |
| Learning Rate (D) | 0.0002 |
| Optimizer | AdamW (β1=0.8, β2=0.99) |
| Epochs | 1000 max |
| Early Stopping | patience=100 |
| Gradient Clipping | max_norm=5.0 (G only) |

### Training Loop (V2)

```
For each epoch:
    For each batch (units, pitch, audio):
        
        # ===== Train Discriminator =====
        fake_audio = G(units, pitch).detach()
        
        real_scores, real_features = D(audio)
        fake_scores, fake_features = D(fake_audio)
        
        d_loss = discriminator_loss(real_scores, fake_scores)
        d_loss.backward()
        optimizer_d.step()
        
        # ===== Train Generator =====
        fake_audio = G(units, pitch)
        
        fake_scores, fake_features = D(fake_audio)
        _, real_features = D(audio)
        
        mel_loss = mel_spectrogram_loss(audio, fake_audio)
        stft_loss = multi_resolution_stft_loss(audio, fake_audio)
        fm_loss = feature_matching_loss(real_features, fake_features)
        adv_loss = generator_adversarial_loss(fake_scores)
        
        g_loss = 45*mel_loss + 2*stft_loss + 2*fm_loss + adv_loss
        g_loss.backward()
        clip_grad_norm_(G.parameters(), max_norm=5.0)
        optimizer_g.step()
```

### Common Issues & Solutions

| Issue | Symptom | Solution |
|-------|---------|----------|
| **D Collapse** | D_loss → 0 | Add gradient penalty, reduce D lr |
| **Mode Collapse** | All outputs similar | Increase FM loss weight |
| **Robotic Audio** | Flat pitch | Use V2 with pitch conditioning |
| **Metallic Sound** | High-freq artifacts | Increase STFT loss weight |
| **OOM Error** | CUDA memory | Reduce batch size or segment length |

---

## References

- **wav2vec 2.0**: [Baevski et al., 2020](https://arxiv.org/abs/2006.11477)
- **XLSR-53**: [Conneau et al., 2020](https://arxiv.org/abs/2006.13979)
- **HiFi-GAN**: [Kong et al., 2020](https://arxiv.org/abs/2010.05646)
- **MelGAN**: [Kumar et al., 2019](https://arxiv.org/abs/1910.06711)
- **Spectral Norm**: [Miyato et al., 2018](https://arxiv.org/abs/1802.05957)
- **Weight Norm**: [Salimans & Kingma, 2016](https://arxiv.org/abs/1602.07868)
