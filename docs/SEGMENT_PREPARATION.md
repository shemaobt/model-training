# Segment Size Impact & Preparation Guide

## 📋 Overview

This document explains how **audio segment size and quality** critically impact training performance, memory usage, and model capacity. Proper segment preparation is essential for successful vocoder training.

---

## 🎯 Current Segmentation Strategy

### What We Do Now

1. **Pre-segmentation** (Local, before upload):
   - Silence-based segmentation using RMS energy
   - Minimum segment: **2 seconds**
   - Maximum segment: **120 seconds**
   - Splits long segments at natural pauses

2. **Training-time Segmentation** (Modal, during training):
   - Fixed window: **16,000 samples = 1 second** at 16kHz
   - Random cropping from longer segments
   - Unit alignment: **50 units per segment** (16000 / 320 hop_size)

### Current Parameters

```python
# Pre-segmentation (scripts/segment_audio.py)
MIN_SEGMENT_DURATION = 2.0   # seconds
MAX_SEGMENT_DURATION = 120.0  # seconds
SILENCE_THRESHOLD = -40       # dB
MIN_SILENCE_DURATION = 0.5    # seconds

# Training segmentation (phase3_vocoder.py)
segment_length = 16000        # samples (1 second)
hop_size = 320                # XLSR-53 frame rate
unit_length = 50              # units per segment (16000 / 320)
```

---

## 🔍 How Segment Size Impacts Training

### 1. **Memory Constraints**

#### GPU Memory Usage

**Formula:**
```
Memory per batch = batch_size × (
    audio_memory + 
    unit_memory + 
    model_memory + 
    gradient_memory
)
```

**Current Setup:**
- Segment length: **16,000 samples** (1 second)
- Batch size: **16**
- Audio memory: `16 × 16,000 × 4 bytes = 1.024 MB`
- Unit memory: `16 × 50 × 4 bytes = 3.2 KB`
- Model memory: ~**500 MB** (generator + discriminator)
- Gradient memory: ~**500 MB** (same as model)
- **Total per batch: ~1 GB**

#### Impact of Changing Segment Length

| Segment Length | Duration | Units | Memory/Batch | Max Batch Size (24GB GPU) |
|---------------|----------|-------|--------------|---------------------------|
| 8,000 samples | 0.5 sec  | 25    | ~0.6 GB      | **32-40**                 |
| 16,000 samples| 1.0 sec  | 50    | ~1.0 GB      | **16-20** (current)       |
| 32,000 samples| 2.0 sec  | 100   | ~1.8 GB      | **8-12**                  |
| 48,000 samples| 3.0 sec  | 150   | ~2.5 GB      | **6-8**                   |

**Key Insight**: Longer segments = larger batches possible, but more memory per sample.

---

### 2. **Model Capacity & Context**

#### What the Model Can Learn

**Current Architecture:**
- Generator: **No attention mechanism**
- Receptive field: Limited by convolution kernel sizes
- Context window: ~**200-300ms** (local patterns only)

**Segment Length Implications:**

| Segment Length | Context Available | What Model Learns |
|---------------|-------------------|-------------------|
| 0.5 sec (8K)  | Very short        | Phoneme-level patterns, basic transitions |
| 1.0 sec (16K) | Short             | Word-level patterns, simple prosody (current) |
| 2.0 sec (32K) | Medium            | Phrase-level patterns, better prosody |
| 3.0+ sec (48K+)| Long              | Sentence-level patterns, full prosody |

**Current Limitation:**
- 1-second segments are **too short** for learning:
  - Natural prosody (intonation patterns)
  - Phrase-level rhythm
  - Sentence-level stress patterns

**Trade-off:**
- Longer segments = better prosody learning
- But: Model architecture must support it (attention/RNN)

---

### 3. **Alignment Issues**

#### Critical Problem: Unit-Audio Alignment

**The Challenge:**
- XLSR-53 extracts features at **20ms intervals** (320 samples)
- Units are assigned per 20ms frame
- Audio segments must be **perfectly aligned** with unit sequences

**Current Implementation:**
```python
# Random cropping from longer segments
audio_start = random_position_in_segment
units_start = audio_start // 320  # Align to unit boundaries
```

**Potential Issues:**

1. **Misalignment**:
   - If `audio_start` is not a multiple of 320, units and audio don't match
   - Example: `audio_start = 100` → `units_start = 0` (off by 100 samples!)

2. **Boundary Effects**:
   - Cropping in the middle of a word/phoneme
   - Units at boundaries may not correspond to audio boundaries

3. **Padding Artifacts**:
   - Short segments padded with zeros
   - Model learns to generate silence at segment ends

**Solution**: Always align to unit boundaries:
```python
# Correct alignment
unit_start = random_unit_index
audio_start = unit_start * 320  # Always multiple of 320
```

---

### 4. **Training Efficiency**

#### Batch Diversity

**Current Approach:**
- Random cropping from segments
- Each epoch sees different crops from same segments
- Good for data augmentation

**Impact of Segment Length:**

| Segment Length | Unique Crops per Segment | Training Diversity |
|---------------|--------------------------|-------------------|
| 0.5 sec       | 1 crop                    | Low (no augmentation) |
| 1.0 sec       | 1 crop                    | Low (current) |
| 2.0 sec       | ~2-3 crops                 | Medium |
| 5.0 sec       | ~5-10 crops                | High |

**Recommendation**: Use longer segments (2-5 seconds) for better diversity.

---

### 5. **Quality of Pre-segmentation**

#### Silence-Based Segmentation

**Current Parameters:**
```python
SILENCE_THRESHOLD = -40 dB
MIN_SILENCE_DURATION = 0.5 seconds
MIN_SEGMENT_DURATION = 2.0 seconds
MAX_SEGMENT_DURATION = 120.0 seconds
```

**Impact on Training:**

✅ **Good Segments:**
- Natural pauses (sentence boundaries)
- Complete phrases/words
- No mid-word cuts
- Consistent audio level

❌ **Bad Segments:**
- Cuts in middle of words
- Very short segments (< 1 second)
- Very long segments (> 10 seconds)
- Inconsistent audio levels
- Background noise

**Common Issues:**

1. **Mid-word Cuts**:
   - Silence detection splits words incorrectly
   - Example: "Hel-lo" instead of "Hello"
   - **Impact**: Model learns broken phoneme sequences

2. **Inconsistent Boundaries**:
   - Some segments start with silence
   - Others start mid-speech
   - **Impact**: Model confused about segment starts

3. **Length Variation**:
   - Segments range from 2-120 seconds
   - Most training uses 1-second crops
   - **Impact**: Wasted data, inconsistent context

---

## 🛠️ Best Practices for Segment Preparation

### 1. **Optimal Segment Length for Training**

#### Recommendation: **2-5 seconds**

**Why:**
- Long enough for prosody patterns
- Short enough for GPU memory
- Allows multiple crops per segment
- Natural phrase/sentence boundaries

**Implementation:**
```python
# Update segmentation parameters
MIN_SEGMENT_DURATION = 2.0   # Keep minimum
MAX_SEGMENT_DURATION = 5.0   # Reduce from 120.0
```

**Training-time cropping:**
```python
# Use 1-2 second crops from 2-5 second segments
segment_length = 16000  # 1 second (or 32000 for 2 seconds)
```

---

### 2. **Alignment Strategy**

#### Always Align to Unit Boundaries

**Correct Implementation:**
```python
def __getitem__(self, idx):
    sample = self.samples[idx % len(self.samples)]
    audio, sr = sf.read(sample['audio_path'])
    units = sample['units']
    
    # Calculate max valid start (aligned to units)
    max_units = len(units) - self.unit_length
    max_audio_samples = len(audio) - self.segment_length
    
    if max_units <= 0 or max_audio_samples <= 0:
        # Pad if too short
        ...
    
    # Choose random unit index
    unit_start = np.random.randint(0, max(1, max_units))
    
    # Align audio to unit boundary
    audio_start = unit_start * self.hop_size  # 320
    
    # Extract aligned segments
    audio_seg = audio[audio_start:audio_start + self.segment_length]
    units_seg = units[unit_start:unit_start + self.unit_length]
    
    return {'audio': audio_seg, 'units': units_seg}
```

**Key Points:**
- ✅ Always start audio at `unit_index × 320`
- ✅ Never crop audio at non-unit boundaries
- ✅ Ensure `len(units_seg) × 320 == len(audio_seg)`

---

### 3. **Pre-segmentation Quality**

#### Improve Silence Detection

**Current Issues:**
- Fixed threshold (-40 dB) may not work for all audio
- Doesn't account for background noise
- May split words incorrectly

**Better Approach:**
```python
# Adaptive threshold based on audio statistics
rms_db = librosa.power_to_db(rms**2, ref=np.max)
mean_energy = np.mean(rms_db)
std_energy = np.std(rms_db)

# Adaptive threshold: mean - 2*std (more robust)
SILENCE_THRESHOLD = mean_energy - 2 * std_energy

# Or use percentile-based threshold
SILENCE_THRESHOLD = np.percentile(rms_db, 10)  # Bottom 10%
```

**Additional Improvements:**
1. **Voice Activity Detection (VAD)**:
   - Use more sophisticated VAD (e.g., `webrtcvad`)
   - Better than simple energy threshold

2. **Phoneme-Aware Segmentation**:
   - Use forced alignment (if transcripts available)
   - Split at word boundaries, not arbitrary silence

3. **Quality Filtering**:
   - Remove segments with:
     - Too much silence (> 30%)
     - Too quiet (mean RMS < -50 dB)
     - Too noisy (high zero-crossing rate)

---

### 4. **Segment Length Strategy by Phase**

#### Phase 1: Acoustic Tokenization

**Current**: Uses full segments (2-120 seconds)
**Impact**: 
- ✅ Good for learning unit vocabulary
- ❌ Long segments may cause memory issues
- ❌ Inconsistent context

**Recommendation**: 
- Keep current approach (full segments)
- But reduce `MAX_SEGMENT_DURATION` to 10-15 seconds
- Filter out very short segments (< 1 second)

---

#### Phase 2: BPE Training

**Current**: Uses full unit sequences
**Impact**:
- ✅ Good for learning motifs
- ✅ No audio needed (just units)

**Recommendation**:
- Keep current approach
- No changes needed

---

#### Phase 3: Vocoder Training

**Current**: 1-second crops from longer segments
**Impact**:
- ❌ Too short for prosody
- ❌ Limited context
- ✅ Fits in GPU memory

**Recommendation**:
- **Increase to 2 seconds** (32,000 samples)
- Use 2-5 second pre-segments
- Crop 2-second windows during training
- Align perfectly to unit boundaries

**Memory Impact**:
- 2-second segments: ~1.8 GB/batch
- Max batch size: 12-16 (still feasible on A10G)

---

### 5. **Data Augmentation**

#### Current: Random Cropping

**What We Do:**
- Random start position within segment
- Different crop each epoch

**Improvements:**

1. **Stride-Based Cropping**:
   ```python
   # Instead of random, use fixed strides
   stride = segment_length // 2  # 50% overlap
   crops = [0, stride, 2*stride, ...]
   ```

2. **Boundary-Aware Cropping**:
   - Prefer crops that start/end at word boundaries
   - Avoid mid-word cuts

3. **Length Variation**:
   - Train with multiple segment lengths (1s, 2s, 3s)
   - Helps model generalize

---

## 📊 Recommended Configuration

### Pre-segmentation (scripts/segment_audio.py)

```python
# Optimal parameters
MIN_SEGMENT_DURATION = 2.0    # Minimum 2 seconds
MAX_SEGMENT_DURATION = 5.0    # Maximum 5 seconds (reduced from 120)
SILENCE_THRESHOLD = "adaptive"  # Use adaptive threshold
MIN_SILENCE_DURATION = 0.3    # Slightly shorter (0.3s instead of 0.5s)

# Quality filters
MIN_ENERGY_DB = -50           # Reject very quiet segments
MAX_SILENCE_RATIO = 0.3       # Reject segments with >30% silence
```

### Training Segmentation (phase3_vocoder.py)

```python
# Recommended for Phase 3 (Vocoder)
segment_length = 32000         # 2 seconds (instead of 16000)
hop_size = 320                # Keep same
unit_length = 100              # 100 units per segment (32000 / 320)

# Batch size adjustment
batch_size = 12                # Reduce from 16 (due to longer segments)
```

**Memory Check:**
- 2-second segments: ~1.8 GB/batch
- Batch size 12: ~21.6 GB total
- ✅ Fits in A10G (24 GB)

---

## 🎯 Implementation Checklist

### Pre-segmentation Improvements

- [ ] Reduce `MAX_SEGMENT_DURATION` to 5 seconds
- [ ] Implement adaptive silence threshold
- [ ] Add quality filters (energy, silence ratio)
- [ ] Validate segment boundaries (no mid-word cuts)
- [ ] Log segment statistics (length distribution)

### Training Improvements

- [ ] Increase `segment_length` to 32000 (2 seconds)
- [ ] Fix alignment to always use unit boundaries
- [ ] Update `unit_length` calculation (32000 / 320 = 100)
- [ ] Adjust batch size (12 instead of 16)
- [ ] Add validation for unit-audio alignment

### Validation

- [ ] Check alignment: `len(audio) == len(units) * 320`
- [ ] Verify no padding artifacts in training
- [ ] Monitor memory usage with new segment length
- [ ] Compare training metrics (loss, quality) before/after

---

## 📈 Expected Improvements

### After Implementing Recommendations

1. **Better Prosody**:
   - 2-second segments capture phrase-level patterns
   - Model learns intonation better

2. **More Stable Training**:
   - Perfect alignment prevents confusion
   - Consistent segment quality

3. **Better Data Utilization**:
   - 2-5 second segments → multiple crops
   - More training diversity

4. **Reduced Artifacts**:
   - Quality filtering removes bad segments
   - Better boundaries reduce edge effects

---

## ⚠️ Trade-offs

### Longer Segments

**Upside:**
- Better prosody learning
- More context for model
- Better alignment with natural speech

**Downside:**
- More GPU memory
- Smaller batch sizes
- Longer training time per epoch

### Shorter Segments

**Upside:**
- Less memory
- Larger batches
- Faster training

**Downside:**
- Limited prosody
- Less context
- May miss phrase-level patterns

**Recommendation**: **2 seconds** is the sweet spot for current architecture.

---

## 📏 Maximum Segment Size Analysis

### GPU Memory Constraints

The primary constraint for segment size is **GPU memory**. Here's an analysis for the A10G (24GB):

#### Memory Usage Formula

```
Total Memory ≈ 
    Model Parameters (G + D) +
    Audio Batch (batch × segment × 4 bytes) +
    Generator Activations (grows with segment) +
    Discriminator Activations (grows with segment) +
    Gradients (≈ same as activations)
```

#### Practical Limits on A10G (24GB)

| Segment Length | Duration | Units | Max Batch Size | Memory Usage | Notes |
|---------------|----------|-------|----------------|--------------|-------|
| 16,000 | 1.0 sec | 50 | 16-20 | ~16 GB | V1 default |
| 32,000 | 2.0 sec | 100 | 10-12 | ~20 GB | **V2 default** |
| 48,000 | 3.0 sec | 150 | 6-8 | ~22 GB | Good for phrases |
| 64,000 | 4.0 sec | 200 | 4-6 | ~23 GB | Sentence-level |
| 80,000 | 5.0 sec | 250 | 3-4 | ~24 GB | Near limit |
| 160,000 | 10.0 sec | 500 | 1-2 | OOM | Requires changes |

#### Practical Maximum: **5 seconds (80,000 samples)**

Beyond 5 seconds on A10G, you need architectural changes (see below).

### Effective Batch Size

Even with smaller batch sizes, you can maintain training stability with **gradient accumulation**:

```python
# Accumulate gradients over multiple mini-batches
accumulation_steps = 4
for i, batch in enumerate(dataloader):
    loss = compute_loss(batch)
    loss = loss / accumulation_steps  # Normalize
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

This gives an effective batch size of `batch_size × accumulation_steps`.

---

## 🏗️ Architectural Changes for Larger Segments

If your use case requires segments **longer than 5 seconds**, you need architectural changes:

### 1. Mixed Precision Training (Easy, +40% memory savings)

Use float16 instead of float32 for most computations.

```python
# Enable automatic mixed precision
scaler = torch.cuda.amp.GradScaler()

with torch.cuda.amp.autocast():
    fake_audio = generator(units, pitch)
    loss = compute_loss(fake_audio, real_audio)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

**Impact**: 
- ✅ Reduces memory by ~40%
- ✅ Faster training (Tensor Cores)
- ✅ Minimal code changes
- Allows: **~8 seconds** on A10G

### 2. Gradient Checkpointing (Medium, +50% memory savings)

Trade compute for memory by recomputing activations during backward pass.

```python
from torch.utils.checkpoint import checkpoint

class GeneratorWithCheckpoint(nn.Module):
    def forward(self, units, pitch):
        x = self.pre_conv(self.embed(units, pitch))
        
        # Checkpoint each upsample block
        for up, mrf in zip(self.ups, self.mrfs):
            x = checkpoint(lambda x: mrf(up(x)), x)
        
        return self.post_conv(x)
```

**Impact**:
- ✅ Reduces memory by ~50%
- ❌ Slower training (recomputation)
- ✅ Works with any segment length
- Allows: **~15 seconds** on A10G

### 3. Chunked/Streaming Processing (Hard, unlimited segment length)

Process audio in chunks during inference (and optionally training).

```python
def generate_chunked(units, pitch, chunk_size=100, overlap=10):
    """Generate audio in overlapping chunks and blend."""
    outputs = []
    
    for i in range(0, len(units), chunk_size - overlap):
        chunk_units = units[i:i + chunk_size]
        chunk_pitch = pitch[i:i + chunk_size]
        
        chunk_audio = generator(chunk_units, chunk_pitch)
        outputs.append(chunk_audio)
    
    # Cross-fade overlapping regions
    return blend_chunks(outputs, overlap * 320)
```

**Impact**:
- ✅ Unlimited segment length
- ❌ Complex implementation
- ⚠️ Potential artifacts at chunk boundaries
- ⚠️ Requires careful overlap handling

### 4. Attention Mechanisms (Hard, better long-range context)

Add Transformer layers for global context. Current convolutions have limited receptive field.

```python
class AttentiveGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        # ... existing layers ...
        
        # Add attention after upsampling
        self.attention = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=256, nhead=8),
            num_layers=2
        )
    
    def forward(self, x):
        # ... upsampling ...
        
        # Apply attention at bottleneck (before final upsampling)
        x = x.transpose(1, 2)  # [B, T, C]
        x = self.attention(x)
        x = x.transpose(1, 2)  # [B, C, T]
        
        # ... rest of generation ...
```

**Impact**:
- ✅ Global context (entire segment)
- ❌ Quadratic memory with sequence length
- ❌ Significantly slower
- ✅ Better prosody modeling

### 5. Larger GPU (Easiest, if budget allows)

| GPU | VRAM | Max Segment (batch=4) |
|-----|------|----------------------|
| A10G | 24 GB | ~5 seconds |
| A100 (40GB) | 40 GB | ~10 seconds |
| A100 (80GB) | 80 GB | ~20 seconds |
| H100 | 80 GB | ~25 seconds (faster) |

---

## 🤔 Does Larger Segments Solve Robotic Audio?

### Short Answer: **Partially, but not the main solution.**

### Analysis

The "robotic audio" problem has **multiple causes**:

| Cause | Impact | Solution | Segment Size Helps? |
|-------|--------|----------|-------------------|
| **Pitch loss** | Major | Pitch conditioning | ❌ No |
| **Discriminator collapse** | Major | Spectral norm + better losses | ❌ No |
| **Limited context** | Minor | Longer segments | ✅ Yes, but diminishing returns |
| **Poor training** | Variable | Better hyperparameters | ❌ No |

### Context Requirements for Different Patterns

| Pattern | Typical Duration | Segment Needed |
|---------|------------------|----------------|
| Phoneme | 50-200 ms | 0.5 sec |
| Syllable | 100-400 ms | 1.0 sec |
| Word | 200-800 ms | 1.5 sec |
| Phrase | 1-3 sec | **2-3 sec** |
| Sentence | 2-5 sec | **3-5 sec** |
| Paragraph | 5-15 sec | 10+ sec |

### Recommendation

```
Segment Length vs Improvement Curve:

Prosody
Quality
  ^
  |                    _______________
  |                ___/
  |            ___/
  |        ___/
  |    ___/
  |___/
  +---------------------------------> Segment Length
     0.5s  1s   2s   3s   5s   10s
          ↑              ↑
        V1 default    Diminishing returns
```

**Key Insights:**
1. **1 → 2 seconds**: Significant improvement (phrase-level patterns)
2. **2 → 3 seconds**: Moderate improvement (better sentence flow)
3. **3 → 5 seconds**: Small improvement (sentence boundaries)
4. **5+ seconds**: Diminishing returns, memory becomes limiting

### Final Recommendation

| Segment Length | Recommendation |
|---------------|----------------|
| **2 seconds** | ✅ Best balance for most use cases |
| **3 seconds** | ✅ If memory allows, slight improvement |
| **5 seconds** | ⚠️ Only if specific need for long prosody |
| **10+ seconds** | ❌ Not recommended (requires arch changes, minimal benefit) |

**The V2 improvements (pitch conditioning, enhanced losses) will have a much bigger impact on audio quality than increasing segment length beyond 3 seconds.**

---

## 📋 Implementation Checklist for Larger Segments

### For 3-5 second segments (current architecture)

- [ ] Reduce batch size to 4-6
- [ ] Enable gradient accumulation (4-8 steps)
- [ ] Monitor GPU memory usage
- [ ] Adjust learning rate (may need lower)
- [ ] Increase patience (longer to converge)

### For 5-10 second segments (with optimizations)

- [ ] Enable mixed precision (AMP)
- [ ] Reduce batch size to 2-3
- [ ] Enable gradient accumulation (8-16 steps)
- [ ] Consider gradient checkpointing
- [ ] Monitor for training instability

### For 10+ second segments (major changes)

- [ ] Implement chunked processing
- [ ] Add attention mechanisms
- [ ] Consider A100 GPU
- [ ] Extensive testing required
- [ ] May need custom training loop

---

## 🔗 Related Documents

- [ARCHITECTURE.md](ARCHITECTURE.md) - Model architecture details
- [ROBOTIC_AUDIO_ANALYSIS.md](ROBOTIC_AUDIO_ANALYSIS.md) - Why audio sounds robotic
- [PIPELINE.md](PIPELINE.md) - Full pipeline execution guide

---

## 📝 Summary

**Key Takeaways:**

1. **Segment length directly impacts**:
   - GPU memory usage
   - Model capacity (what it can learn)
   - Training efficiency

2. **Current setup (1 second)** is:
   - ✅ Memory efficient
   - ❌ Too short for prosody
   - ❌ Limited context

3. **Recommended changes**:
   - Pre-segmentation: 2-5 seconds (instead of 2-120)
   - Training: 2 seconds (instead of 1)
   - Perfect alignment to unit boundaries
   - Quality filtering for better segments

4. **Expected improvement**:
   - Better prosody learning
   - More natural audio
   - More stable training
