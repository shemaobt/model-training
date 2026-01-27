# AudioLM Integration Analysis

## How Google's AudioLM Could Transform Our Speech Pipeline

This document provides a comprehensive technical analysis of **AudioLM** (Google Research, 2022) and explores how its architecture could be integrated into or replace components of our current vocoder training pipeline for low-resource languages like Sateré-Mawé.

**Source**: [AudioLM: a Language Modeling Approach to Audio Generation](https://research.google/blog/audiolm-a-language-modeling-approach-to-audio-generation/)

---

## 1. Executive Summary

AudioLM represents a paradigm shift in audio generation by treating speech synthesis as a **language modeling problem**. Instead of our current approach (discrete units → vocoder), AudioLM uses **hierarchical token generation** with Transformers to produce both semantically coherent and acoustically natural audio.

### Key Insight

Our current pipeline:
```
Audio → XLSR-53 → K-Means Units → Vocoder → Reconstructed Audio
```

AudioLM's approach:
```
Audio → w2v-BERT Semantic Tokens + SoundStream Acoustic Tokens
     → Transformer LM (3 stages) → High-Quality Audio
```

AudioLM's key innovation is combining **semantic tokens** (for meaning/structure) with **acoustic tokens** (for quality/speaker identity) in a unified language model.

---

## 2. AudioLM Architecture Deep Dive

### 2.1 The Two-Token System

AudioLM's core innovation is using two complementary token types:

```
┌─────────────────────────────────────────────────────────────────┐
│                        RAW AUDIO WAVEFORM                        │
│                         (16kHz, 1 channel)                       │
└─────────────────────────────────────────────────────────────────┘
                    │                           │
                    ▼                           ▼
    ┌───────────────────────────┐   ┌───────────────────────────┐
    │      SEMANTIC TOKENS      │   │      ACOUSTIC TOKENS      │
    │       (from w2v-BERT)     │   │     (from SoundStream)    │
    ├───────────────────────────┤   ├───────────────────────────┤
    │ • High-level meaning      │   │ • Low-level waveform      │
    │ • Phonetic content        │   │ • Speaker characteristics │
    │ • Linguistic structure    │   │ • Recording conditions    │
    │ • Heavily downsampled     │   │ • Fine acoustic details   │
    │ • 50 tokens/sec           │   │ • 75 tokens/sec × 8 layers│
    │ • ~1024 vocabulary        │   │ • 1024 vocab per layer    │
    └───────────────────────────┘   └───────────────────────────┘
              │                                   │
              └──────────────┬────────────────────┘
                             ▼
              ┌───────────────────────────────┐
              │    HIERARCHICAL TRANSFORMER   │
              │       LANGUAGE MODEL          │
              └───────────────────────────────┘
                             │
                             ▼
              ┌───────────────────────────────┐
              │   HIGH-QUALITY SYNTHESIZED    │
              │           AUDIO               │
              └───────────────────────────────┘
```

#### Semantic Tokens (What we say)

**Extracted from**: w2v-BERT (similar to XLSR-53, self-supervised)  
**Purpose**: Capture linguistic content, phonetics, and long-term structure  
**Properties**:
- Heavily compressed (50 tokens/second)
- Speaker-invariant (same word → same tokens regardless of speaker)
- Captures syntax, semantics, melody structure
- Poor audio reconstruction quality alone

**Comparison to our current approach**:
| Aspect | Our Pipeline (XLSR-53 + K-Means) | AudioLM (w2v-BERT) |
|--------|----------------------------------|-------------------|
| Model | XLSR-53 (300M params) | w2v-BERT (600M params) |
| Tokenization | K-Means clustering (100 units) | Learned RVQ codebook (~1024) |
| Rate | 50 units/sec | 50 tokens/sec |
| Training | Fixed (pretrained) | Self-supervised |

#### Acoustic Tokens (How we say it)

**Extracted from**: SoundStream neural codec  
**Purpose**: Capture fine-grained audio details for high-fidelity reconstruction  
**Properties**:
- Multi-layer Residual Vector Quantization (RVQ)
- 8 quantizer layers, each with 1024 codebook entries
- 75 frames/second × 8 tokens = 600 tokens/second
- Enables near-lossless audio reconstruction
- Captures speaker identity, prosody, recording quality

**This is what our vocoder tries to do, but AudioLM does it with discrete tokens!**

### 2.2 Three-Stage Hierarchical Generation

AudioLM generates audio in three stages, each modeled by a Transformer:

```
STAGE 1: Semantic Modeling
┌─────────────────────────────────────────────────────────────┐
│                    SEMANTIC TRANSFORMER                      │
│ Input: [Past semantic tokens]                                │
│ Output: [Future semantic tokens]                             │
│                                                              │
│ Purpose: Generate linguistically coherent content            │
│ - Decides WHAT to say next                                   │
│ - Models syntax, semantics, melody                           │
│ - No speaker/acoustic information yet                        │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
STAGE 2: Coarse Acoustic Modeling
┌─────────────────────────────────────────────────────────────┐
│                 COARSE ACOUSTIC TRANSFORMER                  │
│ Input: [All semantic tokens] + [Past coarse acoustic tokens]│
│ Output: [Future coarse acoustic tokens (layers 1-4)]        │
│                                                              │
│ Purpose: Add speaker and prosodic characteristics            │
│ - Decides HOW to say it                                      │
│ - Models speaker identity, emotion, pace                     │
│ - Conditioned on semantic content                            │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
STAGE 3: Fine Acoustic Modeling
┌─────────────────────────────────────────────────────────────┐
│                  FINE ACOUSTIC TRANSFORMER                   │
│ Input: [Coarse acoustic tokens]                              │
│ Output: [Fine acoustic tokens (layers 5-8)]                  │
│                                                              │
│ Purpose: Add high-frequency details                          │
│ - Refines audio quality                                      │
│ - Fills in acoustic texture                                  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    SOUNDSTREAM DECODER
┌─────────────────────────────────────────────────────────────┐
│ Input: [All acoustic tokens (8 layers)]                      │
│ Output: [Waveform at 16kHz]                                  │
└─────────────────────────────────────────────────────────────┘
```

### 2.3 SoundStream Neural Codec

SoundStream is a crucial component that we don't currently have. It's a neural audio codec that:

```
ENCODER (Analysis)                    DECODER (Synthesis)
┌─────────────┐                       ┌─────────────┐
│ Audio Input │                       │ Audio Output│
│   (16kHz)   │                       │   (16kHz)   │
└─────┬───────┘                       └─────▲───────┘
      │                                     │
      ▼                                     │
┌─────────────┐                       ┌─────────────┐
│ CNN Encoder │                       │ CNN Decoder │
│ (Strided)   │                       │ (Transpose) │
└─────┬───────┘                       └─────▲───────┘
      │                                     │
      ▼                                     │
┌─────────────────────────────────────────────────┐
│          RESIDUAL VECTOR QUANTIZATION           │
│                                                 │
│  Layer 1: [Coarse features] ──────────────────► │
│  Layer 2: [Residual 1] ───────────────────────► │
│  Layer 3: [Residual 2] ───────────────────────► │
│  ...                                            │
│  Layer 8: [Fine residual] ────────────────────► │
│                                                 │
│  Each layer: 1024 codebook entries              │
│  Total: 8 × 10 bits = 80 bits/frame             │
│  At 75 fps: 6 kbps (vs 256 kbps raw PCM)        │
└─────────────────────────────────────────────────┘
```

**Key properties**:
- **Compression**: 40x reduction (256 kbps → 6 kbps)
- **Quality**: Near-transparent (MOS > 4.0)
- **Discretization**: Enables language modeling
- **Residual structure**: Coarse-to-fine hierarchy

---

## 3. Comparison: Our Pipeline vs AudioLM

### 3.1 Architecture Comparison

| Component | Our Current Pipeline | AudioLM |
|-----------|---------------------|---------|
| **Semantic Features** | XLSR-53 Layer 14 | w2v-BERT |
| **Discretization** | K-Means (100 units) | Learned codebook (~1024) |
| **Acoustic Modeling** | None (discarded) | SoundStream (8-layer RVQ) |
| **Generation Model** | HiFi-GAN Vocoder | Transformer LM (3 stages) |
| **Pitch Handling** | External F0 extraction | Implicit in acoustic tokens |
| **Speaker Identity** | Lost in K-Means | Preserved in acoustic tokens |
| **Training Data** | Audio only | Audio only |

### 3.2 Information Flow Comparison

**Our Pipeline**:
```
Audio → [XLSR-53] → 1024-dim features → [K-Means] → 100 discrete units
                                                           │
                                                    (INFORMATION LOSS)
                                                    - Speaker identity
                                                    - Prosody/F0
                                                    - Fine acoustic details
                                                           │
                                                           ▼
                                            [HiFi-GAN Vocoder] → Audio
                                                     │
                                              (Tries to recover
                                               lost information)
```

**AudioLM**:
```
Audio → [w2v-BERT] → Semantic tokens (linguistic content)
     → [SoundStream] → Acoustic tokens (ALL acoustic information preserved)
                                    │
                    (NO INFORMATION LOSS - acoustic tokens are near-lossless)
                                    │
                                    ▼
                         [3-Stage Transformer LM]
                                    │
                                    ▼
                         [SoundStream Decoder] → Audio
```

### 3.3 Quality Comparison

| Quality Aspect | Our Pipeline (V2) | AudioLM |
|----------------|-------------------|---------|
| **Intelligibility** | High | High |
| **Speaker Preservation** | Limited (pitch helps) | Excellent (acoustic tokens) |
| **Prosody/Intonation** | Moderate (external F0) | Excellent (implicit) |
| **Audio Fidelity** | Good (HiFi-GAN) | Excellent (SoundStream) |
| **Long-term Coherence** | N/A (frame-by-frame) | Excellent (LM modeling) |
| **Robotic Sound** | Some (despite V2) | Minimal (proven human-like) |

---

## 4. Where AudioLM Could Be Integrated

### 4.1 Option A: Replace K-Means with Learned Semantic Tokens

**Current**: XLSR-53 → K-Means (100 clusters)  
**Proposed**: w2v-BERT → Learned VQ codebook

```python
# Current approach
features = xlsr_model(audio)  # [T, 1024]
units = kmeans.predict(features)  # [T] in range [0, 99]

# AudioLM-style approach
semantic_tokens = w2v_bert_tokenizer(audio)  # [T] in range [0, 1023]
# Advantage: Learned tokenization, larger vocabulary, better coverage
```

**Benefits**:
- Larger vocabulary (1024 vs 100) = finer phonetic distinctions
- Learned tokenization optimized for reconstruction
- Better handling of rare phonemes in low-resource languages

**Implementation Complexity**: Low
**Quality Improvement**: Moderate

### 4.2 Option B: Add SoundStream Acoustic Tokens

**Current**: No acoustic tokens, vocoder must infer all acoustic details  
**Proposed**: Add SoundStream tokens to preserve speaker/prosody

```
┌─────────────────────────────────────────────────────────────┐
│                   ENHANCED PIPELINE                          │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Audio Input                                                 │
│       │                                                      │
│       ├────────────► XLSR-53/MMS → Semantic Units            │
│       │              (linguistic content)                    │
│       │                                                      │
│       └────────────► SoundStream → Acoustic Tokens           │
│                      (speaker, prosody, quality)             │
│                                                              │
│                           │                                  │
│                           ▼                                  │
│              ┌─────────────────────────┐                     │
│              │   Combined Conditioning │                     │
│              │   [Semantic + Acoustic] │                     │
│              └────────────┬────────────┘                     │
│                           │                                  │
│                           ▼                                  │
│              ┌─────────────────────────┐                     │
│              │   Enhanced Vocoder      │                     │
│              │   (HiFi-GAN V2 + Cond)  │                     │
│              └────────────┬────────────┘                     │
│                           │                                  │
│                           ▼                                  │
│                    Output Audio                              │
└─────────────────────────────────────────────────────────────┘
```

**Benefits**:
- Preserves speaker identity in tokens (not lost in K-Means)
- Prosody and intonation encoded explicitly
- Vocoder has clear target to reconstruct

**Implementation Complexity**: Medium-High (need to train SoundStream)
**Quality Improvement**: High

### 4.3 Option C: Full AudioLM-Style Pipeline

**Replace entire pipeline** with AudioLM architecture:

```
┌─────────────────────────────────────────────────────────────┐
│                 FULL AUDIOLM PIPELINE                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  TRAINING PHASE:                                             │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │ 1. Train SoundStream codec on target language audio     │ │
│  │ 2. Extract semantic tokens (w2v-BERT/MMS)               │ │
│  │ 3. Extract acoustic tokens (SoundStream)                │ │
│  │ 4. Train Stage 1: Semantic Transformer                  │ │
│  │ 5. Train Stage 2: Coarse Acoustic Transformer           │ │
│  │ 6. Train Stage 3: Fine Acoustic Transformer             │ │
│  └─────────────────────────────────────────────────────────┘ │
│                                                              │
│  INFERENCE PHASE:                                            │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │ Input: [Semantic tokens from translation/text]          │ │
│  │        OR [Short audio prompt for continuation]         │ │
│  │                         │                                │ │
│  │                         ▼                                │ │
│  │        [Semantic Transformer] → More semantic tokens     │ │
│  │                         │                                │ │
│  │                         ▼                                │ │
│  │        [Coarse Acoustic Transformer] → Coarse tokens     │ │
│  │                         │                                │ │
│  │                         ▼                                │ │
│  │        [Fine Acoustic Transformer] → Fine tokens         │ │
│  │                         │                                │ │
│  │                         ▼                                │ │
│  │        [SoundStream Decoder] → Waveform                  │ │
│  └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

**Benefits**:
- State-of-the-art quality (indistinguishable from real speech)
- Long-term coherence (language model captures structure)
- Unified framework for continuation, synthesis, translation
- Can be prompted with audio (voice cloning)

**Implementation Complexity**: High (multiple models to train)
**Quality Improvement**: Very High

---

## 5. Specific Integration Points for Our Pipeline

### 5.1 Phase 1: Replace K-Means with Learned Tokenization

**Current Phase 1**:
```python
# phase1_acoustic.py
features = xlsr_model(audio, output_hidden_states=True).hidden_states[14]
kmeans = MiniBatchKMeans(n_clusters=100)
units = kmeans.fit_predict(features)
```

**AudioLM-Style Phase 1**:
```python
# phase1_audiolm.py

# Option 1: Use pretrained w2v-BERT tokenizer
from transformers import Wav2Vec2BertModel

model = Wav2Vec2BertModel.from_pretrained("facebook/w2v-bert-2.0")
# w2v-BERT already has learned discrete representations

# Option 2: Train VQ-VAE on top of XLSR-53/MMS
class LearnedTokenizer(nn.Module):
    def __init__(self, input_dim=1024, num_codes=1024, code_dim=256):
        super().__init__()
        self.encoder = nn.Linear(input_dim, code_dim)
        self.codebook = nn.Embedding(num_codes, code_dim)
    
    def forward(self, features):
        encoded = self.encoder(features)
        distances = torch.cdist(encoded, self.codebook.weight)
        tokens = distances.argmin(dim=-1)
        return tokens

# Train with commitment loss + reconstruction loss
```

### 5.2 Phase 3: Add SoundStream Conditioning

**Current Phase 3 Generator**:
```python
class GeneratorV2(nn.Module):
    def __init__(self):
        self.unit_embedding = nn.Embedding(100, 256)   # Semantic only
        self.pitch_embedding = nn.Embedding(33, 64)    # External F0
        # ... HiFi-GAN architecture
```

**AudioLM-Enhanced Generator**:
```python
class GeneratorV3(nn.Module):
    def __init__(self):
        # Semantic conditioning (same as V2)
        self.unit_embedding = nn.Embedding(1024, 256)  # Larger vocab
        
        # Acoustic conditioning (NEW - from SoundStream)
        self.acoustic_embedding = nn.ModuleList([
            nn.Embedding(1024, 64) for _ in range(8)  # 8 RVQ layers
        ])
        
        # Combined conditioning
        self.condition_proj = nn.Linear(256 + 64*8, 512)
        
        # ... rest of HiFi-GAN
    
    def forward(self, semantic_units, acoustic_tokens):
        # Semantic embedding
        sem_emb = self.unit_embedding(semantic_units)
        
        # Acoustic embedding (all 8 layers)
        acoustic_embs = []
        for i, layer in enumerate(self.acoustic_embedding):
            acoustic_embs.append(layer(acoustic_tokens[:, i]))
        acoustic_emb = torch.cat(acoustic_embs, dim=-1)
        
        # Combined conditioning
        condition = self.condition_proj(
            torch.cat([sem_emb, acoustic_emb], dim=-1)
        )
        
        # Generate audio with full information
        return self.generator(condition)
```

### 5.3 Adding a Language Model for Coherent Generation

For speech-to-speech translation, add a Transformer to predict acoustic tokens:

```python
class AcousticLanguageModel(nn.Module):
    """Predicts acoustic tokens from semantic tokens."""
    
    def __init__(self, semantic_vocab=1024, acoustic_vocab=1024, 
                 num_acoustic_layers=8):
        super().__init__()
        
        self.semantic_embedding = nn.Embedding(semantic_vocab, 512)
        self.acoustic_embedding = nn.Embedding(
            acoustic_vocab * num_acoustic_layers, 512
        )
        
        self.transformer = nn.Transformer(
            d_model=512,
            nhead=8,
            num_encoder_layers=6,
            num_decoder_layers=6
        )
        
        self.acoustic_heads = nn.ModuleList([
            nn.Linear(512, acoustic_vocab) for _ in range(num_acoustic_layers)
        ])
    
    def forward(self, semantic_tokens, acoustic_prefix=None):
        """
        Given semantic tokens, predict acoustic tokens autoregressively.
        acoustic_prefix can be from a prompt (for voice cloning).
        """
        semantic_emb = self.semantic_embedding(semantic_tokens)
        
        if acoustic_prefix is not None:
            # Start from prompt (speaker conditioning)
            prefix_emb = self.acoustic_embedding(acoustic_prefix)
        else:
            # Start from scratch
            prefix_emb = torch.zeros_like(semantic_emb[:, :1])
        
        # Autoregressive generation
        output = self.transformer(semantic_emb, prefix_emb)
        
        # Predict each acoustic layer
        acoustic_logits = [head(output) for head in self.acoustic_heads]
        
        return acoustic_logits
```

---

## 6. Implementation Roadmap

### Phase 1: SoundStream Training (Foundation)

**Goal**: Train a neural codec for our target languages

```bash
# Step 1: Collect audio data (already done - segmented_audio/)
# Step 2: Train SoundStream
python3 -m modal run src/training/soundstream_train.py \
    --language portuguese \
    --codebook-size 1024 \
    --num-quantizers 8 \
    --epochs 100

# Step 3: Extract acoustic tokens for all audio
python3 -m modal run src/training/extract_acoustic_tokens.py \
    --language portuguese
```

**Estimated Time**: 2-4 days training  
**Compute**: A100 GPU recommended

### Phase 2: Enhanced Vocoder (Quick Win)

**Goal**: Add acoustic token conditioning to existing HiFi-GAN

```bash
# Use acoustic tokens alongside semantic units
python3 -m modal run src/training/phase3_vocoder_v3.py \
    --language portuguese \
    --acoustic-conditioning \
    --epochs 500
```

**Estimated Time**: 4-8 hours  
**Expected Improvement**: Significant (speaker preservation, prosody)

### Phase 3: Acoustic Language Model (Full AudioLM)

**Goal**: Train Transformer to generate acoustic tokens from semantic tokens

```bash
# Train the acoustic language model
python3 -m modal run src/training/acoustic_lm.py \
    --language portuguese \
    --epochs 100

# Use for speech-to-speech translation
python3 -m modal run src/training/s2s_translate.py \
    --source-audio input.wav \
    --target-language satere \
    --output output.wav
```

**Estimated Time**: 1-2 weeks training  
**Expected Improvement**: State-of-the-art quality

---

## 7. Use Cases Enabled by AudioLM

### 7.1 Voice Cloning / Speaker Adaptation

With acoustic tokens, we can clone a speaker from a short prompt:

```python
# Extract acoustic tokens from 3-second prompt
prompt_audio = load_audio("native_speaker.wav")
prompt_acoustic = soundstream.encode(prompt_audio)

# Generate new speech in that voice
semantic_tokens = get_semantic_tokens(new_text_or_translation)
acoustic_tokens = acoustic_lm.generate(
    semantic_tokens, 
    acoustic_prefix=prompt_acoustic[:150]  # 2 seconds of prompt
)
output_audio = soundstream.decode(acoustic_tokens)
```

**Application**: Preserve indigenous speakers' voices for future generations.

### 7.2 Speech-to-Speech Translation

```python
def translate_speech(source_audio, target_language):
    # 1. Extract semantic tokens from source
    source_semantic = semantic_tokenizer.encode(source_audio)
    
    # 2. Translate semantic tokens (separate model)
    target_semantic = translation_model.translate(
        source_semantic, 
        target_language
    )
    
    # 3. Generate acoustic tokens for target language
    target_acoustic = acoustic_lm.generate(target_semantic)
    
    # 4. Decode to audio
    output_audio = soundstream.decode(target_acoustic)
    
    return output_audio

# Example: Portuguese → Sateré-Mawé
satere_audio = translate_speech(portuguese_audio, "satere")
```

### 7.3 Audio Bible Narration

```python
def narrate_verse(text, reference_speaker_audio):
    # 1. Convert text to semantic tokens
    semantic = text_to_semantic_model(text)  # Separate model needed
    
    # 2. Extract speaker characteristics from reference
    speaker_acoustic = soundstream.encode(reference_speaker_audio)[:150]
    
    # 3. Generate in reference speaker's voice
    acoustic = acoustic_lm.generate(
        semantic,
        acoustic_prefix=speaker_acoustic
    )
    
    # 4. Synthesize
    audio = soundstream.decode(acoustic)
    
    return audio

# Generate entire book
for verse in bible_book:
    audio = narrate_verse(verse.text, narrator_sample)
    save_audio(audio, f"{verse.reference}.wav")
```

---

## 8. Trade-offs and Considerations

### 8.1 Advantages of AudioLM Approach

| Advantage | Description |
|-----------|-------------|
| **Quality** | Near-indistinguishable from real speech (51.2% detection rate = random chance) |
| **Speaker Preservation** | Acoustic tokens capture all speaker characteristics |
| **Prosody** | Natural intonation and rhythm (not external F0) |
| **Long-term Coherence** | Transformer LM models dependencies across utterance |
| **Unified Framework** | Same architecture for continuation, synthesis, translation |
| **Prompt-based** | Can clone any speaker with 2-3 second prompt |

### 8.2 Disadvantages / Challenges

| Challenge | Description | Mitigation |
|-----------|-------------|------------|
| **Complexity** | 4+ models to train (SoundStream, 3 Transformers) | Start with hybrid approach (Option B) |
| **Compute** | Training requires significant GPU resources | Use Modal with A100 GPUs |
| **Data Requirements** | Transformers need more data than vocoders | MMS/XLSR-53 pretraining helps |
| **Latency** | Autoregressive generation is slower | Parallel decoding, caching |
| **Low-Resource** | May struggle with very limited data | Leverage multilingual pretraining |

### 8.3 Data Requirements Comparison

| Model | Minimum Data | Recommended Data |
|-------|--------------|------------------|
| **Our Current Vocoder** | 1-2 hours | 10+ hours |
| **SoundStream** | 10+ hours | 100+ hours |
| **Acoustic LM** | 50+ hours | 500+ hours |
| **Full AudioLM** | 100+ hours | 1000+ hours |

For Sateré-Mawé with limited data, the hybrid approach (Option B) is more feasible.

---

## 9. Recommended Implementation Strategy

### For Our Use Case (Low-Resource Languages)

Given our constraints (limited data, need for rapid deployment), I recommend:

**Short-term (1-2 weeks)**:
1. Keep current Phase 1 & 2 (XLSR-53/MMS + K-Means + BPE)
2. Experiment with larger vocabulary (500-1000 units instead of 100)
3. Continue improving V2 vocoder

**Medium-term (1-2 months)**:
1. Train SoundStream codec on Portuguese (more data)
2. Add acoustic token conditioning to vocoder (Option B)
3. Test if quality improves significantly

**Long-term (3-6 months)**:
1. Train acoustic language model
2. Implement voice cloning capability
3. Build speech-to-speech translation pipeline
4. Apply to Sateré-Mawé and other languages

### Priority Order

```
[HIGHEST]  1. MMS migration (see MMS_VS_XLSR53.md)
           2. Larger semantic vocabulary (100 → 500-1000)
           3. SoundStream codec training
           4. Acoustic token conditioning in vocoder
           5. Acoustic language model
[LOWEST]   6. Full AudioLM replication
```

---

## 10. Code Sketch: SoundStream Integration

Here's a sketch for integrating SoundStream-style acoustic tokens:

```python
# src/models/soundstream.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualVectorQuantizer(nn.Module):
    """8-layer residual vector quantization."""
    
    def __init__(self, dim=512, codebook_size=1024, num_quantizers=8):
        super().__init__()
        self.num_quantizers = num_quantizers
        self.codebooks = nn.ParameterList([
            nn.Parameter(torch.randn(codebook_size, dim))
            for _ in range(num_quantizers)
        ])
    
    def forward(self, x):
        """
        Args:
            x: [B, T, D] continuous features
        Returns:
            codes: [B, T, num_quantizers] discrete codes
            quantized: [B, T, D] quantized features
        """
        residual = x
        codes = []
        quantized = torch.zeros_like(x)
        
        for i, codebook in enumerate(self.codebooks):
            # Find nearest codebook entry
            distances = torch.cdist(residual, codebook)
            indices = distances.argmin(dim=-1)
            codes.append(indices)
            
            # Quantize and compute residual
            quantized_layer = F.embedding(indices, codebook)
            quantized = quantized + quantized_layer
            residual = residual - quantized_layer
        
        return torch.stack(codes, dim=-1), quantized


class SoundStream(nn.Module):
    """Neural audio codec inspired by AudioLM's SoundStream."""
    
    def __init__(self, sample_rate=16000):
        super().__init__()
        
        # Encoder: audio → continuous latent
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 32, 7, padding=3),
            nn.ELU(),
            # Downsample 320x: 16kHz → 50 fps
            self._make_downsample_block(32, 64, 4),    # /4
            self._make_downsample_block(64, 128, 4),   # /16
            self._make_downsample_block(128, 256, 4),  # /64
            self._make_downsample_block(256, 512, 5),  # /320
        )
        
        # Quantizer: continuous → discrete
        self.quantizer = ResidualVectorQuantizer(
            dim=512, codebook_size=1024, num_quantizers=8
        )
        
        # Decoder: discrete → audio
        self.decoder = nn.Sequential(
            self._make_upsample_block(512, 256, 5),
            self._make_upsample_block(256, 128, 4),
            self._make_upsample_block(128, 64, 4),
            self._make_upsample_block(64, 32, 4),
            nn.Conv1d(32, 1, 7, padding=3),
            nn.Tanh(),
        )
    
    def _make_downsample_block(self, in_ch, out_ch, stride):
        return nn.Sequential(
            nn.Conv1d(in_ch, out_ch, stride*2, stride, stride//2),
            nn.ELU(),
            nn.Conv1d(out_ch, out_ch, 3, padding=1),
            nn.ELU(),
        )
    
    def _make_upsample_block(self, in_ch, out_ch, stride):
        return nn.Sequential(
            nn.ConvTranspose1d(in_ch, out_ch, stride*2, stride, stride//2),
            nn.ELU(),
            nn.Conv1d(out_ch, out_ch, 3, padding=1),
            nn.ELU(),
        )
    
    def encode(self, audio):
        """Audio → discrete tokens [B, T, 8]"""
        features = self.encoder(audio)
        features = features.transpose(1, 2)  # [B, T, D]
        codes, _ = self.quantizer(features)
        return codes
    
    def decode(self, codes):
        """Discrete tokens → audio"""
        # Dequantize
        quantized = torch.zeros(codes.shape[0], codes.shape[1], 512)
        for i in range(self.quantizer.num_quantizers):
            quantized += F.embedding(codes[:, :, i], 
                                     self.quantizer.codebooks[i])
        
        # Decode to audio
        quantized = quantized.transpose(1, 2)
        audio = self.decoder(quantized)
        return audio


# Training loop
def train_soundstream(audio_data, epochs=100):
    model = SoundStream()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    for epoch in range(epochs):
        for audio in audio_data:
            # Encode and decode
            codes = model.encode(audio)
            reconstructed = model.decode(codes)
            
            # Losses
            recon_loss = F.l1_loss(reconstructed, audio)
            # Add adversarial loss, commitment loss, etc.
            
            optimizer.zero_grad()
            recon_loss.backward()
            optimizer.step()
```

---

## 11. Conclusion

AudioLM represents a significant advancement in audio generation that could substantially improve our pipeline's quality. The key insights are:

1. **Two-token system** (semantic + acoustic) is superior to semantic-only
2. **Language modeling** enables long-term coherence
3. **SoundStream** provides near-lossless audio discretization
4. **Hierarchical generation** naturally handles coarse-to-fine details

### For Our Project

| Timeline | Action | Expected Benefit |
|----------|--------|------------------|
| **Now** | MMS migration | Better low-resource features |
| **Soon** | Larger vocabulary | Finer phonetic distinctions |
| **Medium** | SoundStream + acoustic conditioning | Speaker/prosody preservation |
| **Long** | Full AudioLM | State-of-the-art quality |

The hybrid approach (keeping XLSR-53/MMS + adding SoundStream) offers the best balance of feasibility and quality improvement for our low-resource language use case.

---

## References

1. Borsos, Z., et al. (2022). "AudioLM: a Language Modeling Approach to Audio Generation." [Google Research Blog](https://research.google/blog/audiolm-a-language-modeling-approach-to-audio-generation/)

2. Zeghidour, N., et al. (2021). "SoundStream: An End-to-End Neural Audio Codec." arXiv:2107.03312

3. Chung, Y.-A., et al. (2021). "w2v-BERT: Combining Contrastive Learning and Masked Language Modeling for Self-Supervised Speech Pre-Training." arXiv:2108.06209

4. Défossez, A., et al. (2022). "High Fidelity Neural Audio Compression." (EnCodec) arXiv:2210.13438

5. Wang, C., et al. (2023). "Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers." (VALL-E) arXiv:2301.02111
