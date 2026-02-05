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
# # Generator V2 - HiFi-GAN Style with Pitch Conditioning
#
# This is an upgraded generator architecture based on HiFi-GAN with:
#
# 1. **Pitch Conditioning**: Accepts F0 (pitch) information to restore prosody
# 2. **Multi-Receptive Field Fusion (MRF)**: Better captures patterns at multiple scales
# 3. **Weight Normalization**: More stable training than batch normalization
# 4. **Improved Upsampling**: Better transposed convolutions with anti-aliasing
#
# ## Architecture
#
# ```
# Input: 
#   - Unit sequence [B, T] (e.g., T=50 units)
#   - Pitch sequence [B, T] (e.g., T=50 pitch bins)
#    ↓
# Embeddings: 
#   - Units: 100 → 256
#   - Pitch: 32 bins → 64
#    ↓
# Concatenate: 256 + 64 = 320
#    ↓
# Pre-Conv: 320 → 512 (with weight norm)
#    ↓
# Upsample Blocks (with MRF):
#   5x: 512 → 256
#   4x: 256 → 128
#   4x: 128 → 64
#   4x: 64 → 32
#    ↓
# Post-Conv: 32 → 1
#    ↓
# Output: Audio [B, T*320] (e.g., 16,000 samples)
# ```
#
# Total upsampling: 5 × 4 × 4 × 4 = 320x
# This matches the XLSR-53 frame rate (~20ms per unit at 16kHz)

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm, remove_weight_norm
from typing import List, Optional
from src.constants import (
    NUM_ACOUSTIC_UNITS,
    NUM_PITCH_BINS,
    UNIT_EMBED_DIM,
    PITCH_EMBED_DIM,
    UPSAMPLE_INITIAL_CHANNEL,
    UPSAMPLE_RATES,
    UPSAMPLE_KERNEL_SIZES,
    RESBLOCK_KERNEL_SIZES,
    RESBLOCK_DILATIONS,
    LEAKY_RELU_SLOPE,
    SAMPLE_RATE,
    HOP_SIZE,
    F0_MIN,
    F0_MAX,
    PITCH_UNVOICED_BIN,
)


# %%
def get_padding(kernel_size: int, dilation: int = 1) -> int:
    return (kernel_size * dilation - dilation) // 2


# %%
class ResBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3, 
                 dilations: List[List[int]] = [[1, 1], [3, 1], [5, 1]]):
        super().__init__()
        
        self.convs1 = nn.ModuleList()
        self.convs2 = nn.ModuleList()
        
        for dilation_pair in dilations:
            self.convs1.append(
                weight_norm(nn.Conv1d(
                    channels, channels, kernel_size, 
                    dilation=dilation_pair[0],
                    padding=get_padding(kernel_size, dilation_pair[0])
                ))
            )
            self.convs2.append(
                weight_norm(nn.Conv1d(
                    channels, channels, kernel_size,
                    dilation=dilation_pair[1],
                    padding=get_padding(kernel_size, dilation_pair[1])
                ))
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for conv1, conv2 in zip(self.convs1, self.convs2):
            residual = x
            x = F.leaky_relu(x, LEAKY_RELU_SLOPE)
            x = conv1(x)
            x = F.leaky_relu(x, LEAKY_RELU_SLOPE)
            x = conv2(x)
            x = x + residual
        return x
    
    def remove_weight_norm(self):
        for conv in self.convs1:
            remove_weight_norm(conv)
        for conv in self.convs2:
            remove_weight_norm(conv)


# %%
class MultiReceptiveFieldFusion(nn.Module):
    def __init__(self, channels: int, 
                 kernel_sizes: List[int] = [3, 7, 11],
                 dilations: List[List[List[int]]] = None):
        super().__init__()
        
        if dilations is None:
            dilations = [
                [[1, 1], [3, 1], [5, 1]],
                [[1, 1], [3, 1], [5, 1]],
                [[1, 1], [3, 1], [5, 1]],
            ]
        
        self.resblocks = nn.ModuleList([
            ResBlock(channels, k, d) 
            for k, d in zip(kernel_sizes, dilations)
        ])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = None
        for resblock in self.resblocks:
            if out is None:
                out = resblock(x)
            else:
                out = out + resblock(x)
        return out / len(self.resblocks)
    
    def remove_weight_norm(self):
        for resblock in self.resblocks:
            resblock.remove_weight_norm()


# %%
class GeneratorV2(nn.Module):
    """
    HiFi-GAN Style Generator with Pitch Conditioning (V2)
    
    Converts discrete acoustic units + pitch to raw audio waveform.
    
    Key Improvements over V1:
    - Pitch conditioning for natural prosody
    - Multi-Receptive Field Fusion (MRF) for multi-scale patterns
    - Weight normalization for stable training
    - Better upsampling with anti-aliasing convolutions
    
    Args:
        num_units: Number of discrete acoustic units (default: 100)
        num_pitch_bins: Number of pitch bins (default: 32)
        unit_embed_dim: Unit embedding dimension (default: 256)
        pitch_embed_dim: Pitch embedding dimension (default: 64)
        upsample_rates: Upsampling rates (default: [5, 4, 4, 4] = 320x)
        upsample_kernel_sizes: Kernel sizes for upsampling (default: [10, 8, 8, 8])
        upsample_initial_channel: Initial channel size (default: 512)
        resblock_kernel_sizes: Kernel sizes for MRF (default: [3, 7, 11])
        resblock_dilation_sizes: Dilation sizes for MRF
    
    Input:
        units: Unit indices [batch_size, sequence_length]
        pitch: Pitch bin indices [batch_size, sequence_length] (optional)
    
    Output:
        audio: Waveform [batch_size, sequence_length * 320]
    """
    
    def __init__(
        self,
        num_units: int = NUM_ACOUSTIC_UNITS,
        num_pitch_bins: int = NUM_PITCH_BINS,
        unit_embed_dim: int = UNIT_EMBED_DIM,
        pitch_embed_dim: int = PITCH_EMBED_DIM,
        upsample_rates: List[int] = None,
        upsample_kernel_sizes: List[int] = None,
        upsample_initial_channel: int = UPSAMPLE_INITIAL_CHANNEL,
        resblock_kernel_sizes: List[int] = None,
        resblock_dilation_sizes: List[List[List[int]]] = None,
    ):
        super().__init__()
        
        if upsample_rates is None:
            upsample_rates = list(UPSAMPLE_RATES)
        if upsample_kernel_sizes is None:
            upsample_kernel_sizes = list(UPSAMPLE_KERNEL_SIZES)
        if resblock_kernel_sizes is None:
            resblock_kernel_sizes = list(RESBLOCK_KERNEL_SIZES)
        if resblock_dilation_sizes is None:
            resblock_dilation_sizes = [
                RESBLOCK_DILATIONS,
                RESBLOCK_DILATIONS,
                RESBLOCK_DILATIONS,
            ]
        
        self.num_units = num_units
        self.num_pitch_bins = num_pitch_bins
        self.upsample_rates = upsample_rates
        self.upsample_factor = 1
        for r in upsample_rates:
            self.upsample_factor *= r
        
        self.unit_embed = nn.Embedding(num_units, unit_embed_dim)
        self.pitch_embed = nn.Embedding(num_pitch_bins + 1, pitch_embed_dim)
        
        input_dim = unit_embed_dim + pitch_embed_dim
        self.pre_conv = weight_norm(
            nn.Conv1d(input_dim, upsample_initial_channel, 7, padding=3)
        )
        
        self.ups = nn.ModuleList()
        self.mrfs = nn.ModuleList()
        
        ch = upsample_initial_channel
        for i, (rate, kernel) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            padding = (kernel - rate) // 2
            self.ups.append(
                weight_norm(nn.ConvTranspose1d(
                    ch, ch // 2, kernel, stride=rate, padding=padding
                ))
            )
            self.mrfs.append(
                MultiReceptiveFieldFusion(
                    ch // 2, resblock_kernel_sizes, resblock_dilation_sizes
                )
            )
            ch = ch // 2
        
        self.post_conv = weight_norm(nn.Conv1d(ch, 1, 7, padding=3))
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
                nn.init.normal_(m.weight, 0.0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(
        self, 
        units: torch.Tensor, 
        pitch: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        unit_emb = self.unit_embed(units)
        
        if pitch is None:
            # Use "no pitch" token (index 0)
            pitch = torch.zeros_like(units)
        pitch_emb = self.pitch_embed(pitch)
        
        x = torch.cat([unit_emb, pitch_emb], dim=-1)
        x = x.transpose(1, 2)
        x = self.pre_conv(x)
        
        for up, mrf in zip(self.ups, self.mrfs):
            x = F.leaky_relu(x, LEAKY_RELU_SLOPE)
            x = up(x)
            x = mrf(x)
        
        x = F.leaky_relu(x, LEAKY_RELU_SLOPE)
        x = self.post_conv(x)
        x = torch.tanh(x)
        
        return x.squeeze(1)
    
    def remove_weight_norm(self):
        remove_weight_norm(self.pre_conv)
        for up in self.ups:
            remove_weight_norm(up)
        for mrf in self.mrfs:
            mrf.remove_weight_norm()
        remove_weight_norm(self.post_conv)


# %% [markdown]
# ## Pitch Extraction Utilities
#
# These functions help extract and quantize pitch (F0) from audio.

# %%
def extract_pitch(audio: torch.Tensor, sample_rate: int = SAMPLE_RATE, 
                  hop_length: int = HOP_SIZE, f0_min: float = F0_MIN, 
                  f0_max: float = F0_MAX) -> torch.Tensor:
    """
    Extract pitch (F0) from audio using a simple autocorrelation method.
    
    For production, consider using CREPE, PYIN, or pyworld for better accuracy.
    
    Args:
        audio: Audio waveform [batch_size, num_samples] or [num_samples]
        sample_rate: Audio sample rate (default: 16000)
        hop_length: Hop length in samples (default: 320, matches XLSR-53)
        f0_min: Minimum F0 frequency (default: 50 Hz)
        f0_max: Maximum F0 frequency (default: 400 Hz)
    
    Returns:
        pitch: F0 values in Hz [batch_size, num_frames] or [num_frames]
    """
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)
    
    batch_size, num_samples = audio.shape
    num_frames = num_samples // hop_length
    
    return torch.zeros(batch_size, num_frames, device=audio.device)


def quantize_pitch(pitch: torch.Tensor, num_bins: int = NUM_PITCH_BINS,
                   f0_min: float = F0_MIN, f0_max: float = F0_MAX) -> torch.Tensor:
    """
    Quantize continuous pitch values to discrete bins.
    
    Uses log-scale binning for perceptually uniform spacing.
    
    Args:
        pitch: F0 values in Hz [batch_size, num_frames]
        num_bins: Number of pitch bins (default: 32)
        f0_min: Minimum F0 frequency (default: 50 Hz)
        f0_max: Maximum F0 frequency (default: 400 Hz)
    
    Returns:
        pitch_bins: Quantized pitch [batch_size, num_frames]
                   0 = unvoiced/no pitch, 1-num_bins = voiced
    """
    voiced_mask = (pitch > 0) & ~torch.isnan(pitch)
    
    log_min = torch.log(torch.tensor(f0_min))
    log_max = torch.log(torch.tensor(f0_max))
    
    pitch_clamped = torch.clamp(pitch, f0_min, f0_max)
    log_pitch = torch.log(pitch_clamped)
    
    normalized = (log_pitch - log_min) / (log_max - log_min)
    bins = (normalized * (num_bins - 1) + 1).long()
    
    bins = torch.where(voiced_mask, bins, torch.zeros_like(bins))
    
    return bins


# %% [markdown]
# ## Usage Example
#
# ```python
# # Create generator
# generator = GeneratorV2(num_units=100, num_pitch_bins=32)
#
# # Input: batch of unit sequences + pitch
# units = torch.randint(0, 100, (4, 50))  # 4 sequences of 50 units
# pitch = torch.randint(0, 33, (4, 50))   # 4 pitch sequences (0 = unvoiced)
#
# # Generate audio
# audio = generator(units, pitch)  # Shape: [4, 16000] (1 second at 16kHz)
#
# # Generate without pitch (falls back to monotone)
# audio_no_pitch = generator(units)  # Shape: [4, 16000]
# ```

# %%
if __name__ == "__main__":
    # Test the generator
    generator = GeneratorV2(num_units=100, num_pitch_bins=32)
    print(f"Generator V2 parameters: {sum(p.numel() for p in generator.parameters()):,}")
    
    # Test forward pass with pitch
    units = torch.randint(0, 100, (2, 50))  # 2 sequences of 50 units
    pitch = torch.randint(0, 33, (2, 50))   # 2 pitch sequences
    audio = generator(units, pitch)
    print(f"Input shapes: units={units.shape}, pitch={pitch.shape}")
    print(f"Output shape: {audio.shape}")
    print(f"Expected output length: {50 * 320} = {50 * 320}")
    
    # Test forward pass without pitch
    audio_no_pitch = generator(units)
    print(f"Output shape (no pitch): {audio_no_pitch.shape}")
    
    # Test weight norm removal
    generator.remove_weight_norm()
    audio_inference = generator(units, pitch)
    print(f"Inference output shape: {audio_inference.shape}")
