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
# # Generator Model
#
# Unit-to-Waveform Generator (Vocoder) that converts discrete acoustic units 
# to continuous audio waveforms.
#
# ## Architecture
#
# ```
# Input: Unit sequence [B, T] (e.g., T=50 units)
#    ↓
# Embedding: 100 units → 256 dimensions
#    ↓
# Pre-Conv: 256 → 512 channels
#    ↓
# Upsample 5x: 512 → 256 (ConvTranspose + Conv)
#    ↓
# Upsample 4x: 256 → 128
#    ↓
# Upsample 4x: 128 → 64
#    ↓
# Upsample 4x: 64 → 32
#    ↓
# Residual Blocks: 3x (refine details)
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
from src.constants import NUM_ACOUSTIC_UNITS, UNIT_EMBED_DIM, HOP_SIZE, LEAKY_RELU_SLOPE


# %%
class Generator(nn.Module):
    """
    Unit-to-Waveform Generator (Vocoder)
    
    Converts discrete acoustic units to raw audio waveform.
    Upsampling factor: 5 × 4 × 4 × 4 = 320x (matches XLSR-53 hop size)
    
    Args:
        num_units: Number of discrete acoustic units (default: 100)
        embed_dim: Embedding dimension (default: 256)
    
    Input:
        x: Unit indices [batch_size, sequence_length]
    
    Output:
        audio: Waveform [batch_size, sequence_length * 320]
    """
    
    def __init__(self, num_units: int = NUM_ACOUSTIC_UNITS, embed_dim: int = UNIT_EMBED_DIM):
        super().__init__()
        
        self.num_units = num_units
        self.embed_dim = embed_dim
        self.upsample_factor = HOP_SIZE  # 5 * 4 * 4 * 4 = 320
        
        self.unit_embed = nn.Embedding(num_units, embed_dim)
        self.pre_conv = nn.Conv1d(embed_dim, 512, kernel_size=7, padding=3)
        
        self.upsamples = nn.ModuleList([
            self._make_upsample_block(512, 256, kernel=10, stride=5),
            self._make_upsample_block(256, 128, kernel=8, stride=4),
            self._make_upsample_block(128, 64, kernel=8, stride=4),
            self._make_upsample_block(64, 32, kernel=8, stride=4),
        ])
        
        self.res_blocks = nn.ModuleList([
            self._make_res_block(32) for _ in range(3)
        ])
        
        self.post_conv = nn.Conv1d(32, 1, kernel_size=7, padding=3)
        
    def _make_upsample_block(self, in_ch: int, out_ch: int, kernel: int, stride: int) -> nn.Sequential:
        padding = (kernel - stride) // 2
        return nn.Sequential(
            nn.ConvTranspose1d(in_ch, out_ch, kernel, stride=stride, padding=padding),
            nn.LeakyReLU(LEAKY_RELU_SLOPE),
            nn.Conv1d(out_ch, out_ch, kernel_size=7, padding=3),  # Reduce artifacts
            nn.LeakyReLU(LEAKY_RELU_SLOPE),
        )
    
    def _make_res_block(self, channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.LeakyReLU(LEAKY_RELU_SLOPE),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.unit_embed(x)
        x = x.transpose(1, 2)
        x = self.pre_conv(x)
        x = F.leaky_relu(x, LEAKY_RELU_SLOPE)
        
        for upsample in self.upsamples:
            x = upsample(x)
        
        for res_block in self.res_blocks:
            x = x + res_block(x) * LEAKY_RELU_SLOPE
        
        x = self.post_conv(x)
        x = torch.tanh(x)
        
        return x.squeeze(1)


# %% [markdown]
# ## Usage Example
#
# ```python
# # Create generator
# generator = Generator(num_units=100)
#
# # Input: batch of unit sequences
# units = torch.randint(0, 100, (4, 50))  # 4 sequences of 50 units
#
# # Generate audio
# audio = generator(units)  # Shape: [4, 16000] (1 second at 16kHz)
# ```

# %%
if __name__ == "__main__":
    # Test the generator
    generator = Generator(num_units=NUM_ACOUSTIC_UNITS)
    print(f"Generator parameters: {sum(p.numel() for p in generator.parameters()):,}")
    
    # Test forward pass
    units = torch.randint(0, NUM_ACOUSTIC_UNITS, (2, 50))  # 2 sequences of 50 units
    audio = generator(units)
    print(f"Input shape: {units.shape}")
    print(f"Output shape: {audio.shape}")
    print(f"Expected output length: {50 * HOP_SIZE} = {50 * HOP_SIZE}")
