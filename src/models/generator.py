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
    
    def __init__(self, num_units: int = 100, embed_dim: int = 256):
        super().__init__()
        
        self.num_units = num_units
        self.embed_dim = embed_dim
        self.upsample_factor = 320  # 5 * 4 * 4 * 4
        
        # Unit embedding
        self.unit_embed = nn.Embedding(num_units, embed_dim)
        
        # Pre-processing convolution
        self.pre_conv = nn.Conv1d(embed_dim, 512, kernel_size=7, padding=3)
        
        # Upsampling blocks: 5 × 4 × 4 × 4 = 320x total
        self.upsamples = nn.ModuleList([
            self._make_upsample_block(512, 256, kernel=10, stride=5),
            self._make_upsample_block(256, 128, kernel=8, stride=4),
            self._make_upsample_block(128, 64, kernel=8, stride=4),
            self._make_upsample_block(64, 32, kernel=8, stride=4),
        ])
        
        # Residual blocks for refinement
        self.res_blocks = nn.ModuleList([
            self._make_res_block(32) for _ in range(3)
        ])
        
        # Post-processing convolution
        self.post_conv = nn.Conv1d(32, 1, kernel_size=7, padding=3)
        
    def _make_upsample_block(self, in_ch: int, out_ch: int, kernel: int, stride: int) -> nn.Sequential:
        """Create an upsampling block with transposed convolution."""
        padding = (kernel - stride) // 2
        return nn.Sequential(
            nn.ConvTranspose1d(in_ch, out_ch, kernel, stride=stride, padding=padding),
            nn.LeakyReLU(0.1),
            nn.Conv1d(out_ch, out_ch, kernel_size=7, padding=3),  # Reduce artifacts
            nn.LeakyReLU(0.1),
        )
    
    def _make_res_block(self, channels: int) -> nn.Sequential:
        """Create a residual block for detail refinement."""
        return nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Unit indices [batch_size, sequence_length]
            
        Returns:
            audio: Waveform [batch_size, sequence_length * 320]
        """
        # Embed units: [B, T] -> [B, T, embed_dim]
        x = self.unit_embed(x)
        
        # Transpose for conv1d: [B, T, C] -> [B, C, T]
        x = x.transpose(1, 2)
        
        # Pre-processing
        x = self.pre_conv(x)
        x = F.leaky_relu(x, 0.1)
        
        # Upsampling (320x total)
        for upsample in self.upsamples:
            x = upsample(x)
        
        # Residual refinement
        for res_block in self.res_blocks:
            x = x + res_block(x) * 0.1
        
        # Post-processing
        x = self.post_conv(x)
        x = torch.tanh(x)
        
        # Remove channel dimension: [B, 1, T*320] -> [B, T*320]
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
    generator = Generator(num_units=100)
    print(f"Generator parameters: {sum(p.numel() for p in generator.parameters()):,}")
    
    # Test forward pass
    units = torch.randint(0, 100, (2, 50))  # 2 sequences of 50 units
    audio = generator(units)
    print(f"Input shape: {units.shape}")
    print(f"Output shape: {audio.shape}")
    print(f"Expected output length: {50 * 320} = {50 * 320}")
