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
# # Discriminator Model
#
# Multi-scale discriminator for GAN-based vocoder training.
# Operates at multiple audio resolutions for better quality assessment.
#
# ## Architecture
#
# ```
# Input: Audio waveform [B, T]
#    ↓
# Scale 1: Original resolution
#    ├─ Conv layers → Score 1
#    │
# Scale 2: 2x downsampled (AvgPool)
#    ├─ Conv layers → Score 2
#    │
# Scale 3: 4x downsampled (AvgPool)
#    └─ Conv layers → Score 3
#
# Output: List of scores [Score1, Score2, Score3]
# ```
#
# Multi-scale discrimination helps capture:
# - Fine details (high frequency) at original scale
# - Global structure (low frequency) at downsampled scales

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


# %%
class MultiScaleDiscriminator(nn.Module):
    """
    Multi-scale discriminator for audio quality assessment.
    
    Uses 3 discriminators at different audio resolutions to capture
    both fine details and global structure.
    
    Input:
        x: Audio waveform [batch_size, sequence_length]
    
    Output:
        outputs: List of discriminator outputs at each scale
    """
    
    def __init__(self):
        super().__init__()
        
        # Three discriminators for different scales
        self.discriminators = nn.ModuleList([
            self._make_discriminator(),
            self._make_discriminator(),
            self._make_discriminator(),
        ])
        
        # Downsampling for multi-scale
        self.downsample = nn.AvgPool1d(kernel_size=4, stride=2, padding=1)
        
    def _make_discriminator(self) -> nn.Sequential:
        """Create a single discriminator network."""
        return nn.Sequential(
            # Initial convolution
            nn.Conv1d(1, 16, kernel_size=15, stride=1, padding=7),
            nn.LeakyReLU(0.1),
            
            # Strided convolutions (downsample)
            nn.Conv1d(16, 64, kernel_size=41, stride=4, padding=20, groups=4),
            nn.LeakyReLU(0.1),
            
            nn.Conv1d(64, 256, kernel_size=41, stride=4, padding=20, groups=16),
            nn.LeakyReLU(0.1),
            
            nn.Conv1d(256, 512, kernel_size=41, stride=4, padding=20, groups=16),
            nn.LeakyReLU(0.1),
            
            # Final layers
            nn.Conv1d(512, 512, kernel_size=5, stride=1, padding=2),
            nn.LeakyReLU(0.1),
            
            # Output score
            nn.Conv1d(512, 1, kernel_size=3, stride=1, padding=1),
        )
        
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Forward pass at multiple scales.
        
        Args:
            x: Audio waveform [batch_size, sequence_length]
            
        Returns:
            outputs: List of discriminator outputs at each scale
        """
        # Add channel dimension: [B, T] -> [B, 1, T]
        x = x.unsqueeze(1)
        
        outputs = []
        for i, disc in enumerate(self.discriminators):
            outputs.append(disc(x))
            if i < len(self.discriminators) - 1:
                x = self.downsample(x)
        
        return outputs


# %% [markdown]
# ## Multi-Period Discriminator (HiFi-GAN Style)
#
# An alternative discriminator design that looks at different periodic patterns.
# This helps detect artifacts in voiced speech which has strong periodicity.

# %%
class MultiPeriodDiscriminator(nn.Module):
    """
    Multi-period discriminator (HiFi-GAN style).
    
    Reshapes audio into 2D based on different periods and applies
    2D convolutions to detect periodic artifacts.
    
    Periods: [2, 3, 5, 7, 11] (prime numbers for diverse patterns)
    """
    
    def __init__(self, periods: List[int] = None):
        super().__init__()
        
        if periods is None:
            periods = [2, 3, 5, 7, 11]
        
        self.periods = periods
        self.discriminators = nn.ModuleList([
            self._make_period_discriminator() for _ in periods
        ])
    
    def _make_period_discriminator(self) -> nn.Sequential:
        """Create a period discriminator with 2D convolutions."""
        return nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(5, 1), stride=(3, 1), padding=(2, 0)),
            nn.LeakyReLU(0.1),
            nn.Conv2d(32, 128, kernel_size=(5, 1), stride=(3, 1), padding=(2, 0)),
            nn.LeakyReLU(0.1),
            nn.Conv2d(128, 256, kernel_size=(5, 1), stride=(3, 1), padding=(2, 0)),
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, 512, kernel_size=(5, 1), stride=(3, 1), padding=(2, 0)),
            nn.LeakyReLU(0.1),
            nn.Conv2d(512, 1, kernel_size=(3, 1), padding=(1, 0)),
        )
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Forward pass with periodic reshaping.
        
        Args:
            x: Audio waveform [batch_size, sequence_length]
            
        Returns:
            outputs: List of discriminator outputs for each period
        """
        outputs = []
        
        for period, disc in zip(self.periods, self.discriminators):
            # Pad to make divisible by period
            batch_size, seq_len = x.shape
            if seq_len % period != 0:
                pad_len = period - (seq_len % period)
                x_padded = F.pad(x, (0, pad_len), mode='reflect')
            else:
                x_padded = x
            
            # Reshape: [B, T] -> [B, 1, T/period, period]
            x_reshaped = x_padded.view(batch_size, 1, -1, period)
            
            # Apply 2D convolutions
            output = disc(x_reshaped)
            outputs.append(output)
        
        return outputs


# %% [markdown]
# ## Usage Example
#
# ```python
# # Create discriminators
# msd = MultiScaleDiscriminator()
# mpd = MultiPeriodDiscriminator()
#
# # Input: batch of audio
# audio = torch.randn(4, 16000)  # 4 samples, 1 second each
#
# # Get scores
# msd_outputs = msd(audio)  # List of 3 tensors
# mpd_outputs = mpd(audio)  # List of 5 tensors
# ```

# %%
if __name__ == "__main__":
    # Test multi-scale discriminator
    msd = MultiScaleDiscriminator()
    print(f"MSD parameters: {sum(p.numel() for p in msd.parameters()):,}")
    
    audio = torch.randn(2, 16000)
    outputs = msd(audio)
    print(f"MSD output shapes: {[o.shape for o in outputs]}")
    
    # Test multi-period discriminator
    mpd = MultiPeriodDiscriminator()
    print(f"\nMPD parameters: {sum(p.numel() for p in mpd.parameters()):,}")
    
    outputs = mpd(audio)
    print(f"MPD output shapes: {[o.shape for o in outputs]}")
