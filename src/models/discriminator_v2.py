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
# # Discriminator V2 - HiFi-GAN Style with Feature Matching
#
# This is an upgraded discriminator architecture based on HiFi-GAN with:
#
# 1. **Multi-Period Discriminator (MPD)**: Detects periodic artifacts in voiced speech
# 2. **Multi-Scale Discriminator (MSD)**: Captures patterns at multiple resolutions
# 3. **Spectral Normalization**: Prevents discriminator from becoming too strong
# 4. **Feature Extraction**: Returns intermediate features for feature matching loss
#
# ## Architecture
#
# ```
# Input: Audio waveform [B, T]
#    ↓
# ┌─────────────────────────────────────────────┐
# │  Multi-Period Discriminator (MPD)            │
# │  - Period 2: Detects 8kHz patterns          │
# │  - Period 3: Detects 5.3kHz patterns        │
# │  - Period 5: Detects 3.2kHz patterns        │
# │  - Period 7: Detects 2.3kHz patterns        │
# │  - Period 11: Detects 1.5kHz patterns       │
# │  Each returns: (score, features)            │
# └─────────────────────────────────────────────┘
#    +
# ┌─────────────────────────────────────────────┐
# │  Multi-Scale Discriminator (MSD)            │
# │  - Scale 1: Original resolution             │
# │  - Scale 2: 2x downsampled                  │
# │  - Scale 3: 4x downsampled                  │
# │  Each returns: (score, features)            │
# └─────────────────────────────────────────────┘
#    ↓
# Output: List of (scores, features) tuples
# ```

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm, weight_norm
from typing import List, Tuple, Optional
from src.constants import LEAKY_RELU_SLOPE, MPD_PERIODS


# %%
def get_padding(kernel_size: int, dilation: int = 1) -> int:
    return (kernel_size * dilation - dilation) // 2


# %% [markdown]
# ## Period Discriminator
#
# A single discriminator that reshapes audio by a specific period and applies 2D convolutions.
# This helps detect periodic artifacts that are common in synthesized speech.

# %%
class PeriodDiscriminator(nn.Module):
    def __init__(self, period: int, use_spectral_norm: bool = True):
        super().__init__()
        
        self.period = period
        norm_fn = spectral_norm if use_spectral_norm else weight_norm
        
        self.convs = nn.ModuleList([
            norm_fn(nn.Conv2d(1, 32, (5, 1), (3, 1), padding=(2, 0))),
            norm_fn(nn.Conv2d(32, 128, (5, 1), (3, 1), padding=(2, 0))),
            norm_fn(nn.Conv2d(128, 512, (5, 1), (3, 1), padding=(2, 0))),
            norm_fn(nn.Conv2d(512, 1024, (5, 1), (3, 1), padding=(2, 0))),
            norm_fn(nn.Conv2d(1024, 1024, (5, 1), 1, padding=(2, 0))),
        ])
        
        self.conv_post = norm_fn(nn.Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        features = []
        
        batch_size, channels, seq_len = x.shape
        
        if seq_len % self.period != 0:
            n_pad = self.period - (seq_len % self.period)
            x = F.pad(x, (0, n_pad), mode='reflect')
            seq_len = seq_len + n_pad
        
        x = x.view(batch_size, channels, seq_len // self.period, self.period)
        
        for conv in self.convs:
            x = conv(x)
            x = F.leaky_relu(x, LEAKY_RELU_SLOPE)
            features.append(x)
        
        x = self.conv_post(x)
        features.append(x)
        x = torch.flatten(x, 1, -1)
        
        return x, features


# %% [markdown]
# ## Multi-Period Discriminator (MPD)
#
# Combines multiple period discriminators with different periods.

# %%
class MultiPeriodDiscriminatorV2(nn.Module):
    """Prime periods ensure diverse and non-overlapping patterns."""
    
    def __init__(self, periods: List[int] = None, use_spectral_norm: bool = True):
        super().__init__()
        
        if periods is None:
            periods = list(MPD_PERIODS)
        
        self.discriminators = nn.ModuleList([
            PeriodDiscriminator(p, use_spectral_norm) for p in periods
        ])
    
    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], List[List[torch.Tensor]]]:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        scores = []
        features = []
        
        for disc in self.discriminators:
            score, feat = disc(x)
            scores.append(score)
            features.append(feat)
        
        return scores, features


# %% [markdown]
# ## Scale Discriminator
#
# A single discriminator that operates at a specific audio resolution.

# %%
class ScaleDiscriminator(nn.Module):
    def __init__(self, use_spectral_norm: bool = True):
        super().__init__()
        
        norm_fn = spectral_norm if use_spectral_norm else weight_norm
        
        self.convs = nn.ModuleList([
            norm_fn(nn.Conv1d(1, 128, 15, 1, padding=7)),
            norm_fn(nn.Conv1d(128, 128, 41, 2, groups=4, padding=20)),
            norm_fn(nn.Conv1d(128, 256, 41, 2, groups=16, padding=20)),
            norm_fn(nn.Conv1d(256, 512, 41, 4, groups=16, padding=20)),
            norm_fn(nn.Conv1d(512, 1024, 41, 4, groups=16, padding=20)),
            norm_fn(nn.Conv1d(1024, 1024, 41, 1, groups=16, padding=20)),
            norm_fn(nn.Conv1d(1024, 1024, 5, 1, padding=2)),
        ])
        
        self.conv_post = norm_fn(nn.Conv1d(1024, 1, 3, 1, padding=1))
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        features = []
        
        for conv in self.convs:
            x = conv(x)
            x = F.leaky_relu(x, LEAKY_RELU_SLOPE)
            features.append(x)
        
        x = self.conv_post(x)
        features.append(x)
        x = torch.flatten(x, 1, -1)
        
        return x, features


# %% [markdown]
# ## Multi-Scale Discriminator (MSD)
#
# Combines multiple scale discriminators at different resolutions.

# %%
class MultiScaleDiscriminatorV2(nn.Module):
    def __init__(self, use_spectral_norm: bool = True):
        super().__init__()
        
        self.discriminators = nn.ModuleList([
            ScaleDiscriminator(use_spectral_norm),
            ScaleDiscriminator(use_spectral_norm),
            ScaleDiscriminator(use_spectral_norm),
        ])
        
        self.pools = nn.ModuleList([
            nn.AvgPool1d(4, 2, padding=2),
            nn.AvgPool1d(4, 2, padding=2),
        ])
    
    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], List[List[torch.Tensor]]]:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        scores = []
        features = []
        
        for i, disc in enumerate(self.discriminators):
            score, feat = disc(x)
            scores.append(score)
            features.append(feat)
            
            if i < len(self.pools):
                x = self.pools[i](x)
        
        return scores, features


# %% [markdown]
# ## Combined Discriminator
#
# Combines MPD and MSD for comprehensive audio quality assessment.

# %%
class CombinedDiscriminatorV2(nn.Module):
    def __init__(self, mpd_periods: List[int] = None, use_spectral_norm: bool = True):
        super().__init__()
        
        self.mpd = MultiPeriodDiscriminatorV2(mpd_periods, use_spectral_norm)
        self.msd = MultiScaleDiscriminatorV2(use_spectral_norm)
    
    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], List[List[torch.Tensor]]]:
        mpd_scores, mpd_features = self.mpd(x)
        msd_scores, msd_features = self.msd(x)
        
        scores = mpd_scores + msd_scores
        features = mpd_features + msd_features
        
        return scores, features


# %% [markdown]
# ## Loss Functions
#
# Helper functions for computing discriminator and generator losses.

# %%
def discriminator_loss(
    real_outputs: List[torch.Tensor], 
    fake_outputs: List[torch.Tensor]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """LSGAN discriminator loss."""
    real_loss = 0
    fake_loss = 0
    
    for real_out, fake_out in zip(real_outputs, fake_outputs):
        real_loss += torch.mean((1 - real_out) ** 2)
        fake_loss += torch.mean(fake_out ** 2)
    
    loss = real_loss + fake_loss
    return loss, real_loss, fake_loss


def generator_adversarial_loss(fake_outputs: List[torch.Tensor]) -> torch.Tensor:
    """LSGAN generator loss."""
    loss = 0
    for fake_out in fake_outputs:
        loss += torch.mean((1 - fake_out) ** 2)
    return loss


def feature_matching_loss(
    real_features: List[List[torch.Tensor]], 
    fake_features: List[List[torch.Tensor]]
) -> torch.Tensor:
    loss = 0
    for real_feat_list, fake_feat_list in zip(real_features, fake_features):
        for real_feat, fake_feat in zip(real_feat_list, fake_feat_list):
            loss += F.l1_loss(fake_feat, real_feat.detach())
    return loss


def gradient_penalty(
    discriminator: nn.Module,
    real_audio: torch.Tensor,
    fake_audio: torch.Tensor,
    lambda_gp: float = 10.0
) -> torch.Tensor:
    """WGAN-GP gradient penalty for training stability."""
    batch_size = real_audio.size(0)
    device = real_audio.device
    
    alpha = torch.rand(batch_size, 1, device=device)
    alpha = alpha.expand_as(real_audio)
    
    interpolated = alpha * real_audio + (1 - alpha) * fake_audio
    interpolated.requires_grad_(True)
    
    scores, _ = discriminator(interpolated)
    
    gradients = torch.autograd.grad(
        outputs=scores[0].sum(),  # Use first discriminator
        inputs=interpolated,
        create_graph=True,
        retain_graph=True,
    )[0]
    
    gradients = gradients.view(batch_size, -1)
    gradient_norm = gradients.norm(2, dim=1)
    penalty = lambda_gp * ((gradient_norm - 1) ** 2).mean()
    
    return penalty


# %% [markdown]
# ## Usage Example
#
# ```python
# # Create discriminator
# discriminator = CombinedDiscriminatorV2()
#
# # Input: batch of audio
# real_audio = torch.randn(4, 16000)  # 4 samples, 1 second each
# fake_audio = torch.randn(4, 16000)
#
# # Get scores and features
# real_scores, real_features = discriminator(real_audio)
# fake_scores, fake_features = discriminator(fake_audio)
#
# # Compute losses
# d_loss, _, _ = discriminator_loss(real_scores, fake_scores)
# g_adv_loss = generator_adversarial_loss(fake_scores)
# fm_loss = feature_matching_loss(real_features, fake_features)
# ```

# %%
if __name__ == "__main__":
    # Test multi-period discriminator
    mpd = MultiPeriodDiscriminatorV2()
    print(f"MPD parameters: {sum(p.numel() for p in mpd.parameters()):,}")
    
    audio = torch.randn(2, 16000)
    scores, features = mpd(audio)
    print(f"MPD scores: {len(scores)}, features per disc: {[len(f) for f in features]}")
    
    # Test multi-scale discriminator
    msd = MultiScaleDiscriminatorV2()
    print(f"\nMSD parameters: {sum(p.numel() for p in msd.parameters()):,}")
    
    scores, features = msd(audio)
    print(f"MSD scores: {len(scores)}, features per disc: {[len(f) for f in features]}")
    
    # Test combined discriminator
    combined = CombinedDiscriminatorV2()
    print(f"\nCombined parameters: {sum(p.numel() for p in combined.parameters()):,}")
    
    real_scores, real_features = combined(audio)
    fake_scores, fake_features = combined(torch.randn(2, 16000))
    
    print(f"Total scores: {len(real_scores)}")
    print(f"Total feature lists: {len(real_features)}")
    
    # Test losses
    d_loss, real_loss, fake_loss = discriminator_loss(real_scores, fake_scores)
    g_loss = generator_adversarial_loss(fake_scores)
    fm_loss = feature_matching_loss(real_features, fake_features)
    
    print(f"\nD Loss: {d_loss.item():.4f}")
    print(f"G Adv Loss: {g_loss.item():.4f}")
    print(f"FM Loss: {fm_loss.item():.4f}")
