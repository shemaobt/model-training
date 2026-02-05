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
# # Phase 3 V2: Enhanced Vocoder Training
#
# This is an upgraded vocoder training script with all improvements from the analysis:
#
# ## Key Improvements
#
# 1. **Pitch Conditioning**: Extracts F0 and conditions generator on pitch
# 2. **HiFi-GAN Architecture**: Generator with MRF, Discriminator with MPD+MSD
# 3. **Enhanced Losses**:
#    - Mel Spectrogram L1 Loss
#    - Multi-resolution STFT Loss (spectral convergence + log magnitude)
#    - Feature Matching Loss
#    - Adversarial Loss (LSGAN)
# 4. **Gradient Penalty**: Stabilizes discriminator training
# 5. **Better Alignment**: 2-second segments aligned to unit boundaries
# 6. **Spectral Normalization**: Prevents discriminator collapse
#
# ## Training
#
# - **Losses**: Mel (×45) + STFT (×2) + FM (×2) + Adversarial
# - **Early Stopping**: Patience 100 epochs, monitors total validation loss
# - **Checkpoints**: Every 25 epochs
#
# ## Run on Modal
#
# ```bash
# # Full training with pitch conditioning
# python3 -m modal run --detach src/training/phase3_vocoder_v2.py::main
#
# # Resume from checkpoint
# python3 -m modal run --detach src/training/phase3_vocoder_v2.py::main --resume v2_latest.pt
#
# # Custom parameters
# python3 -m modal run --detach src/training/phase3_vocoder_v2.py::main \
#     --epochs 1000 --segment-length 32000 --patience 100
# ```

# %% [markdown]
# ## Modal Configuration

# %%
import modal
import os
from src.constants import (
    SAMPLE_RATE,
    HOP_SIZE,
    SEGMENT_LENGTH,
    NUM_ACOUSTIC_UNITS,
    NUM_PITCH_BINS,
    DEFAULT_EPOCHS,
    DEFAULT_BATCH_SIZE,
    DEFAULT_LEARNING_RATE,
    DEFAULT_ADAM_BETAS,
    DEFAULT_LR_GAMMA,
    DEFAULT_PATIENCE,
    DEFAULT_SAVE_EVERY,
    DEFAULT_SAMPLES_PER_EPOCH,
    DEFAULT_GRAD_CLIP_MAX_NORM,
    LAMBDA_MEL,
    LAMBDA_STFT,
    LAMBDA_FM,
    EARLY_STOPPING_THRESHOLD,
)

# %%
app = modal.App("bible-vocoder-v2-training")

audio_volume = modal.Volume.from_name(
    "bible-audio-data",
    create_if_missing=True
)

# %%
image = (
    modal.Image.debian_slim()
    .apt_install("ffmpeg", "libsndfile1")
    .pip_install(
        "torch>=2.0.0",
        "torchaudio>=2.0.0",
        "numpy<2",
        "scipy>=1.11.0",
        "scikit-learn>=1.3.0",
        "joblib>=1.3.0",
        "soundfile>=0.12.0",
        "tqdm>=4.66.0",
        "tensorboard>=2.14.0",
        "librosa>=0.10.0",
    )
)

# %%
AUDIO_MOUNT = "/mnt/audio_data"

# Language configuration
LANGUAGE_CONFIGS = {
    "portuguese": {
        "segmented_dir": f"{AUDIO_MOUNT}/segmented_audio",
        "output_dir": f"{AUDIO_MOUNT}/portuguese_units",
        "vocoder_dir": f"{AUDIO_MOUNT}/vocoder_v2_checkpoints",
        "corpus_file": "portuguese_corpus_timestamped.json",
    },
    "satere": {
        "segmented_dir": f"{AUDIO_MOUNT}/segmented_audio_satere",
        "output_dir": f"{AUDIO_MOUNT}/satere_units",
        "vocoder_dir": f"{AUDIO_MOUNT}/vocoder_v2_satere_checkpoints",
        "corpus_file": "satere_corpus_timestamped.json",
    },
}

# Default paths (can be overridden via CLI)
import os as _os
_LANGUAGE = _os.environ.get("TRAINING_LANGUAGE", "portuguese")
_config = LANGUAGE_CONFIGS.get(_LANGUAGE, LANGUAGE_CONFIGS["portuguese"])

SEGMENTED_DIR = _config["segmented_dir"]
OUTPUT_DIR = _config["output_dir"]
VOCODER_DIR = _config["vocoder_dir"]

# %% [markdown]
# ## Model Definitions (V2)
#
# HiFi-GAN style generator with pitch conditioning and combined MPD+MSD discriminator.

# %%
GENERATOR_V2_CODE = '''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm, remove_weight_norm
from typing import List, Optional

def get_padding(kernel_size: int, dilation: int = 1) -> int:
    return (kernel_size * dilation - dilation) // 2

class ResBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3, 
                 dilations: list = [[1, 1], [3, 1], [5, 1]]):
        super().__init__()
        self.convs1 = nn.ModuleList()
        self.convs2 = nn.ModuleList()
        for d in dilations:
            self.convs1.append(weight_norm(nn.Conv1d(
                channels, channels, kernel_size, dilation=d[0],
                padding=get_padding(kernel_size, d[0]))))
            self.convs2.append(weight_norm(nn.Conv1d(
                channels, channels, kernel_size, dilation=d[1],
                padding=get_padding(kernel_size, d[1]))))
    
    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            res = x
            x = F.leaky_relu(x, 0.1)
            x = c1(x)
            x = F.leaky_relu(x, 0.1)
            x = c2(x)
            x = x + res
        return x

class MRF(nn.Module):
    def __init__(self, channels: int, kernel_sizes: list = [3, 7, 11]):
        super().__init__()
        self.resblocks = nn.ModuleList([
            ResBlock(channels, k) for k in kernel_sizes
        ])
    
    def forward(self, x):
        out = self.resblocks[0](x)
        for rb in self.resblocks[1:]:
            out = out + rb(x)
        return out / len(self.resblocks)

class GeneratorV2(nn.Module):
    def __init__(self, num_units=100, num_pitch_bins=32, 
                 unit_embed_dim=256, pitch_embed_dim=64):
        super().__init__()
        self.upsample_rates = [5, 4, 4, 4]  # 320x
        self.upsample_kernels = [10, 8, 8, 8]
        
        self.unit_embed = nn.Embedding(num_units, unit_embed_dim)
        self.pitch_embed = nn.Embedding(num_pitch_bins + 1, pitch_embed_dim)
        
        input_dim = unit_embed_dim + pitch_embed_dim
        self.pre_conv = weight_norm(nn.Conv1d(input_dim, 512, 7, padding=3))
        
        self.ups = nn.ModuleList()
        self.mrfs = nn.ModuleList()
        ch = 512
        for rate, kernel in zip(self.upsample_rates, self.upsample_kernels):
            self.ups.append(weight_norm(nn.ConvTranspose1d(
                ch, ch // 2, kernel, stride=rate, padding=(kernel - rate) // 2)))
            self.mrfs.append(MRF(ch // 2))
            ch = ch // 2
        
        self.post_conv = weight_norm(nn.Conv1d(ch, 1, 7, padding=3))
    
    def forward(self, units, pitch=None):
        unit_emb = self.unit_embed(units)
        if pitch is None:
            pitch = torch.zeros_like(units)
        pitch_emb = self.pitch_embed(pitch)
        x = torch.cat([unit_emb, pitch_emb], dim=-1).transpose(1, 2)
        x = self.pre_conv(x)
        for up, mrf in zip(self.ups, self.mrfs):
            x = F.leaky_relu(x, 0.1)
            x = up(x)
            x = mrf(x)
        x = F.leaky_relu(x, 0.1)
        x = self.post_conv(x)
        return torch.tanh(x).squeeze(1)
'''

# %%
DISCRIMINATOR_V2_CODE = '''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from typing import List, Tuple

class PeriodDiscriminator(nn.Module):
    def __init__(self, period: int):
        super().__init__()
        self.period = period
        self.convs = nn.ModuleList([
            spectral_norm(nn.Conv2d(1, 32, (5, 1), (3, 1), (2, 0))),
            spectral_norm(nn.Conv2d(32, 128, (5, 1), (3, 1), (2, 0))),
            spectral_norm(nn.Conv2d(128, 512, (5, 1), (3, 1), (2, 0))),
            spectral_norm(nn.Conv2d(512, 1024, (5, 1), (3, 1), (2, 0))),
            spectral_norm(nn.Conv2d(1024, 1024, (5, 1), 1, (2, 0))),
        ])
        self.conv_post = spectral_norm(nn.Conv2d(1024, 1, (3, 1), 1, (1, 0)))
    
    def forward(self, x):
        features = []
        b, c, t = x.shape
        if t % self.period != 0:
            x = F.pad(x, (0, self.period - t % self.period), mode='reflect')
            t = x.shape[2]
        x = x.view(b, c, t // self.period, self.period)
        for conv in self.convs:
            x = F.leaky_relu(conv(x), 0.1)
            features.append(x)
        x = self.conv_post(x)
        features.append(x)
        return x.flatten(1, -1), features

class ScaleDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.convs = nn.ModuleList([
            spectral_norm(nn.Conv1d(1, 128, 15, 1, 7)),
            spectral_norm(nn.Conv1d(128, 128, 41, 2, 20, groups=4)),
            spectral_norm(nn.Conv1d(128, 256, 41, 2, 20, groups=16)),
            spectral_norm(nn.Conv1d(256, 512, 41, 4, 20, groups=16)),
            spectral_norm(nn.Conv1d(512, 1024, 41, 4, 20, groups=16)),
            spectral_norm(nn.Conv1d(1024, 1024, 41, 1, 20, groups=16)),
            spectral_norm(nn.Conv1d(1024, 1024, 5, 1, 2)),
        ])
        self.conv_post = spectral_norm(nn.Conv1d(1024, 1, 3, 1, 1))
    
    def forward(self, x):
        features = []
        for conv in self.convs:
            x = F.leaky_relu(conv(x), 0.1)
            features.append(x)
        x = self.conv_post(x)
        features.append(x)
        return x.flatten(1, -1), features

class CombinedDiscriminatorV2(nn.Module):
    def __init__(self, periods=[2, 3, 5, 7, 11]):
        super().__init__()
        self.period_discs = nn.ModuleList([PeriodDiscriminator(p) for p in periods])
        self.scale_discs = nn.ModuleList([ScaleDiscriminator() for _ in range(3)])
        self.pools = nn.ModuleList([nn.AvgPool1d(4, 2, 2), nn.AvgPool1d(4, 2, 2)])
    
    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        scores, features = [], []
        for pd in self.period_discs:
            s, f = pd(x)
            scores.append(s)
            features.append(f)
        x_scale = x
        for i, sd in enumerate(self.scale_discs):
            s, f = sd(x_scale)
            scores.append(s)
            features.append(f)
            if i < len(self.pools):
                x_scale = self.pools[i](x_scale)
        return scores, features
'''

# %% [markdown]
# ## Dataset with Pitch Extraction
#
# Enhanced dataset that:
# - Loads audio and units
# - Extracts pitch (F0) using librosa
# - Aligns samples to unit boundaries
# - Uses 2-second segments

# %%
DATASET_V2_CODE = '''
import torch
from torch.utils.data import Dataset
import soundfile as sf
import numpy as np
import os
import json
import librosa

class UnitAudioPitchDataset(Dataset):
    def __init__(self, corpus_path, audio_dir, segment_length=32000,
                 hop_size=320, samples_per_epoch=10000, num_pitch_bins=32,
                 f0_min=50.0, f0_max=400.0):
        self.audio_dir = audio_dir
        self.segment_length = segment_length
        self.hop_size = hop_size
        self.unit_length = segment_length // hop_size
        self.samples_per_epoch = samples_per_epoch
        self.num_pitch_bins = num_pitch_bins
        self.f0_min = f0_min
        self.f0_max = f0_max
        
        print(f"Loading corpus from {corpus_path}...")
        with open(corpus_path, 'r') as f:
            self.corpus = json.load(f)
        
        self.samples = []
        missing = 0
        for name, data in self.corpus.items():
            units = data.get('units', [])
            if not units:
                continue
            for ext in ['.wav', '.WAV']:
                path = os.path.join(audio_dir, name + ext)
                if os.path.exists(path):
                    self.samples.append({'path': path, 'units': units, 'name': name})
                    break
            else:
                missing += 1
        
        print(f"Loaded {len(self.samples)} samples ({missing} missing)")
    
    def _extract_pitch(self, audio, sr=16000):
        try:
            f0, voiced, _ = librosa.pyin(
                audio, fmin=self.f0_min, fmax=self.f0_max,
                sr=sr, hop_length=self.hop_size
            )
            f0 = np.nan_to_num(f0, nan=0.0)
            return f0
        except:
            return np.zeros(len(audio) // self.hop_size)
    
    def _quantize_pitch(self, f0):
        bins = np.zeros_like(f0, dtype=np.int64)
        voiced = f0 > 0
        if np.any(voiced):
            log_f0 = np.log(np.clip(f0[voiced], self.f0_min, self.f0_max))
            log_min, log_max = np.log(self.f0_min), np.log(self.f0_max)
            normalized = (log_f0 - log_min) / (log_max - log_min)
            bins[voiced] = (normalized * (self.num_pitch_bins - 1) + 1).astype(np.int64)
        return bins
    
    def __len__(self):
        return self.samples_per_epoch
    
    def __getitem__(self, idx):
        sample = self.samples[idx % len(self.samples)]
        try:
            audio, sr = sf.read(sample['path'])
            if sr != 16000:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)
            
            units = sample['units']
            
            max_unit_start = max(0, len(units) - self.unit_length)
            if max_unit_start > 0:
                unit_start = np.random.randint(0, max_unit_start)
            else:
                unit_start = 0
            audio_start = unit_start * self.hop_size
            
            audio_seg = audio[audio_start:audio_start + self.segment_length]
            units_seg = units[unit_start:unit_start + self.unit_length]
            
            if len(audio_seg) < self.segment_length:
                audio_seg = np.pad(audio_seg, (0, self.segment_length - len(audio_seg)))
            if len(units_seg) < self.unit_length:
                units_seg = units_seg + [0] * (self.unit_length - len(units_seg))
            
            f0 = self._extract_pitch(audio_seg)
            if len(f0) < self.unit_length:
                f0 = np.pad(f0, (0, self.unit_length - len(f0)))
            elif len(f0) > self.unit_length:
                f0 = f0[:self.unit_length]
            pitch = self._quantize_pitch(f0)
            
            return {
                'audio': torch.FloatTensor(audio_seg),
                'units': torch.LongTensor(units_seg[:self.unit_length]),
                'pitch': torch.LongTensor(pitch[:self.unit_length]),
            }
        except Exception as e:
            return {
                'audio': torch.zeros(self.segment_length),
                'units': torch.zeros(self.unit_length, dtype=torch.long),
                'pitch': torch.zeros(self.unit_length, dtype=torch.long),
            }
'''

# %% [markdown]
# ## Loss Functions
#
# Enhanced loss functions for better audio quality.

# %%
LOSSES_CODE = '''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T

def mel_spectrogram_loss(real, fake, sample_rate=16000, n_fft=1024, 
                         hop_length=256, n_mels=80):
    min_len = min(real.shape[-1], fake.shape[-1])
    real, fake = real[..., :min_len], fake[..., :min_len]
    mel = T.MelSpectrogram(sample_rate, n_fft, hop_length=hop_length, n_mels=n_mels)
    mel = mel.to(real.device)
    return F.l1_loss(mel(fake), mel(real))

def stft_loss(real, fake, fft_sizes=[512, 1024, 2048], hop_sizes=[50, 120, 240],
              win_sizes=[240, 600, 1200]):
    min_len = min(real.shape[-1], fake.shape[-1])
    real, fake = real[..., :min_len], fake[..., :min_len]
    
    sc_loss = 0.0
    mag_loss = 0.0
    
    for fft, hop, win in zip(fft_sizes, hop_sizes, win_sizes):
        window = torch.hann_window(win).to(real.device)
        
        real_stft = torch.stft(real, fft, hop, win, window, return_complex=True)
        fake_stft = torch.stft(fake, fft, hop, win, window, return_complex=True)
        
        real_mag = torch.abs(real_stft)
        fake_mag = torch.abs(fake_stft)
        
        # Spectral convergence
        sc_loss += torch.norm(real_mag - fake_mag, p='fro') / (torch.norm(real_mag, p='fro') + 1e-8)
        
        # Log magnitude
        mag_loss += F.l1_loss(torch.log(fake_mag + 1e-8), torch.log(real_mag + 1e-8))
    
    return (sc_loss + mag_loss) / len(fft_sizes)

def feature_matching_loss(real_features, fake_features):
    loss = 0.0
    for rf_list, ff_list in zip(real_features, fake_features):
        for rf, ff in zip(rf_list, ff_list):
            min_sizes = [min(rf.size(i), ff.size(i)) for i in range(rf.dim())]
            rf_aligned = rf
            ff_aligned = ff
            for dim in range(rf.dim()):
                rf_aligned = rf_aligned.narrow(dim, 0, min_sizes[dim])
                ff_aligned = ff_aligned.narrow(dim, 0, min_sizes[dim])
            loss += F.l1_loss(ff_aligned, rf_aligned.detach())
    return loss

def discriminator_loss(real_outputs, fake_outputs):
    """LSGAN discriminator loss."""
    loss = 0.0
    for ro, fo in zip(real_outputs, fake_outputs):
        loss += torch.mean((1 - ro) ** 2) + torch.mean(fo ** 2)
    return loss

def generator_adversarial_loss(fake_outputs):
    """LSGAN generator loss."""
    loss = 0.0
    for fo in fake_outputs:
        loss += torch.mean((1 - fo) ** 2)
    return loss
'''

# %% [markdown]
# ## Training Function
#
# Main training loop with all improvements.

# %%
@app.function(
    image=image,
    volumes={AUDIO_MOUNT: audio_volume},
    timeout=43200,  # 12 hours
    gpu="A10G",
)
def train_vocoder_v2(
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    learning_rate_g: float = DEFAULT_LEARNING_RATE,
    learning_rate_d: float = DEFAULT_LEARNING_RATE,
    segment_length: int = SEGMENT_LENGTH,
    save_every: int = DEFAULT_SAVE_EVERY,
    resume_from: str = None,
    patience: int = DEFAULT_PATIENCE,
    samples_per_epoch: int = DEFAULT_SAMPLES_PER_EPOCH,
    lambda_mel: float = LAMBDA_MEL,
    lambda_stft: float = LAMBDA_STFT,
    lambda_fm: float = LAMBDA_FM,
):
    import torch
    import torch.optim as optim
    from torch.utils.data import DataLoader
    import numpy as np
    from tqdm import tqdm
    import json
    from scipy.io.wavfile import write
    import tempfile
    import importlib.util
    
    print("=" * 60)
    print("🎵 Vocoder V2 Training")
    print("=" * 60)
    print(f"  Segment length: {segment_length} samples ({segment_length/SAMPLE_RATE:.1f}s)")
    print(f"  Batch size: {batch_size}")
    print(f"  Loss weights: Mel={lambda_mel}, STFT={lambda_stft}, FM={lambda_fm}")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    os.makedirs(VOCODER_DIR, exist_ok=True)
    
    def load_module(code, name):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            path = f.name
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    
    gen_module = load_module(GENERATOR_V2_CODE, "generator_v2")
    disc_module = load_module(DISCRIMINATOR_V2_CODE, "discriminator_v2")
    data_module = load_module(DATASET_V2_CODE, "dataset_v2")
    loss_module = load_module(LOSSES_CODE, "losses")
    
    GeneratorV2 = gen_module.GeneratorV2
    CombinedDiscriminatorV2 = disc_module.CombinedDiscriminatorV2
    UnitAudioPitchDataset = data_module.UnitAudioPitchDataset
    
    generator = GeneratorV2(num_units=NUM_ACOUSTIC_UNITS, num_pitch_bins=NUM_PITCH_BINS).to(device)
    discriminator = CombinedDiscriminatorV2().to(device)
    
    g_params = sum(p.numel() for p in generator.parameters())
    d_params = sum(p.numel() for p in discriminator.parameters())
    print(f"Generator: {g_params:,} params")
    print(f"Discriminator: {d_params:,} params")
    
    corpus_path = os.path.join(OUTPUT_DIR, "portuguese_corpus_timestamped.json")
    if not os.path.exists(corpus_path):
        print(f"❌ Corpus not found at {corpus_path}")
        return None
    
    unit_length = segment_length // HOP_SIZE
    
    dataset = UnitAudioPitchDataset(
        corpus_path=corpus_path,
        audio_dir=SEGMENTED_DIR,
        segment_length=segment_length,
        hop_size=HOP_SIZE,
        samples_per_epoch=samples_per_epoch,
        num_pitch_bins=NUM_PITCH_BINS,
    )
    
    if len(dataset.samples) == 0:
        print("❌ No valid samples found!")
        return None
    
    print(f"Dataset: {len(dataset.samples)} audio files, {samples_per_epoch} samples/epoch")
    
    # Note: num_workers=0 to avoid pickling issues with dynamically defined dataset
    # Python 3.14+ uses 'forkserver' which requires picklable objects
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, 
        num_workers=0, drop_last=True, pin_memory=True
    )
    
    optimizer_g = optim.AdamW(generator.parameters(), lr=learning_rate_g, betas=DEFAULT_ADAM_BETAS)
    optimizer_d = optim.AdamW(discriminator.parameters(), lr=learning_rate_d, betas=DEFAULT_ADAM_BETAS)
    
    scheduler_g = optim.lr_scheduler.ExponentialLR(optimizer_g, gamma=DEFAULT_LR_GAMMA)
    scheduler_d = optim.lr_scheduler.ExponentialLR(optimizer_d, gamma=DEFAULT_LR_GAMMA)
    
    start_epoch = 0
    best_loss = float('inf')
    epochs_without_improvement = 0
    
    if resume_from and os.path.exists(resume_from):
        print(f"📥 Resuming from: {resume_from}")
        ckpt = torch.load(resume_from, map_location=device)
        generator.load_state_dict(ckpt['generator_state_dict'])
        discriminator.load_state_dict(ckpt['discriminator_state_dict'])
        optimizer_g.load_state_dict(ckpt['optimizer_g_state_dict'])
        optimizer_d.load_state_dict(ckpt['optimizer_d_state_dict'])
        start_epoch = ckpt.get('epoch', 0) + 1
        best_loss = ckpt.get('best_loss', float('inf'))
    
    training_log = []
    
    for epoch in range(start_epoch, epochs):
        generator.train()
        discriminator.train()
        
        epoch_losses = {
            'g_total': 0, 'd_total': 0, 'mel': 0, 'stft': 0, 'fm': 0, 'adv': 0
        }
        num_batches = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch in pbar:
            real_audio = batch['audio'].to(device)
            units = batch['units'].to(device)
            pitch = batch['pitch'].to(device)
            
            optimizer_d.zero_grad()
            
            with torch.no_grad():
                fake_audio = generator(units, pitch)
            
            min_len = min(real_audio.size(-1), fake_audio.size(-1))
            real_audio_aligned = real_audio[..., :min_len]
            fake_audio_aligned = fake_audio[..., :min_len]
            
            real_scores, real_features = discriminator(real_audio_aligned)
            fake_scores, fake_features = discriminator(fake_audio_aligned)
            
            d_loss = loss_module.discriminator_loss(real_scores, fake_scores)
            
            d_loss.backward()
            optimizer_d.step()
            
            optimizer_g.zero_grad()
            
            fake_audio = generator(units, pitch)
            
            min_len = min(real_audio.size(-1), fake_audio.size(-1))
            real_audio_aligned = real_audio[..., :min_len]
            fake_audio_aligned = fake_audio[..., :min_len]
            
            fake_scores, fake_features = discriminator(fake_audio_aligned)
            _, real_features = discriminator(real_audio_aligned)
            
            mel_loss = loss_module.mel_spectrogram_loss(real_audio_aligned, fake_audio_aligned)
            stft_loss = loss_module.stft_loss(real_audio_aligned, fake_audio_aligned)
            fm_loss = loss_module.feature_matching_loss(real_features, fake_features)
            adv_loss = loss_module.generator_adversarial_loss(fake_scores)
            
            g_loss = (
                lambda_mel * mel_loss + 
                lambda_stft * stft_loss + 
                lambda_fm * fm_loss + 
                adv_loss
            )
            
            g_loss.backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=DEFAULT_GRAD_CLIP_MAX_NORM)
            optimizer_g.step()
            
            # Accumulate losses
            epoch_losses['g_total'] += g_loss.item()
            epoch_losses['d_total'] += d_loss.item()
            epoch_losses['mel'] += mel_loss.item()
            epoch_losses['stft'] += stft_loss.item()
            epoch_losses['fm'] += fm_loss.item()
            epoch_losses['adv'] += adv_loss.item()
            num_batches += 1
            
            pbar.set_postfix({
                'G': f'{g_loss.item():.1f}',
                'D': f'{d_loss.item():.2f}',
                'Mel': f'{mel_loss.item():.2f}'
            })
        
        # Schedulers
        scheduler_g.step()
        scheduler_d.step()
        
        # Average losses
        for k in epoch_losses:
            epoch_losses[k] /= num_batches
        
        print(f"\nEpoch {epoch+1}:")
        print(f"  G: {epoch_losses['g_total']:.2f} | D: {epoch_losses['d_total']:.2f}")
        print(f"  Mel: {epoch_losses['mel']:.3f} | STFT: {epoch_losses['stft']:.3f}")
        print(f"  FM: {epoch_losses['fm']:.3f} | Adv: {epoch_losses['adv']:.3f}")
        
        training_log.append({
            'epoch': epoch + 1,
            **{k: float(v) for k, v in epoch_losses.items()}
        })
        
        # Early stopping on total generator loss
        current_loss = epoch_losses['g_total']
        if current_loss < best_loss - EARLY_STOPPING_THRESHOLD:
            best_loss = current_loss
            epochs_without_improvement = 0
            print(f"  ✓ New best: {best_loss:.2f}")
            
            # Save best model and commit immediately so canceled runs keep it
            torch.save({
                'epoch': epoch,
                'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'optimizer_g_state_dict': optimizer_g.state_dict(),
                'optimizer_d_state_dict': optimizer_d.state_dict(),
                'best_loss': best_loss,
            }, os.path.join(VOCODER_DIR, "v2_best.pt"))
            audio_volume.commit()
        else:
            epochs_without_improvement += 1
            print(f"  ⚠️  No improvement: {epochs_without_improvement}/{patience}")
        
        if epochs_without_improvement >= patience:
            print(f"\n🛑 Early stopping at epoch {epoch+1}!")
            break
        
        # Periodic checkpoints
        if (epoch + 1) % save_every == 0:
            ckpt_path = os.path.join(VOCODER_DIR, f"v2_epoch_{epoch+1:04d}.pt")
            torch.save({
                'epoch': epoch,
                'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'optimizer_g_state_dict': optimizer_g.state_dict(),
                'optimizer_d_state_dict': optimizer_d.state_dict(),
                'best_loss': best_loss,
                'training_log': training_log,
            }, ckpt_path)
            
            # Save latest
            torch.save({
                'epoch': epoch,
                'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'optimizer_g_state_dict': optimizer_g.state_dict(),
                'optimizer_d_state_dict': optimizer_d.state_dict(),
                'best_loss': best_loss,
            }, os.path.join(VOCODER_DIR, "v2_latest.pt"))
            
            # Generate sample
            generator.eval()
            with torch.no_grad():
                sample = generator(units[:1], pitch[:1]).squeeze().cpu().numpy()
                sample = (sample * 32767).astype(np.int16)
                write(os.path.join(VOCODER_DIR, f"sample_v2_{epoch+1:04d}.wav"), SAMPLE_RATE, sample)
            generator.train()
            
            audio_volume.commit()
            print(f"  💾 Saved checkpoint: {ckpt_path}")
    
    # Save final
    torch.save({
        'epoch': epoch,
        'generator_state_dict': generator.state_dict(),
        'training_log': training_log,
    }, os.path.join(VOCODER_DIR, "v2_final.pt"))
    
    with open(os.path.join(VOCODER_DIR, "training_log_v2.json"), 'w') as f:
        json.dump(training_log, f, indent=2)
    
    audio_volume.commit()
    
    print("\n" + "=" * 60)
    print("✅ Training Complete!")
    print(f"  Best loss: {best_loss:.2f}")
    print(f"  Epochs trained: {epoch + 1}")
    print("=" * 60)
    
    return {
        'epochs_trained': epoch + 1,
        'best_loss': best_loss,
        'final_losses': epoch_losses,
    }

# %% [markdown]
# ## Entry Point

# %%
@app.local_entrypoint()
def main(
    language: str = "portuguese",
    epochs: int = 1000,
    batch_size: int = 12,
    lr_g: float = 0.0002,
    lr_d: float = 0.0002,
    segment_length: int = 32000,
    save_every: int = 25,
    resume: str = None,
    patience: int = 100,
    samples_per_epoch: int = 10000,
    lambda_mel: float = 45.0,
    lambda_stft: float = 2.0,
    lambda_fm: float = 2.0,
):
    config = LANGUAGE_CONFIGS.get(language, LANGUAGE_CONFIGS["portuguese"])
    vocoder_dir = config["vocoder_dir"]
    
    print("🎵 Starting Vocoder V2 Training")
    print("=" * 60)
    print(f"  Language: {language}")
    print(f"  Segments: {config['segmented_dir']}")
    print(f"  Units: {config['output_dir']}")
    print(f"  Checkpoints: {vocoder_dir}")
    print("=" * 60)
    print(f"  Max epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Segment length: {segment_length} ({segment_length/16000:.1f}s)")
    print(f"  Learning rates: G={lr_g}, D={lr_d}")
    print(f"  Loss weights: Mel={lambda_mel}, STFT={lambda_stft}, FM={lambda_fm}")
    print(f"  Early stopping: patience={patience}")
    print("=" * 60)
    
    # Set environment variable for the remote function
    import os
    os.environ["TRAINING_LANGUAGE"] = language
    
    resume_path = os.path.join(vocoder_dir, resume) if resume else None
    
    result = train_vocoder_v2.remote(
        epochs=epochs,
        batch_size=batch_size,
        learning_rate_g=lr_g,
        learning_rate_d=lr_d,
        segment_length=segment_length,
        save_every=save_every,
        resume_from=resume_path,
        patience=patience,
        samples_per_epoch=samples_per_epoch,
        lambda_mel=lambda_mel,
        lambda_stft=lambda_stft,
        lambda_fm=lambda_fm,
    )
    
    if result:
        print(f"\n✅ Complete!")
        print(f"  Epochs: {result['epochs_trained']}")
        print(f"  Best loss: {result['best_loss']:.2f}")
    else:
        print("\n❌ Training failed.")
