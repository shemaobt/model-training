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
# # Phase 3: Vocoder Training
#
# This script trains a GAN-based vocoder to convert discrete acoustic units
# back to continuous audio waveforms.
#
# ## Architecture
#
# ```
# Generator (3.2M params):
#   Input: Unit indices [B, T]
#     ↓ Embedding (100 → 256)
#     ↓ Pre-Conv (256 → 512)
#     ↓ Upsample 5x (512 → 256)
#     ↓ Upsample 4x (256 → 128)
#     ↓ Upsample 4x (128 → 64)
#     ↓ Upsample 4x (64 → 32)
#     ↓ Residual Blocks × 3
#     ↓ Post-Conv (32 → 1)
#   Output: Audio [B, T×320]
#
# Discriminator (5.1M params):
#   Multi-scale: 3 discriminators at different resolutions
#   Captures fine details AND global structure
# ```
#
# ## Training
#
# - **Loss**: Mel Spectrogram L1 (×45) + Adversarial (LSGAN)
# - **Early Stopping**: Patience 50 epochs, min_delta 0.5
# - **Checkpoints**: Every 20 epochs
#
# ## Run on Modal
#
# ```bash
# # Full training (500 epochs max, early stopping)
# python3 -m modal run --detach src/training/phase3_vocoder.py::main
#
# # Resume from checkpoint
# python3 -m modal run --detach src/training/phase3_vocoder.py::main --resume vocoder_latest.pt
#
# # Custom parameters
# python3 -m modal run --detach src/training/phase3_vocoder.py::main --epochs 300 --patience 30
# ```

# %% [markdown]
# ## Modal Configuration

# %%
import modal
import os
from src.constants import (
    SAMPLE_RATE,
    NUM_ACOUSTIC_UNITS,
    DEFAULT_LEARNING_RATE,
    DEFAULT_SAMPLES_PER_EPOCH,
    LAMBDA_MEL,
    MEL_N_FFT,
    MEL_HOP_LENGTH,
    MEL_N_MELS,
)

# %%
app = modal.App("bible-vocoder-training")

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
    )
)

# %%
AUDIO_MOUNT = "/mnt/audio_data"

LANGUAGE_CONFIGS = {
    "portuguese": {
        "segmented_dir": f"{AUDIO_MOUNT}/segmented_audio",
        "output_dir": f"{AUDIO_MOUNT}/portuguese_units",
        "vocoder_dir": f"{AUDIO_MOUNT}/vocoder_checkpoints",
        "corpus_file": "portuguese_corpus_timestamped.json",
    },
    "satere": {
        "segmented_dir": f"{AUDIO_MOUNT}/segmented_audio_satere",
        "output_dir": f"{AUDIO_MOUNT}/satere_units",
        "vocoder_dir": f"{AUDIO_MOUNT}/vocoder_satere_checkpoints",
        "corpus_file": "satere_corpus_timestamped.json",
    },
}

import os as _os
_LANGUAGE = _os.environ.get("TRAINING_LANGUAGE", "portuguese")
_config = LANGUAGE_CONFIGS.get(_LANGUAGE, LANGUAGE_CONFIGS["portuguese"])

SEGMENTED_DIR = _config["segmented_dir"]
OUTPUT_DIR = _config["output_dir"]
VOCODER_DIR = _config["vocoder_dir"]

# %% [markdown]
# ## Model Definitions
#
# Models are defined as code strings to be executed in the Modal environment.
# See `src/models/generator.py` and `src/models/discriminator.py` for 
# standalone implementations.

# %%
GENERATOR_CODE = '''
import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    """
    Unit-to-Waveform Generator (Vocoder)
    
    Upsampling: 5 × 4 × 4 × 4 = 320x (matches XLSR-53 hop size)
    Input: [B, T] unit indices
    Output: [B, T×320] audio at 16kHz
    """
    def __init__(self, num_units=100, embed_dim=256):
        super().__init__()
        
        self.num_units = num_units
        self.unit_embed = nn.Embedding(num_units, embed_dim)
        self.pre_conv = nn.Conv1d(embed_dim, 512, 7, padding=3)
        
        self.upsamples = nn.ModuleList([
            self._make_upsample_block(512, 256, kernel=10, stride=5),
            self._make_upsample_block(256, 128, kernel=8, stride=4),
            self._make_upsample_block(128, 64, kernel=8, stride=4),
            self._make_upsample_block(64, 32, kernel=8, stride=4),
        ])
        
        self.res_blocks = nn.ModuleList([
            self._make_res_block(32) for _ in range(3)
        ])
        
        self.post_conv = nn.Conv1d(32, 1, 7, padding=3)
        
    def _make_upsample_block(self, in_ch, out_ch, kernel, stride):
        padding = (kernel - stride) // 2
        return nn.Sequential(
            nn.ConvTranspose1d(in_ch, out_ch, kernel, stride=stride, padding=padding),
            nn.LeakyReLU(0.1),
            nn.Conv1d(out_ch, out_ch, 7, padding=3),
            nn.LeakyReLU(0.1),
        )
    
    def _make_res_block(self, channels):
        return nn.Sequential(
            nn.Conv1d(channels, channels, 3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv1d(channels, channels, 3, padding=1),
        )
        
    def forward(self, x):
        x = self.unit_embed(x)
        x = x.transpose(1, 2)
        
        x = self.pre_conv(x)
        x = F.leaky_relu(x, 0.1)
        
        for upsample in self.upsamples:
            x = upsample(x)
        
        for res_block in self.res_blocks:
            x = x + res_block(x) * 0.1
        
        x = self.post_conv(x)
        x = torch.tanh(x)
        
        return x.squeeze(1)
'''

# %%
DISCRIMINATOR_CODE = '''
import torch
import torch.nn as nn

class MultiScaleDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.discriminators = nn.ModuleList([
            self._make_discriminator(),
            self._make_discriminator(),
            self._make_discriminator(),
        ])
        
        self.downsample = nn.AvgPool1d(4, stride=2, padding=1)
        
    def _make_discriminator(self):
        return nn.Sequential(
            nn.Conv1d(1, 16, 15, stride=1, padding=7),
            nn.LeakyReLU(0.1),
            nn.Conv1d(16, 64, 41, stride=4, padding=20, groups=4),
            nn.LeakyReLU(0.1),
            nn.Conv1d(64, 256, 41, stride=4, padding=20, groups=16),
            nn.LeakyReLU(0.1),
            nn.Conv1d(256, 512, 41, stride=4, padding=20, groups=16),
            nn.LeakyReLU(0.1),
            nn.Conv1d(512, 512, 5, stride=1, padding=2),
            nn.LeakyReLU(0.1),
            nn.Conv1d(512, 1, 3, stride=1, padding=1),
        )
        
    def forward(self, x):
        x = x.unsqueeze(1)
        
        outputs = []
        for disc in self.discriminators:
            outputs.append(disc(x))
            x = self.downsample(x)
        
        return outputs
'''

# %% [markdown]
# ## Dataset
#
# Pairs unit sequences with their corresponding audio segments.

# %%
DATASET_CODE = '''
import torch
from torch.utils.data import Dataset
import soundfile as sf
import numpy as np
import os
import json

class UnitAudioDataset(Dataset):
    def __init__(self, corpus_path, audio_dir, segment_length=16000, 
                 hop_size=320, samples_per_epoch=10000):
        self.audio_dir = audio_dir
        self.segment_length = segment_length
        self.hop_size = hop_size
        self.unit_length = segment_length // hop_size
        self.samples_per_epoch = samples_per_epoch
        
        print(f"Loading corpus from {corpus_path}...")
        with open(corpus_path, 'r') as f:
            self.corpus = json.load(f)
        
        print(f"Corpus has {len(self.corpus)} entries")
        
        self.samples = []
        missing_count = 0
        
        for segment_name, data in self.corpus.items():
            units = data.get('units', [])
            if not units:
                continue
            
            audio_path = self._find_audio_file(segment_name)
            if audio_path and os.path.exists(audio_path):
                self.samples.append({
                    'audio_path': audio_path,
                    'units': units,
                    'name': segment_name,
                })
            else:
                missing_count += 1
        
        print(f"Loaded {len(self.samples)} valid samples ({missing_count} missing)")
        
    def _find_audio_file(self, segment_name):
        for ext in ['.wav', '.WAV']:
            path = os.path.join(self.audio_dir, segment_name + ext)
            if os.path.exists(path):
                return path
        return None
    
    def __len__(self):
        return self.samples_per_epoch
    
    def __getitem__(self, idx):
        sample_idx = idx % len(self.samples)
        sample = self.samples[sample_idx]
        
        try:
            audio, sr = sf.read(sample['audio_path'])
            if sr != 16000:
                import scipy.signal
                audio = scipy.signal.resample(audio, int(len(audio) * 16000 / sr))
            
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)
            
            units = sample['units']
            
            max_start_audio = max(0, len(audio) - self.segment_length)
            max_start_units = max(0, len(units) - self.unit_length)
            
            if max_start_audio > 0 and max_start_units > 0:
                audio_ratio = max_start_audio / len(audio)
                units_ratio = max_start_units / len(units)
                start_ratio = np.random.random() * min(audio_ratio, units_ratio)
                audio_start = int(start_ratio * len(audio))
                units_start = int(start_ratio * len(units))
            else:
                audio_start = 0
                units_start = 0
            
            audio_seg = audio[audio_start:audio_start + self.segment_length]
            units_seg = units[units_start:units_start + self.unit_length]
            
            if len(audio_seg) < self.segment_length:
                audio_seg = np.pad(audio_seg, (0, self.segment_length - len(audio_seg)))
            if len(units_seg) < self.unit_length:
                units_seg = units_seg + [0] * (self.unit_length - len(units_seg))
            
            return {
                'audio': torch.FloatTensor(audio_seg),
                'units': torch.LongTensor(units_seg[:self.unit_length]),
            }
            
        except Exception:
            return {
                'audio': torch.zeros(self.segment_length),
                'units': torch.zeros(self.unit_length, dtype=torch.long),
            }
'''

# %% [markdown]
# ## Training Function
#
# Main training loop with:
# - Adversarial loss (LSGAN)
# - Mel spectrogram loss
# - Early stopping
# - Periodic checkpointing

# %%
@app.function(
    image=image,
    volumes={AUDIO_MOUNT: audio_volume},
    timeout=43200,  # 12 hours
    gpu="A10G",
)
def train_vocoder(
    epochs: int = 500,
    batch_size: int = 16,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    save_every: int = 20,
    resume_from: str = None,
    patience: int = 50,
    min_delta: float = 0.5,
    samples_per_epoch: int = DEFAULT_SAMPLES_PER_EPOCH,
    language: str = "portuguese",
):
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader
    import numpy as np
    from tqdm import tqdm
    import json
    from scipy.io.wavfile import write
    import tempfile
    import importlib.util
    
    print("=" * 60)
    print("🎵 Vocoder Training")
    print(f"Language: {language}")
    print("=" * 60)
    
    config = LANGUAGE_CONFIGS.get(language, LANGUAGE_CONFIGS["portuguese"])
    segmented_dir = config["segmented_dir"]
    output_dir = config["output_dir"]
    vocoder_dir = config["vocoder_dir"]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    os.makedirs(vocoder_dir, exist_ok=True)
    
    def load_module(code, name):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            path = f.name
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    
    gen_module = load_module(GENERATOR_CODE, "generator")
    disc_module = load_module(DISCRIMINATOR_CODE, "discriminator")
    data_module = load_module(DATASET_CODE, "dataset")
    
    Generator = gen_module.Generator
    MultiScaleDiscriminator = disc_module.MultiScaleDiscriminator
    UnitAudioDataset = data_module.UnitAudioDataset
    
    generator = Generator(num_units=NUM_ACOUSTIC_UNITS).to(device)
    discriminator = MultiScaleDiscriminator().to(device)
    
    print(f"Generator: {sum(p.numel() for p in generator.parameters()):,} params")
    print(f"Discriminator: {sum(p.numel() for p in discriminator.parameters()):,} params")
    
    corpus_file = config["corpus_file"]
    corpus_path = os.path.join(output_dir, corpus_file)
    if not os.path.exists(corpus_path):
        print(f"❌ Corpus not found at {corpus_path}. Run Phase 1 first!")
        return None
    
    dataset = UnitAudioDataset(
        corpus_path=corpus_path,
        audio_dir=segmented_dir,
        segment_length=SAMPLE_RATE,
        samples_per_epoch=samples_per_epoch,
    )
    
    if len(dataset.samples) == 0:
        print("❌ No valid samples found!")
        return None
    
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True
    )
    
    optimizer_g = optim.AdamW(generator.parameters(), lr=learning_rate, betas=(0.5, 0.9))
    optimizer_d = optim.AdamW(discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.9))
    scheduler_g = optim.lr_scheduler.ExponentialLR(optimizer_g, gamma=0.999)
    scheduler_d = optim.lr_scheduler.ExponentialLR(optimizer_d, gamma=0.999)
    
    start_epoch = 0
    if resume_from and os.path.exists(resume_from):
        print(f"📥 Resuming from: {resume_from}")
        checkpoint = torch.load(resume_from, map_location=device)
        generator.load_state_dict(checkpoint['generator_state_dict'])
        discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        optimizer_g.load_state_dict(checkpoint['optimizer_g_state_dict'])
        optimizer_d.load_state_dict(checkpoint['optimizer_d_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1
    
    def adversarial_loss(outputs, target_is_real):
        target = torch.ones_like(outputs) if target_is_real else torch.zeros_like(outputs)
        return nn.MSELoss()(outputs, target)
    
    def mel_spectrogram_loss(real_audio, fake_audio):
        import torchaudio.transforms as T
        
        min_len = min(real_audio.shape[-1], fake_audio.shape[-1])
        real_audio = real_audio[..., :min_len]
        fake_audio = fake_audio[..., :min_len]
        
        mel_transform = T.MelSpectrogram(
            sample_rate=SAMPLE_RATE, n_fft=MEL_N_FFT, hop_length=MEL_HOP_LENGTH, n_mels=MEL_N_MELS
        ).to(real_audio.device)
        
        return nn.L1Loss()(mel_transform(fake_audio), mel_transform(real_audio))
    
    training_log = []
    best_mel_loss = float('inf')
    epochs_without_improvement = 0
    
    for epoch in range(start_epoch, epochs):
        generator.train()
        discriminator.train()
        
        epoch_g_loss = epoch_d_loss = epoch_mel_loss = 0
        num_batches = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch in pbar:
            real_audio = batch['audio'].to(device)
            units = batch['units'].to(device)
            
            optimizer_d.zero_grad()
            with torch.no_grad():
                fake_audio = generator(units)
            
            real_outputs = discriminator(real_audio)
            fake_outputs = discriminator(fake_audio)
            
            d_loss = 0
            for real_out, fake_out in zip(real_outputs, fake_outputs):
                d_loss += adversarial_loss(real_out, True)
                d_loss += adversarial_loss(fake_out, False)
            d_loss = d_loss / len(real_outputs)
            
            d_loss.backward()
            optimizer_d.step()
            
            optimizer_g.zero_grad()
            fake_audio = generator(units)
            fake_outputs = discriminator(fake_audio)
            
            g_adv_loss = sum(adversarial_loss(o, True) for o in fake_outputs) / len(fake_outputs)
            mel_loss = mel_spectrogram_loss(real_audio, fake_audio)
            g_loss = g_adv_loss + 45.0 * mel_loss
            
            g_loss.backward()
            optimizer_g.step()
            
            epoch_g_loss += g_loss.item()
            epoch_d_loss += d_loss.item()
            epoch_mel_loss += mel_loss.item()
            num_batches += 1
            
            pbar.set_postfix({'G': f'{g_loss.item():.3f}', 'D': f'{d_loss.item():.3f}', 'Mel': f'{mel_loss.item():.3f}'})
        
        scheduler_g.step()
        scheduler_d.step()
        
        avg_g_loss = epoch_g_loss / num_batches
        avg_d_loss = epoch_d_loss / num_batches
        avg_mel_loss = epoch_mel_loss / num_batches
        
        print(f"\nEpoch {epoch+1}: G={avg_g_loss:.4f}, D={avg_d_loss:.4f}, Mel={avg_mel_loss:.4f}")
        
        training_log.append({'epoch': epoch+1, 'g_loss': avg_g_loss, 'd_loss': avg_d_loss, 'mel_loss': avg_mel_loss})
        
        if avg_mel_loss < best_mel_loss - min_delta:
            best_mel_loss = avg_mel_loss
            epochs_without_improvement = 0
            print(f"  ✓ New best: {best_mel_loss:.4f}")
        else:
            epochs_without_improvement += 1
            print(f"  ⚠️  No improvement: {epochs_without_improvement}/{patience}")
        
        if epochs_without_improvement >= patience:
            print(f"\n🛑 Early stopping!")
            break
        
        if (epoch + 1) % save_every == 0:
            checkpoint_path = os.path.join(vocoder_dir, f"vocoder_epoch_{epoch+1:04d}.pt")
            torch.save({
                'epoch': epoch,
                'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'optimizer_g_state_dict': optimizer_g.state_dict(),
                'optimizer_d_state_dict': optimizer_d.state_dict(),
                'training_log': training_log,
            }, checkpoint_path)
            
            torch.save({
                'epoch': epoch,
                'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'optimizer_g_state_dict': optimizer_g.state_dict(),
                'optimizer_d_state_dict': optimizer_d.state_dict(),
            }, os.path.join(vocoder_dir, "vocoder_latest.pt"))
            
            generator.eval()
            with torch.no_grad():
                sample_audio = generator(units[:1]).squeeze().cpu().numpy()
                sample_audio = (sample_audio * 32767).astype(np.int16)
                write(os.path.join(vocoder_dir, f"sample_epoch_{epoch+1:04d}.wav"), SAMPLE_RATE, sample_audio)
            generator.train()
            
            audio_volume.commit()
    
    torch.save({'epoch': epochs, 'generator_state_dict': generator.state_dict(), 'training_log': training_log},
               os.path.join(vocoder_dir, "vocoder_final.pt"))
    
    with open(os.path.join(vocoder_dir, "training_log.json"), 'w') as f:
        json.dump(training_log, f, indent=2)
    
    audio_volume.commit()
    
    print("\n✅ Training Complete!")
    return {'epochs_trained': len(training_log), 'final_mel_loss': training_log[-1]['mel_loss']}

# %% [markdown]
# ## Entry Point

# %%
@app.local_entrypoint()
def main(
    epochs: int = 500,
    batch_size: int = 16,
    lr: float = 0.0002,
    save_every: int = 20,
    resume: str = None,
    patience: int = 50,
    min_delta: float = 0.5,
    samples_per_epoch: int = 10000,
    language: str = "portuguese",
):
    print("🎵 Starting Vocoder Training")
    print("=" * 60)
    print(f"  Language: {language}")
    print(f"  Max epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {lr}")
    print(f"  Early stopping: patience={patience}")
    print("=" * 60)
    
    config = LANGUAGE_CONFIGS.get(language, LANGUAGE_CONFIGS["portuguese"])
    vocoder_dir = config["vocoder_dir"]
    resume_path = os.path.join(vocoder_dir, resume) if resume else None
    
    result = train_vocoder.remote(
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=lr,
        save_every=save_every,
        resume_from=resume_path,
        patience=patience,
        min_delta=min_delta,
        samples_per_epoch=samples_per_epoch,
        language=language,
    )
    
    if result:
        print(f"\n✅ Complete! Epochs: {result['epochs_trained']}, Mel Loss: {result['final_mel_loss']:.4f}")
    else:
        print("\n❌ Training failed.")
