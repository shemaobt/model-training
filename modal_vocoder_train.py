"""
Modal deployment for Vocoder (Generator) Training
Trains a GAN-based vocoder to convert acoustic units back to audio

Run with: python3 -m modal run --detach modal_vocoder_train.py::main
"""

import modal
import os

# Create Modal app
app = modal.App("bible-vocoder-training")

# Create persistent volume (same as Phase 1 & 2)
audio_volume = modal.Volume.from_name(
    "bible-audio-data",
    create_if_missing=True
)

# Create image with dependencies
image = (
    modal.Image.debian_slim()
    .apt_install("ffmpeg")
    .pip_install(
        "torch>=2.0.0",
        "torchaudio>=2.0.0",
        "numpy<2",
        "scipy",
        "scikit-learn",
        "joblib",
        "soundfile>=0.12.0",
        "tqdm",
        "tensorboard",
    )
)

# Mount points
AUDIO_MOUNT = "/mnt/audio_data"
SEGMENTED_DIR = f"{AUDIO_MOUNT}/segmented_audio"
OUTPUT_DIR = f"{AUDIO_MOUNT}/portuguese_units"
VOCODER_DIR = f"{AUDIO_MOUNT}/vocoder_checkpoints"


# ============================================================================
# Model Definitions
# ============================================================================

GENERATOR_CODE = '''
import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    """
    Unit-to-Waveform Generator (Vocoder)
    Converts discrete acoustic units to raw audio waveform
    
    Upsampling factor: 5 * 5 * 4 * 4 * 2 = 800x
    Input: [B, T] unit indices
    Output: [B, T*800] audio samples at 16kHz
    """
    def __init__(self, num_units=100, embed_dim=256):
        super().__init__()
        
        self.num_units = num_units
        self.unit_embed = nn.Embedding(num_units, embed_dim)
        self.pre_conv = nn.Conv1d(embed_dim, 512, 7, padding=3)
        
        # Upsample blocks: 5*5*4*4*2 = 800x total
        self.upsamples = nn.ModuleList([
            self._make_upsample_block(512, 256, kernel=10, stride=5),
            self._make_upsample_block(256, 128, kernel=10, stride=5),
            self._make_upsample_block(128, 64, kernel=8, stride=4),
            self._make_upsample_block(64, 32, kernel=8, stride=4),
            self._make_upsample_block(32, 32, kernel=4, stride=2),
        ])
        
        # Residual blocks for better quality
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
        # x: [B, T] unit indices
        x = self.unit_embed(x)  # [B, T, embed_dim]
        x = x.transpose(1, 2)   # [B, embed_dim, T]
        
        x = self.pre_conv(x)    # [B, 512, T]
        x = F.leaky_relu(x, 0.1)
        
        for upsample in self.upsamples:
            x = upsample(x)
        
        # Residual blocks
        for res_block in self.res_blocks:
            x = x + res_block(x) * 0.1
        
        x = self.post_conv(x)   # [B, 1, T*800]
        x = torch.tanh(x)
        
        return x.squeeze(1)     # [B, T*800]
'''

DISCRIMINATOR_CODE = '''
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiScaleDiscriminator(nn.Module):
    """
    Multi-scale discriminator for GAN training
    Operates at multiple audio resolutions for better quality
    """
    def __init__(self):
        super().__init__()
        
        self.discriminators = nn.ModuleList([
            self._make_discriminator(),
            self._make_discriminator(),
            self._make_discriminator(),
        ])
        
        # Downsampling for multi-scale
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
        # x: [B, T] audio
        x = x.unsqueeze(1)  # [B, 1, T]
        
        outputs = []
        for disc in self.discriminators:
            outputs.append(disc(x))
            x = self.downsample(x)
        
        return outputs
'''


# ============================================================================
# Dataset
# ============================================================================

DATASET_CODE = '''
import torch
from torch.utils.data import Dataset
import soundfile as sf
import numpy as np
import os
import json

class UnitAudioDataset(Dataset):
    """
    Dataset that pairs unit sequences with their source audio
    
    Corpus format:
    {
        "segment_name_seg_0000": {"units": [...], "timestamps": [...]},
        "segment_name_seg_0001": {"units": [...], "timestamps": [...]},
        ...
    }
    """
    def __init__(self, corpus_path, audio_dir, segment_length=16000, hop_size=320):
        self.audio_dir = audio_dir
        self.segment_length = segment_length  # 1 second at 16kHz
        self.hop_size = hop_size  # ~20ms per frame (matches XLSR-53)
        self.unit_length = segment_length // hop_size  # units per segment
        
        # Load corpus - it's a dict with segment names as keys
        print(f"Loading corpus from {corpus_path}...")
        with open(corpus_path, 'r') as f:
            self.corpus = json.load(f)
        
        print(f"Corpus has {len(self.corpus)} entries")
        
        # Build list of valid samples
        self.samples = []
        missing_count = 0
        
        for segment_name, data in self.corpus.items():
            units = data.get('units', [])
            
            if not units:
                continue
            
            # Find corresponding audio file
            audio_path = self._find_audio_file(segment_name)
            if audio_path and os.path.exists(audio_path):
                self.samples.append({
                    'audio_path': audio_path,
                    'units': units,
                    'name': segment_name,
                })
            else:
                missing_count += 1
        
        print(f"Loaded {len(self.samples)} valid samples ({missing_count} missing audio files)")
        
        # Debug: show sample of what we're looking for vs what exists
        if len(self.samples) == 0 and missing_count > 0:
            print("\\n🔍 Debug: Checking file matching...")
            # List actual files in audio_dir
            try:
                actual_files = os.listdir(audio_dir)[:5]
                print(f"   Sample actual files: {actual_files[:3]}")
            except Exception as e:
                print(f"   Could not list {audio_dir}: {e}")
            
            # Sample corpus entries
            sample_keys = list(self.corpus.keys())[:3]
            print(f"   Sample corpus keys: {sample_keys}")
        
    def _find_audio_file(self, segment_name):
        """Find audio file in segmented directory"""
        # Try .wav extension
        for ext in ['.wav', '.WAV']:
            path = os.path.join(self.audio_dir, segment_name + ext)
            if os.path.exists(path):
                return path
        
        return None
    
    def __len__(self):
        return len(self.samples) * 10  # Multiple segments per file
    
    def __getitem__(self, idx):
        sample_idx = idx % len(self.samples)
        sample = self.samples[sample_idx]
        
        try:
            # Load audio
            audio, sr = sf.read(sample['audio_path'])
            if sr != 16000:
                # Simple resampling (for robustness)
                import scipy.signal
                audio = scipy.signal.resample(audio, int(len(audio) * 16000 / sr))
            
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)
            
            units = sample['units']
            
            # Random segment
            max_start_audio = max(0, len(audio) - self.segment_length)
            max_start_units = max(0, len(units) - self.unit_length)
            
            if max_start_audio > 0 and max_start_units > 0:
                # Align audio and units
                audio_ratio = max_start_audio / len(audio)
                units_ratio = max_start_units / len(units)
                
                start_ratio = np.random.random() * min(audio_ratio, units_ratio)
                
                audio_start = int(start_ratio * len(audio))
                units_start = int(start_ratio * len(units))
            else:
                audio_start = 0
                units_start = 0
            
            # Extract segment
            audio_seg = audio[audio_start:audio_start + self.segment_length]
            units_seg = units[units_start:units_start + self.unit_length]
            
            # Pad if needed
            if len(audio_seg) < self.segment_length:
                audio_seg = np.pad(audio_seg, (0, self.segment_length - len(audio_seg)))
            if len(units_seg) < self.unit_length:
                units_seg = units_seg + [0] * (self.unit_length - len(units_seg))
            
            return {
                'audio': torch.FloatTensor(audio_seg),
                'units': torch.LongTensor(units_seg[:self.unit_length]),
            }
            
        except Exception as e:
            # Return zeros on error
            return {
                'audio': torch.zeros(self.segment_length),
                'units': torch.zeros(self.unit_length, dtype=torch.long),
            }
'''


# ============================================================================
# Training Function
# ============================================================================

@app.function(
    image=image,
    volumes={AUDIO_MOUNT: audio_volume},
    timeout=14400,  # 4 hours
    gpu="A10G",
)
def train_vocoder(
    epochs: int = 100,
    batch_size: int = 16,
    learning_rate: float = 0.0002,
    save_every: int = 10,
    resume_from: str = None,
):
    """
    Train the vocoder (Generator) using GAN training
    """
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
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    os.makedirs(VOCODER_DIR, exist_ok=True)
    
    # ========== Load Models ==========
    print("\n📦 Loading models...")
    
    # Write code to temp files and import
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
    
    # Create models
    generator = Generator(num_units=100).to(device)
    discriminator = MultiScaleDiscriminator().to(device)
    
    print(f"Generator params: {sum(p.numel() for p in generator.parameters()):,}")
    print(f"Discriminator params: {sum(p.numel() for p in discriminator.parameters()):,}")
    
    # ========== Load Dataset ==========
    print("\n📂 Loading dataset...")
    
    corpus_path = os.path.join(OUTPUT_DIR, "portuguese_corpus_timestamped.json")
    if not os.path.exists(corpus_path):
        print(f"❌ Corpus not found: {corpus_path}")
        print("   Please run Phase 1 first!")
        return None
    
    dataset = UnitAudioDataset(
        corpus_path=corpus_path,
        audio_dir=SEGMENTED_DIR,
        segment_length=16000,  # 1 second
    )
    
    if len(dataset.samples) == 0:
        print("❌ No valid samples found!")
        print("   Check that segmented audio files exist and match corpus entries")
        return None
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        drop_last=True,
    )
    
    print(f"Dataset: {len(dataset.samples)} files, ~{len(dataset)} segments")
    print(f"Batches per epoch: {len(dataloader)}")
    
    # ========== Optimizers ==========
    optimizer_g = optim.AdamW(generator.parameters(), lr=learning_rate, betas=(0.5, 0.9))
    optimizer_d = optim.AdamW(discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.9))
    
    # Learning rate schedulers
    scheduler_g = optim.lr_scheduler.ExponentialLR(optimizer_g, gamma=0.999)
    scheduler_d = optim.lr_scheduler.ExponentialLR(optimizer_d, gamma=0.999)
    
    # ========== Resume from checkpoint ==========
    start_epoch = 0
    if resume_from and os.path.exists(resume_from):
        print(f"\n📥 Resuming from: {resume_from}")
        checkpoint = torch.load(resume_from, map_location=device)
        generator.load_state_dict(checkpoint['generator_state_dict'])
        discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        optimizer_g.load_state_dict(checkpoint['optimizer_g_state_dict'])
        optimizer_d.load_state_dict(checkpoint['optimizer_d_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        print(f"   Resumed at epoch {start_epoch}")
    
    # ========== Loss Functions ==========
    def adversarial_loss(outputs, target_is_real):
        """LSGAN loss"""
        target = torch.ones_like(outputs) if target_is_real else torch.zeros_like(outputs)
        return nn.MSELoss()(outputs, target)
    
    def feature_matching_loss(real_features, fake_features):
        """Feature matching loss for stable training"""
        loss = 0
        for real_feat, fake_feat in zip(real_features, fake_features):
            loss += nn.L1Loss()(fake_feat, real_feat.detach())
        return loss / len(real_features)
    
    def mel_spectrogram_loss(real_audio, fake_audio):
        """L1 loss on mel spectrograms"""
        import torchaudio.transforms as T
        
        mel_transform = T.MelSpectrogram(
            sample_rate=16000,
            n_fft=1024,
            hop_length=256,
            n_mels=80,
        ).to(real_audio.device)
        
        real_mel = mel_transform(real_audio)
        fake_mel = mel_transform(fake_audio)
        
        return nn.L1Loss()(fake_mel, real_mel)
    
    # ========== Training Loop ==========
    print(f"\n🚀 Starting training for {epochs} epochs...")
    
    training_log = []
    
    for epoch in range(start_epoch, epochs):
        generator.train()
        discriminator.train()
        
        epoch_g_loss = 0
        epoch_d_loss = 0
        epoch_mel_loss = 0
        num_batches = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch in pbar:
            real_audio = batch['audio'].to(device)
            units = batch['units'].to(device)
            
            # ===== Train Discriminator =====
            optimizer_d.zero_grad()
            
            # Generate fake audio
            with torch.no_grad():
                fake_audio = generator(units)
            
            # Discriminator outputs
            real_outputs = discriminator(real_audio)
            fake_outputs = discriminator(fake_audio)
            
            # Discriminator loss
            d_loss = 0
            for real_out, fake_out in zip(real_outputs, fake_outputs):
                d_loss += adversarial_loss(real_out, True)
                d_loss += adversarial_loss(fake_out, False)
            d_loss = d_loss / len(real_outputs)
            
            d_loss.backward()
            optimizer_d.step()
            
            # ===== Train Generator =====
            optimizer_g.zero_grad()
            
            # Generate fake audio
            fake_audio = generator(units)
            
            # Adversarial loss
            fake_outputs = discriminator(fake_audio)
            g_adv_loss = 0
            for fake_out in fake_outputs:
                g_adv_loss += adversarial_loss(fake_out, True)
            g_adv_loss = g_adv_loss / len(fake_outputs)
            
            # Mel spectrogram loss
            mel_loss = mel_spectrogram_loss(real_audio, fake_audio)
            
            # Total generator loss
            g_loss = g_adv_loss + 45.0 * mel_loss  # Weight mel loss higher
            
            g_loss.backward()
            optimizer_g.step()
            
            # Logging
            epoch_g_loss += g_loss.item()
            epoch_d_loss += d_loss.item()
            epoch_mel_loss += mel_loss.item()
            num_batches += 1
            
            pbar.set_postfix({
                'G': f'{g_loss.item():.3f}',
                'D': f'{d_loss.item():.3f}',
                'Mel': f'{mel_loss.item():.3f}',
            })
        
        # Update learning rate
        scheduler_g.step()
        scheduler_d.step()
        
        # Epoch summary
        avg_g_loss = epoch_g_loss / num_batches
        avg_d_loss = epoch_d_loss / num_batches
        avg_mel_loss = epoch_mel_loss / num_batches
        
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  G Loss: {avg_g_loss:.4f}")
        print(f"  D Loss: {avg_d_loss:.4f}")
        print(f"  Mel Loss: {avg_mel_loss:.4f}")
        
        training_log.append({
            'epoch': epoch + 1,
            'g_loss': avg_g_loss,
            'd_loss': avg_d_loss,
            'mel_loss': avg_mel_loss,
        })
        
        # ===== Save Checkpoint =====
        if (epoch + 1) % save_every == 0 or (epoch + 1) == epochs:
            checkpoint_path = os.path.join(VOCODER_DIR, f"vocoder_epoch_{epoch+1:04d}.pt")
            torch.save({
                'epoch': epoch,
                'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'optimizer_g_state_dict': optimizer_g.state_dict(),
                'optimizer_d_state_dict': optimizer_d.state_dict(),
                'training_log': training_log,
            }, checkpoint_path)
            print(f"  ✓ Saved checkpoint: {checkpoint_path}")
            
            # Save latest as well
            latest_path = os.path.join(VOCODER_DIR, "vocoder_latest.pt")
            torch.save({
                'epoch': epoch,
                'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'optimizer_g_state_dict': optimizer_g.state_dict(),
                'optimizer_d_state_dict': optimizer_d.state_dict(),
                'training_log': training_log,
            }, latest_path)
            
            # Generate sample audio
            generator.eval()
            with torch.no_grad():
                sample_units = units[:1]  # First sample from last batch
                sample_audio = generator(sample_units).squeeze().cpu().numpy()
                sample_audio = (sample_audio * 32767).astype(np.int16)
                
                sample_path = os.path.join(VOCODER_DIR, f"sample_epoch_{epoch+1:04d}.wav")
                write(sample_path, 16000, sample_audio)
                print(f"  ✓ Saved sample: {sample_path}")
            generator.train()
            
            # Commit volume
            audio_volume.commit()
    
    # ===== Save Final Model ==========
    final_path = os.path.join(VOCODER_DIR, "vocoder_final.pt")
    torch.save({
        'epoch': epochs,
        'generator_state_dict': generator.state_dict(),
        'training_log': training_log,
    }, final_path)
    
    # Save training log
    log_path = os.path.join(VOCODER_DIR, "training_log.json")
    with open(log_path, 'w') as f:
        json.dump(training_log, f, indent=2)
    
    audio_volume.commit()
    
    print("\n" + "=" * 60)
    print("✅ Training Complete!")
    print(f"  Final checkpoint: {final_path}")
    print(f"  Training log: {log_path}")
    print("=" * 60)
    
    return {
        'epochs_trained': epochs,
        'final_g_loss': training_log[-1]['g_loss'],
        'final_mel_loss': training_log[-1]['mel_loss'],
        'checkpoint_path': final_path,
    }


@app.local_entrypoint()
def main(
    epochs: int = 100,
    batch_size: int = 16,
    lr: float = 0.0002,
    save_every: int = 10,
    resume: str = None,
):
    """
    Train the vocoder model
    
    Args:
        epochs: Number of training epochs (default: 100)
        batch_size: Batch size (default: 16)
        lr: Learning rate (default: 0.0002)
        save_every: Save checkpoint every N epochs (default: 10)
        resume: Path to checkpoint to resume from (optional)
    """
    print("🎵 Starting Vocoder Training on Modal")
    print("=" * 60)
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {lr}")
    print(f"  Save every: {save_every} epochs")
    if resume:
        print(f"  Resume from: {resume}")
    print("=" * 60)
    
    resume_path = None
    if resume:
        resume_path = os.path.join(VOCODER_DIR, resume)
    
    result = train_vocoder.remote(
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=lr,
        save_every=save_every,
        resume_from=resume_path,
    )
    
    if result:
        print("\n✅ Vocoder Training Complete!")
        print(f"  Epochs trained: {result['epochs_trained']}")
        print(f"  Final G Loss: {result['final_g_loss']:.4f}")
        print(f"  Final Mel Loss: {result['final_mel_loss']:.4f}")
        print(f"\nCheckpoint saved at: {result['checkpoint_path']}")
        print("\nTo synthesize audio, run:")
        print("  python3 -m modal run modal_checking.py::main --generator-checkpoint /mnt/audio_data/vocoder_checkpoints/vocoder_final.pt")
    else:
        print("\n❌ Training failed. Check logs above.")
