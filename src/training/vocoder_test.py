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
# # Vocoder Quality Test
#
# Tests the trained vocoder by synthesizing audio from unit sequences
# and comparing with original audio using quality metrics.
#
# ## Metrics
#
# - **SNR (Signal-to-Noise Ratio)**: Measures reconstruction fidelity
#   - \> 10 dB: Excellent
#   - 5-10 dB: Good
#   - 0-5 dB: Fair
#   - < 0 dB: Poor
#
# - **MCD (Mel Cepstral Distortion)**: Measures spectral similarity
#   - < 5: Excellent
#   - 5-8: Good
#   - 8-12: Acceptable
#   - \> 12: Needs improvement
#
# ## Run on Modal
#
# ```bash
# python3 -m modal run src/training/vocoder_test.py::main --num-samples 20
# ```

# %% [markdown]
# ## Modal Configuration

# %%
import modal
import os

# %%
app = modal.App("bible-vocoder-test")

audio_volume = modal.Volume.from_name(
    "bible-audio-data",
    create_if_missing=True
)

# %%
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
        "librosa>=0.10.0",
        "tqdm",
    )
)

# %%
AUDIO_MOUNT = "/mnt/audio_data"
SEGMENTED_DIR = f"{AUDIO_MOUNT}/segmented_audio"
OUTPUT_DIR = f"{AUDIO_MOUNT}/portuguese_units"
VOCODER_DIR = f"{AUDIO_MOUNT}/vocoder_checkpoints"
TEST_OUTPUT_DIR = f"{AUDIO_MOUNT}/vocoder_test_output"

# %% [markdown]
# ## Generator Model
#
# Same architecture as training (320x upsampling).

# %%
GENERATOR_CODE = '''
import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    """Upsampling: 5 × 4 × 4 × 4 = 320x"""
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

# %% [markdown]
# ## Test Function
#
# Synthesizes audio and computes quality metrics.

# %%
@app.function(
    image=image,
    volumes={AUDIO_MOUNT: audio_volume},
    timeout=1800,
    gpu="A10G",
)
def test_vocoder(num_samples: int = 20):
    import torch
    import numpy as np
    import json
    import soundfile as sf
    import librosa
    from scipy.io.wavfile import write
    from tqdm import tqdm
    import tempfile
    import importlib.util
    
    print("=" * 60)
    print("🎵 Vocoder Quality Test")
    print("=" * 60)
    
    os.makedirs(TEST_OUTPUT_DIR, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(GENERATOR_CODE)
        gen_path = f.name
    
    spec = importlib.util.spec_from_file_location("generator", gen_path)
    gen_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen_module)
    Generator = gen_module.Generator
    
    generator = Generator(num_units=100).to(device)
    
    checkpoint_path = os.path.join(VOCODER_DIR, "vocoder_final.pt")
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return None
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    generator.load_state_dict(checkpoint['generator_state_dict'])
    generator.eval()
    print(f"✓ Generator loaded")
    
    corpus_path = os.path.join(OUTPUT_DIR, "portuguese_corpus_timestamped.json")
    with open(corpus_path, 'r') as f:
        corpus = json.load(f)
    
    sample_keys = list(corpus.keys())
    np.random.shuffle(sample_keys)
    selected_keys = sample_keys[:num_samples]
    
    results = {"num_samples": num_samples, "samples": [], "aggregate_metrics": {}}
    all_snr = []
    all_mcd = []
    
    print(f"\n🔬 Testing {num_samples} samples...")
    
    for idx, segment_name in enumerate(tqdm(selected_keys, desc="Testing")):
        data = corpus[segment_name]
        units = data['units']
        
        audio_path = os.path.join(SEGMENTED_DIR, segment_name + ".wav")
        if not os.path.exists(audio_path):
            continue
        
        try:
            original_audio, sr = sf.read(audio_path)
            if sr != 16000:
                original_audio = librosa.resample(original_audio, orig_sr=sr, target_sr=16000)
            
            if len(original_audio.shape) > 1:
                original_audio = original_audio.mean(axis=1)
            
            units_tensor = torch.LongTensor(units).unsqueeze(0).to(device)
            
            with torch.no_grad():
                synth_audio = generator(units_tensor).squeeze().cpu().numpy()
            
            synth_audio = synth_audio / (np.max(np.abs(synth_audio)) + 1e-8) * 0.95
            
            min_len = min(len(original_audio), len(synth_audio))
            original_segment = original_audio[:min_len]
            synth_segment = synth_audio[:min_len]
            
            diff = original_segment - synth_segment
            signal_power = np.mean(original_segment ** 2)
            noise_power = np.mean(diff ** 2)
            snr = 10 * np.log10(signal_power / (noise_power + 1e-10))
            all_snr.append(snr)
            
            try:
                orig_mfcc = librosa.feature.mfcc(y=original_segment, sr=16000, n_mfcc=13)
                synth_mfcc = librosa.feature.mfcc(y=synth_segment, sr=16000, n_mfcc=13)
                
                min_frames = min(orig_mfcc.shape[1], synth_mfcc.shape[1])
                mcd = np.mean(np.sqrt(2 * np.sum((orig_mfcc[:, :min_frames] - synth_mfcc[:, :min_frames]) ** 2, axis=0)))
                all_mcd.append(mcd)
            except:
                mcd = None
            
            if idx < 10:
                write(os.path.join(TEST_OUTPUT_DIR, f"synth_{idx:04d}.wav"), 16000, (synth_audio * 32767).astype(np.int16))
                write(os.path.join(TEST_OUTPUT_DIR, f"orig_{idx:04d}.wav"), 16000, (original_segment * 32767).astype(np.int16))
            
            results["samples"].append({
                "segment_name": segment_name[:50],
                "snr_db": float(snr),
                "mcd": float(mcd) if mcd else None,
            })
            
        except Exception as e:
            print(f"⚠️  Error: {e}")
            continue
    
    results["aggregate_metrics"] = {
        "mean_snr_db": float(np.mean(all_snr)) if all_snr else None,
        "std_snr_db": float(np.std(all_snr)) if all_snr else None,
        "mean_mcd": float(np.mean(all_mcd)) if all_mcd else None,
        "std_mcd": float(np.std(all_mcd)) if all_mcd else None,
        "samples_tested": len(results["samples"]),
    }
    
    print("\n" + "=" * 60)
    print("📊 Results")
    print("=" * 60)
    if results['aggregate_metrics']['mean_snr_db']:
        print(f"  SNR: {results['aggregate_metrics']['mean_snr_db']:.2f} dB (±{results['aggregate_metrics']['std_snr_db']:.2f})")
    if results['aggregate_metrics']['mean_mcd']:
        print(f"  MCD: {results['aggregate_metrics']['mean_mcd']:.2f} (±{results['aggregate_metrics']['std_mcd']:.2f})")
    
    with open(os.path.join(TEST_OUTPUT_DIR, "test_results.json"), 'w') as f:
        json.dump(results, f, indent=2)
    
    audio_volume.commit()
    return results

# %% [markdown]
# ## Entry Point

# %%
@app.local_entrypoint()
def main(num_samples: int = 20):
    print("🔬 Starting Vocoder Quality Test")
    
    results = test_vocoder.remote(num_samples=num_samples)
    
    if results:
        print(f"\n✅ Tested {results['aggregate_metrics']['samples_tested']} samples")
        if results['aggregate_metrics']['mean_snr_db']:
            print(f"  SNR: {results['aggregate_metrics']['mean_snr_db']:.2f} dB")
        if results['aggregate_metrics']['mean_mcd']:
            print(f"  MCD: {results['aggregate_metrics']['mean_mcd']:.2f}")
        
        print("\n📥 Download results:")
        print("  modal volume get bible-audio-data vocoder_test_output/ ./modal_downloads/vocoder_test/")
    else:
        print("\n❌ Test failed.")
