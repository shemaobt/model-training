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
# # Vocoder V2 Quality Test
#
# Tests the V2 vocoder with pitch conditioning.
#
# ## Metrics
#
# - **SNR (Signal-to-Noise Ratio)**: Higher is better
# - **MCD (Mel Cepstral Distortion)**: Lower is better (< 5.0 is good)
# - **F0 RMSE**: Pitch accuracy (lower is better)
#
# ## Run on Modal
#
# ```bash
# # Test V2 vocoder
# python3 -m modal run src/training/vocoder_test_v2.py::main --num-samples 50
#
# # Download results
# modal volume get bible-audio-data vocoder_v2_test_output/ ./modal_downloads/vocoder_v2_test/
# ```

# %%
import modal
import os
from src.constants import (
    SAMPLE_RATE,
    HOP_SIZE,
    NUM_ACOUSTIC_UNITS,
    NUM_PITCH_BINS,
    F0_MIN,
    F0_MAX,
)

# %%
app = modal.App("vocoder-v2-test")

audio_volume = modal.Volume.from_name("bible-audio-data", create_if_missing=True)

# %%
image = (
    modal.Image.debian_slim()
    .apt_install("ffmpeg", "libsndfile1")
    .pip_install(
        "torch>=2.0.0",
        "torchaudio>=2.0.0",
        "numpy<2",
        "scipy",
        "soundfile>=0.12.0",
        "tqdm",
        "librosa>=0.10.0",
    )
    .add_local_dir("src", remote_path="/root/src")
)

# %%
AUDIO_MOUNT = "/mnt/audio_data"

LANGUAGE_CONFIGS = {
    "portuguese": {
        "segmented_dir": f"{AUDIO_MOUNT}/segmented_audio",
        "output_dir": f"{AUDIO_MOUNT}/portuguese_units",
        "vocoder_dir": f"{AUDIO_MOUNT}/vocoder_v2_checkpoints",
        "test_output_dir": f"{AUDIO_MOUNT}/vocoder_v2_test_output",
        "corpus_file": "portuguese_corpus_timestamped.json",
    },
    "satere": {
        "segmented_dir": f"{AUDIO_MOUNT}/segmented_audio_satere",
        "output_dir": f"{AUDIO_MOUNT}/satere_units",
        "vocoder_dir": f"{AUDIO_MOUNT}/vocoder_v2_satere_checkpoints",
        "test_output_dir": f"{AUDIO_MOUNT}/vocoder_v2_satere_test_output",
        "corpus_file": "satere_corpus_timestamped.json",
    },
}

import os as _os
_LANGUAGE = _os.environ.get("TRAINING_LANGUAGE", "portuguese")
_config = LANGUAGE_CONFIGS.get(_LANGUAGE, LANGUAGE_CONFIGS["portuguese"])

SEGMENTED_DIR = _config["segmented_dir"]
OUTPUT_DIR = _config["output_dir"]
VOCODER_DIR = _config["vocoder_dir"]
TEST_OUTPUT_DIR = _config["test_output_dir"]

# %% [markdown]
# ## Generator V2 Definition

# %%
GENERATOR_V2_CODE = '''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm

def get_padding(kernel_size, dilation=1):
    return (kernel_size * dilation - dilation) // 2

class ResBlock(nn.Module):
    def __init__(self, channels, kernel_size=3, dilations=[[1,1],[3,1],[5,1]]):
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
    def __init__(self, channels, kernel_sizes=[3, 7, 11]):
        super().__init__()
        self.resblocks = nn.ModuleList([ResBlock(channels, k) for k in kernel_sizes])
    
    def forward(self, x):
        out = self.resblocks[0](x)
        for rb in self.resblocks[1:]:
            out = out + rb(x)
        return out / len(self.resblocks)

class GeneratorV2(nn.Module):
    def __init__(self, num_units=100, num_pitch_bins=32, 
                 unit_embed_dim=256, pitch_embed_dim=64):
        super().__init__()
        self.upsample_rates = [5, 4, 4, 4]
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

# %% [markdown]
# ## Test Function

# %%
@app.function(
    image=image,
    volumes={AUDIO_MOUNT: audio_volume},
    timeout=3600,
    gpu="A10G",
)
def test_vocoder_v2(num_samples: int = 50, checkpoint: str = "v2_best.pt", language: str = "portuguese", model_key: str = "mms-300m"):
    import torch
    import numpy as np
    import soundfile as sf
    import librosa
    import json
    from scipy.io.wavfile import write
    from tqdm import tqdm
    import tempfile
    import importlib.util
    
    print("=" * 60)
    print("🔬 Vocoder V2 Quality Test")
    print("=" * 60)
    
    config = LANGUAGE_CONFIGS.get(language, LANGUAGE_CONFIGS["portuguese"])
    segmented_dir = config["segmented_dir"]
    output_dir = config["output_dir"]
    vocoder_dir = config["vocoder_dir"]
    test_output_dir = config["test_output_dir"]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Language: {language}")
    
    os.makedirs(test_output_dir, exist_ok=True)
    
    def load_module(code, name):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            path = f.name
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    
    gen_module = load_module(GENERATOR_V2_CODE, "generator_v2")
    GeneratorV2 = gen_module.GeneratorV2
    
    generator = GeneratorV2(num_units=NUM_ACOUSTIC_UNITS, num_pitch_bins=NUM_PITCH_BINS).to(device)
    
    ckpt_path = os.path.join(vocoder_dir, checkpoint)
    if not os.path.exists(ckpt_path):
        print(f"❌ Checkpoint not found: {ckpt_path}")
        ckpt_path = os.path.join(vocoder_dir, "v2_latest.pt")
        if not os.path.exists(ckpt_path):
            print(f"❌ No V2 checkpoint found!")
            return None
    
    print(f"📥 Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    generator.load_state_dict(ckpt['generator_state_dict'])
    generator.eval()
    print(f"✓ Generator loaded (epoch {ckpt.get('epoch', '?')})")
    
    corpus_path = os.path.join(output_dir, f"{language}_corpus_timestamped_{model_key}.json")
    with open(corpus_path, 'r') as f:
        corpus = json.load(f)
    
    sample_keys = list(corpus.keys())
    np.random.shuffle(sample_keys)
    selected_keys = sample_keys[:num_samples]
    
    all_snr = []
    all_mcd = []
    all_f0_rmse = []
    results = {"samples": [], "aggregate_metrics": {}}
    
    hop_size = HOP_SIZE
    
    print(f"\n🔬 Testing {num_samples} samples...")
    
    for idx, segment_name in enumerate(tqdm(selected_keys, desc="Testing")):
        data = corpus[segment_name]
        units = data['units']
        
        audio_path = None
        for ext in ['.wav', '.WAV']:
            p = os.path.join(segmented_dir, segment_name + ext)
            if os.path.exists(p):
                audio_path = p
                break
        
        if not audio_path:
            continue
        
        try:
            original_audio, sr = sf.read(audio_path)
            if sr != SAMPLE_RATE:
                original_audio = librosa.resample(original_audio, orig_sr=sr, target_sr=SAMPLE_RATE)
            if len(original_audio.shape) > 1:
                original_audio = original_audio.mean(axis=1)
            
            f0_orig, voiced, _ = librosa.pyin(
                original_audio, fmin=F0_MIN, fmax=F0_MAX, sr=SAMPLE_RATE, hop_length=hop_size
            )
            f0_orig = np.nan_to_num(f0_orig, nan=0.0)
            
            num_units = len(units)
            if len(f0_orig) < num_units:
                f0_orig = np.pad(f0_orig, (0, num_units - len(f0_orig)))
            else:
                f0_orig = f0_orig[:num_units]
            
            pitch_bins = np.zeros_like(f0_orig, dtype=np.int64)
            voiced_mask = f0_orig > 0
            if np.any(voiced_mask):
                log_f0 = np.log(np.clip(f0_orig[voiced_mask], F0_MIN, F0_MAX))
                log_min, log_max = np.log(F0_MIN), np.log(F0_MAX)
                normalized = (log_f0 - log_min) / (log_max - log_min)
                pitch_bins[voiced_mask] = (normalized * (NUM_PITCH_BINS - 1) + 1).astype(np.int64)
            
            units_tensor = torch.LongTensor(units).unsqueeze(0).to(device)
            pitch_tensor = torch.LongTensor(pitch_bins).unsqueeze(0).to(device)
            
            with torch.no_grad():
                synth_audio = generator(units_tensor, pitch_tensor).squeeze().cpu().numpy()
            
            synth_audio = synth_audio / (np.max(np.abs(synth_audio)) + 1e-8) * 0.95
            
            min_len = min(len(original_audio), len(synth_audio))
            orig_seg = original_audio[:min_len]
            synth_seg = synth_audio[:min_len]
            
            diff = orig_seg - synth_seg
            signal_power = np.mean(orig_seg ** 2)
            noise_power = np.mean(diff ** 2)
            snr = 10 * np.log10(signal_power / (noise_power + 1e-10))
            all_snr.append(snr)
            
            try:
                orig_mfcc = librosa.feature.mfcc(y=orig_seg, sr=SAMPLE_RATE, n_mfcc=13)
                synth_mfcc = librosa.feature.mfcc(y=synth_seg, sr=SAMPLE_RATE, n_mfcc=13)
                min_frames = min(orig_mfcc.shape[1], synth_mfcc.shape[1])
                mcd = np.mean(np.sqrt(2 * np.sum(
                    (orig_mfcc[:, :min_frames] - synth_mfcc[:, :min_frames]) ** 2, axis=0
                )))
                all_mcd.append(mcd)
            except:
                mcd = None
            
            try:
                f0_synth, _, _ = librosa.pyin(
                    synth_seg, fmin=F0_MIN, fmax=F0_MAX, sr=SAMPLE_RATE, hop_length=hop_size
                )
                f0_synth = np.nan_to_num(f0_synth, nan=0.0)
                
                min_f0_len = min(len(f0_orig), len(f0_synth))
                f0_orig_aligned = f0_orig[:min_f0_len]
                f0_synth_aligned = f0_synth[:min_f0_len]
                
                voiced = (f0_orig_aligned > 0) & (f0_synth_aligned > 0)
                if np.sum(voiced) > 0:
                    f0_rmse = np.sqrt(np.mean(
                        (f0_orig_aligned[voiced] - f0_synth_aligned[voiced]) ** 2
                    ))
                    all_f0_rmse.append(f0_rmse)
                else:
                    f0_rmse = None
            except:
                f0_rmse = None
            
            if idx < 20:
                write(
                    os.path.join(test_output_dir, f"v2_synth_{idx:04d}.wav"),
                    SAMPLE_RATE, (synth_audio * 32767).astype(np.int16)
                )
                write(
                    os.path.join(test_output_dir, f"v2_orig_{idx:04d}.wav"),
                    SAMPLE_RATE, (orig_seg * 32767).astype(np.int16)
                )
            
            results["samples"].append({
                "segment": segment_name[:50],
                "snr_db": float(snr),
                "mcd": float(mcd) if mcd else None,
                "f0_rmse": float(f0_rmse) if f0_rmse else None,
            })
            
        except Exception as e:
            print(f"⚠️  Error on {segment_name}: {e}")
            continue
    
    results["aggregate_metrics"] = {
        "mean_snr_db": float(np.mean(all_snr)) if all_snr else None,
        "std_snr_db": float(np.std(all_snr)) if all_snr else None,
        "mean_mcd": float(np.mean(all_mcd)) if all_mcd else None,
        "std_mcd": float(np.std(all_mcd)) if all_mcd else None,
        "mean_f0_rmse": float(np.mean(all_f0_rmse)) if all_f0_rmse else None,
        "std_f0_rmse": float(np.std(all_f0_rmse)) if all_f0_rmse else None,
        "samples_tested": len(results["samples"]),
    }
    
    print("\n" + "=" * 60)
    print("📊 Results")
    print("=" * 60)
    m = results['aggregate_metrics']
    if m['mean_snr_db']:
        print(f"  SNR: {m['mean_snr_db']:.2f} dB (±{m['std_snr_db']:.2f})")
    if m['mean_mcd']:
        print(f"  MCD: {m['mean_mcd']:.2f} (±{m['std_mcd']:.2f})")
        if m['mean_mcd'] < 5.0:
            print("    ✓ Good MCD (< 5.0)")
        elif m['mean_mcd'] < 8.0:
            print("    ⚠️  Moderate MCD (5.0-8.0)")
        else:
            print("    ❌ High MCD (> 8.0)")
    if m['mean_f0_rmse']:
        print(f"  F0 RMSE: {m['mean_f0_rmse']:.1f} Hz (±{m['std_f0_rmse']:.1f})")
        if m['mean_f0_rmse'] < 20.0:
            print("    ✓ Good pitch accuracy (< 20 Hz)")
        elif m['mean_f0_rmse'] < 50.0:
            print("    ⚠️  Moderate pitch accuracy (20-50 Hz)")
        else:
            print("    ❌ Poor pitch accuracy (> 50 Hz)")
    print("=" * 60)
    
    with open(os.path.join(test_output_dir, "test_results_v2.json"), 'w') as f:
        json.dump(results, f, indent=2)
    
    audio_volume.commit()
    
    return results

# %% [markdown]
# ## Entry Point

# %%
@app.local_entrypoint()
def main(language: str = "portuguese", num_samples: int = 50, checkpoint: str = "v2_best.pt"):
    import os
    os.environ["TRAINING_LANGUAGE"] = language
    
    config = LANGUAGE_CONFIGS.get(language, LANGUAGE_CONFIGS["portuguese"])
    
    print("🔬 Starting Vocoder V2 Quality Test")
    print(f"  Language: {language}")
    print(f"  Samples: {num_samples}")
    print(f"  Checkpoint: {config['vocoder_dir']}/{checkpoint}")
    
    results = test_vocoder_v2.remote(num_samples=num_samples, checkpoint=checkpoint, language=language)
    
    if results:
        m = results['aggregate_metrics']
        print(f"\n✅ Tested {m['samples_tested']} samples")
        if m['mean_snr_db']:
            print(f"  SNR: {m['mean_snr_db']:.2f} dB")
        if m['mean_mcd']:
            print(f"  MCD: {m['mean_mcd']:.2f}")
        if m['mean_f0_rmse']:
            print(f"  F0 RMSE: {m['mean_f0_rmse']:.1f} Hz")
        
        print("\n📥 Download results:")
        print("  modal volume get bible-audio-data vocoder_v2_test_output/ ./modal_downloads/vocoder_v2_test/")
    else:
        print("\n❌ Test failed.")
