"""
Modal script to test the trained vocoder
Synthesizes audio from unit sequences and compares with original

Run with: python3 -m modal run modal_vocoder_test.py::main
"""

import modal
import os

# Create Modal app
app = modal.App("bible-vocoder-test")

# Create persistent volume
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
        "librosa>=0.10.0",
        "tqdm",
    )
)

# Mount points
AUDIO_MOUNT = "/mnt/audio_data"
SEGMENTED_DIR = f"{AUDIO_MOUNT}/segmented_audio"
OUTPUT_DIR = f"{AUDIO_MOUNT}/portuguese_units"
VOCODER_DIR = f"{AUDIO_MOUNT}/vocoder_checkpoints"
TEST_OUTPUT_DIR = f"{AUDIO_MOUNT}/vocoder_test_output"


# Generator model definition (same as training)
GENERATOR_CODE = '''
import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    """
    Upsampling factor: 5 * 4 * 4 * 4 = 320x (matches XLSR-53 hop size)
    """
    def __init__(self, num_units=100, embed_dim=256):
        super().__init__()
        
        self.num_units = num_units
        self.unit_embed = nn.Embedding(num_units, embed_dim)
        self.pre_conv = nn.Conv1d(embed_dim, 512, 7, padding=3)
        
        # Upsample blocks: 5*4*4*4 = 320x total
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


@app.function(
    image=image,
    volumes={AUDIO_MOUNT: audio_volume},
    timeout=1800,
    gpu="A10G",
)
def test_vocoder(num_samples: int = 20):
    """
    Test the trained vocoder by synthesizing audio and comparing quality
    """
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
    
    # Load Generator
    print("\n📦 Loading Generator model...")
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(GENERATOR_CODE)
        gen_path = f.name
    
    spec = importlib.util.spec_from_file_location("generator", gen_path)
    gen_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen_module)
    Generator = gen_module.Generator
    
    generator = Generator(num_units=100).to(device)
    
    # Load checkpoint
    checkpoint_path = os.path.join(VOCODER_DIR, "vocoder_final.pt")
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return None
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    generator.load_state_dict(checkpoint['generator_state_dict'])
    generator.eval()
    print(f"✓ Generator loaded from: {checkpoint_path}")
    
    # Load corpus
    corpus_path = os.path.join(OUTPUT_DIR, "portuguese_corpus_timestamped.json")
    with open(corpus_path, 'r') as f:
        corpus = json.load(f)
    
    print(f"✓ Corpus loaded: {len(corpus)} entries")
    
    # Select random samples
    sample_keys = list(corpus.keys())
    np.random.shuffle(sample_keys)
    selected_keys = sample_keys[:num_samples]
    
    # Test results
    results = {
        "num_samples": num_samples,
        "samples": [],
        "aggregate_metrics": {}
    }
    
    all_snr = []
    all_mcd = []  # Mel Cepstral Distortion
    
    print(f"\n🔬 Testing {num_samples} samples...")
    
    for idx, segment_name in enumerate(tqdm(selected_keys, desc="Testing")):
        data = corpus[segment_name]
        units = data['units']
        
        # Find original audio
        audio_path = os.path.join(SEGMENTED_DIR, segment_name + ".wav")
        if not os.path.exists(audio_path):
            continue
        
        try:
            # Load original audio
            original_audio, sr = sf.read(audio_path)
            if sr != 16000:
                original_audio = librosa.resample(original_audio, orig_sr=sr, target_sr=16000)
            
            if len(original_audio.shape) > 1:
                original_audio = original_audio.mean(axis=1)
            
            # Synthesize audio
            units_tensor = torch.LongTensor(units).unsqueeze(0).to(device)
            
            with torch.no_grad():
                synth_audio = generator(units_tensor)
                synth_audio = synth_audio.squeeze().cpu().numpy()
            
            # Normalize
            synth_audio = synth_audio / (np.max(np.abs(synth_audio)) + 1e-8) * 0.95
            
            # Match lengths for comparison
            min_len = min(len(original_audio), len(synth_audio))
            original_segment = original_audio[:min_len]
            synth_segment = synth_audio[:min_len]
            
            # Calculate metrics
            # 1. Signal-to-Noise Ratio (treating original as signal, difference as noise)
            diff = original_segment - synth_segment
            signal_power = np.mean(original_segment ** 2)
            noise_power = np.mean(diff ** 2)
            snr = 10 * np.log10(signal_power / (noise_power + 1e-10))
            all_snr.append(snr)
            
            # 2. Mel Cepstral Distortion
            try:
                orig_mfcc = librosa.feature.mfcc(y=original_segment, sr=16000, n_mfcc=13)
                synth_mfcc = librosa.feature.mfcc(y=synth_segment, sr=16000, n_mfcc=13)
                
                min_frames = min(orig_mfcc.shape[1], synth_mfcc.shape[1])
                orig_mfcc = orig_mfcc[:, :min_frames]
                synth_mfcc = synth_mfcc[:, :min_frames]
                
                mcd = np.mean(np.sqrt(2 * np.sum((orig_mfcc - synth_mfcc) ** 2, axis=0)))
                all_mcd.append(mcd)
            except:
                mcd = None
            
            # Save synthesized audio
            if idx < 10:  # Save first 10 samples
                output_path = os.path.join(TEST_OUTPUT_DIR, f"synth_{idx:04d}.wav")
                write(output_path, 16000, (synth_audio * 32767).astype(np.int16))
                
                orig_output_path = os.path.join(TEST_OUTPUT_DIR, f"orig_{idx:04d}.wav")
                write(orig_output_path, 16000, (original_segment * 32767).astype(np.int16))
            
            results["samples"].append({
                "segment_name": segment_name[:50] + "..." if len(segment_name) > 50 else segment_name,
                "original_length_sec": len(original_audio) / 16000,
                "synth_length_sec": len(synth_audio) / 16000,
                "num_units": len(units),
                "snr_db": float(snr) if snr else None,
                "mcd": float(mcd) if mcd else None,
            })
            
        except Exception as e:
            print(f"⚠️  Error processing {segment_name[:30]}...: {e}")
            continue
    
    # Aggregate metrics
    results["aggregate_metrics"] = {
        "mean_snr_db": float(np.mean(all_snr)) if all_snr else None,
        "std_snr_db": float(np.std(all_snr)) if all_snr else None,
        "mean_mcd": float(np.mean(all_mcd)) if all_mcd else None,
        "std_mcd": float(np.std(all_mcd)) if all_mcd else None,
        "samples_tested": len(results["samples"]),
    }
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 Test Results Summary")
    print("=" * 60)
    print(f"  Samples tested: {results['aggregate_metrics']['samples_tested']}")
    if results['aggregate_metrics']['mean_snr_db']:
        print(f"  Mean SNR: {results['aggregate_metrics']['mean_snr_db']:.2f} dB (±{results['aggregate_metrics']['std_snr_db']:.2f})")
    if results['aggregate_metrics']['mean_mcd']:
        print(f"  Mean MCD: {results['aggregate_metrics']['mean_mcd']:.2f} (±{results['aggregate_metrics']['std_mcd']:.2f})")
    
    print("\n📁 Quality Interpretation:")
    if results['aggregate_metrics']['mean_snr_db']:
        snr = results['aggregate_metrics']['mean_snr_db']
        if snr > 10:
            print("  SNR > 10 dB: Excellent reconstruction")
        elif snr > 5:
            print("  SNR 5-10 dB: Good reconstruction")
        elif snr > 0:
            print("  SNR 0-5 dB: Fair reconstruction")
        else:
            print("  SNR < 0 dB: Poor reconstruction - needs more training")
    
    if results['aggregate_metrics']['mean_mcd']:
        mcd = results['aggregate_metrics']['mean_mcd']
        if mcd < 5:
            print("  MCD < 5: Excellent audio quality")
        elif mcd < 8:
            print("  MCD 5-8: Good audio quality")
        elif mcd < 12:
            print("  MCD 8-12: Acceptable audio quality")
        else:
            print("  MCD > 12: Needs improvement")
    
    # Save results
    results_path = os.path.join(TEST_OUTPUT_DIR, "test_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to: {results_path}")
    print(f"✓ Sample audio saved to: {TEST_OUTPUT_DIR}")
    
    # Commit volume
    audio_volume.commit()
    
    return results


@app.local_entrypoint()
def main(num_samples: int = 20):
    """
    Test the trained vocoder
    
    Args:
        num_samples: Number of samples to test (default: 20)
    """
    print("🔬 Starting Vocoder Quality Test")
    print("=" * 60)
    
    results = test_vocoder.remote(num_samples=num_samples)
    
    if results:
        print("\n✅ Test Complete!")
        print(f"  Samples tested: {results['aggregate_metrics']['samples_tested']}")
        if results['aggregate_metrics']['mean_snr_db']:
            print(f"  Mean SNR: {results['aggregate_metrics']['mean_snr_db']:.2f} dB")
        if results['aggregate_metrics']['mean_mcd']:
            print(f"  Mean MCD: {results['aggregate_metrics']['mean_mcd']:.2f}")
        
        print("\n📥 Download test audio with:")
        print("  python3 -m modal volume get bible-audio-data vocoder_test_output/ ./modal_downloads/vocoder_test/")
    else:
        print("\n❌ Test failed. Check logs above.")
