"""
Modal deployment for Checking1 notebook - Audio Synthesis Validation
Tests acoustic units by synthesizing audio using a Generator (vocoder) model

Run with: /usr/local/bin/python3 -m modal run modal_checking.py::main
"""

import modal
import os

# Create Modal app
app = modal.App("bible-audio-training")

# Create persistent volume (same as Phase 1 & 2)
audio_volume = modal.Volume.from_name(
    "bible-audio-data",
    create_if_missing=True
)

# Create image with dependencies
image = (
    modal.Image.debian_slim()
    .pip_install(
        "torch>=2.0.0",
        "torchaudio>=2.0.0",
        "numpy<2",
        "scipy",
        "scikit-learn",
        "joblib",
        "soundfile>=0.12.0",
        "tqdm",
    )
)

# Mount points
AUDIO_MOUNT = "/mnt/audio_data"
OUTPUT_DIR = f"{AUDIO_MOUNT}/portuguese_units"
CHECKING_OUTPUT_DIR = f"{AUDIO_MOUNT}/checking_output"


# Generator model definition (same as notebook)
GENERATOR_CODE = """
import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self, num_units=100):
        super().__init__()

        self.unit_embed = nn.Embedding(num_units, 256)
        self.pre_conv = nn.Conv1d(256, 512, 7, padding=3)

        self.upsamples = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose1d(512, 256, 10, stride=5, padding=2, output_padding=0),
                nn.LeakyReLU(0.1),
                nn.Conv1d(256, 256, 7, padding=3)
            ),
            nn.Sequential(
                nn.ConvTranspose1d(256, 128, 10, stride=5, padding=2, output_padding=0),
                nn.LeakyReLU(0.1),
                nn.Conv1d(128, 128, 7, padding=3)
            ),
            nn.Sequential(
                nn.ConvTranspose1d(128, 64, 8, stride=4, padding=2, output_padding=0),
                nn.LeakyReLU(0.1),
                nn.Conv1d(64, 64, 7, padding=3)
            ),
            nn.Sequential(
                nn.ConvTranspose1d(64, 32, 8, stride=4, padding=2, output_padding=0),
                nn.LeakyReLU(0.1),
                nn.Conv1d(32, 32, 7, padding=3)
            ),
            nn.Sequential(
                nn.ConvTranspose1d(32, 64, 4, stride=2, padding=1, output_padding=0),
                nn.LeakyReLU(0.1),
                nn.Conv1d(64, 64, 7, padding=3)
            ),
        ])

        self.post_conv = nn.Conv1d(64, 1, 7, padding=3)

    def forward(self, x):
        x = self.unit_embed(x).transpose(1, 2)  # [B, 256, T]
        x = self.pre_conv(x)  # [B, 512, T]
        x = F.leaky_relu(x, 0.1)

        for upsample in self.upsamples:
            x = upsample(x)
            x = F.leaky_relu(x, 0.1)

        x = self.post_conv(x)  # [B, 1, T*800]
        x = torch.tanh(x)
        return x.squeeze(1)
"""


@app.function(
    image=image,
    volumes={AUDIO_MOUNT: audio_volume},
    timeout=1800,  # 30 minutes
    gpu="A10G",  # GPU needed for Generator inference
)
def validate_and_synthesize(
    num_samples: int = 10,
    generator_checkpoint_path: str = None,
    use_random_units: bool = False
):
    """
    Validate acoustic units and optionally synthesize audio
    
    Args:
        num_samples: Number of unit sequences to test
        generator_checkpoint_path: Path to Generator checkpoint (if None, will validate units only)
        use_random_units: If True, generate random unit sequences for testing
    """
    import torch
    import numpy as np
    import joblib
    import json
    from scipy.io.wavfile import write
    from tqdm import tqdm
    import tempfile
    
    os.makedirs(CHECKING_OUTPUT_DIR, exist_ok=True)
    
    # Load K-Means model
    kmeans_path = os.path.join(OUTPUT_DIR, "portuguese_kmeans.pkl")
    if not os.path.exists(kmeans_path):
        print(f"❌ K-Means model not found: {kmeans_path}")
        print("   Please run Phase 1 first!")
        return None
    
    print(f"✓ Loading K-Means model from: {kmeans_path}")
    kmeans = joblib.load(kmeans_path)
    num_units = kmeans.n_clusters
    print(f"✓ K-Means loaded: {num_units} acoustic units")
    
    # Load Generator model if checkpoint provided
    generator = None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if generator_checkpoint_path and os.path.exists(generator_checkpoint_path):
        print(f"\n📦 Loading Generator from: {generator_checkpoint_path}")
        
        # Write Generator code to temp file and import
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(GENERATOR_CODE)
            gen_module_path = f.name
        
        import importlib.util
        spec = importlib.util.spec_from_file_location("generator", gen_module_path)
        gen_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(gen_module)
        Generator = gen_module.Generator
        
        # Create and load Generator
        generator = Generator(num_units=num_units).to(device)
        
        try:
            checkpoint = torch.load(generator_checkpoint_path, map_location=device)
            if 'generator_state_dict' in checkpoint:
                generator.load_state_dict(checkpoint['generator_state_dict'])
            else:
                generator.load_state_dict(checkpoint)
            generator.eval()
            print(f"✓ Generator loaded successfully!")
        except Exception as e:
            print(f"⚠️  Error loading Generator checkpoint: {e}")
            print("   Will validate units only (no synthesis)")
            generator = None
    else:
        print(f"\n⚠️  No Generator checkpoint provided")
        print("   Will validate units only (no audio synthesis)")
    
    # Load unit sequences
    units_file = os.path.join(OUTPUT_DIR, "all_units_for_bpe.txt")
    corpus_file = os.path.join(OUTPUT_DIR, "portuguese_corpus_timestamped.json")
    
    if use_random_units:
        print(f"\n🎲 Generating {num_samples} random unit sequences for testing...")
        unit_sequences = []
        for _ in range(num_samples):
            # Generate random sequence of length 50-200 units
            seq_len = np.random.randint(50, 200)
            random_units = np.random.randint(0, num_units, seq_len).tolist()
            unit_sequences.append(random_units)
    else:
        if not os.path.exists(units_file):
            print(f"❌ Units file not found: {units_file}")
            return None
        
        print(f"\n📖 Reading unit sequences from: {units_file}")
        unit_sequences = []
        with open(units_file, 'r') as f:
            for line in tqdm(f, desc="Loading sequences", total=num_samples):
                line = line.strip()
                if line:
                    units = [int(u) for u in line.split()]
                    unit_sequences.append(units)
                    if len(unit_sequences) >= num_samples:
                        break
        
        print(f"✓ Loaded {len(unit_sequences)} unit sequences")
    
    # Validate units
    print(f"\n🔍 Validating acoustic units...")
    validation_results = {
        "total_sequences": len(unit_sequences),
        "unit_stats": {},
        "synthesis_results": []
    }
    
    all_units_flat = []
    for seq in unit_sequences:
        all_units_flat.extend(seq)
    
    unique_units = set(all_units_flat)
    validation_results["unit_stats"] = {
        "total_units": len(all_units_flat),
        "unique_units": len(unique_units),
        "expected_unique": num_units,
        "unit_range": (min(all_units_flat), max(all_units_flat)),
        "avg_sequence_length": np.mean([len(s) for s in unit_sequences]),
        "min_sequence_length": min(len(s) for s in unit_sequences),
        "max_sequence_length": max(len(s) for s in unit_sequences),
    }
    
    print(f"  Total units: {validation_results['unit_stats']['total_units']:,}")
    print(f"  Unique units: {validation_results['unit_stats']['unique_units']}/{num_units}")
    print(f"  Unit range: {validation_results['unit_stats']['unit_range']}")
    print(f"  Avg sequence length: {validation_results['unit_stats']['avg_sequence_length']:.1f}")
    
    # Synthesize audio if Generator available
    if generator is not None:
        print(f"\n🎵 Synthesizing audio from units...")
        
        for idx, units in enumerate(tqdm(unit_sequences[:num_samples], desc="Synthesizing")):
            try:
                # Convert to tensor
                units_tensor = torch.LongTensor(units).unsqueeze(0).to(device)
                
                # Synthesize
                with torch.no_grad():
                    synth_audio = generator(units_tensor)
                    synth_audio = synth_audio.squeeze().cpu().numpy()
                
                # Normalize
                if len(synth_audio) > 0:
                    max_val = np.max(np.abs(synth_audio))
                    if max_val > 0:
                        synth_audio = synth_audio / max_val * 0.95
                
                # Save audio
                output_path = os.path.join(CHECKING_OUTPUT_DIR, f"synthesized_{idx:04d}.wav")
                write(output_path, 16000, (synth_audio * 32767).astype(np.int16))
                
                validation_results["synthesis_results"].append({
                    "sequence_idx": idx,
                    "num_units": len(units),
                    "audio_length_samples": len(synth_audio),
                    "audio_length_sec": len(synth_audio) / 16000,
                    "output_path": output_path
                })
                
            except Exception as e:
                print(f"⚠️  Error synthesizing sequence {idx}: {e}")
                validation_results["synthesis_results"].append({
                    "sequence_idx": idx,
                    "error": str(e)
                })
        
        print(f"✓ Synthesized {len([r for r in validation_results['synthesis_results'] if 'error' not in r])} audio files")
    else:
        print(f"\n⏭️  Skipping synthesis (no Generator model)")
    
    # Save validation results
    results_path = os.path.join(CHECKING_OUTPUT_DIR, "validation_results.json")
    with open(results_path, 'w') as f:
        json.dump(validation_results, f, indent=2)
    
    print(f"\n✓ Validation complete!")
    print(f"  Results saved to: {results_path}")
    if generator:
        print(f"  Synthesized audio saved to: {CHECKING_OUTPUT_DIR}")
    
    # Commit volume changes
    audio_volume.commit()
    
    return validation_results


@app.local_entrypoint()
def main(num_samples: int = 10, generator_checkpoint: str = None, use_random: bool = False):
    """
    Run validation and synthesis
    
    Args:
        num_samples: Number of unit sequences to test (default: 10)
        generator_checkpoint: Path to Generator checkpoint file (optional)
        use_random: Use random unit sequences instead of real ones (default: False)
    """
    print("🚀 Starting Acoustic Units Validation & Synthesis")
    print("=" * 60)
    
    checkpoint_path = None
    if generator_checkpoint:
        # If provided, assume it's a local path that needs to be uploaded
        # For now, we'll expect it to be already in the Modal volume
        checkpoint_path = generator_checkpoint
    
    results = validate_and_synthesize.remote(
        num_samples=num_samples,
        generator_checkpoint_path=checkpoint_path,
        use_random_units=use_random
    )
    
    if results:
        print("\n✅ Validation complete!")
        print(f"  Validated {results['total_sequences']} sequences")
        if results['synthesis_results']:
            successful = len([r for r in results['synthesis_results'] if 'error' not in r])
            print(f"  Successfully synthesized {successful} audio files")
    else:
        print("\n❌ Validation failed. Check logs above for details.")
