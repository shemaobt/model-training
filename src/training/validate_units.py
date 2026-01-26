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
# # Acoustic Units Validation
#
# Validates the acoustic units from Phase 1 and optionally synthesizes
# audio using a pre-trained vocoder.
#
# ## Features
#
# - Load and verify K-Means model
# - Analyze unit sequence statistics
# - Optionally synthesize audio from units
#
# ## Run on Modal
#
# ```bash
# # Validate units only (no synthesis)
# python3 -m modal run src/training/validate_units.py::main
#
# # Validate and synthesize with vocoder
# python3 -m modal run src/training/validate_units.py::main \
#     --generator-checkpoint /mnt/audio_data/vocoder_checkpoints/vocoder_final.pt
# ```

# %% [markdown]
# ## Modal Configuration

# %%
import modal
import os

# %%
app = modal.App("bible-audio-training")

audio_volume = modal.Volume.from_name(
    "bible-audio-data",
    create_if_missing=True
)

# %%
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

# %%
AUDIO_MOUNT = "/mnt/audio_data"
OUTPUT_DIR = f"{AUDIO_MOUNT}/portuguese_units"
CHECKING_OUTPUT_DIR = f"{AUDIO_MOUNT}/checking_output"

# %% [markdown]
# ## Generator Model (Legacy)
#
# This is an older generator architecture (800x upsampling).
# For the trained vocoder, use `vocoder_test.py` instead.

# %%
GENERATOR_CODE = """
import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    \"\"\"Legacy generator with 800x upsampling.\"\"\"
    def __init__(self, num_units=100):
        super().__init__()

        self.unit_embed = nn.Embedding(num_units, 256)
        self.pre_conv = nn.Conv1d(256, 512, 7, padding=3)

        self.upsamples = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose1d(512, 256, 10, stride=5, padding=2),
                nn.LeakyReLU(0.1),
                nn.Conv1d(256, 256, 7, padding=3)
            ),
            nn.Sequential(
                nn.ConvTranspose1d(256, 128, 10, stride=5, padding=2),
                nn.LeakyReLU(0.1),
                nn.Conv1d(128, 128, 7, padding=3)
            ),
            nn.Sequential(
                nn.ConvTranspose1d(128, 64, 8, stride=4, padding=2),
                nn.LeakyReLU(0.1),
                nn.Conv1d(64, 64, 7, padding=3)
            ),
            nn.Sequential(
                nn.ConvTranspose1d(64, 32, 8, stride=4, padding=2),
                nn.LeakyReLU(0.1),
                nn.Conv1d(32, 32, 7, padding=3)
            ),
            nn.Sequential(
                nn.ConvTranspose1d(32, 64, 4, stride=2, padding=1),
                nn.LeakyReLU(0.1),
                nn.Conv1d(64, 64, 7, padding=3)
            ),
        ])

        self.post_conv = nn.Conv1d(64, 1, 7, padding=3)

    def forward(self, x):
        x = self.unit_embed(x).transpose(1, 2)
        x = self.pre_conv(x)
        x = F.leaky_relu(x, 0.1)

        for upsample in self.upsamples:
            x = upsample(x)
            x = F.leaky_relu(x, 0.1)

        x = self.post_conv(x)
        x = torch.tanh(x)
        return x.squeeze(1)
"""

# %% [markdown]
# ## Validation Function

# %%
@app.function(
    image=image,
    volumes={AUDIO_MOUNT: audio_volume},
    timeout=1800,
    gpu="A10G",
)
def validate_and_synthesize(
    num_samples: int = 10,
    generator_checkpoint_path: str = None,
    use_random_units: bool = False
):
    """Validate acoustic units and optionally synthesize audio."""
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
        print(f"❌ K-Means model not found. Run Phase 1 first!")
        return None
    
    print(f"✓ Loading K-Means from: {kmeans_path}")
    kmeans = joblib.load(kmeans_path)
    num_units = kmeans.n_clusters
    print(f"✓ K-Means loaded: {num_units} acoustic units")
    
    # Load Generator if checkpoint provided
    generator = None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if generator_checkpoint_path and os.path.exists(generator_checkpoint_path):
        print(f"\n📦 Loading Generator from: {generator_checkpoint_path}")
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(GENERATOR_CODE)
            gen_module_path = f.name
        
        import importlib.util
        spec = importlib.util.spec_from_file_location("generator", gen_module_path)
        gen_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(gen_module)
        Generator = gen_module.Generator
        
        generator = Generator(num_units=num_units).to(device)
        
        try:
            checkpoint = torch.load(generator_checkpoint_path, map_location=device)
            if 'generator_state_dict' in checkpoint:
                generator.load_state_dict(checkpoint['generator_state_dict'])
            else:
                generator.load_state_dict(checkpoint)
            generator.eval()
            print(f"✓ Generator loaded!")
        except Exception as e:
            print(f"⚠️  Error loading Generator: {e}")
            generator = None
    else:
        print(f"\n⚠️  No Generator checkpoint provided")
        print("   Will validate units only")
    
    # Load unit sequences
    units_file = os.path.join(OUTPUT_DIR, "all_units_for_bpe.txt")
    
    if use_random_units:
        print(f"\n🎲 Generating {num_samples} random sequences...")
        unit_sequences = []
        for _ in range(num_samples):
            seq_len = np.random.randint(50, 200)
            random_units = np.random.randint(0, num_units, seq_len).tolist()
            unit_sequences.append(random_units)
    else:
        if not os.path.exists(units_file):
            print(f"❌ Units file not found: {units_file}")
            return None
        
        print(f"\n📖 Loading unit sequences...")
        unit_sequences = []
        with open(units_file, 'r') as f:
            for line in tqdm(f, desc="Loading", total=num_samples):
                line = line.strip()
                if line:
                    units = [int(u) for u in line.split()]
                    unit_sequences.append(units)
                    if len(unit_sequences) >= num_samples:
                        break
        
        print(f"✓ Loaded {len(unit_sequences)} sequences")
    
    # Validate units
    print(f"\n🔍 Validating units...")
    all_units_flat = [u for seq in unit_sequences for u in seq]
    unique_units = set(all_units_flat)
    
    validation_results = {
        "total_sequences": len(unit_sequences),
        "unit_stats": {
            "total_units": len(all_units_flat),
            "unique_units": len(unique_units),
            "expected_unique": num_units,
            "unit_range": (min(all_units_flat), max(all_units_flat)),
            "avg_sequence_length": np.mean([len(s) for s in unit_sequences]),
        },
        "synthesis_results": []
    }
    
    print(f"  Total units: {validation_results['unit_stats']['total_units']:,}")
    print(f"  Unique: {validation_results['unit_stats']['unique_units']}/{num_units}")
    print(f"  Range: {validation_results['unit_stats']['unit_range']}")
    
    # Synthesize if Generator available
    if generator is not None:
        print(f"\n🎵 Synthesizing audio...")
        
        for idx, units in enumerate(tqdm(unit_sequences[:num_samples], desc="Synthesizing")):
            try:
                units_tensor = torch.LongTensor(units).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    synth_audio = generator(units_tensor).squeeze().cpu().numpy()
                
                if len(synth_audio) > 0:
                    max_val = np.max(np.abs(synth_audio))
                    if max_val > 0:
                        synth_audio = synth_audio / max_val * 0.95
                
                output_path = os.path.join(CHECKING_OUTPUT_DIR, f"synthesized_{idx:04d}.wav")
                write(output_path, 16000, (synth_audio * 32767).astype(np.int16))
                
                validation_results["synthesis_results"].append({
                    "sequence_idx": idx,
                    "num_units": len(units),
                    "audio_length_sec": len(synth_audio) / 16000,
                })
                
            except Exception as e:
                print(f"⚠️  Error: {e}")
                validation_results["synthesis_results"].append({"sequence_idx": idx, "error": str(e)})
        
        successful = len([r for r in validation_results['synthesis_results'] if 'error' not in r])
        print(f"✓ Synthesized {successful} files")
    else:
        print(f"\n⏭️  Skipping synthesis")
    
    # Save results
    results_path = os.path.join(CHECKING_OUTPUT_DIR, "validation_results.json")
    with open(results_path, 'w') as f:
        json.dump(validation_results, f, indent=2)
    
    print(f"\n✓ Results saved: {results_path}")
    
    audio_volume.commit()
    return validation_results

# %% [markdown]
# ## Entry Point

# %%
@app.local_entrypoint()
def main(num_samples: int = 10, generator_checkpoint: str = None, use_random: bool = False):
    """Run validation and synthesis."""
    print("🚀 Starting Acoustic Units Validation")
    print("=" * 60)
    
    results = validate_and_synthesize.remote(
        num_samples=num_samples,
        generator_checkpoint_path=generator_checkpoint,
        use_random_units=use_random
    )
    
    if results:
        print(f"\n✅ Validated {results['total_sequences']} sequences")
        if results['synthesis_results']:
            successful = len([r for r in results['synthesis_results'] if 'error' not in r])
            print(f"  Synthesized {successful} files")
    else:
        print("\n❌ Validation failed.")
