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
# # Full Training Pipeline
#
# Run all training phases sequentially or selectively.
#
# ## Usage
#
# ```bash
# # Run ALL phases (1 + 2 + 3 V2)
# python3 -m modal run --detach src/training/run_full_pipeline.py::main
#
# # Run specific phases
# python3 -m modal run --detach src/training/run_full_pipeline.py::main --phases 1,2,3
# python3 -m modal run --detach src/training/run_full_pipeline.py::main --phases 3  # Only vocoder
#
# # Use V1 vocoder instead of V2
# python3 -m modal run --detach src/training/run_full_pipeline.py::main --vocoder-version v1
#
# # Skip phases that are already complete
# python3 -m modal run --detach src/training/run_full_pipeline.py::main --phases 2,3
# ```
#
# ## What Each Phase Does
#
# - **Phase 1**: Extract XLSR-53 features → K-Means clustering → Unit sequences
# - **Phase 2**: Train BPE tokenizer on unit sequences → Discover motifs
# - **Phase 3**: Train vocoder (V1 or V2) → Unit sequences → Audio

# %%
import modal
import os
import subprocess
import sys

# %%
app = modal.App("full-training-pipeline")

# %%
@app.local_entrypoint()
def main(
    phases: str = "1,2,3",
    vocoder_version: str = "v2",
    detach: bool = True,
    # Phase 1 options
    skip_segmentation: bool = True,
    # Phase 3 options
    epochs: int = 1000,
    batch_size: int = 12,
    segment_length: int = 32000,
    patience: int = 100,
):
    """
    Run the full training pipeline.
    
    Args:
        phases: Comma-separated list of phases to run (1,2,3)
        vocoder_version: 'v1' or 'v2' (default: v2)
        detach: Run in detached mode (default: True)
        skip_segmentation: Skip audio segmentation in Phase 1 (default: True)
        epochs: Max epochs for vocoder training
        batch_size: Batch size for vocoder training
        segment_length: Audio segment length in samples (16000=1s, 32000=2s)
        patience: Early stopping patience
    """
    
    # Parse phases
    phase_list = [int(p.strip()) for p in phases.split(",")]
    
    print("=" * 60)
    print("🚀 Full Training Pipeline")
    print("=" * 60)
    print(f"  Phases to run: {phase_list}")
    print(f"  Vocoder version: {vocoder_version.upper()}")
    print(f"  Detached mode: {detach}")
    print("=" * 60)
    
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Build commands for each phase
    commands = []
    
    if 1 in phase_list:
        phase1_script = os.path.join(script_dir, "phase1_acoustic.py")
        if skip_segmentation:
            cmd = f"python3 -m modal run {'--detach ' if detach else ''}{phase1_script}::main_skip_segmentation"
        else:
            cmd = f"python3 -m modal run {'--detach ' if detach else ''}{phase1_script}::main"
        commands.append(("Phase 1: Acoustic Tokenization", cmd))
    
    if 2 in phase_list:
        phase2_script = os.path.join(script_dir, "phase2_bpe.py")
        cmd = f"python3 -m modal run {'--detach ' if detach else ''}{phase2_script}::main"
        commands.append(("Phase 2: BPE Training", cmd))
    
    if 3 in phase_list:
        if vocoder_version.lower() == "v2":
            phase3_script = os.path.join(script_dir, "phase3_vocoder_v2.py")
            cmd = (
                f"python3 -m modal run {'--detach ' if detach else ''}{phase3_script}::main "
                f"--epochs {epochs} --batch-size {batch_size} "
                f"--segment-length {segment_length} --patience {patience}"
            )
        else:
            phase3_script = os.path.join(script_dir, "phase3_vocoder.py")
            cmd = (
                f"python3 -m modal run {'--detach ' if detach else ''}{phase3_script}::main "
                f"--epochs {epochs} --batch-size {batch_size}"
            )
        commands.append((f"Phase 3: Vocoder Training ({vocoder_version.upper()})", cmd))
    
    # Execute commands
    for phase_name, cmd in commands:
        print(f"\n{'=' * 60}")
        print(f"🔄 Starting: {phase_name}")
        print(f"{'=' * 60}")
        print(f"Command: {cmd}\n")
        
        if detach:
            # In detached mode, just print the command and let user run it
            # because we can't chain detached processes
            print(f"⚠️  In detached mode, run each phase separately:")
            print(f"    {cmd}")
        else:
            # In attached mode, run sequentially
            result = subprocess.run(cmd, shell=True)
            if result.returncode != 0:
                print(f"❌ {phase_name} failed with code {result.returncode}")
                print("Stopping pipeline.")
                return
            print(f"✅ {phase_name} complete!")
    
    if detach:
        print("\n" + "=" * 60)
        print("📋 Commands to Run (execute in order)")
        print("=" * 60)
        for phase_name, cmd in commands:
            print(f"\n# {phase_name}")
            print(cmd)
        print("\n" + "=" * 60)
        print("💡 TIP: Wait for each phase to complete before starting the next.")
        print("   Check Modal dashboard: https://modal.com/apps")
        print("=" * 60)
    else:
        print("\n✅ All phases complete!")


# %% [markdown]
# ## Quick Commands Reference
#
# ### Run Everything (V2)
# ```bash
# python3 -m modal run src/training/run_full_pipeline.py::main
# ```
#
# ### Run Everything (V1 - Original)
# ```bash
# python3 -m modal run src/training/run_full_pipeline.py::main --vocoder-version v1
# ```
#
# ### Run Only Vocoder V2
# ```bash
# python3 -m modal run src/training/run_full_pipeline.py::main --phases 3
# ```
#
# ### Run Phases 2 and 3 (skip Phase 1)
# ```bash
# python3 -m modal run src/training/run_full_pipeline.py::main --phases 2,3
# ```
