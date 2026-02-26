import os
import subprocess

import modal


app = modal.App("pair-translation-pipeline-rfc11")


def run_command(command: str):
    subprocess.run(command, shell=True, check=True)


@app.local_entrypoint()
def main(
    model: str = "xeus",
    epochs: int = 20,
    batch_size: int = 16,
    learning_rate: float = 3e-4,
    semantic_weight: float = 0.4,
    fleurs_split: str = "train",
    max_pairs: int = 0,
    detach: bool = True,
):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    phase1_script = os.path.join(script_dir, "phase1_acoustic.py")
    phase4_fetch_script = os.path.join(script_dir, "phase4_pair_translation.py")
    phase4_rfc11_script = os.path.join(script_dir, "phase4_pair_translation_rfc11.py")
    detach_flag = "--detach " if detach else ""

    commands = [
        (
            "Fetch PT/EN FLEURS paired data",
            "python3 -m modal run "
            f"{detach_flag}{phase4_fetch_script}::fetch_parallel_bible_pt_en "
            "--source-codes FLEURS_PT_BR --target-codes FLEURS_EN_US "
            "--data-source fleurs "
            f"--fleurs-split {fleurs_split} --fleurs-max-records 3000 --max-pairs {max_pairs}",
        ),
        (
            "Phase 1 acoustic units (portuguese_fleurs)",
            "python3 -m modal run "
            f"{detach_flag}{phase1_script}::main_skip_segmentation "
            f"--language portuguese_fleurs --model {model}",
        ),
        (
            "Phase 1 acoustic units (english_fleurs)",
            "python3 -m modal run "
            f"{detach_flag}{phase1_script}::main_skip_segmentation "
            f"--language english_fleurs --model {model}",
        ),
        (
            "RFC11 pair translator training",
            "python3 -m modal run "
            f"{detach_flag}{phase4_rfc11_script} "
            f"--model {model} --epochs {epochs} --batch-size {batch_size} "
            f"--learning-rate {learning_rate} --semantic-weight {semantic_weight} "
            f"--fleurs-split {fleurs_split}",
        ),
    ]

    print("\n" + "=" * 70)
    print("RFC11 PIPELINE COMMANDS")
    print("=" * 70)
    for name, command in commands:
        print(f"\n# {name}")
        print(command)
    print("\n" + "=" * 70)

    if detach:
        print("Detached mode enabled. Run commands in order and monitor Modal dashboard.")
        print("Modal dashboard: https://modal.com/apps")
        return

    for name, command in commands:
        print(f"\nRunning: {name}")
        run_command(command)

    print("\nRFC11 pipeline completed.")
