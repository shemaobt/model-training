# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: "1.3"
#       jupytext_version: 1.20.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Pair Translation Pipeline Orchestrator

# %%
import os
import subprocess

import modal


# %%
app = modal.App("pair-translation-pipeline")


# %%
WILDERNESS_LANGUAGE_STATS = {
    "PORARA": {
        "language_name": "Portuguese",
        "duration_hms": "13:51:16",
        "duration_hours": 13.85,
        "pass1_utterances": 9475,
    },
    "EN1NIV": {
        "language_name": "English",
        "duration_hms": "13:41:58",
        "duration_hours": 13.70,
        "pass1_utterances": 10251,
    },
}


def run_command(command: str):
    subprocess.run(command, shell=True, check=True)


def parse_code_list(values: str):
    return [value.strip().upper() for value in values.split(",") if value.strip()]


def print_data_summary(source_codes: str, target_codes: str, max_files_per_language: int):
    source_code_list = parse_code_list(source_codes)
    target_code_list = parse_code_list(target_codes)

    print("=" * 70)
    print("DATA PLAN FOR TRAINING")
    print("=" * 70)
    print("Source corpus: CMU Wilderness aligned Bible audio")
    print(f"Source codes: {', '.join(source_code_list)}")
    print(f"Target codes: {', '.join(target_code_list)}")
    print(f"Max files per language: {max_files_per_language}")

    source_hours = []
    target_hours = []

    print()
    print("CMU Wilderness reference durations (Pass 1 alignments):")
    for code in source_code_list:
        stats = WILDERNESS_LANGUAGE_STATS.get(code)
        if not stats:
            print(f"- {code}: stats unavailable")
            continue
        source_hours.append(stats["duration_hours"])
        print(
            f"- {stats['language_name']} ({code}): "
            f"{stats['duration_hms']} ({stats['pass1_utterances']} utterances)"
        )

    for code in target_code_list:
        stats = WILDERNESS_LANGUAGE_STATS.get(code)
        if not stats:
            print(f"- {code}: stats unavailable")
            continue
        target_hours.append(stats["duration_hours"])
        print(
            f"- {stats['language_name']} ({code}): "
            f"{stats['duration_hms']} ({stats['pass1_utterances']} utterances)"
        )

    print()
    if source_hours and target_hours:
        source_cap = sum(source_hours)
        target_cap = sum(target_hours)
        theoretical_overlap_hours = min(source_cap, target_cap)
        print(
            "Theoretical upper bound before pair intersection/filtering: "
            f"~{theoretical_overlap_hours:.2f}h"
        )
        print(
            "Expected practical paired hours after overlap + quality filtering: "
            "~10-20h (depends on code overlap and duration-ratio constraints)"
        )
    else:
        print("Stats are partial/unavailable for at least one selected code.")
        print("Practical hours will be measured after manifest generation.")
    print("=" * 70)


# %%
@app.local_entrypoint()
def main(
    source_codes: str = "PORARA,PORARC",
    target_codes: str = "EN1NIV",
    data_source: str = "auto",
    fleurs_split: str = "train",
    fleurs_max_records: int = 3000,
    source_language: str = "portuguese",
    target_language: str = "english",
    model: str = "xeus",
    max_files_per_language: int = 4000,
    max_pairs: int = 0,
    run_phase2: bool = False,
    epochs: int = 20,
    batch_size: int = 16,
    detach: bool = True,
):
    if data_source == "fleurs":
        if source_language == "portuguese":
            source_language = "portuguese_fleurs"
        if target_language == "english":
            target_language = "english_fleurs"

    print_data_summary(
        source_codes=source_codes,
        target_codes=target_codes,
        max_files_per_language=max_files_per_language,
    )

    script_dir = os.path.dirname(os.path.abspath(__file__))
    phase1_script = os.path.join(script_dir, "phase1_acoustic.py")
    phase2_script = os.path.join(script_dir, "phase2_bpe.py")
    phase4_script = os.path.join(script_dir, "phase4_pair_translation.py")
    detach_flag = "--detach " if detach else ""

    commands = [
        (
            "Fetch PT/EN aligned Bible pairs",
            "python3 -m modal run "
            f"{detach_flag}{phase4_script}::fetch_parallel_bible_pt_en "
            f"--source-codes {source_codes} --target-codes {target_codes} "
            f"--data-source {data_source} "
            f"--max-files-per-language {max_files_per_language} --max-pairs {max_pairs} "
            f"--fleurs-split {fleurs_split} --fleurs-max-records {fleurs_max_records}",
        ),
        (
            "Phase 1 acoustic units (source)",
            "python3 -m modal run "
            f"{detach_flag}{phase1_script}::main_skip_segmentation "
            f"--language {source_language} --model {model}",
        ),
        (
            "Phase 1 acoustic units (target)",
            "python3 -m modal run "
            f"{detach_flag}{phase1_script}::main_skip_segmentation "
            f"--language {target_language} --model {model}",
        ),
    ]

    if run_phase2:
        commands.extend(
            [
                (
                    "Phase 2 BPE (source)",
                    "python3 -m modal run "
                    f"{detach_flag}{phase2_script}::main --language {source_language} --model {model}",
                ),
                (
                    "Phase 2 BPE (target)",
                    "python3 -m modal run "
                    f"{detach_flag}{phase2_script}::main --language {target_language} --model {model}",
                ),
            ]
        )

    commands.append(
        (
            "Phase 4 pair translator training",
            "python3 -m modal run "
            f"{detach_flag}{phase4_script}::main --fetch-data false "
            f"--source-codes {source_codes} --target-codes {target_codes} "
            f"--data-source {data_source} "
            f"--fleurs-split {fleurs_split} --fleurs-max-records {fleurs_max_records} "
            f"--max-pairs {max_pairs} "
            f"--model {model} --epochs {epochs} --batch-size {batch_size}",
        )
    )

    print()
    print("=" * 70)
    print("PAIR TRANSLATION PIPELINE COMMANDS")
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

    print("\nPipeline completed.")
