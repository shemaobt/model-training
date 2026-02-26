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
# # Phase 4: Pair Translation (Portuguese -> English)
#
# This phase trains a single-direction motif translator from paired Bible audio.
# It follows RFC 010 decisions:
# - paired audio supervision
# - motif-level translation with BPE over acoustemes
# - single-direction training first
#
# ## Workflow
#
# 1. Fetch paired Bible audio alignments from CMU Wilderness for:
#    - `PORARA` (Portuguese)
#    - `EN1NIV` (English)
# 2. Reuse Phase 1 acoustic units from both languages.
# 3. Train source/target BPE motif tokenizers on paired unit sequences.
# 4. Train a Seq2Seq Transformer (Portuguese -> English motifs).
#
# ## Run on Modal
#
# ```bash
# python3 -m modal run --detach src/training/phase4_pair_translation.py::main
# ```

# %%
import json
import math
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import modal


# %%
app = modal.App("bible-audio-pair-translation")

audio_volume = modal.Volume.from_name("bible-audio-data", create_if_missing=True)

image = (
    modal.Image.debian_slim()
    .apt_install(
        "git",
        "build-essential",
        "libncurses5-dev",
        "sox",
        "wget",
        "csh",
        "ffmpeg",
        "html2text",
    )
    .pip_install(
        "torch>=2.0.0",
        "sentencepiece>=0.1.99",
        "numpy<2",
        "tqdm>=4.66.0",
        "soundfile>=0.12.0",
        "datasets==2.19.2",
    )
)


# %%
AUDIO_MOUNT = "/mnt/audio_data"
PAIR_ROOT = f"{AUDIO_MOUNT}/parallel_pt_en"
PAIR_MANIFEST = f"{PAIR_ROOT}/pt_en_manifest.json"

LANGUAGE_DIRS = {
    "portuguese": {
        "segmented_dir": f"{AUDIO_MOUNT}/segmented_audio",
        "units_dir": f"{AUDIO_MOUNT}/portuguese_units",
    },
    "english": {
        "segmented_dir": f"{AUDIO_MOUNT}/segmented_audio_english",
        "units_dir": f"{AUDIO_MOUNT}/english_units",
    },
    "portuguese_fleurs": {
        "segmented_dir": f"{PAIR_ROOT}/segmented_audio_portuguese_fleurs",
        "units_dir": f"{AUDIO_MOUNT}/portuguese_fleurs_units",
    },
    "english_fleurs": {
        "segmented_dir": f"{PAIR_ROOT}/segmented_audio_english_fleurs",
        "units_dir": f"{AUDIO_MOUNT}/english_fleurs_units",
    },
}


# %%
def run_command(command: str, cwd: str | None = None):
    subprocess.run(command, shell=True, check=True, cwd=cwd)


def list_wav_files(directory: str, max_files: int) -> List[str]:
    paths = sorted(str(p) for p in Path(directory).glob("*.wav"))
    if max_files > 0:
        return paths[:max_files]
    return paths


def parse_code_list(values: str) -> List[str]:
    return [value.strip().upper() for value in values.split(",") if value.strip()]


def aligned_id_from_filename(path: str) -> str:
    return Path(path).stem


def canonical_pair_key(path: str, language_code: str) -> str:
    stem = Path(path).stem.upper()
    code = language_code.upper()

    if stem.startswith(f"{code}_"):
        stem = stem[len(code) + 1 :]

    book_chapter = re.search(r"(B\d+___\d+)", stem)
    segment = re.search(r"_(\d{5})$", stem)
    if book_chapter and segment:
        return f"{book_chapter.group(1)}_{segment.group(1)}"
    if book_chapter:
        return book_chapter.group(1)

    simplified = re.sub(rf"{re.escape(code)}[A-Z0-9]*", "", stem)
    simplified = re.sub(r"_+", "_", simplified).strip("_")
    return simplified or aligned_id_from_filename(path)


def prefixed_copy_name(language_code: str, path: str) -> str:
    return f"{language_code}_{Path(path).name}"


def wav_duration_seconds(path: str) -> float:
    import soundfile as sf

    return float(sf.info(path).duration)


def write_lines(path: str, values: List[str]):
    with open(path, "w") as file:
        for value in values:
            file.write(f"{value}\n")


def infer_spm_vocab_size(corpus_path: str, requested_size: int) -> int:
    token_set = set()
    with open(corpus_path, "r") as file:
        for line in file:
            token_set.update(piece for piece in line.strip().split() if piece)
    return max(32, min(requested_size, len(token_set) + 3))


def build_manifest_from_candidates(
    source_by_id: dict,
    target_by_id: dict,
    min_duration_sec: float,
    max_duration_sec: float,
    min_duration_ratio: float,
    max_duration_ratio: float,
    max_pairs: int,
) -> List[dict]:
    shared_ids = sorted(set(source_by_id.keys()).intersection(target_by_id.keys()))
    manifest = []

    for shared_id in shared_ids:
        best_pair = None
        best_score = None

        for src_item in source_by_id[shared_id]:
            for tgt_item in target_by_id[shared_id]:
                src_duration = src_item["duration_sec"]
                tgt_duration = tgt_item["duration_sec"]
                if src_duration <= 0 or tgt_duration <= 0:
                    continue
                if src_duration < min_duration_sec or tgt_duration < min_duration_sec:
                    continue
                if src_duration > max_duration_sec or tgt_duration > max_duration_sec:
                    continue

                ratio = src_duration / tgt_duration
                if ratio < min_duration_ratio or ratio > max_duration_ratio:
                    continue

                score = abs(math.log(ratio))
                if best_score is None or score < best_score:
                    best_score = score
                    best_pair = {
                        "id": shared_id,
                        "source_raw_id": src_item["raw_id"],
                        "target_raw_id": tgt_item["raw_id"],
                        "source_code": src_item["code"],
                        "target_code": tgt_item["code"],
                        "source_wav": src_item["wav"],
                        "target_wav": tgt_item["wav"],
                        "source_stem": src_item["stem"],
                        "target_stem": tgt_item["stem"],
                        "source_duration_sec": src_duration,
                        "target_duration_sec": tgt_duration,
                        "duration_ratio": ratio,
                    }

        if best_pair is not None:
            manifest.append(best_pair)

    if max_pairs > 0:
        manifest = manifest[:max_pairs]
    return manifest


def build_fleurs_candidates(
    source_segmented: str,
    target_segmented: str,
    fleurs_split: str,
    fleurs_max_records: int,
) -> Tuple[dict, dict]:
    import io
    import soundfile as sf
    from datasets import Audio, load_dataset

    pt_dataset = load_dataset("google/fleurs", "pt_br", split=fleurs_split)
    en_dataset = load_dataset("google/fleurs", "en_us", split=fleurs_split)
    pt_dataset = pt_dataset.cast_column("audio", Audio(decode=False))
    en_dataset = en_dataset.cast_column("audio", Audio(decode=False))

    en_by_id = {str(item["id"]): item for item in en_dataset}
    source_by_id = {}
    target_by_id = {}
    count = 0

    def read_audio_blob(audio_blob: dict) -> Tuple[object, int]:
        audio_bytes = audio_blob.get("bytes")
        if audio_bytes:
            return sf.read(io.BytesIO(audio_bytes))

        audio_path = audio_blob.get("path")
        if audio_path and os.path.exists(audio_path):
            return sf.read(audio_path)
        if audio_path:
            maybe_relative = os.path.join(os.getcwd(), audio_path)
            if os.path.exists(maybe_relative):
                return sf.read(maybe_relative)
        raise FileNotFoundError(f"Audio blob path not found: {audio_path}")

    for pt_item in pt_dataset:
        if fleurs_max_records > 0 and count >= fleurs_max_records:
            break

        pair_id = str(pt_item["id"])
        en_item = en_by_id.get(pair_id)
        if en_item is None:
            continue

        pt_wav = os.path.join(source_segmented, f"FLEURS_PTBR_{pair_id}.wav")
        en_wav = os.path.join(target_segmented, f"FLEURS_ENUS_{pair_id}.wav")

        if not os.path.exists(pt_wav):
            pt_array, pt_sr = read_audio_blob(pt_item["audio"])
            sf.write(pt_wav, pt_array, pt_sr)
        if not os.path.exists(en_wav):
            en_array, en_sr = read_audio_blob(en_item["audio"])
            sf.write(en_wav, en_array, en_sr)

        pair_key = f"FLEURS_{pair_id}"
        source_by_id[pair_key] = [
            {
                "code": "FLEURS_PT_BR",
                "wav": pt_wav,
                "stem": Path(pt_wav).stem,
                "duration_sec": wav_duration_seconds(pt_wav),
                "raw_id": pair_id,
            }
        ]
        target_by_id[pair_key] = [
            {
                "code": "FLEURS_EN_US",
                "wav": en_wav,
                "stem": Path(en_wav).stem,
                "duration_sec": wav_duration_seconds(en_wav),
                "raw_id": pair_id,
            }
        ]
        count += 1

    return source_by_id, target_by_id


# %%
@app.function(
    image=image,
    volumes={AUDIO_MOUNT: audio_volume},
    timeout=60 * 60 * 6,
    cpu=8,
)
def fetch_parallel_bible_pt_en(
    source_codes: str = "PORARA",
    target_codes: str = "EN1NIV",
    data_source: str = "auto",
    max_files_per_language: int = 4000,
    max_pairs: int = 0,
    min_duration_sec: float = 1.0,
    max_duration_sec: float = 40.0,
    min_duration_ratio: float = 0.6,
    max_duration_ratio: float = 1.8,
    fleurs_split: str = "train",
    fleurs_max_records: int = 3000,
):
    source_code_list = parse_code_list(source_codes)
    target_code_list = parse_code_list(target_codes)

    os.makedirs(PAIR_ROOT, exist_ok=True)
    wilderness_dir = "/tmp/datasets-CMU_Wilderness"

    if not os.path.exists(wilderness_dir):
        run_command(
            "git clone --depth 1 https://github.com/festvox/datasets-CMU_Wilderness",
            cwd="/tmp",
        )

    available_aligned_counts = {}

    if data_source in ("auto", "cmu"):
        all_codes = sorted(set(source_code_list + target_code_list))
        dependencies_ready = False
        for lang_code in all_codes:
            lang_dir = os.path.join(wilderness_dir, lang_code)
            aligned_dir = os.path.join(lang_dir, "aligned", "wav")
            preexisting = len(list_wav_files(aligned_dir, 0))
            if preexisting > 0:
                available_aligned_counts[lang_code] = preexisting
                continue

            first_attempt = subprocess.run(
                f"./bin/do_found fast_make_align indices/{lang_code}.tar.gz",
                shell=True,
                cwd=wilderness_dir,
            )

            if first_attempt.returncode != 0 and not dependencies_ready:
                run_command("./bin/do_found make_dependencies", cwd=wilderness_dir)
                dependencies_ready = True
                subprocess.run(
                    f"./bin/do_found fast_make_align indices/{lang_code}.tar.gz",
                    shell=True,
                    cwd=wilderness_dir,
                )

            available_aligned_counts[lang_code] = len(list_wav_files(aligned_dir, 0))

        source_code_list = [
            code for code in source_code_list if available_aligned_counts.get(code, 0) > 0
        ]
        target_code_list = [
            code for code in target_code_list if available_aligned_counts.get(code, 0) > 0
        ]

    source_language_key = "portuguese_fleurs" if data_source == "fleurs" else "portuguese"
    target_language_key = "english_fleurs" if data_source == "fleurs" else "english"
    source_segmented = LANGUAGE_DIRS[source_language_key]["segmented_dir"]
    target_segmented = LANGUAGE_DIRS[target_language_key]["segmented_dir"]
    os.makedirs(source_segmented, exist_ok=True)
    os.makedirs(target_segmented, exist_ok=True)

    source_by_id = {}
    target_by_id = {}
    if data_source in ("auto", "cmu") and source_code_list and target_code_list:
        for lang_code in source_code_list:
            source_aligned = os.path.join(wilderness_dir, lang_code, "aligned", "wav")
            for src_wav in list_wav_files(source_aligned, max_files_per_language):
                copy_name = prefixed_copy_name(lang_code, src_wav)
                dst_wav = os.path.join(source_segmented, copy_name)
                if not os.path.exists(dst_wav):
                    shutil.copy2(src_wav, dst_wav)

                pair_key = canonical_pair_key(src_wav, lang_code)
                source_by_id.setdefault(pair_key, []).append(
                    {
                        "code": lang_code,
                        "wav": dst_wav,
                        "stem": Path(dst_wav).stem,
                        "duration_sec": wav_duration_seconds(dst_wav),
                        "raw_id": aligned_id_from_filename(src_wav),
                    }
                )

        for lang_code in target_code_list:
            target_aligned = os.path.join(wilderness_dir, lang_code, "aligned", "wav")
            for tgt_wav in list_wav_files(target_aligned, max_files_per_language):
                copy_name = prefixed_copy_name(lang_code, tgt_wav)
                dst_wav = os.path.join(target_segmented, copy_name)
                if not os.path.exists(dst_wav):
                    shutil.copy2(tgt_wav, dst_wav)

                pair_key = canonical_pair_key(tgt_wav, lang_code)
                target_by_id.setdefault(pair_key, []).append(
                    {
                        "code": lang_code,
                        "wav": dst_wav,
                        "stem": Path(dst_wav).stem,
                        "duration_sec": wav_duration_seconds(dst_wav),
                        "raw_id": aligned_id_from_filename(tgt_wav),
                    }
                )

    manifest = build_manifest_from_candidates(
        source_by_id=source_by_id,
        target_by_id=target_by_id,
        min_duration_sec=min_duration_sec,
        max_duration_sec=max_duration_sec,
        min_duration_ratio=min_duration_ratio,
        max_duration_ratio=max_duration_ratio,
        max_pairs=max_pairs,
    )

    if (data_source in ("auto", "fleurs")) and len(manifest) == 0:
        source_by_id, target_by_id = build_fleurs_candidates(
            source_segmented=source_segmented,
            target_segmented=target_segmented,
            fleurs_split=fleurs_split,
            fleurs_max_records=fleurs_max_records,
        )
        manifest = build_manifest_from_candidates(
            source_by_id=source_by_id,
            target_by_id=target_by_id,
            min_duration_sec=min_duration_sec,
            max_duration_sec=max_duration_sec,
            min_duration_ratio=min_duration_ratio,
            max_duration_ratio=max_duration_ratio,
            max_pairs=max_pairs,
        )
        source_code_list = ["FLEURS_PT_BR"]
        target_code_list = ["FLEURS_EN_US"]
        available_aligned_counts["FLEURS_PT_BR"] = len(source_by_id)
        available_aligned_counts["FLEURS_EN_US"] = len(target_by_id)

    with open(PAIR_MANIFEST, "w") as file:
        json.dump(manifest, file, indent=2)

    audio_volume.commit()
    return {
        "manifest": PAIR_MANIFEST,
        "pairs": len(manifest),
        "source_codes": source_code_list,
        "target_codes": target_code_list,
        "source_language": source_language_key,
        "target_language": target_language_key,
        "aligned_wav_counts": available_aligned_counts,
        "source_segmented_dir": source_segmented,
        "target_segmented_dir": target_segmented,
    }


# %%
@dataclass
class PairExample:
    source_ids: List[int]
    target_ids: List[int]


# %%
@app.function(
    image=image,
    volumes={AUDIO_MOUNT: audio_volume},
    timeout=60 * 60 * 6,
    gpu="A10G",
)
def train_pair_translator(
    model_key: str = "xeus",
    source_language: str = "portuguese",
    target_language: str = "english",
    src_vocab_size: int = 2000,
    tgt_vocab_size: int = 2000,
    epochs: int = 20,
    batch_size: int = 16,
    learning_rate: float = 3e-4,
):
    import sentencepiece as spm
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset

    if not os.path.exists(PAIR_MANIFEST):
        raise FileNotFoundError(
            f"Pair manifest not found at {PAIR_MANIFEST}. Run fetch_parallel_bible_pt_en first."
        )

    source_units_dir = LANGUAGE_DIRS[source_language]["units_dir"]
    target_units_dir = LANGUAGE_DIRS[target_language]["units_dir"]
    os.makedirs(PAIR_ROOT, exist_ok=True)

    with open(PAIR_MANIFEST, "r") as file:
        manifest = json.load(file)

    paired_source_units = []
    paired_target_units = []

    for item in manifest:
        source_units_path = os.path.join(
            source_units_dir, f"{item['source_stem']}.units_{model_key}.txt"
        )
        target_units_path = os.path.join(
            target_units_dir, f"{item['target_stem']}.units_{model_key}.txt"
        )

        if not os.path.exists(source_units_path) or not os.path.exists(target_units_path):
            continue

        with open(source_units_path, "r") as src_file:
            source_units = src_file.read().strip()
        with open(target_units_path, "r") as tgt_file:
            target_units = tgt_file.read().strip()

        if source_units and target_units:
            paired_source_units.append(source_units)
            paired_target_units.append(target_units)

    if len(paired_source_units) == 0:
        raise RuntimeError(
            "No paired unit files found. Run Phase 1 for portuguese and english first."
        )

    src_corpus_path = os.path.join(PAIR_ROOT, f"src_units_{model_key}.txt")
    tgt_corpus_path = os.path.join(PAIR_ROOT, f"tgt_units_{model_key}.txt")
    write_lines(src_corpus_path, paired_source_units)
    write_lines(tgt_corpus_path, paired_target_units)

    src_bpe_prefix = os.path.join(PAIR_ROOT, f"src_bpe_{model_key}")
    tgt_bpe_prefix = os.path.join(PAIR_ROOT, f"tgt_bpe_{model_key}")
    src_effective_vocab_size = infer_spm_vocab_size(src_corpus_path, src_vocab_size)
    tgt_effective_vocab_size = infer_spm_vocab_size(tgt_corpus_path, tgt_vocab_size)

    spm.SentencePieceTrainer.train(
        input=src_corpus_path,
        model_prefix=src_bpe_prefix,
        vocab_size=src_effective_vocab_size,
        model_type="bpe",
        character_coverage=1.0,
        input_sentence_size=len(paired_source_units),
        shuffle_input_sentence=True,
    )
    spm.SentencePieceTrainer.train(
        input=tgt_corpus_path,
        model_prefix=tgt_bpe_prefix,
        vocab_size=tgt_effective_vocab_size,
        model_type="bpe",
        character_coverage=1.0,
        input_sentence_size=len(paired_target_units),
        shuffle_input_sentence=True,
    )

    src_sp = spm.SentencePieceProcessor(model_file=f"{src_bpe_prefix}.model")
    tgt_sp = spm.SentencePieceProcessor(model_file=f"{tgt_bpe_prefix}.model")

    bos_src, eos_src, pad_src = src_sp.bos_id(), src_sp.eos_id(), src_sp.pad_id()
    bos_tgt, eos_tgt, pad_tgt = tgt_sp.bos_id(), tgt_sp.eos_id(), tgt_sp.pad_id()
    if pad_src < 0:
        pad_src = src_sp.unk_id()
    if pad_tgt < 0:
        pad_tgt = tgt_sp.unk_id()

    def encode_pairs() -> List[PairExample]:
        examples = []
        for src_text, tgt_text in zip(paired_source_units, paired_target_units):
            src_ids = [bos_src] + src_sp.encode(src_text, out_type=int) + [eos_src]
            tgt_ids = [bos_tgt] + tgt_sp.encode(tgt_text, out_type=int) + [eos_tgt]
            examples.append(PairExample(source_ids=src_ids, target_ids=tgt_ids))
        return examples

    examples = encode_pairs()

    class PairDataset(Dataset):
        def __init__(self, values: List[PairExample]):
            self.values = values

        def __len__(self):
            return len(self.values)

        def __getitem__(self, idx: int):
            value = self.values[idx]
            return (
                torch.tensor(value.source_ids, dtype=torch.long),
                torch.tensor(value.target_ids, dtype=torch.long),
            )

    def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]]):
        src_batch = [item[0] for item in batch]
        tgt_batch = [item[1] for item in batch]
        src_padded = nn.utils.rnn.pad_sequence(src_batch, batch_first=True, padding_value=pad_src)
        tgt_padded = nn.utils.rnn.pad_sequence(tgt_batch, batch_first=True, padding_value=pad_tgt)
        return src_padded, tgt_padded

    class PositionalEncoding(nn.Module):
        def __init__(self, d_model: int, max_len: int = 5000):
            super().__init__()
            position = torch.arange(max_len).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
            pe = torch.zeros(max_len, d_model)
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            self.register_buffer("pe", pe.unsqueeze(0))

        def forward(self, x):
            return x + self.pe[:, : x.size(1), :]

    class Seq2SeqTransformer(nn.Module):
        def __init__(
            self,
            src_vocab: int,
            tgt_vocab: int,
            d_model: int = 512,
            nhead: int = 8,
            layers: int = 4,
            dim_ff: int = 1024,
            dropout: float = 0.1,
        ):
            super().__init__()
            self.src_embed = nn.Embedding(src_vocab, d_model, padding_idx=pad_src)
            self.tgt_embed = nn.Embedding(tgt_vocab, d_model, padding_idx=pad_tgt)
            self.positional = PositionalEncoding(d_model=d_model)
            self.transformer = nn.Transformer(
                d_model=d_model,
                nhead=nhead,
                num_encoder_layers=layers,
                num_decoder_layers=layers,
                dim_feedforward=dim_ff,
                dropout=dropout,
                batch_first=True,
            )
            self.out = nn.Linear(d_model, tgt_vocab)

        def forward(self, src_ids, tgt_ids, src_pad_mask, tgt_pad_mask, tgt_mask):
            src = self.positional(self.src_embed(src_ids))
            tgt = self.positional(self.tgt_embed(tgt_ids))
            hidden = self.transformer(
                src,
                tgt,
                tgt_mask=tgt_mask,
                src_key_padding_mask=src_pad_mask,
                tgt_key_padding_mask=tgt_pad_mask,
                memory_key_padding_mask=src_pad_mask,
            )
            return self.out(hidden)

    def causal_mask(size: int, device: torch.device):
        return torch.triu(torch.full((size, size), float("-inf"), device=device), diagonal=1)

    dataset = PairDataset(examples)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Seq2SeqTransformer(
        src_vocab=src_sp.get_piece_size(),
        tgt_vocab=tgt_sp.get_piece_size(),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_tgt)

    model.train()
    training_history = []
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        for src_ids, tgt_ids in loader:
            src_ids = src_ids.to(device)
            tgt_ids = tgt_ids.to(device)

            tgt_input = tgt_ids[:, :-1]
            tgt_output = tgt_ids[:, 1:]

            src_pad_mask = src_ids.eq(pad_src)
            tgt_pad_mask = tgt_input.eq(pad_tgt)
            tgt_mask = causal_mask(tgt_input.size(1), device=device)

            logits = model(src_ids, tgt_input, src_pad_mask, tgt_pad_mask, tgt_mask)
            loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_output.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / max(len(loader), 1)
        training_history.append({"epoch": epoch, "loss": avg_loss})
        print(f"Epoch {epoch}/{epochs} loss={avg_loss:.4f}")

    checkpoint_path = os.path.join(PAIR_ROOT, f"pt_en_pair_translator_{model_key}.pt")
    metadata_path = os.path.join(PAIR_ROOT, f"pt_en_pair_translator_{model_key}.json")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "src_vocab_size": src_sp.get_piece_size(),
            "tgt_vocab_size": tgt_sp.get_piece_size(),
            "source_language": source_language,
            "target_language": target_language,
            "model_key": model_key,
        },
        checkpoint_path,
    )
    with open(metadata_path, "w") as file:
        json.dump(
            {
                "pairs_used": len(examples),
                "history": training_history,
                "src_bpe_model": f"{src_bpe_prefix}.model",
                "tgt_bpe_model": f"{tgt_bpe_prefix}.model",
                "checkpoint": checkpoint_path,
            },
            file,
            indent=2,
        )

    audio_volume.commit()
    return {
        "checkpoint": checkpoint_path,
        "metadata": metadata_path,
        "pairs_used": len(examples),
        "epochs": epochs,
    }


# %%
@app.local_entrypoint()
def main(
    fetch_data: bool = True,
    source_codes: str = "PORARA",
    target_codes: str = "EN1NIV",
    data_source: str = "auto",
    max_files_per_language: int = 4000,
    max_pairs: int = 0,
    fleurs_split: str = "train",
    fleurs_max_records: int = 3000,
    model: str = "xeus",
    epochs: int = 20,
    batch_size: int = 16,
):
    source_language = "portuguese_fleurs" if data_source == "fleurs" else "portuguese"
    target_language = "english_fleurs" if data_source == "fleurs" else "english"

    if fetch_data:
        fetch_result = fetch_parallel_bible_pt_en.remote(
            source_codes=source_codes,
            target_codes=target_codes,
            data_source=data_source,
            max_files_per_language=max_files_per_language,
            max_pairs=max_pairs,
            fleurs_split=fleurs_split,
            fleurs_max_records=fleurs_max_records,
        )
        print(f"Paired data prepared: {fetch_result}")

    print("Run Phase 1 before translation training:")
    print(
        "python3 -m modal run --detach src/training/phase1_acoustic.py::main_skip_segmentation "
        f"--language {source_language} --model {model}"
    )
    print(
        "python3 -m modal run --detach src/training/phase1_acoustic.py::main_skip_segmentation "
        f"--language {target_language} --model {model}"
    )

    result = train_pair_translator.remote(
        model_key=model,
        source_language=source_language,
        target_language=target_language,
        epochs=epochs,
        batch_size=batch_size,
    )
    print(f"Training finished: {result}")
