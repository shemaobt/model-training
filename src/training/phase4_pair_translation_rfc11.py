import json
import math
import os
from dataclasses import dataclass
from typing import List, Tuple

import modal


app = modal.App("bible-audio-pair-translation-rfc11")

audio_volume = modal.Volume.from_name("bible-audio-data", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git")
    .pip_install(
        "torch>=2.0.0",
        "sentencepiece>=0.1.99",
        "numpy<2",
        "tqdm>=4.66.0",
        "datasets==2.19.2",
        "transformers>=4.44.0",
        "soundfile>=0.12.0",
    )
)


AUDIO_MOUNT = "/mnt/audio_data"
PAIR_ROOT = f"{AUDIO_MOUNT}/parallel_pt_en"
PAIR_MANIFEST = f"{PAIR_ROOT}/pt_en_manifest.json"

LANGUAGE_DIRS = {
    "portuguese_fleurs": {
        "units_dir": f"{AUDIO_MOUNT}/portuguese_fleurs_units",
    },
    "english_fleurs": {
        "units_dir": f"{AUDIO_MOUNT}/english_fleurs_units",
    },
}


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


def mean_pool_hidden_states(hidden_states, attention_mask):
    expanded_mask = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
    masked = hidden_states * expanded_mask
    summed = masked.sum(dim=1)
    counts = expanded_mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def build_fleurs_text_lookup(split: str):
    from datasets import load_dataset

    ds = load_dataset("google/fleurs", "en_us", split=split)
    if "audio" in ds.column_names:
        ds = ds.remove_columns(["audio"])
    lookup = {}
    for row in ds:
        text = row.get("transcription") or row.get("raw_transcription")
        if not text:
            continue
        row_id = row.get("id")
        if row_id is not None:
            lookup[str(row_id)] = text
    return lookup


@dataclass
class PairExample:
    source_ids: List[int]
    target_ids: List[int]
    text_embedding: List[float]


@app.function(
    image=image,
    volumes={AUDIO_MOUNT: audio_volume},
    timeout=60 * 60 * 8,
    gpu="A10G",
)
def train_pair_translator_rfc11(
    model_key: str = "xeus",
    source_language: str = "portuguese_fleurs",
    target_language: str = "english_fleurs",
    fleurs_split: str = "train",
    src_vocab_size: int = 2000,
    tgt_vocab_size: int = 2000,
    epochs: int = 20,
    batch_size: int = 16,
    learning_rate: float = 3e-4,
    semantic_weight: float = 0.4,
    early_stopping_patience: int = 5,
    early_stopping_min_delta: float = 1e-4,
):
    import sentencepiece as spm
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoModel, AutoTokenizer

    if not os.path.exists(PAIR_MANIFEST):
        raise FileNotFoundError(
            f"Pair manifest not found at {PAIR_MANIFEST}. Run fetch step before training."
        )

    with open(PAIR_MANIFEST, "r") as file:
        manifest = json.load(file)

    text_lookup = build_fleurs_text_lookup(split=fleurs_split)

    source_units_dir = LANGUAGE_DIRS[source_language]["units_dir"]
    target_units_dir = LANGUAGE_DIRS[target_language]["units_dir"]
    os.makedirs(PAIR_ROOT, exist_ok=True)

    paired_source_units = []
    paired_target_units = []
    paired_target_texts = []

    for item in manifest:
        pair_id = str(item.get("target_raw_id") or item.get("source_raw_id") or "")
        target_text = text_lookup.get(pair_id)
        if not target_text:
            continue

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
        if not source_units or not target_units:
            continue

        paired_source_units.append(source_units)
        paired_target_units.append(target_units)
        paired_target_texts.append(target_text)

    if len(paired_source_units) == 0:
        raise RuntimeError("No paired samples with units and HRL text were found.")

    src_corpus_path = os.path.join(PAIR_ROOT, f"src_units_rfc11_{model_key}.txt")
    tgt_corpus_path = os.path.join(PAIR_ROOT, f"tgt_units_rfc11_{model_key}.txt")
    write_lines(src_corpus_path, paired_source_units)
    write_lines(tgt_corpus_path, paired_target_units)

    src_bpe_prefix = os.path.join(PAIR_ROOT, f"src_bpe_rfc11_{model_key}")
    tgt_bpe_prefix = os.path.join(PAIR_ROOT, f"tgt_bpe_rfc11_{model_key}")
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    text_model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    text_tokenizer = AutoTokenizer.from_pretrained(text_model_name)
    text_model = AutoModel.from_pretrained(text_model_name).to(device)
    text_model.eval()
    text_embedding_dim = text_model.config.hidden_size

    target_text_embeddings = []
    text_batch_size = 64
    with torch.no_grad():
        for start in range(0, len(paired_target_texts), text_batch_size):
            batch_texts = paired_target_texts[start : start + text_batch_size]
            tokens = text_tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt",
            ).to(device)
            outputs = text_model(**tokens)
            pooled = mean_pool_hidden_states(outputs.last_hidden_state, tokens["attention_mask"])
            target_text_embeddings.extend(pooled.detach().cpu().tolist())

    def build_examples() -> List[PairExample]:
        values = []
        for src_text, tgt_text, text_embedding in zip(
            paired_source_units,
            paired_target_units,
            target_text_embeddings,
        ):
            src_ids = [bos_src] + src_sp.encode(src_text, out_type=int) + [eos_src]
            tgt_ids = [bos_tgt] + tgt_sp.encode(tgt_text, out_type=int) + [eos_tgt]
            values.append(
                PairExample(
                    source_ids=src_ids,
                    target_ids=tgt_ids,
                    text_embedding=text_embedding,
                )
            )
        return values

    examples = build_examples()

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
                torch.tensor(value.text_embedding, dtype=torch.float32),
            )

    def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
        src_batch = [item[0] for item in batch]
        tgt_batch = [item[1] for item in batch]
        text_batch = torch.stack([item[2] for item in batch], dim=0)
        src_padded = nn.utils.rnn.pad_sequence(src_batch, batch_first=True, padding_value=pad_src)
        tgt_padded = nn.utils.rnn.pad_sequence(tgt_batch, batch_first=True, padding_value=pad_tgt)
        return src_padded, tgt_padded, text_batch

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

    class Seq2SeqTransformerRfc11(nn.Module):
        def __init__(
            self,
            src_vocab: int,
            tgt_vocab: int,
            text_dim: int,
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
            self.semantic_head = nn.Linear(d_model, text_dim)

        def forward(self, src_ids, tgt_ids, src_pad_mask, tgt_pad_mask, tgt_mask):
            src_emb = self.positional(self.src_embed(src_ids))
            tgt_emb = self.positional(self.tgt_embed(tgt_ids))

            memory = self.transformer.encoder(
                src_emb,
                src_key_padding_mask=src_pad_mask,
            )
            decoded = self.transformer.decoder(
                tgt_emb,
                memory,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_pad_mask,
                memory_key_padding_mask=src_pad_mask,
            )
            token_logits = self.out(decoded)

            src_valid = (~src_pad_mask).unsqueeze(-1).float()
            src_lengths = src_valid.sum(dim=1).clamp(min=1e-6)
            pooled_memory = (memory * src_valid).sum(dim=1) / src_lengths
            semantic_pred = self.semantic_head(pooled_memory)
            return token_logits, semantic_pred

    def causal_mask(size: int, device_: torch.device):
        return torch.triu(torch.full((size, size), float("-inf"), device=device_), diagonal=1)

    dataset = PairDataset(examples)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    model = Seq2SeqTransformerRfc11(
        src_vocab=src_sp.get_piece_size(),
        tgt_vocab=tgt_sp.get_piece_size(),
        text_dim=text_embedding_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    unit_criterion = nn.CrossEntropyLoss(ignore_index=pad_tgt)
    semantic_criterion = nn.MSELoss()

    model.train()
    training_history = []
    best_loss = float("inf")
    best_epoch = 0
    epochs_without_improve = 0
    best_checkpoint_path = os.path.join(PAIR_ROOT, f"pt_en_pair_translator_rfc11_{model_key}_best.pt")
    latest_checkpoint_path = os.path.join(PAIR_ROOT, f"pt_en_pair_translator_rfc11_{model_key}_latest.pt")
    stopped_early = False

    for epoch in range(1, epochs + 1):
        epoch_total = 0.0
        epoch_unit = 0.0
        epoch_semantic = 0.0
        for src_ids, tgt_ids, text_targets in loader:
            src_ids = src_ids.to(device)
            tgt_ids = tgt_ids.to(device)
            text_targets = text_targets.to(device)

            tgt_input = tgt_ids[:, :-1]
            tgt_output = tgt_ids[:, 1:]

            src_pad_mask = src_ids.eq(pad_src)
            tgt_pad_mask = tgt_input.eq(pad_tgt)
            tgt_mask = causal_mask(tgt_input.size(1), device_=device)

            token_logits, semantic_pred = model(
                src_ids,
                tgt_input,
                src_pad_mask,
                tgt_pad_mask,
                tgt_mask,
            )
            unit_loss = unit_criterion(
                token_logits.reshape(-1, token_logits.size(-1)),
                tgt_output.reshape(-1),
            )
            semantic_loss = semantic_criterion(
                F.normalize(semantic_pred, p=2, dim=-1),
                F.normalize(text_targets, p=2, dim=-1),
            )
            loss = unit_loss + semantic_weight * semantic_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_total += loss.item()
            epoch_unit += unit_loss.item()
            epoch_semantic += semantic_loss.item()

        batches = max(len(loader), 1)
        avg_total = epoch_total / batches
        avg_unit = epoch_unit / batches
        avg_semantic = epoch_semantic / batches
        training_history.append(
            {
                "epoch": epoch,
                "loss": avg_total,
                "unit_loss": avg_unit,
                "semantic_loss": avg_semantic,
            }
        )
        print(
            f"Epoch {epoch}/{epochs} total={avg_total:.4f} "
            f"unit={avg_unit:.4f} semantic={avg_semantic:.4f}"
        )

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "src_vocab_size": src_sp.get_piece_size(),
                "tgt_vocab_size": tgt_sp.get_piece_size(),
                "text_embedding_dim": text_embedding_dim,
                "source_language": source_language,
                "target_language": target_language,
                "model_key": model_key,
                "semantic_weight": semantic_weight,
                "epoch": epoch,
            },
            latest_checkpoint_path,
        )

        improved = avg_total < (best_loss - early_stopping_min_delta)
        if improved:
            best_loss = avg_total
            best_epoch = epoch
            epochs_without_improve = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "src_vocab_size": src_sp.get_piece_size(),
                    "tgt_vocab_size": tgt_sp.get_piece_size(),
                    "text_embedding_dim": text_embedding_dim,
                    "source_language": source_language,
                    "target_language": target_language,
                    "model_key": model_key,
                    "semantic_weight": semantic_weight,
                    "epoch": epoch,
                    "best_loss": best_loss,
                },
                best_checkpoint_path,
            )
            print(f"  ✓ New best loss at epoch {epoch}: {best_loss:.4f}")
        else:
            epochs_without_improve += 1
            print(
                f"  Early stopping counter: {epochs_without_improve}/{early_stopping_patience} "
                f"(best={best_loss:.4f} @ epoch {best_epoch})"
            )
            if epochs_without_improve >= early_stopping_patience:
                stopped_early = True
                print(
                    f"  ⏹ Early stopping triggered at epoch {epoch}. "
                    f"Best epoch={best_epoch}, best loss={best_loss:.4f}"
                )
                break

    checkpoint_path = best_checkpoint_path
    metadata_path = os.path.join(PAIR_ROOT, f"pt_en_pair_translator_rfc11_{model_key}.json")
    with open(metadata_path, "w") as file:
        json.dump(
            {
                "pairs_used": len(examples),
                "epochs": epochs,
                "best_epoch": best_epoch,
                "best_loss": best_loss,
                "stopped_early": stopped_early,
                "early_stopping_patience": early_stopping_patience,
                "early_stopping_min_delta": early_stopping_min_delta,
                "history": training_history,
                "src_bpe_model": f"{src_bpe_prefix}.model",
                "tgt_bpe_model": f"{tgt_bpe_prefix}.model",
                "checkpoint": checkpoint_path,
                "latest_checkpoint": latest_checkpoint_path,
                "fleurs_split": fleurs_split,
                "rfc": "011",
            },
            file,
            indent=2,
        )

    audio_volume.commit()
    return {
        "checkpoint": checkpoint_path,
        "latest_checkpoint": latest_checkpoint_path,
        "metadata": metadata_path,
        "pairs_used": len(examples),
        "epochs_requested": epochs,
        "epochs_trained": len(training_history),
        "best_epoch": best_epoch,
        "best_loss": best_loss,
        "stopped_early": stopped_early,
    }


@app.local_entrypoint()
def main(
    model: str = "xeus",
    epochs: int = 20,
    batch_size: int = 16,
    learning_rate: float = 3e-4,
    semantic_weight: float = 0.4,
    fleurs_split: str = "train",
    early_stopping_patience: int = 5,
    early_stopping_min_delta: float = 1e-4,
):
    result = train_pair_translator_rfc11.remote(
        model_key=model,
        source_language="portuguese_fleurs",
        target_language="english_fleurs",
        fleurs_split=fleurs_split,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        semantic_weight=semantic_weight,
        early_stopping_patience=early_stopping_patience,
        early_stopping_min_delta=early_stopping_min_delta,
    )
    print(f"RFC11 training finished: {result}")
