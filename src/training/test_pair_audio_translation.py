import json
import os
import tempfile
from pathlib import Path

import modal

from src.constants import ACOUSTIC_MODELS, SAMPLE_RATE


app = modal.App("pair-audio-single-inference")

audio_volume = modal.Volume.from_name("bible-audio-data", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "ffmpeg", "libsndfile1")
    .pip_install(
        "torch>=2.0.0",
        "torchaudio>=2.0.0",
        "numpy<2",
        "soundfile>=0.12.0",
        "joblib>=1.3.0",
        "sentencepiece==0.1.97",
        "scikit-learn>=1.3.0",
        "huggingface_hub>=0.23.0",
        "espnet @ git+https://github.com/wanchichen/espnet.git@ssl",
        "pyyaml>=6.0.0",
    )
    .add_local_python_source("src")
)

AUDIO_MOUNT = "/mnt/audio_data"
PAIR_ROOT = f"{AUDIO_MOUNT}/parallel_pt_en"
PT_KMEANS_CANDIDATES = [
    f"{AUDIO_MOUNT}/portuguese_fleurs_units/checkpoint_kmeans_xeus.pkl",
    f"{AUDIO_MOUNT}/portuguese_fleurs_units/portuguese_fleurs_kmeans_xeus.pkl",
    f"{AUDIO_MOUNT}/portuguese_units/checkpoint_kmeans_xeus.pkl",
    f"{AUDIO_MOUNT}/portuguese_units/portuguese_kmeans.pkl",
]
TRANSLATOR_CHECKPOINT_CANDIDATES = [
    f"{PAIR_ROOT}/pt_en_pair_translator_rfc11_xeus_best.pt",
    f"{PAIR_ROOT}/pt_en_pair_translator_rfc11_xeus_latest.pt",
    f"{PAIR_ROOT}/pt_en_pair_translator_xeus.pt",
]
SRC_BPE_MODEL_CANDIDATES = [
    f"{PAIR_ROOT}/src_bpe_rfc11_xeus.model",
    f"{PAIR_ROOT}/src_bpe_xeus.model",
]
TGT_BPE_MODEL_CANDIDATES = [
    f"{PAIR_ROOT}/tgt_bpe_rfc11_xeus.model",
    f"{PAIR_ROOT}/tgt_bpe_xeus.model",
]
OUTPUT_DIR = f"{PAIR_ROOT}/inference_outputs"


@app.function(
    image=image,
    volumes={AUDIO_MOUNT: audio_volume},
    timeout=60 * 60,
    gpu="A10G",
)
def translate_single_audio(
    audio_bytes: bytes,
    input_filename: str,
    model_key: str = "xeus",
):
    import argparse
    import joblib
    import numpy as np
    import sentencepiece as spm
    import soundfile as sf
    import subprocess
    import torch
    import torchaudio
    import yaml
    from espnet2.tasks.ssl import SSLTask
    from huggingface_hub import snapshot_download
    from torch.nn.utils.rnn import pad_sequence

    if model_key != "xeus":
        raise ValueError("This inference entrypoint currently supports only model_key='xeus'.")

    kmeans_path = next((path for path in PT_KMEANS_CANDIDATES if os.path.exists(path)), None)
    translator_checkpoint = next(
        (path for path in TRANSLATOR_CHECKPOINT_CANDIDATES if os.path.exists(path)),
        None,
    )
    src_bpe_model = next((path for path in SRC_BPE_MODEL_CANDIDATES if os.path.exists(path)), None)
    tgt_bpe_model = next((path for path in TGT_BPE_MODEL_CANDIDATES if os.path.exists(path)), None)

    if translator_checkpoint is None:
        raise FileNotFoundError("Required artifact not found: no translator checkpoint was found.")
    if src_bpe_model is None:
        raise FileNotFoundError("Required artifact not found: no source BPE model was found.")
    if tgt_bpe_model is None:
        raise FileNotFoundError("Required artifact not found: no target BPE model was found.")
    if kmeans_path is None:
        raise FileNotFoundError(
            "Required artifact not found: no Portuguese XEUS KMeans checkpoint was found."
        )

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    stem = Path(input_filename).stem
    tmp_dir = tempfile.mkdtemp(prefix="pair-audio-infer-")
    raw_path = os.path.join(tmp_dir, input_filename)
    wav_path = os.path.join(tmp_dir, f"{stem}.wav")

    with open(raw_path, "wb") as file:
        file.write(audio_bytes)

    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            raw_path,
            "-ac",
            "1",
            "-ar",
            str(SAMPLE_RATE),
            wav_path,
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = ACOUSTIC_MODELS[model_key]["model_name"]
    target_layer = ACOUSTIC_MODELS[model_key]["layer"]

    xeus_repo_path = snapshot_download(repo_id=model_name)
    config_path = os.path.join(xeus_repo_path, "model", "config.yaml")
    with open(config_path, "r") as file:
        xeus_config = yaml.safe_load(file)
    if isinstance(xeus_config.get("frontend_conf"), dict):
        xeus_config["frontend_conf"].pop("normalize_output", None)
    if isinstance(xeus_config.get("loss"), list) and xeus_config["loss"]:
        first_loss = xeus_config["loss"][0]
        if isinstance(first_loss, dict):
            xeus_config["loss"] = first_loss.get("name")
            xeus_config["loss_conf"] = first_loss.get("conf", {})
    xeus_config.setdefault("masker", "hubert")
    xeus_config.setdefault(
        "masker_conf",
        {
            "mask_prob": 0.8,
            "mask_selection": "static",
            "mask_other": 0.0,
            "mask_length": 10,
            "no_mask_overlap": False,
            "mask_min_space": 1,
            "mask_channel_prob": 0.0,
            "mask_channel_selection": "static",
            "mask_channel_other": 0.0,
            "mask_channel_length": 10,
            "no_mask_channel_overlap": False,
            "mask_channel_min_space": 1,
        },
    )

    candidate_checkpoints = [
        os.path.join(xeus_repo_path, "model", "xeus_checkpoint_old.pth"),
        os.path.join(xeus_repo_path, "model", "xeus_checkpoint_new.pth"),
    ]
    xeus_model = None
    for checkpoint_path in candidate_checkpoints:
        if not os.path.exists(checkpoint_path):
            continue
        args = argparse.Namespace(**xeus_config)
        model = SSLTask.build_model(args)
        checkpoint = torch.load(checkpoint_path, map_location=str(device))
        state_dict = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
        model.load_state_dict(state_dict, strict=False)
        xeus_model = model.to(device)
        break
    if xeus_model is None:
        raise RuntimeError("Unable to load XEUS checkpoint.")
    xeus_model.eval()

    waveform_data, rate = sf.read(wav_path, dtype="float32")
    if waveform_data.ndim > 1:
        waveform_data = waveform_data.mean(axis=1)
    if rate != SAMPLE_RATE:
        waveform_tensor = torch.from_numpy(waveform_data).unsqueeze(0).float()
        resampler = torchaudio.transforms.Resample(rate, SAMPLE_RATE)
        waveform_tensor = resampler(waveform_tensor)
        waveform_data = waveform_tensor.squeeze().numpy()

    wav_tensor = torch.tensor(waveform_data, dtype=torch.float32, device=device)
    wav_lengths = torch.LongTensor([wav_tensor.numel()]).to(device)
    wavs = pad_sequence([wav_tensor], batch_first=True)
    with torch.no_grad():
        encoded = xeus_model.encode(
            wavs,
            wav_lengths,
            use_mask=False,
            use_final_output=False,
        )
    hidden_states = encoded[0]
    layer_idx = target_layer if target_layer >= 0 else (len(hidden_states) + target_layer)
    layer_idx = min(max(layer_idx, 0), len(hidden_states) - 1)
    feats = hidden_states[layer_idx].squeeze(0).detach().cpu().numpy()

    kmeans = joblib.load(kmeans_path)
    source_units = kmeans.predict(feats).tolist()
    source_unit_text = " ".join(str(unit) for unit in source_units)

    src_sp = spm.SentencePieceProcessor(model_file=src_bpe_model)
    tgt_sp = spm.SentencePieceProcessor(model_file=tgt_bpe_model)
    bos_src = src_sp.bos_id()
    eos_src = src_sp.eos_id()
    pad_src = src_sp.pad_id()
    bos_tgt = tgt_sp.bos_id()
    eos_tgt = tgt_sp.eos_id()
    pad_tgt = tgt_sp.pad_id()
    if pad_src < 0:
        pad_src = src_sp.unk_id()
    if pad_tgt < 0:
        pad_tgt = tgt_sp.unk_id()

    src_ids = [bos_src] + src_sp.encode(source_unit_text, out_type=int) + [eos_src]

    class PositionalEncoding(torch.nn.Module):
        def __init__(self, d_model: int, max_len: int = 5000):
            super().__init__()
            position = torch.arange(max_len).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
            pe = torch.zeros(max_len, d_model)
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            self.register_buffer("pe", pe.unsqueeze(0))

        def forward(self, x):
            return x + self.pe[:, : x.size(1), :]

    class Seq2SeqTransformer(torch.nn.Module):
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
            self.src_embed = torch.nn.Embedding(src_vocab, d_model, padding_idx=pad_src)
            self.tgt_embed = torch.nn.Embedding(tgt_vocab, d_model, padding_idx=pad_tgt)
            self.positional = PositionalEncoding(d_model=d_model)
            self.transformer = torch.nn.Transformer(
                d_model=d_model,
                nhead=nhead,
                num_encoder_layers=layers,
                num_decoder_layers=layers,
                dim_feedforward=dim_ff,
                dropout=dropout,
                batch_first=True,
            )
            self.out = torch.nn.Linear(d_model, tgt_vocab)

        def forward(self, src_ids_tensor, tgt_ids_tensor, src_pad_mask, tgt_pad_mask, tgt_mask):
            src = self.positional(self.src_embed(src_ids_tensor))
            tgt = self.positional(self.tgt_embed(tgt_ids_tensor))
            hidden = self.transformer(
                src,
                tgt,
                tgt_mask=tgt_mask,
                src_key_padding_mask=src_pad_mask,
                tgt_key_padding_mask=tgt_pad_mask,
                memory_key_padding_mask=src_pad_mask,
            )
            return self.out(hidden)

    def causal_mask(size: int, device_: torch.device):
        return torch.triu(torch.full((size, size), float("-inf"), device=device_), diagonal=1)

    checkpoint = torch.load(translator_checkpoint, map_location=device)
    model = Seq2SeqTransformer(
        src_vocab=checkpoint["src_vocab_size"],
        tgt_vocab=checkpoint["tgt_vocab_size"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.eval()

    src_tensor = torch.tensor(src_ids, dtype=torch.long, device=device).unsqueeze(0)
    src_pad_mask = src_tensor.eq(pad_src)

    generated = [bos_tgt]
    max_decode_len = 1024
    with torch.no_grad():
        for _ in range(max_decode_len):
            tgt_tensor = torch.tensor(generated, dtype=torch.long, device=device).unsqueeze(0)
            tgt_pad_mask = tgt_tensor.eq(pad_tgt)
            tgt_mask = causal_mask(tgt_tensor.size(1), device)
            logits = model(src_tensor, tgt_tensor, src_pad_mask, tgt_pad_mask, tgt_mask)
            next_token = int(torch.argmax(logits[0, -1]).item())
            generated.append(next_token)
            if next_token == eos_tgt:
                break

    decoded_ids = [token for token in generated[1:] if token not in {eos_tgt, pad_tgt}]
    target_unit_text = tgt_sp.decode(decoded_ids)
    try:
        target_units = [int(piece) for piece in target_unit_text.split() if piece.strip()]
    except ValueError:
        target_units = []

    result = {
        "input_filename": input_filename,
        "model_key": model_key,
        "kmeans_checkpoint": kmeans_path,
        "translator_checkpoint": translator_checkpoint,
        "src_bpe_model": src_bpe_model,
        "tgt_bpe_model": tgt_bpe_model,
        "source_units_count": len(source_units),
        "target_units_count": len(target_units),
        "source_units_preview": source_units[:80],
        "target_units_preview": target_units[:80],
        "target_units_text": target_unit_text,
    }

    output_path = os.path.join(OUTPUT_DIR, f"{stem}_translation_{model_key}.json")
    with open(output_path, "w") as file:
        json.dump(result, file, indent=2)

    audio_volume.commit()
    return {"result": result, "output_path": output_path}


@app.local_entrypoint()
def main(input_audio_path: str, model: str = "xeus"):
    if not os.path.exists(input_audio_path):
        raise FileNotFoundError(f"Input audio not found: {input_audio_path}")

    with open(input_audio_path, "rb") as file:
        audio_bytes = file.read()

    response = translate_single_audio.remote(
        audio_bytes=audio_bytes,
        input_filename=os.path.basename(input_audio_path),
        model_key=model,
    )

    result = response["result"]
    print("Single-audio translation complete")
    print(f"Input: {result['input_filename']}")
    print(f"Model: {result['model_key']}")
    print(f"Source units: {result['source_units_count']}")
    print(f"Target units: {result['target_units_count']}")
    print(f"Saved JSON: {response['output_path']}")
