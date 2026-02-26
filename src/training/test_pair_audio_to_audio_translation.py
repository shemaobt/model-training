import json
import os
import tempfile
from pathlib import Path

import modal

from src.constants import (
    ACOUSTIC_MODELS,
    F0_MAX,
    F0_MIN,
    HOP_SIZE,
    NUM_ACOUSTIC_UNITS,
    NUM_PITCH_BINS,
    SAMPLE_RATE,
)


app = modal.App("pair-audio-to-audio-inference")

audio_volume = modal.Volume.from_name("bible-audio-data", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "ffmpeg", "libsndfile1")
    .pip_install(
        "torch>=2.0.0",
        "torchaudio>=2.0.0",
        "numpy<2",
        "soundfile>=0.12.0",
        "librosa==0.9.2",
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
VOCODER_DIR = f"{AUDIO_MOUNT}/vocoder_v2_english_fleurs_checkpoints"
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
VOCODER_CHECKPOINT_CANDIDATES = [
    os.path.join(VOCODER_DIR, "v2_best.pt"),
    os.path.join(VOCODER_DIR, "v2_latest.pt"),
]
OUTPUT_DIR = f"{PAIR_ROOT}/inference_outputs"


@app.function(
    image=image,
    volumes={AUDIO_MOUNT: audio_volume},
    timeout=60 * 60,
    gpu="A10G",
)
def translate_single_audio_to_audio(
    audio_bytes: bytes,
    input_filename: str,
    model_key: str = "xeus",
):
    import argparse
    import importlib.util
    import joblib
    import librosa
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
    vocoder_checkpoint = next((path for path in VOCODER_CHECKPOINT_CANDIDATES if os.path.exists(path)), None)

    if translator_checkpoint is None:
        raise FileNotFoundError("Required artifact not found: no translator checkpoint was found.")
    if src_bpe_model is None:
        raise FileNotFoundError("Required artifact not found: no source BPE model was found.")
    if tgt_bpe_model is None:
        raise FileNotFoundError("Required artifact not found: no target BPE model was found.")
    if kmeans_path is None:
        raise FileNotFoundError("Required artifact not found: no Portuguese XEUS KMeans checkpoint was found.")
    if vocoder_checkpoint is None:
        raise FileNotFoundError("Required artifact not found: no English vocoder checkpoint was found.")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    stem = Path(input_filename).stem
    tmp_dir = tempfile.mkdtemp(prefix="pair-audio-to-audio-infer-")
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
    translator_model = Seq2SeqTransformer(
        src_vocab=checkpoint["src_vocab_size"],
        tgt_vocab=checkpoint["tgt_vocab_size"],
    ).to(device)
    translator_model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    translator_model.eval()

    src_tensor = torch.tensor(src_ids, dtype=torch.long, device=device).unsqueeze(0)
    src_pad_mask = src_tensor.eq(pad_src)

    source_len = len(source_units)
    min_decode_len = max(24, int(source_len * 0.65))
    max_decode_len = min(320, max(72, int(source_len * 1.45)))
    temperature = 0.85
    top_k = 16
    repetition_penalty = 1.22
    max_repeat_run = 8
    num_decode_candidates = 12

    def has_valid_probabilities(values: torch.Tensor) -> bool:
        if values.numel() == 0:
            return False
        if torch.all(torch.isinf(values)):
            return False
        return True

    def decode_one_candidate(seed: int):
        torch.manual_seed(seed)
        generated = [bos_tgt]
        repeat_run = 0
        with torch.no_grad():
            for _ in range(max_decode_len):
                tgt_tensor = torch.tensor(generated, dtype=torch.long, device=device).unsqueeze(0)
                tgt_pad_mask = tgt_tensor.eq(pad_tgt)
                tgt_mask = causal_mask(tgt_tensor.size(1), device)
                logits = translator_model(src_tensor, tgt_tensor, src_pad_mask, tgt_pad_mask, tgt_mask)
                step_logits = logits[0, -1].clone()
                step_logits[pad_tgt] = float("-inf")
                if len(generated) <= min_decode_len:
                    step_logits[eos_tgt] = float("-inf")
                for token in set(generated[-96:]):
                    step_logits[token] = step_logits[token] / repetition_penalty
                if len(generated) >= 3:
                    prefix = (generated[-2], generated[-1])
                    blocked = set()
                    for idx in range(len(generated) - 2):
                        if (generated[idx], generated[idx + 1]) == prefix:
                            blocked.add(generated[idx + 2])
                    for token in blocked:
                        step_logits[token] = float("-inf")
                k = min(top_k, step_logits.size(0))
                top_values, top_indices = torch.topk(step_logits, k=k)
                if not has_valid_probabilities(top_values):
                    next_token = int(torch.argmax(logits[0, -1]).item())
                else:
                    top_probs = torch.softmax(top_values / temperature, dim=-1)
                    sampled_index = int(torch.multinomial(top_probs, num_samples=1).item())
                    next_token = int(top_indices[sampled_index].item())
                if len(generated) > 1 and next_token == generated[-1]:
                    repeat_run += 1
                else:
                    repeat_run = 0
                if repeat_run >= max_repeat_run:
                    for token in top_indices.tolist():
                        token_int = int(token)
                        if token_int != generated[-1]:
                            next_token = token_int
                            break
                    repeat_run = 0
                generated.append(next_token)
                if next_token == eos_tgt and len(generated) > min_decode_len:
                    break
        decoded_ids = [token for token in generated[1:] if token not in {eos_tgt, pad_tgt}]
        target_text = tgt_sp.decode(decoded_ids)
        try:
            units = [int(piece) for piece in target_text.split() if piece.strip()]
        except ValueError:
            units = []
        return units, target_text

    def score_candidate(units: list[int]) -> float:
        if not units:
            return float("-inf")
        length_ratio = len(units) / max(1, source_len)
        diversity = len(set(units)) / max(1, len(units))
        repeat_ratio = (
            sum(1 for i in range(1, len(units)) if units[i] == units[i - 1]) / max(1, len(units) - 1)
        )
        length_penalty = abs(1.0 - length_ratio)
        return (2.2 * diversity) - (2.0 * repeat_ratio) - (1.4 * length_penalty)

    best_units = []
    best_text = ""
    best_score = float("-inf")
    for seed_offset in range(num_decode_candidates):
        candidate_units, candidate_text = decode_one_candidate(1234 + seed_offset)
        candidate_score = score_candidate(candidate_units)
        if candidate_score > best_score:
            best_units = candidate_units
            best_text = candidate_text
            best_score = candidate_score

    target_units = best_units
    target_unit_text = best_text
    if target_units and (len(target_units) < int(source_len * 0.55) or len(target_units) > int(source_len * 1.75)):
        mapped_positions = np.linspace(0, len(target_units) - 1, num=source_len)
        target_units = [target_units[min(len(target_units) - 1, max(0, int(round(pos))))] for pos in mapped_positions]
        target_unit_text = " ".join(str(unit) for unit in target_units)

    if not target_units:
        raise RuntimeError("Translation produced no target units.")

    generator_code = """
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
    def __init__(self, num_units=100, num_pitch_bins=32, unit_embed_dim=256, pitch_embed_dim=64):
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
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as module_file:
        module_file.write(generator_code)
        module_path = module_file.name
    module_spec = importlib.util.spec_from_file_location("generator_v2_module", module_path)
    generator_module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(generator_module)
    GeneratorV2 = generator_module.GeneratorV2

    vocoder = GeneratorV2(num_units=NUM_ACOUSTIC_UNITS, num_pitch_bins=NUM_PITCH_BINS).to(device)
    vocoder_state = torch.load(vocoder_checkpoint, map_location=device)
    vocoder.load_state_dict(vocoder_state["generator_state_dict"])
    vocoder.eval()

    source_f0_raw, _, _ = librosa.pyin(
        waveform_data,
        fmin=F0_MIN,
        fmax=F0_MAX,
        sr=SAMPLE_RATE,
        hop_length=HOP_SIZE,
    )
    source_f0_raw = np.nan_to_num(source_f0_raw, nan=0.0)
    if len(source_f0_raw) < 2:
        source_f0 = np.zeros(len(target_units), dtype=np.float32)
    else:
        old_positions = np.linspace(0.0, 1.0, num=len(source_f0_raw))
        new_positions = np.linspace(0.0, 1.0, num=len(target_units))
        source_f0 = np.interp(new_positions, old_positions, source_f0_raw)
    pitch_bins = np.zeros_like(source_f0, dtype=np.int64)
    voiced_mask = source_f0 > 0
    if np.any(voiced_mask):
        log_f0 = np.log(np.clip(source_f0[voiced_mask], F0_MIN, F0_MAX))
        log_min = np.log(F0_MIN)
        log_max = np.log(F0_MAX)
        normalized = (log_f0 - log_min) / (log_max - log_min)
        pitch_bins[voiced_mask] = (normalized * (NUM_PITCH_BINS - 1) + 1).astype(np.int64)

    units_tensor = torch.LongTensor(target_units).unsqueeze(0).to(device)
    pitch_tensor = torch.LongTensor(pitch_bins).unsqueeze(0).to(device)
    with torch.no_grad():
        audio = vocoder(units_tensor, pitch_tensor).squeeze(0).detach().cpu().numpy()
    peak = float(np.max(np.abs(audio)))
    if peak > 0:
        audio = audio / peak * 0.95

    output_audio_path = os.path.join(OUTPUT_DIR, f"{stem}_translation_{model_key}_en.wav")
    sf.write(output_audio_path, audio, SAMPLE_RATE)

    result = {
        "input_filename": input_filename,
        "model_key": model_key,
        "kmeans_checkpoint": kmeans_path,
        "translator_checkpoint": translator_checkpoint,
        "src_bpe_model": src_bpe_model,
        "tgt_bpe_model": tgt_bpe_model,
        "vocoder_checkpoint": vocoder_checkpoint,
        "source_units_count": len(source_units),
        "target_units_count": len(target_units),
        "decode_score": best_score,
        "source_units_preview": source_units[:80],
        "target_units_preview": target_units[:80],
        "target_units_text": target_unit_text,
        "audio_output_path": output_audio_path,
    }

    output_json_path = os.path.join(OUTPUT_DIR, f"{stem}_translation_{model_key}_en.json")
    with open(output_json_path, "w") as file:
        json.dump(result, file, indent=2)

    audio_volume.commit()
    return {"result": result, "output_path": output_json_path, "audio_path": output_audio_path}


@app.local_entrypoint()
def main(input_audio_path: str, model: str = "xeus"):
    if not os.path.exists(input_audio_path):
        raise FileNotFoundError(f"Input audio not found: {input_audio_path}")

    with open(input_audio_path, "rb") as file:
        audio_bytes = file.read()

    response = translate_single_audio_to_audio.remote(
        audio_bytes=audio_bytes,
        input_filename=os.path.basename(input_audio_path),
        model_key=model,
    )

    result = response["result"]
    print("Single-audio translation + vocoder synthesis complete")
    print(f"Input: {result['input_filename']}")
    print(f"Model: {result['model_key']}")
    print(f"Source units: {result['source_units_count']}")
    print(f"Target units: {result['target_units_count']}")
    print(f"Saved audio: {response['audio_path']}")
    print(f"Saved JSON: {response['output_path']}")
