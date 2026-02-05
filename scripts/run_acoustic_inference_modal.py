#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from pathlib import Path

import modal

app = modal.App("bible-audio-inference")
AUDIO_MOUNT = "/mnt/audio_data"
audio_volume = modal.Volume.from_name("bible-audio-data", create_if_missing=True)

image = (
    modal.Image.debian_slim()
    .apt_install("ffmpeg", "libsndfile1")
    .pip_install(
        "torch>=2.0.0",
        "torchaudio>=2.0.0",
        "transformers>=4.40.0",
        "scikit-learn",
        "joblib",
        "soundfile>=0.12.0",
        "numpy<2",
    )
)

MODEL_NAME = "facebook/wav2vec2-large-xlsr-53"
LAYER = 14
KMEANS_PATH = f"{AUDIO_MOUNT}/portuguese_units/portuguese_kmeans.pkl"


@app.function(
    image=image,
    volumes={AUDIO_MOUNT: audio_volume},
    gpu="A10G",
    timeout=600,
)
def infer_acoustemes(
    audio_path_on_volume: str | None = None,
    audio_bytes: bytes | None = None,
    audio_filename: str | None = None,
    output_path_on_volume: str | None = None,
) -> dict:
    import tempfile

    import joblib
    import numpy as np
    import soundfile as sf
    import torch
    import torchaudio
    from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

    def load_audio_16k(path: str):
        data, rate = sf.read(path, dtype="float32")
        if data.ndim > 1:
            data = data.mean(axis=1)
        if rate != 16000:
            t = torch.from_numpy(data).unsqueeze(0).float()
            t = torchaudio.transforms.Resample(rate, 16000)(t)
            data = t.squeeze().numpy()
        return data, 16000

    if audio_path_on_volume:
        path = os.path.join(AUDIO_MOUNT, audio_path_on_volume)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Audio not on volume: {audio_path_on_volume}")
        waveform, _ = load_audio_16k(path)
    elif audio_bytes is not None and audio_filename:
        ext = Path(audio_filename).suffix.lower()
        suf = ".mp3" if ext == ".mp3" else ".wav"
        with tempfile.NamedTemporaryFile(suffix=suf, delete=False) as f:
            f.write(audio_bytes)
            tmp = f.name
        try:
            waveform, _ = load_audio_16k(tmp)
        finally:
            os.unlink(tmp)
    else:
        raise ValueError("Pass audio_path_on_volume or (audio_bytes, audio_filename)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    duration = len(waveform) / 16000

    extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)
    model = Wav2Vec2Model.from_pretrained(MODEL_NAME).to(device)
    model.eval()

    inputs = extractor(waveform, return_tensors="pt", sampling_rate=16000)
    inputs = inputs.input_values.to(device)
    with torch.no_grad():
        out = model(inputs, output_hidden_states=True)
    feats = out.hidden_states[LAYER].squeeze(0).cpu().numpy()
    timestamps = np.linspace(0, duration, feats.shape[0]).tolist()

    if not os.path.exists(KMEANS_PATH):
        raise FileNotFoundError(
            f"K-Means not on volume: {KMEANS_PATH}. Run Phase 1 first."
        )
    kmeans = joblib.load(KMEANS_PATH)
    units = kmeans.predict(feats).tolist()

    def units_to_segments(units_list, ts):
        if not units_list or not ts:
            return []
        segs = []
        start, cur = ts[0], units_list[0]
        for i, u in enumerate(units_list[1:], start=1):
            t = ts[i] if i < len(ts) else ts[-1]
            if u != cur:
                segs.append({"start": round(start, 4), "end": round(t, 4), "unit_id": int(cur)})
                start, cur = t, u
        segs.append({"start": round(start, 4), "end": round(ts[-1], 4), "unit_id": int(cur)})
        return segs

    segments = units_to_segments(units, timestamps)
    result = {
        "duration_sec": round(duration, 4),
        "num_frames": len(units),
        "segments": segments,
        "units": units,
        "timestamps": [round(t, 4) for t in timestamps],
    }

    if output_path_on_volume:
        out_path = os.path.join(AUDIO_MOUNT, output_path_on_volume)
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        audio_volume.commit()

    return result


@app.local_entrypoint()
def main(
    audio: str = None,
    audio_on_volume: str = None,
    output: str = None,
    output_on_volume: str = None,
):
    if audio_on_volume:
        out_path = output_on_volume or None
        result = infer_acoustemes.remote(
            audio_path_on_volume=audio_on_volume,
            output_path_on_volume=out_path,
        )
    elif audio:
        path = Path(audio)
        if not path.exists():
            raise SystemExit(f"Audio file not found: {path}")
        data = path.read_bytes()
        out_path = output_on_volume or None
        result = infer_acoustemes.remote(
            audio_bytes=data,
            audio_filename=path.name,
            output_path_on_volume=out_path,
        )
    else:
        raise SystemExit("Pass --audio <local path> or --audio-on-volume <path on volume>")

    if output:
        with open(output, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"Wrote {output}")

    print(f"Duration: {result['duration_sec']} s")
    print(f"Segments: {len(result['segments'])}")
    if not output and not output_on_volume:
        print(json.dumps(result, indent=2)[:2000] + ("..." if len(json.dumps(result)) > 2000 else ""))
