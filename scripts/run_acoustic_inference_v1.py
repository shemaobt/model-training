#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Add project root for imports if needed
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def load_audio_16k(path: str):
    try:
        import numpy as np
        import soundfile as sf
        import torch
        import torchaudio
    except ImportError:
        raise ImportError("Need soundfile, torch, torchaudio. pip install soundfile torch torchaudio")
    data, rate = sf.read(path, dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    if rate != 16000:
        t = torch.from_numpy(data).unsqueeze(0).float()
        t = torchaudio.transforms.Resample(rate, 16000)(t)
        data = t.squeeze().numpy()
    return data, 16000


def extract_features(waveform, model, extractor, layer: int, device):
    import numpy as np
    import torch
    duration = len(waveform) / 16000
    inputs = extractor(waveform, return_tensors="pt", sampling_rate=16000)
    inputs = inputs.input_values.to(device)
    with torch.no_grad():
        out = model(inputs, output_hidden_states=True)
    feats = out.hidden_states[layer].squeeze(0).cpu().numpy()
    timestamps = np.linspace(0, duration, feats.shape[0]).tolist()
    return feats, timestamps, duration


def units_to_segments(units: list[int], timestamps: list[float]) -> list[dict]:
    if not units or not timestamps:
        return []
    segs = []
    start = timestamps[0]
    cur = units[0]
    for i, u in enumerate(units[1:], start=1):
        t = timestamps[i] if i < len(timestamps) else timestamps[-1]
        if u != cur:
            segs.append({"start": round(start, 4), "end": round(t, 4), "unit_id": int(cur)})
            start = t
            cur = u
    segs.append({
        "start": round(start, 4),
        "end": round(timestamps[-1], 4),
        "unit_id": int(cur),
    })
    return segs


def main():
    ap = argparse.ArgumentParser(description="Run Portuguese v1 acoustic model on one audio file.")
    ap.add_argument("--audio", required=True, help="Path to audio file (mp3 or wav)")
    ap.add_argument(
        "--kmeans",
        default=None,
        help="Path to portuguese_kmeans.pkl (default: modal_downloads/portuguese_units/portuguese_kmeans.pkl)",
    )
    ap.add_argument(
        "--output",
        default=None,
        help="Output JSON path (default: <audio_basename>_acoustemes.json next to audio)",
    )
    ap.add_argument("--cpu", action="store_true", help="Use CPU only (no GPU)")
    args = ap.parse_args()

    audio_path = Path(args.audio)
    if not audio_path.exists():
        print(f"Error: audio file not found: {audio_path}", file=sys.stderr)
        sys.exit(1)

    kmeans_path = args.kmeans
    if not kmeans_path:
        for candidate in [
            REPO_ROOT / "modal_downloads" / "portuguese_units" / "portuguese_kmeans.pkl",
            REPO_ROOT / "modal_downloads" / "portuguese_units" / "portuguese_units" / "portuguese_kmeans.pkl",
            REPO_ROOT / "modal_downloads" / "phase1_outputs" / "portuguese_kmeans.pkl",
            REPO_ROOT / "portuguese_units" / "portuguese_kmeans.pkl",
        ]:
            if candidate.exists():
                kmeans_path = str(candidate)
                break
    if not kmeans_path or not Path(kmeans_path).exists():
        print(
            "Portuguese K-Means not found. Download it from Modal first:\n"
            "  modal volume get bible-audio-data portuguese_units/ ./modal_downloads/portuguese_units/\n"
            "If you see 'output path already exists': remove the path if it's a file, then retry:\n"
            "  rm -f modal_downloads/portuguese_units\n"
            "  modal volume get bible-audio-data portuguese_units/ ./modal_downloads/portuguese_units/\n"
            "Or pass the pkl path explicitly: --kmeans path/to/portuguese_kmeans.pkl",
            file=sys.stderr,
        )
        sys.exit(1)

    out_path = args.output
    if not out_path:
        out_path = str(audio_path.with_suffix("")) + "_acoustemes.json"

    import joblib
    import numpy as np
    import torch
    from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    MODEL_NAME = "facebook/wav2vec2-large-xlsr-53"
    LAYER = 14

    print("Loading XLSR-53...")
    extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)
    model = Wav2Vec2Model.from_pretrained(MODEL_NAME).to(device)
    model.eval()

    print("Loading Portuguese K-Means...")
    kmeans = joblib.load(kmeans_path)

    print(f"Loading audio: {audio_path}")
    waveform, _ = load_audio_16k(str(audio_path))

    print("Extracting features and predicting units...")
    feats, timestamps, duration_sec = extract_features(waveform, model, extractor, LAYER, device)
    units = kmeans.predict(feats).tolist()
    segments = units_to_segments(units, timestamps)

    out = {
        "duration_sec": round(duration_sec, 4),
        "num_frames": len(units),
        "segments": segments,
        "units": units,
        "timestamps": [round(t, 4) for t in timestamps],
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {len(segments)} segments to {out_path}")
    return out_path


if __name__ == "__main__":
    main()
