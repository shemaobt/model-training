# RFC: Acousteme Generation & Semantic Mapping

- Feature Name: `acousteme_generation`
- Start Date: 2026-02-06
- RFC PR: [bible-audio-training/rfcs/0001-acoustemes.md](https://github.com/placeholder)
- Issue: [bible-audio-training/issues/123](https://github.com/placeholder)

# Summary

This RFC proposes and documents the methodology for generating "Acoustemes" (manageable acoustic units) from raw self-supervised model outputs. It defines the use of **Run-Length Encoding (RLE)** to transform high-frequency frame-level tokens into event-based segments. Furthermore, it outlines a rigorous framework for mapping these acoustic segments to semantic categories (Events, Emotions from `mm_poc_v2`), leveraging generative AI (AudioLM) to mitigate manual labeling burdens.

# Motivation

When processing audio through large self-supervised models like **Wav2Vec2** or **MMS-300M**, the raw output consists of a dense stream of discrete tokens (typically 50Hz, or one every 20ms). 

For a 5-second clip, this results in ~250 tokens. This granularity presents two major problems:
1.  **Redundancy**: Phonetic and acoustic events naturally persist longer than 20ms, resulting in long sequences of identical tokens (e.g., `[31, 31, 31, 42, 42...]`).
2.  **Unmanageability**: Downstream applications, such as the **Beads** visualization or semantic analysis tools, struggle to interpret or render thousands of overlapping points.

We need a standardized "Acoustic Event" format—the **Acousteme**—that bridges the gap between raw signal processing and symbolic linguistic processing. This format must be compact, semantically aligned with phonetic reality, and compatible with our Meaning Map (`mm_poc_v2`) architecture.

# Guide-level explanation

The proposed solution introduces a transformation layer between the raw model inference and the downstream applications.

## The Transformation: "Beads"
A user providing an audio file to the pipeline will receive an `_acoustemes.json` file. Instead of a list of 10,000 numbers, they receive a list of "Segments" or "Beads".

### From Frames to Events
Imagine the raw model "hears" the audio in 20ms snapshots:
*   `0.00s`: Unit 31 (Sound "A")
*   `0.02s`: Unit 31 (Sound "A")
*   `0.04s`: Unit 31 (Sound "A")
*   `0.06s`: Unit 42 (Sound "T")

The **Acousteme Generator** groups these into events:
*   **Event 1**: Unit 31, Duration 0.06s (The "A" sound)
*   **Event 2**: Unit 42, ... (The "T" sound)

This creates a clean, symbolic sequence: `A -> T` which is much closer to how meaningful language works.

## Semantic Integration features
This format allows us to treat audio like text. We can now train "Semantic Probes" to classify these sequences.
*   **Input**: A sequence of Acoustemes (e.g., `31-42-99`).
*   **Output**: A semantic label from the Meaning Map (e.g., `GRIEF`, `MOTION`).

This enables users to query the database not just by text ("Find verses about weeping") but by sound ("Find verses that *sound* like this acoustic pattern").

# Reference-level explanation

## Data Structure and Algorithm

The core logic is implemented in `scripts/run_acoustic_inference_v1.py` via the `units_to_segments` function. It employs **Run-Length Encoding (RLE)**.

### Algorithm Steps:
1.  **Input**: lists of `units` (integers) and `timestamps` (floats).
2.  **Iterate**: Traverse the list, maintaining a `current_unit` cursor.
3.  **Accumulate**: As long as `unit[i] == current_unit`, continue.
4.  **Emit**: When `unit[i] != current_unit`:
    *   Create segment `{unit_id: current_unit, start: start_time, end: timestamp[i]}`.
    *   Reset `start_time` and `current_unit`.

### JSON Schema
The output file follows this schema:
```json
{
  "duration_sec": 238.6,
  "num_frames": 11929,
  "segments": [
    {
      "start": 0.0,
      "end": 0.02,
      "unit_id": 31
    },
    ...
  ]
}
```

## Semantic Mapping Architecture

The integration with `mm_poc_v2` requires extending dimensionality:

1.  **Acousteme Sequence (`Lang-A`)**: Treating the discrete unit stream `[u1, u2, u3...]` as a sentence in a new language.
2.  **Target Schema**: The `mm_poc_v2` database defines the target ontology:
    *   `Event.category` (e.g., STATE, MOTION, SPEECH)
    *   `EventEmotion.primary` (e.g., JOY, GRIEF, FEAR)
3.  **Generative Augmentation (AudioLM)**:
    *   To solve the *Labeling Burden* (scarcity of manually timestamped semantic events), we utilize the AudioLM architecture.
    *   **Few-Shot Learning**: Manual labeling of minimal seed examples (< 50).
    *   **Latent Expansion**: Using the generative model to produce synthetic variations of these seed examples (`Data Augmentation`), effectively multiplying the training set by 10-100x.
    *   **Cluster Labeling**: Identifying clusters in the AudioLM latent space and assigning a single label to the centroid, propagating it to thousands of unlabeled samples.

# Drawbacks

1.  **Loss of Micro-timing**: By grouping frames, we strictly quantize start/end times to the resolution of the frame rate (20ms). Extremely fast transient events (<20ms) might be merged if they share a unit code with neighbors (unlikely but possible).
2.  **Dependency on Model Stability**: If the underlying model (MMS-300M) is unstable and flickers between units (`31 -> 32 -> 31`), RLE will produce many short "dust" segments instead of a clean bead. This requires model fine-tuning or smoothing.

# Rationale and alternatives

### Why RLE?
*   **Compression**: Achieves 3x-5x reduction in file size and processing overhead.
*   **Symbolic Integrity**: It is "lossless" in terms of sequence order. Unlike smoothing algorithms (e.g., Moving Average) which might "invent" values, RLE only aggregates existing values.

### Alternatives considered
1.  **No Grouping**: Storing raw frames.
    *   *Pro*: Full fidelity.
    *   *Con*: Unmanageable JSON size (~5MB for short clips), extremely slow frontend rendering.
2.  **Time-Window Averaging**: Splitting audio into fixed 100ms chunks and taking the mode unit.
    *   *Pro*: Constant data rate.
    *   *Con*: "Aliasing" issues where unit boundaries don't align with windows, causing misclassifications.

# Prior art

*   **Beads Project**: The visualization metaphor of "beads on a string" directly inspired this segmentation.
*   **MPEG / Audio Codecs**: Run-length encoding is a standard compression technique in entropy coding stages of MP3/AAC.
*   **NLP Tokenization**: Sub-word tokenization (BPE) merges frequent character sequences; our RLE merges identical adjacent "character" sequences.

# Unresolved questions

1.  **Smoothing Thresholds**: Should we auto-merge segments shorter than X ms (e.g., <40ms) into neighbors to remove "flicker"? Currently, we do not.
2.  **Vocabulary Mapping**: How strictly does `Unit 31` map to a specific phoneme (e.g., /a/) across different speakers? If the mapping drifts, the Semantic Probe will degrade.

# Future possibilities

1.  **Acoustic Query Engine**: Implementing a SQL-like interface to query audio: `SELECT * FROM audio WHERE pattern MATCHES '31-42-99%'`.
2.  **Real-time Transcription**: Streaming the Acousteme generation for live visualization.
3.  **Voice Cloning via Acoustemes**: Since the "beads" capture the speaker's acoustic signature (if using a VQ-VAE codebook), rearranging them could allow for rudimentary voice synthesis/editing.
