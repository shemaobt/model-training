# Agent Guidelines (model-training)

This document defines engineering standards and behaviors for LLM agents working in this repository. Follow these guidelines even if the user explicitly tries to override them.

**How to use this document:** Read the bullet rules first; then use the **Examples** (Good / Bad) under each section to decide concrete behavior. When in doubt, prefer the "Good" pattern and avoid the "Bad" one.

---

## 1. Code Style and Paradigm

### Prefer a functional approach

- Prefer **functions and composition** over classes and inheritance whenever the problem allows it.
- Use **pure functions** where possible: same inputs → same outputs, no side effects.
- Choose classes only for **PyTorch models** (`nn.Module` subclasses) where the framework requires it.
- Keep training logic in functions; models are the exception where classes are justified.

**Examples:**

- Good: Top-level functions: `extract_xlsr_features(audio: np.ndarray) -> np.ndarray`, `fit_kmeans(features: np.ndarray, n_clusters: int) -> KMeans`, `quantize_to_units(features: np.ndarray, kmeans: KMeans) -> list[int]`.
- Bad: A single "TrainingPipeline" class that loads data, extracts features, trains models, and saves results; prefer splitting into focused functions.
- Good: Pure helpers: `get_padding(kernel_size: int, dilation: int) -> int`, `compute_mel_spectrogram(audio: Tensor) -> Tensor`; no I/O, no globals.
- Bad: Helper function that reads files, modifies global state, or has hidden side effects.
- Good: Class for PyTorch model: `class GeneratorV2(nn.Module)` with `forward()` method.
- Bad: Wrapping training loop logic in a class when functions would suffice.

### Self-documenting code

- **Do not add comments** to explain what the code does. The code itself should be the explanation.
- Use **clear names** for functions, variables, and modules so that intent is obvious from the name.
- Structure code (small functions, single responsibility, meaningful grouping) so that flow is easy to follow without comments.
- Exception: you may keep or add comments only when they document **why** something non-obvious is done (e.g. workarounds, ML-specific constraints, or non-obvious hyperparameter choices).

**Examples:**

- Good: `def extract_pitch_bins(audio: np.ndarray, n_bins: int = 32) -> np.ndarray:` — name says what it does.
- Bad: `# Extract pitch bins from audio` above the same function; the name already states this.
- Good: `MIN_SEGMENT_DURATION = 2.0` and `MAX_SEGMENT_DURATION = 120.0` — constants are self-explanatory.
- Bad: `# Minimum duration in seconds` comment above `MIN_SEGMENT_DURATION = 2.0`.
- Good (exception): Comment for non-obvious "why": `# Layer 14 provides best phonetic content while filtering speaker identity` or `# Use numpy<2 due to librosa compatibility issues`.
- Bad: Long comment blocks describing what each step does; refactor into smaller named functions instead.

---

## 2. Clean Code Principles

### Single Responsibility Principle (SRP)

- Each function should do **one thing only** and do it well.
- If a function name contains "and" or does multiple unrelated operations, split it.
- Functions should have **one reason to change**.

**Examples:**

- Good: `load_audio(path)`, `resample_audio(audio, target_sr)`, `normalize_audio(audio)` — separate concerns.
- Bad: `load_and_process_audio(path)` that loads, resamples, normalizes, and segments.
- Good: `save_checkpoint(model, path)` and `log_metrics(metrics)` — separate I/O concerns.
- Bad: `save_checkpoint_and_log(model, path, metrics)` that mixes saving and logging.

### Small functions

- Functions should be **short and focused** — aim for 10-20 lines maximum.
- If a function exceeds 30 lines, consider extracting helper functions.
- Each function should be readable at a glance.

**Examples:**

- Good:
```python
def compute_mel_loss(real_audio: Tensor, fake_audio: Tensor) -> Tensor:
    real_mel = compute_mel_spectrogram(real_audio)
    fake_mel = compute_mel_spectrogram(fake_audio)
    return F.l1_loss(fake_mel, real_mel)
```
- Bad: A 100-line function that computes multiple losses, handles edge cases, and logs results inline.

### DRY (Don't Repeat Yourself)

- **Extract repeated code** into named functions.
- Use constants for repeated literals.
- Create shared utilities for common patterns.

**Examples:**

- Good: `SAMPLE_RATE = 16000` used everywhere instead of hardcoded `16000`.
- Bad: `audio, sr = librosa.load(path, sr=16000)` repeated in 5 different functions.
- Good: Extract `load_audio_16k(path)` and reuse it.
- Bad: Copy-pasting the same 10 lines of setup code in every training function.

### Descriptive naming

- **Function names**: Use verbs — `load_`, `compute_`, `extract_`, `save_`, `train_`, `validate_`.
- **Variable names**: Use nouns — `audio_data`, `feature_matrix`, `checkpoint_path`.
- **Boolean names**: Use `is_`, `has_`, `should_` — `is_valid`, `has_checkpoint`, `should_resume`.
- **Constants**: UPPER_SNAKE_CASE — `SAMPLE_RATE`, `MAX_EPOCHS`, `BATCH_SIZE`.

**Examples:**

- Good: `def extract_xlsr_features(audio: np.ndarray) -> np.ndarray:`
- Bad: `def process(x):` — unclear what it processes or returns.
- Good: `checkpoint_path`, `best_loss`, `is_training`
- Bad: `p`, `l`, `flag` — single letters or generic names.
- Good: `for segment in audio_segments:` — clear iteration variable.
- Bad: `for i in range(len(audio_segments)): segment = audio_segments[i]` — unnecessary indexing.

### Early returns (guard clauses)

- Use **early returns** to handle edge cases at the top of functions.
- Reduces nesting and makes the "happy path" clear.
- Fail fast and return early for invalid inputs.

**Examples:**

- Good:
```python
def process_audio_file(path: str) -> np.ndarray:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Audio file not found: {path}")
    if not path.endswith(('.wav', '.mp3')):
        raise ValueError(f"Unsupported format: {path}")
    return load_and_process(path)
```
- Bad:
```python
def process_audio_file(path: str) -> np.ndarray:
    if os.path.exists(path):
        if path.endswith(('.wav', '.mp3')):
            return load_and_process(path)
        else:
            raise ValueError(...)
    else:
        raise FileNotFoundError(...)
```

### Avoid magic numbers

- Replace **literal numbers** with named constants.
- Constants should explain the meaning of the value.
- Group related constants together.

**Examples:**

- Good:
```python
XLSR_SAMPLE_RATE = 16000
XLSR_LAYER = 14
N_ACOUSTIC_UNITS = 100

features = extract_features(audio, sample_rate=XLSR_SAMPLE_RATE, layer=XLSR_LAYER)
kmeans = fit_kmeans(features, n_clusters=N_ACOUSTIC_UNITS)
```
- Bad:
```python
features = extract_features(audio, sample_rate=16000, layer=14)
kmeans = fit_kmeans(features, n_clusters=100)
```

### Limit function arguments

- Aim for **3 or fewer** arguments per function.
- Use **dataclasses or typed dicts** for related parameters.
- Use **keyword arguments with defaults** for optional parameters.

**Examples:**

- Good: `def train_vocoder(config: TrainingConfig):` with a dataclass.
- Good: `def load_audio(path: str, sample_rate: int = 16000, mono: bool = True):`
- Bad: `def train(model, optimizer, scheduler, train_loader, val_loader, epochs, batch_size, lr, patience, checkpoint_dir, log_dir):`

### Avoid deep nesting

- Keep indentation **shallow** (max 2-3 levels).
- Use early returns, extract functions, or invert conditions to reduce nesting.
- Deeply nested code is hard to read and test.

**Examples:**

- Good:
```python
def process_segments(segments: list) -> list:
    results = []
    for segment in segments:
        if not is_valid(segment):
            continue
        processed = process_segment(segment)
        results.append(processed)
    return results
```
- Bad:
```python
def process_segments(segments: list) -> list:
    results = []
    for segment in segments:
        if is_valid(segment):
            if len(segment) > MIN_LENGTH:
                if segment.dtype == np.float32:
                    processed = process_segment(segment)
                    if processed is not None:
                        results.append(processed)
    return results
```

### Error handling

- **Fail fast**: Validate inputs early and raise meaningful exceptions.
- Use **specific exception types** over generic `Exception`.
- Log errors with context for debugging.

**Examples:**

- Good:
```python
def load_checkpoint(path: str) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    try:
        return torch.load(path, map_location="cpu")
    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint {path}: {e}")
```
- Bad: Silently returning `None` when a checkpoint doesn't exist.
- Bad: Catching all exceptions with bare `except:` clause.

### Jupytext format

- Training scripts use **Jupytext format**: Python files with cell markers that can run as notebooks or scripts.
- Use `# %%` for code cell markers.
- Use `# %% [markdown]` for markdown cells (documentation).
- Do not commit actual `.ipynb` files; keep notebooks as `.py` files.

**Examples:**

- Good: File starts with Jupytext header, uses `# %%` to separate logical sections.
- Bad: Mixing `.ipynb` and `.py` versions of the same notebook; keep only the `.py` version.
- Good: Markdown cell explaining a training phase: `# %% [markdown]` followed by `# # Phase 1: Acoustic Tokenization`.
- Bad: Long inline comments instead of markdown cells for documentation.

---

## 3. Architecture and Design

### Project structure

The repository follows this structure:

```
model-training/
├── src/
│   ├── models/           # Neural network architectures (reference)
│   │   ├── generator.py
│   │   ├── generator_v2.py
│   │   ├── discriminator.py
│   │   └── discriminator_v2.py
│   └── training/         # Modal training scripts (Jupytext)
│       ├── phase1_acoustic.py
│       ├── phase2_bpe.py
│       ├── phase3_vocoder.py
│       └── phase3_vocoder_v2.py
├── scripts/              # Local utilities (run on host)
│   ├── segment_audio.py
│   └── upload_to_modal.py
├── docs/                 # Technical documentation
└── modal_downloads/      # Downloaded results (gitignored)
```

### Phase separation

- The pipeline has **3 distinct phases** that must run in order: Phase 1 → Phase 2 → Phase 3.
- Each phase produces outputs that the next phase consumes.
- Do not mix phase logic; keep each phase in its own script.

**Examples:**

- Good: Phase 1 outputs `kmeans.pkl` and `corpus_timestamped.json`; Phase 3 reads these files.
- Bad: Phase 3 script that re-implements feature extraction from Phase 1.
- Good: Check that Phase 1 outputs exist before running Phase 2.
- Bad: Skipping phases or assuming outputs exist without verification.

### Reuse existing code; avoid overengineering

- **Use current methods or abstractions** instead of creating new ones.
- Create new abstractions **only when necessary** (e.g. no existing function fits).
- **Avoid overengineering:** Do not add layers for "future flexibility" when the current need is simple.

**Examples:**

- Good: Reuse existing `LANGUAGE_CONFIGS` dictionary pattern across all training scripts.
- Bad: Creating a new configuration system when `LANGUAGE_CONFIGS` already works.
- Good: Add a new language by extending the existing dictionary.
- Bad: Creating a "LanguageConfigFactory" class for a simple key-value lookup.

---

## 4. Modal Cloud Training

- **Never run training scripts on the host machine.** All training runs on Modal's cloud GPUs.
- Use `python3 -m modal run <script>` to execute training.
- Use `--detach` for long-running jobs (training can take hours).
- **Local scripts** in `scripts/` are the exception — they run on the host for preprocessing.

**Examples:**

- Good: `python3 -m modal run --detach src/training/phase3_vocoder_v2.py::main`
- Good: `python3 -m modal run --detach src/training/phase1_acoustic.py::main_skip_segmentation --language satere`
- Bad: `python src/training/phase3_vocoder_v2.py` — this will fail or run without GPU.
- Good: Local preprocessing: `python scripts/segment_audio.py --language portuguese`
- Bad: Running `scripts/segment_audio.py` on Modal — it's designed for local execution with local file paths.

### Volume management

- Use `modal.Volume.from_name("bible-audio-data")` for persistent storage.
- Download results with `modal volume get bible-audio-data <remote_path> <local_path>`.
- Upload data with `python3 -m modal run scripts/upload_to_modal.py --language <lang>`.

**Examples:**

- Good: `modal volume get bible-audio-data vocoder_v2_checkpoints/ ./modal_downloads/`
- Bad: Assuming files persist between Modal runs without using volumes.

---

## 5. Dependency Management

- Dependencies are defined **inline in Modal image definitions**.
- There is no `requirements.txt` or `pyproject.toml` for training scripts.
- **Pin versions** for reproducibility (e.g. `torch>=2.0.0`, `numpy<2`).
- System packages use `.apt_install()`.

**Examples:**

- Good:
```python
image = (
    modal.Image.debian_slim()
    .apt_install("ffmpeg", "libsndfile1")
    .pip_install(
        "torch>=2.0.0",
        "torchaudio>=2.0.0",
        "transformers>=4.40.0",
        "numpy<2",
        "librosa>=0.10.0",
    )
)
```
- Bad: Creating a separate `requirements.txt` for Modal scripts.
- Good: Version constraint `numpy<2` to avoid compatibility issues with librosa.
- Bad: Unpinned dependencies like `pip_install("torch")` without version constraints.

---

## 6. Training Pipeline Guidelines

### Language configuration

- Use centralized `LANGUAGE_CONFIGS` dictionaries for multi-language support.
- Override language via CLI (`--language`) or environment variable (`TRAINING_LANGUAGE`).
- Each language has its own directories for segmented audio, units, and checkpoints.

**Examples:**

- Good:
```python
LANGUAGE_CONFIGS = {
    "portuguese": {
        "segmented_dir": f"{AUDIO_MOUNT}/segmented_audio",
        "output_dir": f"{AUDIO_MOUNT}/portuguese_units",
        "vocoder_dir": f"{AUDIO_MOUNT}/vocoder_v2_checkpoints",
    },
    "satere": {
        "segmented_dir": f"{AUDIO_MOUNT}/segmented_audio_satere",
        "output_dir": f"{AUDIO_MOUNT}/satere_units",
        "vocoder_dir": f"{AUDIO_MOUNT}/vocoder_v2_satere_checkpoints",
    },
}
```
- Bad: Hardcoding paths for a single language.
- Good: Adding a new language by extending the dictionary.
- Bad: Creating separate scripts per language.

### Hyperparameters

- Define defaults in function signatures.
- Override via CLI arguments.
- Document hyperparameter choices in markdown cells or docs.

**Examples:**

- Good: `def train_vocoder(epochs: int = 1000, batch_size: int = 12, patience: int = 100):`
- Good: CLI override: `--epochs 500 --batch-size 16`
- Bad: Hardcoding hyperparameters in the middle of training loops.

### Checkpointing

- **Always implement checkpointing** for resume capability.
- Save checkpoints at regular intervals (e.g. every 25 epochs).
- Support `--resume <checkpoint.pt>` to continue training.
- Track best model separately from latest checkpoint.

**Examples:**

- Good: Save `v2_latest.pt` every N epochs; save `v2_best.pt` when validation improves.
- Good: `python3 -m modal run --detach src/training/phase3_vocoder_v2.py::main --resume v2_latest.pt`
- Bad: Training script that cannot resume from interruption.

---

## 7. Model Architecture Guidelines

### PyTorch models

- Models are defined as `nn.Module` subclasses.
- Use **version suffixes** for iterations: `GeneratorV2`, `DiscriminatorV2`.
- Reference model code lives in `src/models/`.
- Training scripts may embed model code as strings for Modal compatibility.

**Examples:**

- Good: `class GeneratorV2(nn.Module):` with clear `__init__` and `forward` methods.
- Good: Model code in `src/models/generator_v2.py` for reference; embedded in training script for Modal.
- Bad: Modifying model architecture without creating a new version (V3).

### GAN patterns

- Generator/Discriminator pairs for GAN training.
- Use embeddings for discrete inputs (acoustic units, pitch bins).
- Multi-scale or multi-period discriminators for audio quality.

**Examples:**

- Good: `self.unit_embedding = nn.Embedding(n_units, embed_dim)` for discrete unit input.
- Good: MPD (Multi-Period Discriminator) + MSD (Multi-Scale Discriminator) combination.
- Bad: Single-scale discriminator for audio (misses multi-scale patterns).

---

## 8. Data Handling

### Audio format

- Processing format: **16kHz mono WAV**.
- Input format: MP3 files (converted during preprocessing).
- Segment length: 2-120 seconds (filtered).

**Examples:**

- Good: `SAMPLE_RATE = 16000` as a constant.
- Good: Silence-based segmentation with configurable thresholds.
- Bad: Hardcoding sample rate in multiple places.

### Data flow

```
Raw MP3 → Local Segmentation → Upload to Modal → Phase 1 → Phase 2 → Phase 3 → Download Results
```

- Local: `scripts/segment_audio.py` runs on host.
- Cloud: Training phases run on Modal.
- Download: `modal volume get` retrieves results.

---

## 9. Testing and Quality

- **No formal test framework** (no pytest). Use dedicated test scripts.
- Test scripts: `vocoder_test.py`, `vocoder_test_v2.py`, `validate_units.py`.
- Quality metrics: MCD (Mel Cepstral Distortion), SNR, F0 RMSE.

**Examples:**

- Good: `python3 -m modal run src/training/vocoder_test_v2.py::main --num-samples 50`
- Good: Test script generates comparison audio samples and reports metrics.
- Bad: No way to evaluate model quality after training.

---

## 10. Secrets and Authentication

- **Modal handles authentication** via `modal token set`.
- No `.env` files or hardcoded secrets.
- Do not commit Modal tokens or API keys.

**Examples:**

- Good: `modal token set --token-id <id> --token-secret <secret>` (one-time setup).
- Bad: Hardcoding Modal credentials in scripts.

---

## 11. Version Control and Commits

### Do not commit unless asked

- **Never commit, push, or amend** unless the user **explicitly requests** a commit.
- Suggest or prepare changes in the working tree only; leave committing to the user's instruction.

**Examples:**

- Good: User says "commit these changes" → run `git status`, group changes, create commits.
- Bad: Automatically running `git commit` after implementing a feature.

### When the user requests a commit

1. **Analyze the working tree**: Run `git status` and `git diff`.
2. **Group by scope**: e.g. "phase1", "vocoder", "docs", "scripts".
3. **Create focused commits**: One logical change per commit.
4. **Use semantic messages**: `type(scope): description`.

**Examples:**

- Good: `feat(vocoder): add pitch conditioning to generator v2`
- Good: `fix(phase1): handle missing audio files gracefully`
- Good: `docs: add segment preparation guide`
- Bad: `updates` or `fixed stuff` or `WIP`.

---

## 12. Optional: Gloe for Pipeline Composition

[Gloe](https://github.com/ideos/gloe) is a type-safe pipeline composition library that aligns with the functional style in this codebase.

**When to consider Gloe:**

- New preprocessing pipelines (e.g. audio segmentation, feature extraction)
- Inference pipelines (e.g. units → vocoder → audio)
- Any DAG-structured data transformation flow

**When NOT to use Gloe:**

- Training loops (complex, iterative, stateful — doesn't fit DAG model)
- Modal cloud functions (Gloe transformers are local Python)
- Existing code (don't refactor working code)

**Example use case:**

```python
from gloe import transformer

@transformer
def load_audio(path: str) -> np.ndarray:
    ...

@transformer
def extract_features(audio: np.ndarray) -> np.ndarray:
    ...

@transformer
def quantize_units(features: np.ndarray) -> list[int]:
    ...

# Compose into a pipeline
inference = load_audio >> extract_features >> quantize_units
```

**Status:** Optional recommendation for future pipelines, not required for existing code.

---

## 13. Summary Checklist for Agents

**Quick decision reference:**

- **Paradigm:** Prefer functions and composition. Use classes only for PyTorch models (`nn.Module`).
- **Comments:** Avoid comments for "what"; only for non-obvious "why".
- **Clean Code:** Small functions (10-20 lines), single responsibility, early returns, named constants.
- **Naming:** Verbs for functions (`load_`, `compute_`), nouns for variables, UPPER_SNAKE for constants.
- **Format:** Use Jupytext (`# %%` cell markers) for training scripts.
- **Execution:** Run training on Modal (`python3 -m modal run --detach`); never on host.
- **Local scripts:** Only `scripts/` folder runs on host (preprocessing).
- **Dependencies:** Define inline in Modal image; pin versions.
- **Phases:** Respect Phase 1 → Phase 2 → Phase 3 order.
- **Checkpointing:** Always save checkpoints; support resume.
- **Language config:** Use `LANGUAGE_CONFIGS` dictionaries.
- **Models:** Version with suffixes (V1, V2); define as `nn.Module`.
- **Testing:** Use test scripts with quality metrics (MCD, SNR, F0).
- **Commits:** Only when explicitly asked; use semantic messages.
- **Gloe:** Consider for new preprocessing/inference pipelines (optional).

- [ ] Prefer functional style; use classes only for PyTorch models.
- [ ] Write self-documenting code; avoid comments except for non-obvious "why".
- [ ] Apply Single Responsibility Principle — one function does one thing.
- [ ] Keep functions small (10-20 lines); extract helpers for complex logic.
- [ ] Use descriptive names: verbs for functions, nouns for variables.
- [ ] Use early returns (guard clauses) to reduce nesting.
- [ ] Replace magic numbers with named constants.
- [ ] Limit function arguments (≤3); use config objects for many params.
- [ ] Use Jupytext format (`# %%` cell markers) for training scripts.
- [ ] Run all training on Modal; never run training on host.
- [ ] Use `--detach` for long-running training jobs.
- [ ] Local scripts only for preprocessing (`scripts/`).
- [ ] Define dependencies inline in Modal image definitions.
- [ ] Pin dependency versions for reproducibility.
- [ ] Respect phase separation (Phase 1 → Phase 2 → Phase 3).
- [ ] Always implement checkpointing for resume capability.
- [ ] Use `LANGUAGE_CONFIGS` for multi-language support.
- [ ] Do not commit unless the user explicitly asks.
- [ ] Use semantic commit messages (`type(scope): description`).

---

*Guidelines for the model-training repository. Built using [agents.md](https://agents.md/) format.*
