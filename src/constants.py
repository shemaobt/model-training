# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.20.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Model Training Constants
#
# Centralized constants for the entire training pipeline.
# Import from this module instead of using magic numbers.

# %%

# Audio processing
SAMPLE_RATE = 16000
HOP_SIZE = 320  # XLSR-53 frame rate: ~20ms at 16kHz

# XLSR-53 feature extraction (Legacy Default)
XLSR_LAYER = 14
XLSR_FEATURE_DIM = 1024

# Acoustic Model Configurations
ACOUSTIC_MODELS = {
    "xlsr-53": {
        "model_name": "facebook/wav2vec2-large-xlsr-53",
        "layer": 14,
        "dim": 1024,
        "description": "Cross-lingual Speech Representations (53 languages)"
    },
    "mms-300m": {
        "model_name": "facebook/mms-300m",
        "layer": 14,  # Aligns with XLSR-53 depth for phonetic features
        "dim": 1024,
        "description": "Massively Multilingual Speech (1,400+ languages)"
    },
    "mms-1b": {
        "model_name": "facebook/mms-1b",
        "layer": 24,  # Deeper model, middle-to-late layers usually best
        "dim": 1280,
        "description": "MMS 1B parameters (High quality, high resource)"
    },
    "xeus": {
        "model_name": "espnet/xeus",
        "layer": 14,
        "dim": 1024,
        "description": "XEUS multilingual speech representations"
    }
}

# Default Model Selection
DEFAULT_MODEL = "xeus"

# Acoustic tokenization
NUM_ACOUSTIC_UNITS = 100
KMEANS_BATCH_SIZE = 1024
KMEANS_RANDOM_STATE = 42
KMEANS_N_INIT = 3

# Pitch conditioning
NUM_PITCH_BINS = 32
PITCH_UNVOICED_BIN = 32  # Index for unvoiced frames
F0_MIN = 50.0
F0_MAX = 400.0

# Segmentation thresholds
SILENCE_THRESHOLD_DB = -40.0
MIN_SILENCE_DURATION = 0.5
MIN_SEGMENT_DURATION = 2.0
MAX_SEGMENT_DURATION = 120.0

# Segment analysis (25ms frame, 10ms hop)
FRAME_DURATION = 0.025
HOP_DURATION = 0.010

# Training segment
SEGMENT_LENGTH = 32000  # 2 seconds at 16kHz
UNIT_LENGTH = SEGMENT_LENGTH // HOP_SIZE  # 100 units per segment

# Generator architecture
UNIT_EMBED_DIM = 256
PITCH_EMBED_DIM = 64
UPSAMPLE_INITIAL_CHANNEL = 512
UPSAMPLE_RATES = [5, 4, 4, 4]  # 320x total upsampling
UPSAMPLE_KERNEL_SIZES = [10, 8, 8, 8]
RESBLOCK_KERNEL_SIZES = [3, 7, 11]
RESBLOCK_DILATIONS = [[1, 1], [3, 1], [5, 1]]
LEAKY_RELU_SLOPE = 0.1

# Discriminator architecture
MPD_PERIODS = [2, 3, 5, 7, 11]  # Prime numbers for period discriminator

# Mel spectrogram
MEL_N_FFT = 1024
MEL_HOP_LENGTH = 256
MEL_N_MELS = 80

# Multi-resolution STFT loss
STFT_FFT_SIZES = [512, 1024, 2048]
STFT_HOP_SIZES = [50, 120, 240]
STFT_WIN_SIZES = [240, 600, 1200]

# Training hyperparameters (defaults)
DEFAULT_EPOCHS = 1000
DEFAULT_BATCH_SIZE = 12
DEFAULT_LEARNING_RATE = 0.0002
DEFAULT_ADAM_BETAS = (0.8, 0.99)
DEFAULT_LR_GAMMA = 0.999
DEFAULT_PATIENCE = 100
DEFAULT_SAVE_EVERY = 25
DEFAULT_SAMPLES_PER_EPOCH = 10000
DEFAULT_GRAD_CLIP_MAX_NORM = 5.0

# Loss weights
LAMBDA_MEL = 45.0
LAMBDA_STFT = 2.0
LAMBDA_FM = 2.0
EARLY_STOPPING_THRESHOLD = 0.1

# BPE tokenization
DEFAULT_BPE_VOCAB_SIZE = 500
DEFAULT_BPE_MIN_FREQUENCY = 5
BPE_MAX_SENTENCE_LENGTH = 10000
BPE_NUM_THREADS = 4
BPE_VOCAB_RETRY_FACTOR = 0.8

# Checkpointing
DEFAULT_CHECKPOINT_INTERVAL = 1000
DEFAULT_BUFFER_LIMIT = 20000
PROGRESS_LOG_INTERVAL = 500
