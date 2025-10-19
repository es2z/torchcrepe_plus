# Codebase Structure

## Directory Layout

```
torchcrepe_plus/
├── torchcrepe/              # Main package
│   ├── __init__.py          # Public API exports
│   ├── __main__.py          # CLI entry point
│   ├── core.py              # Main prediction pipeline (predict, embed, infer, preprocess, postprocess)
│   ├── model.py             # Crepe CNN architecture (6-layer conv net)
│   ├── decode.py            # Decoding algorithms (viterbi, weighted_argmax, argmax)
│   ├── filter.py            # Signal filtering (mean, median for NaN-aware filtering)
│   ├── threshold.py         # Pitch/periodicity thresholding (At, Hysteresis, Silence)
│   ├── convert.py           # Unit conversions (bins ↔ cents ↔ frequency)
│   ├── load.py              # Model and audio loading utilities
│   ├── loudness.py          # A-weighted loudness computation (for silence detection)
│   └── assets/
│       ├── full.pth         # Pretrained full model weights (converted from TensorFlow)
│       └── tiny.pth         # Pretrained tiny model weights
├── tests/
│   ├── conftest.py          # Pytest fixtures (audio, frames, activations)
│   ├── test_core.py         # Tests for prediction and embedding
│   ├── test_decode.py       # Tests for decoders
│   ├── test_filter.py       # Tests for filters
│   ├── test_threshold.py    # Tests for thresholding
│   └── assets/              # Test data (test.wav, activation-*.npy, frames-crepe.npy)
├── setup.py                 # Package configuration
├── README.md                # User documentation
├── CLAUDE.md                # AI assistant guidance (just created)
├── LICENSE                  # MIT License
└── CITATION.cff             # Citation information

```

## Key Modules

### core.py (Main Pipeline)
- **`predict()`**: High-level pitch prediction from audio tensor
- **`predict_from_file()`**: Predict from audio file on disk
- **`predict_from_files_to_files()`**: Batch processing without model reloading
- **`embed()`**: Extract pitch embeddings (layer 5 output)
- **`preprocess()`**: Generator yielding batches of normalized frames
- **`infer()`**: Forward pass through model (with caching)
- **`postprocess()`**: Convert probabilities to pitch and periodicity

### model.py (Neural Network)
- **`Crepe`**: 6-layer CNN (Conv → ReLU → BatchNorm → MaxPool × 6)
  - Input: 1024-sample frames
  - Output: 360-bin probability distribution
  - Embedding extraction: Stop at layer 5

### decode.py (Decoding Algorithms)
- **`viterbi()`**: Viterbi decoding with pitch transition penalty (default, recommended)
- **`weighted_argmax()`**: Weighted average near peak (original CREPE method)
- **`argmax()`**: Simple maximum (fast but error-prone)

### filter.py (Signal Filtering)
- **`mean()`**: Moving average filter (handles NaN)
- **`median()`**: Median filter (robust to outliers)
- Helper utilities: `nanmean()`, `nanmedian()`

### threshold.py (Thresholding)
- **`At(value)`**: Simple threshold on periodicity
- **`Hysteresis()`**: Advanced thresholding to prevent spurious voiced regions
- **`Silence(db_threshold)`**: Set periodicity to 0 in silent regions

### convert.py (Unit Conversions)
- Pitch bins ↔ cents ↔ frequency (Hz)
- Dithering to remove quantization error

## Data Flow

1. **Audio Input** → `preprocess()` → Normalized 1024-sample frames (resampled to 16kHz)
2. **Frames** → `infer()` → 360-bin probabilities (or embeddings if `embed=True`)
3. **Probabilities** → `postprocess()` → Pitch (Hz) + Periodicity (confidence)
4. **Optional**: Filter/Threshold → Final cleaned pitch

## Model Caching
- `infer()` caches loaded model as function attribute
- Only reloads when capacity changes
- Automatically moves model to requested device
