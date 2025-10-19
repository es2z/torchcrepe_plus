# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**torchcrepe** is a PyTorch implementation of the CREPE pitch tracking algorithm. CREPE (Convolutional Representation for Pitch Estimation) is a deep learning approach to fundamental frequency (F0) estimation for audio signals. This library provides a clean, GPU-accelerated implementation with both Python API and command-line interfaces.

**Key Features:**
- PyTorch-based pitch tracking using pretrained CREPE models
- Two model sizes: "tiny" (fast) and "full" (accurate)
- Multiple decoding methods: Viterbi (default), weighted argmax, and argmax
- Pitch embedding extraction for downstream tasks
- Filtering and thresholding utilities for post-processing
- Batch processing support with automatic model caching

## Development Commands

### Running Tests
```bash
# Install pytest if not already installed
pip install pytest

# Run all tests
pytest

# Run specific test file
pytest tests/test_core.py
pytest tests/test_decode.py
pytest tests/test_filter.py
pytest tests/test_threshold.py
```

### Command-Line Usage
```bash
# Basic pitch prediction from audio file
python -m torchcrepe --audio_files input.wav --output_files pitch.pt

# With periodicity output
python -m torchcrepe --audio_files input.wav --output_files pitch.pt --output_periodicity_files periodicity.pt

# Batch processing multiple files
python -m torchcrepe --audio_files file1.wav file2.wav --output_files pitch1.pt pitch2.pt

# Specify decoder and model
python -m torchcrepe --audio_files input.wav --output_files pitch.pt --model tiny --decoder viterbi

# Extract embeddings instead of pitch
python -m torchcrepe --audio_files input.wav --output_files embed.pt --embed

# Use GPU for inference
python -m torchcrepe --audio_files input.wav --output_files pitch.pt --gpu 0
```

### Installation
```bash
# Install PyTorch first (see https://pytorch.org/)
pip install torch torchaudio

# Install torchcrepe
pip install torchcrepe

# For development
pip install -e .
```

## Architecture

### Core Pipeline

The CREPE pipeline consists of four main stages:

1. **Preprocessing** (`preprocess()` in `core.py`):
   - Resamples audio to 16kHz (CREPE's expected sample rate)
   - Creates overlapping 1024-sample windows with configurable hop length
   - Normalizes each frame (mean-center and scale by std deviation)
   - Yields batches of frames for efficient GPU processing

2. **Inference** (`infer()` in `core.py`):
   - Passes frames through the CREPE CNN model
   - Model caching: loads model only once per capacity type
   - Returns either pitch probabilities (360 bins) or embeddings (from 5th layer)

3. **Postprocessing** (`postprocess()` in `core.py`):
   - Applies frequency range filtering (fmin/fmax)
   - Decodes probabilities to pitch using selected decoder
   - Optionally computes periodicity (confidence) metric

4. **Optional Filtering/Thresholding**:
   - `filter.py`: Mean and median filtering for smoothing
   - `threshold.py`: Removes unreliable pitch estimates based on periodicity

### Neural Network Architecture

**File**: `torchcrepe/model.py`

The `Crepe` class implements a 6-layer convolutional network:
- **Input**: 1024-sample audio frames
- **Layers 1-5**: Conv2D → ReLU → BatchNorm → MaxPool
  - Layer 1: Large receptive field (512x1 kernel, stride 4)
  - Layers 2-6: Smaller kernels (64x1, stride 1)
- **Layer 6**: Final convolution followed by flattening
- **Output**: 360-dimensional probability distribution (one per pitch bin)

**Embedding extraction**: Can stop at layer 5 output for pitch-aware embeddings (used in DDSP and other downstream tasks)

**Model sizes**:
- `full`: 1024→128→128→128→256→512 channels, 2048 features before classifier
- `tiny`: 128→16→16→16→32→64 channels, 256 features before classifier

### Pitch Representation

- **Pitch bins**: 360 bins spanning 0-2006 Hz (upper limit of CREPE)
- **Resolution**: 20 cents per bin (1/5 of a semitone)
- **Reference**: Bins mapped to frequency using `10 * 2^(cents/1200)` formula
- **Unvoiced frames**: Represented as `np.nan` when filtered/thresholded

### Decoding Methods

**File**: `torchcrepe/decode.py`

1. **`viterbi`** (default, recommended):
   - Uses librosa's Viterbi algorithm with transition matrix
   - Penalizes large pitch jumps (>12 semitones)
   - Smooths predictions and reduces octave errors
   - Best for musical and speech applications

2. **`weighted_argmax`** (original CREPE):
   - Weighted average of probabilities in 9-bin window around peak
   - Matches original TensorFlow implementation
   - More susceptible to octave/half-octave errors

3. **`argmax`**:
   - Simple maximum probability selection
   - Fastest but most susceptible to errors
   - Useful for debugging or extremely noisy signals

### Filtering and Thresholding

**Filtering** (`filter.py`):
- `mean()`: Moving average filter (handles NaN values)
- `median()`: Median filter (more robust to outliers)
- Common use: Smooth quantization artifacts or noisy periodicity

**Thresholding** (`threshold.py`):
- `At(value)`: Simple threshold on periodicity (typical: 0.21 for clean speech)
- `Hysteresis()`: Advanced thresholding with hysteresis to prevent spurious voiced regions
- `Silence(db)`: Sets periodicity to 0 in silent regions (default: -60 dB)

**Typical post-processing workflow**:
```python
# 1. Median filter noisy periodicity
win_length = 3  # ~15ms for 5ms hop
periodicity = torchcrepe.filter.median(periodicity, win_length)

# 2. Remove silent regions
periodicity = torchcrepe.threshold.Silence(-60.)(periodicity, audio, sr, hop_length)

# 3. Threshold unreliable pitch
pitch = torchcrepe.threshold.At(0.21)(pitch, periodicity)

# 4. Optionally smooth pitch
pitch = torchcrepe.filter.mean(pitch, win_length)
```

### File Organization

```
torchcrepe/
├── __init__.py          # Public API exports
├── __main__.py          # CLI entry point
├── core.py              # Main prediction pipeline (predict, embed, preprocess, infer, postprocess)
├── model.py             # Crepe CNN architecture
├── decode.py            # Decoding algorithms (viterbi, weighted_argmax, argmax)
├── filter.py            # Signal filtering utilities
├── threshold.py         # Pitch/periodicity thresholding
├── convert.py           # Unit conversions (bins ↔ cents ↔ frequency)
├── load.py              # Model and audio loading
├── loudness.py          # A-weighted loudness computation (for silence detection)
└── assets/
    ├── full.pth         # Pretrained full model weights
    └── tiny.pth         # Pretrained tiny model weights

tests/
├── conftest.py          # Pytest fixtures (audio, frames, activations)
├── test_core.py         # Tests for prediction pipeline
├── test_decode.py       # Tests for decoders
├── test_filter.py       # Tests for filters
├── test_threshold.py    # Tests for thresholding
└── assets/              # Test data (audio, reference activations)
```

## Important Implementation Details

### Model Caching
The `infer()` function caches the loaded model as a function attribute:
- Only loads model when capacity changes or first call
- Automatically moves model to requested device (no-op if same)
- Model remains in eval mode throughout

### Batch Processing
For very long audio files:
- `preprocess()` is a generator that yields batches of frames
- Default batch size processes entire file in one batch
- Set `batch_size` parameter to limit GPU memory usage
- Results are concatenated across batches

### Numerical Stability
- Preprocessing normalization can be numerically unstable for silent frames
- During silent frames, std deviation → 0, causing very large normalized values
- The model appears to handle this correctly (as trained)
- Test fixtures load pre-normalized frames to avoid discrepancies

### Audio Format Support
- Uses `torchaudio.load()` for audio loading (supports WAV, MP3, FLAC, etc.)
- All audio internally resampled to 16kHz for model compatibility
- Hop length automatically adjusted to maintain temporal alignment

### Deprecation Notes
- `return_harmonicity` renamed to `return_periodicity` (more accurate term)
- Old argument names still work but emit deprecation warnings
- Rationale: "harmonicity" implies low values for non-harmonic periodic sounds (like sine waves), but this isn't observed

## Common Development Patterns

### Adding a New Decoder
1. Add function to `decode.py` with signature: `decoder(logits) -> (bins, pitch)`
2. Input: `logits` shape `(batch, 360, time/hop_length)`
3. Output: `bins` (integer indices), `pitch` (frequency in Hz)
4. Register in `__main__.py` argument parser if exposing to CLI

### Adding New Filters
1. Add function to `filter.py` with signature: `filter(signals, win_length) -> filtered`
2. Handle NaN values appropriately (use `nanmean`/`nanmedian` utilities)
3. Ensure input/output shapes match: `(batch, time)`

### Working with Embeddings
```python
# Extract embeddings from layer 5
embeddings = torchcrepe.embed(audio, sr, hop_length)
# Shape: (batch, time/hop_length, 32, embedding_size)
# embedding_size = 64 for tiny, 256 for full

# Use for downstream tasks (e.g., DDSP, voice conversion)
```

### Testing Against Original CREPE
- Test fixtures include reference activations from original TensorFlow CREPE
- Use `frames-crepe.npy` (preprocessed frames) to bypass normalization differences
- Compare model outputs to `activation-full.npy` and `activation-tiny.npy`
