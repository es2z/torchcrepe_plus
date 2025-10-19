# Technical Details and Important Notes

## CREPE Model Specifications

### Architecture
- **6-layer convolutional neural network**
- Input: 1024-sample audio frames (16kHz)
- Output: 360-bin probability distribution

### Model Variants
1. **Full model** (`model='full'`):
   - Channels: 1024 → 128 → 128 → 128 → 256 → 512
   - Features before classifier: 2048
   - Embedding size: 64 (at layer 5)
   - More accurate but slower

2. **Tiny model** (`model='tiny'`):
   - Channels: 128 → 16 → 16 → 16 → 32 → 64
   - Features before classifier: 256
   - Embedding size: 8 (at layer 5)
   - Faster but less accurate

### Layer Structure
Each layer: Conv2D → ReLU → BatchNorm2D → MaxPool2D
- Layer 1: kernel=(512, 1), stride=(4, 1), padding=(0, 0, 254, 254)
- Layers 2-6: kernel=(64, 1), stride=(1, 1), padding=(0, 0, 31, 32)
- All max pooling: kernel=(2, 1), stride=(2, 1)

## Pitch Representation

### Frequency Range
- **Maximum frequency**: 2006 Hz (`MAX_FMAX`)
- **Typical range**: 50-550 Hz (speech), 50-2006 Hz (music)
- **Total bins**: 360 (`PITCH_BINS`)

### Resolution
- **Cents per bin**: 20 (`CENTS_PER_BIN`)
- 1 semitone = 100 cents
- Resolution = 1/5 semitone (very fine)

### Conversion Formulas
- **Bins to frequency**: `f = 10 * 2^(cents/1200)` where `cents = 20 * bins + 1997.3794084376191`
- **Frequency to bins**: Inverse of above with quantization
- **Dithering**: Triangular noise added to remove quantization artifacts

### Unvoiced Frames
Represented as `np.nan` (not 0 or -1) to distinguish from low frequencies

## Audio Processing

### Resampling
- **Target sample rate**: 16kHz (CREPE's training rate)
- **Library**: `resampy` (matches original CREPE exactly)
- **Automatic**: All audio resampled internally regardless of input SR

### Windowing
- **Window size**: 1024 samples @ 16kHz (64ms)
- **Default hop length**: 160 samples @ 16kHz (10ms)
- **Padding**: Zero-padding by WINDOW_SIZE/2 on both sides (default)
  - Can disable with `pad=False`

### Normalization (per frame)
1. Mean-center: `frames -= frames.mean()`
2. Scale by std: `frames /= max(1e-10, frames.std())`

**Note**: During silent frames, std → 0, producing very large normalized values.
The model handles this correctly (as trained on similar data).

## Model Caching

The `infer()` function uses **function attributes** for caching:
```python
if not hasattr(infer, 'model') or infer.capacity != model:
    torchcrepe.load.model(device, model)
```

**Behavior**:
- Loads model only on first call or capacity change
- Automatically moves to requested device (no-op if same device)
- Model stays in `eval()` mode
- Cache persists across calls in same Python session

**Implications**:
- Efficient for batch processing multiple files
- No need to manually preload model
- Device switching is handled automatically

## Decoding Algorithms

### Viterbi (Default, Recommended)
- Uses librosa's `sequence.viterbi()`
- Transition matrix: Gaussian-like, penalizes jumps >12 semitones
- **Pros**: Smooths predictions, reduces octave errors
- **Cons**: Slightly slower than argmax
- **Best for**: Most applications (speech, music)

### Weighted Argmax (Original CREPE)
- Weighted average of 9 bins around peak
- Uses sigmoid probabilities (not softmax)
- **Pros**: Matches original TensorFlow implementation
- **Cons**: More octave/half-octave errors
- **Best for**: Replicating original CREPE results

### Argmax
- Simple maximum probability
- **Pros**: Fastest
- **Cons**: Most prone to errors, produces quantization
- **Best for**: Debugging, extremely noisy signals

## Periodicity (Confidence)

**Definition**: Maximum probability at the selected pitch bin

**Not harmonicity**: Despite earlier naming, this measures periodicity, not harmonic content.
Sine waves (non-harmonic but periodic) show high values.

**Typical values**:
- >0.5: High confidence (clear pitch)
- 0.2-0.5: Moderate confidence
- <0.2: Low confidence (probably unvoiced)

**Thresholding**: 0.21 is recommended for clean speech (tune for your data)

## Batch Processing

### Memory Management
- `preprocess()` is a **generator** yielding frame batches
- Default: Process all frames in one batch
- Set `batch_size` to limit GPU memory usage
- Results concatenated across batches

### Device Handling
```python
# Frames sent to inference device
frames = frames.to(device)

# Results moved back to audio device
result = result.to(audio.device)
```

**Why**: Allows very long audio files where audio stays on CPU/disk, 
only frames move to GPU temporarily.

## Embeddings

### Extraction Point
- Stop inference at **layer 5 output** (before final conv and classifier)
- Activation after 5th max pooling layer

### Shape
- **Full model**: `(batch, time/hop_length, 32, 64)`
- **Tiny model**: `(batch, time/hop_length, 32, 8)`
- 32 frequency bands, variable-length embedding per band

### Use Cases
- DDSP (Differentiable Digital Signal Processing)
- Voice conversion
- Pitch-aware audio embeddings
- Transfer learning for pitch-related tasks

## Testing Against Original CREPE

### Test Fixtures
- `activation-full.npy`: Reference output from TensorFlow CREPE (full model)
- `activation-tiny.npy`: Reference output from TensorFlow CREPE (tiny model)
- `frames-crepe.npy`: Pre-normalized frames from original CREPE preprocessing

### Why Pre-normalized Frames?
Normalization is numerically unstable (dividing by std). Using exact same 
preprocessed frames ensures model output comparison is fair.

### Tolerance
- Max difference: < 1e-5
- Mean difference: < 1e-7

This tight tolerance ensures PyTorch implementation matches original exactly.

## Model Weights

### Source
Converted from original TensorFlow CREPE models using MMdnn

### Location
- `torchcrepe/assets/full.pth`
- `torchcrepe/assets/tiny.pth`

### Loading
```python
torch.load(file, map_location=device, weights_only=True)
```

**`weights_only=True`**: Security feature (PyTorch 2.0+) to prevent arbitrary code execution

## Windows Development Notes

- **Git Bash environment**: Unix commands available (ls, grep, find, etc.)
- **File paths**: Use forward slashes in Python, work cross-platform
- **CI runs on Linux**: GitHub Actions uses Ubuntu, but code is platform-agnostic
- **PyTorch**: Same CUDA/CPU backend on Windows and Linux
