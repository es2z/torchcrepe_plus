# torchcrepe Project Overview

## Project Purpose
torchcrepe is a PyTorch implementation of the CREPE pitch tracking algorithm. CREPE (Convolutional Representation for Pitch Estimation) is a deep learning approach to fundamental frequency (F0) estimation from audio signals.

**Key Features:**
- GPU-accelerated pitch tracking using pretrained CREPE models
- Two model sizes: "tiny" (fast, lightweight) and "full" (accurate, larger)
- Multiple decoding methods: Viterbi (default), weighted argmax, argmax
- Pitch embedding extraction for downstream tasks (DDSP, voice conversion, etc.)
- Post-processing utilities: filtering and thresholding
- Batch processing with automatic model caching

## Tech Stack

### Core Dependencies
- **Python**: 3.8 - 3.11 (tested on CI)
- **PyTorch**: Main deep learning framework
- **torchaudio**: Audio loading
- **librosa**: 0.9.1 - 0.10.1 (STFT, Viterbi decoding, A-weighting)
- **resampy**: Audio resampling to 16kHz
- **scipy**: Statistical utilities (dithering)
- **numpy**: Numerical operations
- **tqdm**: Progress bars for batch processing

### Development Dependencies
- **pytest**: Testing framework

## Platform Information
- **Development System**: Windows (Git Bash environment)
- **CI Platform**: Ubuntu Linux (GitHub Actions)
- **Multi-platform Support**: Cross-platform (PyTorch-based)
