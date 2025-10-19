# Suggested Commands for Development

## Testing

### Run all tests
```bash
pytest
# or
python -m pytest
```

### Run specific test file
```bash
pytest tests/test_core.py
pytest tests/test_decode.py
pytest tests/test_filter.py
pytest tests/test_threshold.py
```

### Run specific test function
```bash
pytest tests/test_core.py::test_infer_full
pytest tests/test_decode.py::test_viterbi
```

### Run with verbose output
```bash
pytest -v
pytest -vv  # extra verbose
```

## Installation

### Install package in editable mode (for development)
```bash
pip install -e .
```

### Install with dependencies
```bash
pip install torch torchaudio  # Install PyTorch first
pip install -e .
```

### Install testing dependencies
```bash
pip install pytest
```

## Running the Application

### Command-line interface
```bash
# Basic pitch prediction
python -m torchcrepe --audio_files input.wav --output_files pitch.pt

# With periodicity output
python -m torchcrepe --audio_files input.wav --output_files pitch.pt \
    --output_periodicity_files periodicity.pt

# Using GPU
python -m torchcrepe --audio_files input.wav --output_files pitch.pt --gpu 0

# Using tiny model
python -m torchcrepe --audio_files input.wav --output_files pitch.pt --model tiny

# Extract embeddings
python -m torchcrepe --audio_files input.wav --output_files embed.pt --embed

# Batch processing
python -m torchcrepe --audio_files file1.wav file2.wav \
    --output_files pitch1.pt pitch2.pt
```

### Python API usage
```python
import torchcrepe

# Load audio
audio, sr = torchcrepe.load.audio('input.wav')

# Predict pitch
pitch = torchcrepe.predict(
    audio, sr,
    hop_length=160,
    fmin=50, fmax=550,
    model='full',
    device='cuda:0'
)

# With periodicity
pitch, periodicity = torchcrepe.predict(
    audio, sr,
    hop_length=160,
    return_periodicity=True
)

# Extract embeddings
embeddings = torchcrepe.embed(audio, sr, hop_length=160)
```

## Git Commands (Windows Git Bash)

### Basic workflow
```bash
git status
git add .
git commit -m "message"
git push
```

### Branching
```bash
git checkout -b feature-branch
git checkout master
git merge feature-branch
```

## System Utilities (Windows Git Bash)

### File operations
```bash
ls -la                  # List files
find . -name "*.py"     # Find Python files
grep -r "pattern" .     # Search in files
cat file.py             # Display file contents
```

### Directory operations
```bash
cd /c/dev/torchcrepe_plus
pwd
mkdir new_dir
rm -rf directory
```

## Package Management

### Check installed versions
```bash
python -c "import torch; print(torch.__version__)"
python -c "import librosa; print(librosa.__version__)"
pip list | grep torch
```

### Update dependencies
```bash
pip install --upgrade torch torchaudio
pip install --upgrade librosa
```

## Development Workflow

When completing a task:
1. **Run tests**: `pytest` (ensure all tests pass)
2. **Manual testing**: Test CLI or API with sample audio
3. **Commit changes**: `git add . && git commit -m "message"`

Note: This project does **not** use linters or formatters (no flake8, black, etc.).
Maintain consistency with existing code style manually.
