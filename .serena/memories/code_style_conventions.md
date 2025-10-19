# Code Style and Conventions

## General Style

This project does **not** use automated formatters or linters (no black, flake8, pylint configs found).
Code style is maintained through consistency with existing code.

## Naming Conventions

- **Functions**: `snake_case` (e.g., `predict_from_file`, `a_weighted`)
- **Classes**: `PascalCase` (e.g., `Crepe`, `Hysteresis`)
- **Constants**: `UPPER_CASE` (e.g., `SAMPLE_RATE`, `PITCH_BINS`, `MAX_FMAX`)
- **Private variables**: No leading underscore (small project, minimal encapsulation)
- **Module names**: `snake_case` (e.g., `core.py`, `filter.py`)

## Docstring Style

Uses **simple docstring format** with sections for Arguments and Returns:

```python
def function(arg1, arg2):
    """Brief description of what the function does

    Arguments
        arg1 (type [shape=(...)])
            Description of arg1
        arg2 (type)
            Description of arg2

    Returns
        result (type [shape=(...)])
            Description of return value
    """
```

**Key characteristics:**
- No Google/NumPy-style section headers (no colons)
- Shape information in brackets for tensors: `[shape=(batch, time)]`
- Optional parameters noted with `(Optional)` prefix in Returns section
- Simple indentation (4 spaces)

## Type Hints

**Not widely used** in this codebase. Types are documented in docstrings instead.

Example:
```python
# No type hints:
def predict(audio, sample_rate, hop_length=None):
    """...
    Arguments
        audio (torch.tensor [shape=(1, time)])
        sample_rate (int)
        hop_length (int)
    """
```

## Code Organization

### Section Headers
Use triple-hash comment blocks to separate major sections:

```python
###############################################################################
# Section Title
###############################################################################
```

Common section names:
- `# Constants`
- `# Entry point`
- `# Forward pass utilities`
- `# Testing fixtures`

### Import Order
1. Standard library (e.g., `import os`, `import warnings`)
2. Third-party packages (e.g., `import torch`, `import numpy`)
3. Local imports (e.g., `import torchcrepe`)

Blank line between groups. No alphabetical sorting enforced.

### Module Structure
```python
# Imports

# Constants (if any)

# Public API functions/classes

# Private utilities (if any)
```

## Tensor Conventions

- **Device handling**: Store device, move to CPU for numpy ops, restore device
- **Batch dimension**: Always include (shape=(batch, time) even if batch=1)
- **Detach before numpy**: `tensor.detach().cpu().numpy()`
- **NaN for unvoiced**: Use `np.nan` (not 0 or -1)

Example:
```python
# Save device
device = audio.device

# Convert to numpy
audio = audio.detach().cpu().numpy()

# ... process ...

# Restore device
return torch.tensor(result, device=device)
```

## Testing Conventions

- Test functions start with `test_`
- Use pytest fixtures from `conftest.py`
- Assert with tolerance for numerical comparisons: `assert diff.max() < 1e-5`
- Docstrings describe what is being tested

Example:
```python
def test_infer_tiny(frames, activation_tiny):
    """Test that inference is the same as the original crepe"""
    activation = torchcrepe.infer(frames, 'tiny').detach()
    diff = np.abs(activation - activation_tiny)
    assert diff.max() < 1e-5 and diff.mean() < 1e-7
```

## Common Patterns

### Function Attributes for Caching
```python
def function(...):
    if not hasattr(function, 'cached_value'):
        function.cached_value = expensive_computation()
    return function.cached_value
```

### Generator for Batching
```python
def preprocess(...):
    for i in range(0, total, batch_size):
        # ... prepare batch ...
        yield batch
```

### Deprecation Warnings
```python
if deprecated_arg:
    message = 'The X argument is deprecated...'
    warnings.warn(message, DeprecationWarning)
    new_arg = deprecated_arg
```
