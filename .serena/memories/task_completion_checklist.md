# Task Completion Checklist

When completing a development task in torchcrepe, follow these steps:

## 1. Testing

### Run the test suite
```bash
pytest
```

**What to check:**
- All tests pass
- No new warnings introduced
- If you modified core functionality, verify related tests pass:
  - Modified `core.py` → `pytest tests/test_core.py`
  - Modified `decode.py` → `pytest tests/test_decode.py`
  - Modified `filter.py` → `pytest tests/test_filter.py`
  - Modified `threshold.py` → `pytest tests/test_threshold.py`

### Add tests for new functionality
If you added new features:
- Add test cases to appropriate test file
- Include pytest fixtures from `conftest.py` if needed
- Use numerical tolerance for tensor comparisons: `assert diff.max() < 1e-5`

## 2. Manual Verification

### For core algorithm changes
Test with sample audio using both API and CLI:

```bash
# Test CLI
python -m torchcrepe --audio_files test.wav --output_files pitch.pt

# Test in Python
python -c "
import torchcrepe
audio, sr = torchcrepe.load.audio('test.wav')
pitch = torchcrepe.predict(audio, sr, hop_length=160)
print('Pitch shape:', pitch.shape)
"
```

### For model changes
Verify both model sizes work:
```bash
pytest tests/test_core.py::test_infer_tiny
pytest tests/test_core.py::test_infer_full
```

## 3. Documentation

### Update docstrings
- Ensure new functions have docstrings following project style
- Document arguments with types and shapes
- Document return values

### Update README.md (if needed)
- New API functions → add usage examples
- New CLI options → update command-line section
- Breaking changes → add migration notes

### Update CLAUDE.md (if needed)
- Architectural changes → update architecture section
- New patterns → add to common development patterns

## 4. Code Quality

### Manual style check
- Use `snake_case` for functions
- Use `PascalCase` for classes
- Use `UPPER_CASE` for constants
- Follow existing import order
- Add section headers for major code blocks

### Consistency check
- Match indentation (4 spaces)
- Match docstring style
- Follow existing tensor conventions (device handling, batch dims)

## 5. Version Control

### Commit changes
```bash
git status                    # Review changes
git add <files>               # Stage specific files
git commit -m "message"       # Commit with descriptive message
```

**Commit message guidelines:**
- Use present tense ("Add feature" not "Added feature")
- Be descriptive but concise
- Reference issues if applicable

### Before pushing
```bash
git log -1                    # Review last commit
pytest                        # Final test run
```

## Important Notes

- **No linting/formatting required**: This project does not use automated formatters
- **No type checking**: Type hints are not used; document types in docstrings
- **CI will run**: GitHub Actions tests against Python 3.8-3.11 and librosa 0.9.1-0.10.1
- **Backward compatibility**: Avoid breaking changes to public API
- **Deprecation**: Use warnings.warn() for deprecated features, keep old behavior working

## Quick Checklist

- [ ] Tests pass (`pytest`)
- [ ] Manual testing completed (if applicable)
- [ ] Docstrings added/updated
- [ ] Code style matches existing code
- [ ] README.md updated (if needed)
- [ ] Changes committed with descriptive message
