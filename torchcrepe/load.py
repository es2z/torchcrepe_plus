import os

import numpy as np
import torch
import torchaudio
from scipy.io import wavfile

import torchcrepe


def audio(filename):
    """Load audio from disk"""
    return torchaudio.load(filename)


def model(device, capacity='full', compile_model=False, compile_mode='default'):
    """Preloads model from disk

    Arguments
        device: The device to load the model on
        capacity: One of 'tiny', 'small', 'medium', 'large', 'full'
        compile_model: Whether to use torch.compile for faster inference
                       (requires PyTorch 2.0+, recommended for batch processing)
        compile_mode: The torch.compile mode to use when compile_model is True.
                      One of 'default', 'reduce-overhead', or 'max-autotune'.
                      Default is 'default'.
    """
    # Bind model and capacity
    torchcrepe.infer.capacity = capacity
    torchcrepe.infer.model = torchcrepe.Crepe(capacity)

    # Load weights
    file = os.path.join(os.path.dirname(__file__), 'assets', f'{capacity}.pth')
    torchcrepe.infer.model.load_state_dict(
        torch.load(file, map_location=device, weights_only=True))

    # Place on device
    torchcrepe.infer.model = torchcrepe.infer.model.to(torch.device(device))

    # Eval mode
    torchcrepe.infer.model.eval()

    # Optional: compile model for faster inference (PyTorch 2.0+)
    if compile_model and hasattr(torch, 'compile'):
        torchcrepe.infer.model = torch.compile(
            torchcrepe.infer.model,
            dynamic=True,
            mode=compile_mode
        )
