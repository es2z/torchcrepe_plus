#!/usr/bin/env python
"""Convert the published CREPE-speech Keras weights to torchcrepe format.

Source repository: https://github.com/ardaillon/FCN-f0
Source commit: 8a2b530af821319b6badca93c8a0ed1f14bfee3c
Source path: models/CREPE-speech/weights.h5
"""

import argparse
import hashlib
from pathlib import Path

import h5py
import numpy as np
import torch

from torchcrepe.model import Crepe


SOURCE_SHA256 = (
    "2218880e139f48682c150faf97050befa0735bc848d9d05c2937506315d8d922"
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="FCN-f0 CREPE-speech weights.h5")
    parser.add_argument("output", type=Path, help="Destination full_speech.pth")
    return parser.parse_args()


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_layer_group(file, layer_name):
    """Find a layer group in either a model or weights-only Keras HDF5 file."""
    candidates = [layer_name, f"model_weights/{layer_name}"]
    for candidate in candidates:
        if candidate in file:
            return file[candidate]
    raise KeyError(f"Layer {layer_name!r} was not found in the HDF5 file")


def layer_weights(file, layer_name):
    weights = {}

    def collect(name, item):
        if isinstance(item, h5py.Dataset):
            key = name.rsplit("/", 1)[-1].split(":", 1)[0]
            if key in weights:
                raise ValueError(f"Duplicate {key!r} tensor in layer {layer_name!r}")
            weights[key] = item[()]

    find_layer_group(file, layer_name).visititems(collect)
    return weights


def tensor(array):
    return torch.from_numpy(np.ascontiguousarray(array))


def convert(input_path):
    state_dict = {}
    with h5py.File(input_path, "r") as file:
        for index in range(1, 7):
            conv = layer_weights(file, f"conv{index}")
            state_dict[f"conv{index}.weight"] = tensor(
                np.transpose(conv["kernel"], (3, 2, 0, 1))
            )
            state_dict[f"conv{index}.bias"] = tensor(conv["bias"])

            batch_norm = layer_weights(file, f"conv{index}-BN")
            prefix = f"conv{index}_BN"
            state_dict[f"{prefix}.weight"] = tensor(batch_norm["gamma"])
            state_dict[f"{prefix}.bias"] = tensor(batch_norm["beta"])
            state_dict[f"{prefix}.running_mean"] = tensor(
                batch_norm["moving_mean"]
            )
            state_dict[f"{prefix}.running_var"] = tensor(
                batch_norm["moving_variance"]
            )
            state_dict[f"{prefix}.num_batches_tracked"] = torch.tensor(
                0, dtype=torch.long
            )

        classifier = layer_weights(file, "classifier")
        state_dict["classifier.weight"] = tensor(
            np.transpose(classifier["kernel"], (1, 0))
        )
        state_dict["classifier.bias"] = tensor(classifier["bias"])

    return state_dict


def main():
    args = parse_args()
    source_hash = sha256(args.input)
    if source_hash != SOURCE_SHA256:
        raise ValueError(
            "Unexpected CREPE-speech source weights: "
            f"expected SHA-256 {SOURCE_SHA256}, got {source_hash}"
        )

    state_dict = convert(args.input)
    Crepe("full").load_state_dict(state_dict, strict=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state_dict, args.output)
    print(f"Saved {len(state_dict)} tensors to {args.output}")


if __name__ == "__main__":
    main()
