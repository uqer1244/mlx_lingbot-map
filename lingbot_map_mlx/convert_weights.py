"""Convert PyTorch lingbot-map checkpoint to MLX format.

Usage:
    python -m lingbot_map_mlx.convert_weights --input checkpoint.pt --output weights.npz
"""

import argparse
import re
from pathlib import Path

import numpy as np


def _is_conv_weight(key: str, shape: tuple) -> bool:
    """Detect Conv2d/ConvTranspose2d weight tensors (4D with OIHW layout)."""
    if len(shape) != 4:
        return False
    return bool(re.search(r'\.(proj|conv[0-9]*|head|scratch|refinenet|output_conv)', key)
                and key.endswith('.weight'))


def convert_pytorch_to_mlx(input_path: str, output_path: str):
    """Load a PyTorch checkpoint and save as MLX-compatible .npz."""
    import torch

    print(f"Loading PyTorch checkpoint: {input_path}")
    ckpt = torch.load(input_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("model", ckpt)

    mlx_weights = {}
    conv_transposed = 0
    total = 0

    for key, tensor in state_dict.items():
        arr = tensor.float().numpy()
        total += 1

        if _is_conv_weight(key, arr.shape):
            arr = np.transpose(arr, (0, 2, 3, 1))  # OIHW -> OHWI
            conv_transposed += 1

        mlx_weights[key] = arr

    print(f"Converted {total} tensors ({conv_transposed} conv weights transposed OIHW→OHWI)")
    np.savez(output_path, **mlx_weights)
    print(f"Saved to: {output_path}")

    return mlx_weights


def load_mlx_weights(path: str) -> dict:
    """Load converted weights as a flat dict of mx.array."""
    import mlx.core as mx

    data = np.load(path, allow_pickle=False)
    return {k: mx.array(data[k]) for k in data.files}


def load_weights_into_model(model, weights: dict, strict: bool = False):
    """Load a flat state dict into an MLX nn.Module tree.

    Maps dot-separated keys (e.g. 'aggregator.frame_blocks.0.norm1.weight')
    into nested module attribute access.
    """
    import mlx.core as mx

    loaded, skipped = 0, []

    # Build a map from the model's current parameter tree
    model_params = dict(_flatten_params(model, prefix=""))

    for key, value in weights.items():
        if key in model_params:
            _set_nested_attr(model, key, mx.array(value) if not isinstance(value, mx.array) else value)
            loaded += 1
        else:
            skipped.append(key)

    print(f"Loaded {loaded}/{len(weights)} parameters")
    if skipped and strict:
        print(f"Skipped {len(skipped)} keys (not in model)")
        for k in skipped[:10]:
            print(f"  {k}")
        if len(skipped) > 10:
            print(f"  ... and {len(skipped) - 10} more")

    return loaded, skipped


def _flatten_params(module, prefix=""):
    """Recursively flatten an MLX module's parameters into dot-separated keys."""
    import mlx.core as mx
    import mlx.nn as nn

    if isinstance(module, nn.Module):
        for k, v in module.items():
            full_key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, mx.array):
                yield full_key, v
            elif isinstance(v, nn.Module):
                yield from _flatten_params(v, full_key)
            elif isinstance(v, (list, tuple)):
                for i, item in enumerate(v):
                    yield from _flatten_params(item, f"{full_key}.{i}")
            elif isinstance(v, dict):
                for dk, dv in v.items():
                    yield from _flatten_params(dv, f"{full_key}.{dk}")


def _set_nested_attr(module, key: str, value):
    """Set a nested attribute on an MLX module using dot-separated key."""
    parts = key.split(".")
    obj = module
    for part in parts[:-1]:
        if isinstance(obj, (list, tuple)):
            obj = obj[int(part)]
        elif isinstance(obj, dict):
            obj = obj[part]
        else:
            obj = getattr(obj, part, None) or obj[part]
    final = parts[-1]
    if isinstance(obj, dict):
        obj[final] = value
    else:
        setattr(obj, final, value)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert PyTorch checkpoint to MLX format")
    parser.add_argument("--input", type=str, required=True, help="Path to PyTorch .pt checkpoint")
    parser.add_argument("--output", type=str, default=None, help="Output .npz path (default: same name with .npz)")
    args = parser.parse_args()

    output = args.output or str(Path(args.input).with_suffix(".npz"))
    convert_pytorch_to_mlx(args.input, output)
