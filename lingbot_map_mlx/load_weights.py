"""Load converted PyTorch weights into MLX GCTStream model.

Handles key name mapping between PyTorch state dict and MLX module tree.
"""

import numpy as np
import mlx.core as mx


def load_weights(model, weights_path: str, verbose: bool = True):
    """Load weights from converted .safetensors or .npz into an MLX GCTStream model."""

    if weights_path.endswith(".safetensors"):
        weights = mx.load(weights_path)
    else:
        data = np.load(weights_path, allow_pickle=False)
        weights = {k: mx.array(data[k]) for k in data.files}

    # Build key mapping: PyTorch key -> MLX attribute path
    mapped = {}
    unmapped = []

    for pt_key, value in weights.items():
        mlx_key = _map_key(pt_key)
        if mlx_key is not None:
            mapped[mlx_key] = value
        else:
            unmapped.append(pt_key)

    if verbose:
        print(f"Mapped {len(mapped)}/{len(weights)} weight keys")
        if unmapped:
            print(f"Unmapped: {len(unmapped)} keys")
            for k in unmapped[:10]:
                print(f"  {k}")

    # Load into model using MLX's load_weights
    # Convert flat dict to nested list format expected by MLX
    model.load_weights(list(mapped.items()), strict=False)

    if verbose:
        print("Weights loaded successfully")

    return len(mapped), unmapped


def _map_key(pt_key: str) -> str:
    """Map PyTorch state dict key to MLX module attribute path."""
    k = pt_key

    # camera_head.poseLN_modulation.0 -> SiLU (no weights)
    # camera_head.poseLN_modulation.1 -> poseLN_modulation_1
    k = k.replace("poseLN_modulation.1.", "poseLN_modulation_1.")
    # Skip SiLU (index 0) - has no weights
    if "poseLN_modulation.0." in k:
        return None

    # scratch nested module -> flat attributes
    # depth_head.scratch.layer1_rn -> depth_head.scratch_layer1_rn
    k = k.replace("depth_head.scratch.", "depth_head.scratch_")

    # output_conv2 is Sequential in PyTorch: .0 = Conv, .1 = ReLU, .2 = Conv
    # In MLX: scratch_output_conv2_0 and scratch_output_conv2_1
    k = k.replace("scratch_output_conv2.0.", "scratch_output_conv2_0.")
    k = k.replace("scratch_output_conv2.2.", "scratch_output_conv2_1.")

    # skip_add (FloatFunctional) has no weights
    if "skip_add" in k:
        return None

    # patch_embed is DinoVisionTransformer in MLX
    # aggregator.patch_embed.patch_embed.proj -> aggregator.patch_embed.patch_embed.proj
    # (this maps correctly since MLX DinoVisionTransformer also has patch_embed.proj)

    # Sequential trunk in camera_head -> list
    # camera_head.trunk.0.xxx -> camera_head.trunk.0.xxx (lists work in MLX)

    # norm1/norm2 in blocks have weight+bias

    # Global blocks in PyTorch might be FlashInferBlock or SDPABlock
    # Both have same param names as our SDPABlock

    return k
