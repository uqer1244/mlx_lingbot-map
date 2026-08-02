#!/usr/bin/env python3
import os
import sys
import time
import torch
import numpy as np
import mlx.core as mx

def convert_and_compare(pt_path, mlx_path):
    print("=" * 70)
    print("1. Loading PyTorch weights...")
    print("=" * 70)
    t0 = time.time()
    ckpt = torch.load(pt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("model", ckpt)
    print(f"PyTorch weights loaded successfully! ({len(state_dict)} tensors/layers, {time.time() - t0:.2f}s)")

    print("\n" + "=" * 70)
    print("2. Converting PyTorch -> MLX weights (4D Conv Weight Transpose)...")
    print("=" * 70)
    t1 = time.time()
    mlx_dict = {}
    conv_count = 0
    other_count = 0

    for k, v in state_dict.items():
        v_np = v.detach().cpu().numpy()
        # 4D Conv Weight: PyTorch [out_channels, in_channels, kh, kw] -> MLX [out_channels, kh, kw, in_channels]
        if v.ndim == 4:
            v_np = np.transpose(v_np, (0, 2, 3, 1))
            conv_count += 1
        else:
            other_count += 1
        mlx_dict[k] = mx.array(v_np)

    print(f"Conversion completed! (4D Conv layers: {conv_count}, Other layers: {other_count}, Elapsed: {time.time() - t1:.2f}s)")

    print("\n" + "=" * 70)
    print(f"3. Saving MLX safetensors to {mlx_path}...")
    print("=" * 70)
    t2 = time.time()
    os.makedirs(os.path.dirname(mlx_path), exist_ok=True)
    mx.save_safetensors(mlx_path, mlx_dict)
    file_size_mb = os.path.getsize(mlx_path) / (1024 * 1024)
    print(f"Saved safetensors! (Size: {file_size_mb:.2f} MB, Elapsed: {time.time() - t2:.2f}s)")

    print("\n" + "=" * 70)
    print("4. Re-loading saved MLX safetensors for numerical validation against PyTorch...")
    print("=" * 70)
    t3 = time.time()
    loaded_mlx_dict = mx.load(mlx_path)
    
    total_tensors = len(state_dict)
    matched_keys = 0
    matched_shapes = 0
    max_diffs = []
    cos_sims = []

    print(f"{'Key Name (Sample)':<55} | {'PyTorch Shape':<18} | {'MLX Shape':<18} | {'Max Abs Diff':<12}")
    print("-" * 115)

    sample_keys = list(state_dict.keys())
    # Display 10 representative layers
    display_indices = np.linspace(0, total_tensors - 1, 10, dtype=int)

    for i, (k, pt_v) in enumerate(state_dict.items()):
        if k not in loaded_mlx_dict:
            print(f"[ERROR] Missing key: {k}")
            continue
        matched_keys += 1

        mlx_arr = loaded_mlx_dict[k]
        pt_np = pt_v.detach().cpu().numpy()

        if pt_v.ndim == 4:
            expected_shape = (pt_np.shape[0], pt_np.shape[2], pt_np.shape[3], pt_np.shape[1])
            expected_np = np.transpose(pt_np, (0, 2, 3, 1))
        else:
            expected_shape = pt_np.shape
            expected_np = pt_np

        if tuple(mlx_arr.shape) == expected_shape:
            matched_shapes += 1

        # Numerical accuracy check
        mlx_np = np.array(mlx_arr)
        abs_diff = np.max(np.abs(mlx_np - expected_np))
        max_diffs.append(abs_diff)

        # Cosine similarity
        pt_flat = expected_np.flatten().astype(np.float64)
        mlx_flat = mlx_np.flatten().astype(np.float64)
        norm_pt = np.linalg.norm(pt_flat)
        norm_mlx = np.linalg.norm(mlx_flat)
        if norm_pt > 0 and norm_mlx > 0:
            cos_sim = np.dot(pt_flat, mlx_flat) / (norm_pt * norm_mlx)
            cos_sims.append(cos_sim)

        if i in display_indices:
            key_disp = (k[:52] + "...") if len(k) > 55 else k
            print(f"{key_disp:<55} | {str(tuple(pt_v.shape)):<18} | {str(tuple(mlx_arr.shape)):<18} | {abs_diff:.2e}")

    overall_max_diff = max(max_diffs) if max_diffs else -1
    avg_cos_sim = np.mean(cos_sims) if cos_sims else -1

    print("=" * 115)
    print("Verification Summary")
    print(f" - Total layers: {total_tensors}")
    print(f" - Key match rate: {matched_keys}/{total_tensors} ({matched_keys/total_tensors*100:.1f}%)")
    print(f" - Shape match rate: {matched_shapes}/{total_tensors} ({matched_shapes/total_tensors*100:.1f}%)")
    print(f" - Max absolute difference: {overall_max_diff:.2e}")
    print(f" - Average cosine similarity: {avg_cos_sim:.8f}")
    print("=" * 115)

    if matched_keys == total_tensors and matched_shapes == total_tensors and overall_max_diff < 1e-5:
        print("[SUCCESS] MLX converted weights are 100% identical to original PyTorch weights.")
    else:
        print("[WARNING] Differences detected during weight verification. Please check log above.")

if __name__ == "__main__":
    pt_path = "checkpoints/lingbot-map.pt"
    mlx_path = "checkpoints/lingbot-map-mlx.safetensors"
    convert_and_compare(pt_path, mlx_path)
