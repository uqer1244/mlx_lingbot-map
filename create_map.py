#!/usr/bin/env python3
"""
Create Map — MLX LingBot-MAP 3D Reconstruction Builder

Runs MLX GPU inference, constructs 3D point cloud & camera parameters,
saves the map to disk (.npz / .ply), and exits cleanly to free memory.

Usage:
    python create_map.py --image_folder example/loop --out_map checkpoints/loop_map.npz
"""

import argparse
import glob
import os
import sys
import time
import cv2
import numpy as np
import mlx.core as mx
from PIL import Image, ImageOps
from tqdm.auto import tqdm

# Local package imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from lingbot_map_mlx.models.gct_stream import GCTStream
from lingbot_map_mlx.load_weights import load_weights
from lingbot_map_mlx.utils.pose_enc import pose_encoding_to_extri_intri
from lingbot_map_mlx.utils.geometry import closed_form_inverse_se3_general, unproject_depth_map_to_point_map
from lingbot_map_mlx.utils.exporter import export_point_cloud_map, save_npz


def load_images(image_folder=None, video_path=None, fps=None, max_frames=None, stride=1, image_size=518, rotate_cw90=False):
    paths = []
    if image_folder:
        exts = [".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG"]
        for ext in exts:
            paths.extend(glob.glob(os.path.join(image_folder, "*" + ext)))
        paths = sorted(list(set(paths)))
        if not paths:
            raise ValueError(f"No images found in {image_folder}")
    elif video_path:
        cap = cv2.VideoCapture(video_path)
        fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0
        step = max(1, int(round(fps_in / fps))) if fps else 1
        frame_idx = 0
        temp_dir = "temp_mlx_frames"
        os.makedirs(temp_dir, exist_ok=True)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % step == 0:
                p = os.path.join(temp_dir, f"{len(paths):06d}.png")
                cv2.imwrite(p, frame)
                paths.append(p)
            frame_idx += 1
        cap.release()

    if stride > 1:
        paths = paths[::stride]
    if max_frames and max_frames > 0:
        paths = paths[:max_frames]

    print(f"Loading {len(paths)} images for MLX inference...")
    images_list = []
    patch_size = 14

    sample = Image.open(paths[0])
    sample = ImageOps.exif_transpose(sample)
    w_sample, h_sample = sample.size
    if rotate_cw90 or h_sample > w_sample:
        w_sample, h_sample = h_sample, w_sample
    scale = image_size / max(w_sample, h_sample)
    new_w = (int(w_sample * scale) // patch_size) * patch_size
    new_h = (int(h_sample * scale) // patch_size) * patch_size

    for p in tqdm(paths, desc="Loading & Preprocessing"):
        img = Image.open(p)
        img = ImageOps.exif_transpose(img)
        img = img.convert("RGB")
        w_orig, h_orig = img.size

        if rotate_cw90 or h_orig > w_orig:
            img = img.transpose(Image.ROTATE_270)

        img = img.resize((new_w, new_h), Image.LANCZOS)
        arr = np.array(img, dtype=np.float32) / 255.0  # [H, W, 3]
        images_list.append(arr)

    images_np = np.stack(images_list, axis=0)  # [S, H, W, 3]
    images_mx = mx.array(images_np[None])      # [1, S, H, W, 3]
    return images_mx, (new_h, new_w), paths


def resolve_weight_file(weights_path):
    if os.path.exists(weights_path):
        return weights_path
    checkpoints_dir = "checkpoints"
    os.makedirs(checkpoints_dir, exist_ok=True)
    fallback_npz = os.path.join(checkpoints_dir, "lingbot-map-mlx-f16-fixed.npz")
    if os.path.exists(fallback_npz):
        return fallback_npz
    
    pt_path = os.path.join(checkpoints_dir, "lingbot-map.pt")
    if not os.path.exists(pt_path):
        print("Downloading LingBot-Map weights from HuggingFace...")
        from huggingface_hub import hf_hub_download
        hf_hub_download('robbyant/lingbot-map', 'lingbot-map.pt', local_dir=checkpoints_dir)
        
    print("Converting weights to MLX float16 format...")
    import torch, gc
    ckpt = torch.load(pt_path, map_location='cpu', weights_only=False)
    state_dict = ckpt.get('model', ckpt)
    del ckpt; gc.collect()

    conv_transpose_keys = {'depth_head.resize_layers.0.weight', 'depth_head.resize_layers.1.weight'}
    SKIP_TRANSPOSE = {'aggregator.camera_token', 'aggregator.register_token', 'aggregator.scale_token'}

    weights = {}
    for key, tensor in state_dict.items():
        arr = tensor.float().numpy()
        if key in conv_transpose_keys:
            arr = np.transpose(arr, (1, 2, 3, 0))
        elif arr.ndim == 4 and key.endswith('.weight') and key not in SKIP_TRANSPOSE:
            arr = np.transpose(arr, (0, 2, 3, 1))
        weights[key] = arr.astype(np.float16)

    np.savez(fallback_npz, **weights)
    print(f"Saved converted MLX weights to {fallback_npz}")
    return fallback_npz


def main():
    parser = argparse.ArgumentParser(description="Create 3D Map using MLX LingBot-MAP")
    parser.add_argument("--image_folder", type=str, default=None, help="Folder containing sequence images")
    parser.add_argument("--video_path", type=str, default=None, help="Input video file path")
    parser.add_argument("--fps", type=float, default=None, help="Target FPS for video frame extraction")
    parser.add_argument("--max_frames", type=int, default=20, help="Maximum number of frames to process")
    parser.add_argument("--stride", type=int, default=1, help="Stride interval for frame sampling")
    parser.add_argument("--weights", type=str, default="checkpoints/lingbot-map-mlx-f16-fixed.npz", help="Path to MLX weights file (.npz)")
    parser.add_argument("--image_size", type=int, default=518, help="Target image size (longest side)")
    parser.add_argument("--mode", type=str, default="streaming", choices=["streaming", "windowed"], help="Inference mode")
    parser.add_argument("--window_size", type=int, default=20, help="Frames per window in windowed mode")
    parser.add_argument("--num_scale_frames", type=int, default=4, help="Number of bidirectional scale frames")
    parser.add_argument("--camera_iterations", type=int, default=2, help="Camera head refinement iterations")
    parser.add_argument("--keyframe_interval", type=int, default=1, help="Keyframe cache interval")
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "float32"])
    parser.add_argument("--conf_threshold", type=float, default=1.5, help="Confidence threshold for 3D points")
    parser.add_argument("--voxel_size", type=float, default=0.0, help="Voxel grid downsampling size (0.0 = no downsampling)")
    parser.add_argument("--rotate_cw90", action="store_true", default=False, help="Rotate portrait images 90 deg clockwise")
    parser.add_argument("--out_map", type=str, default="maps/reconstruction_map.npz", help="Output 3D map archive (.npz or .ply)")

    args = parser.parse_args()
    assert args.image_folder or args.video_path, "Specify --image_folder or --video_path"

    print("=" * 60)
    print("MLX LingBot-MAP — 3D Map Building Process")
    print("=" * 60)

    images_mx, (H, W), paths = load_images(
        image_folder=args.image_folder, video_path=args.video_path,
        fps=args.fps, max_frames=args.max_frames, stride=args.stride,
        image_size=args.image_size, rotate_cw90=args.rotate_cw90
    )
    S = images_mx.shape[1]

    print("\nCreating GCTStream model...")
    model = GCTStream(
        img_size=args.image_size, patch_size=14, embed_dim=1024,
        enable_camera=True, enable_depth=True,
        enable_point=False, enable_local_point=False,
        kv_cache_sliding_window=64,
        kv_cache_scale_frames=min(args.num_scale_frames, S),
        camera_num_iterations=args.camera_iterations,
        enable_3d_rope=True, enable_camera_3d_rope=True, max_frame_num=1024
    )
    model.eval()

    weights_path = resolve_weight_file(args.weights)
    print(f"Loading weights from {weights_path}...")
    load_weights(model, weights_path, verbose=True)

    if args.dtype == "float16":
        from mlx.utils import tree_map
        model.update(tree_map(
            lambda x: x.astype(mx.float16) if isinstance(x, mx.array) and x.dtype == mx.float32 else x,
            model.parameters()))
        images_mx = images_mx.astype(mx.float16)
        print("Cast model precision to float16")

    print(f"\nRunning {args.mode} inference ({S} frames)...")
    t0 = time.time()
    if args.mode == "windowed":
        predictions = model.inference_windowed(
            images_mx, window_size=args.window_size,
            num_scale_frames=min(args.num_scale_frames, S),
            keyframe_interval=args.keyframe_interval
        )
    else:
        predictions = model.inference_streaming(
            images_mx, num_scale_frames=min(args.num_scale_frames, S),
            keyframe_interval=args.keyframe_interval
        )

    for v in predictions.values():
        if isinstance(v, mx.array):
            mx.eval(v)
    elapsed = time.time() - t0
    print(f"MLX Metal Inference completed: {S} frames in {elapsed:.1f}s ({S/elapsed:.1f} FPS)")

    # Post-processing: Decode camera extrinsics & intrinsics
    print("\nPost-processing camera poses & unprojecting depth...")
    extri_w2c, intri = pose_encoding_to_extri_intri(predictions['pose_enc'], image_size_hw=(H, W))
    
    top = extri_w2c
    bottom = mx.zeros((*extri_w2c.shape[:-2], 1, 4))
    bottom = bottom.at[..., 0, 3].add(1.0)
    e4x4 = mx.concatenate([top, bottom], axis=-2)
    c2w = closed_form_inverse_se3_general(e4x4)
    extri_c2w = c2w[..., :3, :4]

    images_np = np.array(predictions['images'][0])       # [S, H, W, 3]
    depths_np = np.array(predictions['depth'][0])         # [S, H, W, 1]
    confs_np = np.array(predictions['depth_conf'][0])     # [S, H, W]
    extri_w2c_np = np.array(extri_w2c[0])                 # [S, 3, 4]
    extri_c2w_np = np.array(extri_c2w[0])                 # [S, 3, 4]
    intri_np = np.array(intri[0])                         # [S, 3, 3]

    depths_4d = depths_np if depths_np.ndim == 4 else depths_np[..., None]
    confs_2d = confs_np.squeeze(-1) if (confs_np.ndim == 4 and confs_np.shape[-1] == 1) else confs_np

    print(f"Depth range: [{depths_4d.min():.3f}, {depths_4d.max():.3f}], median={np.median(depths_4d):.3f}")
    print(f"Confidence range: [{confs_2d.min():.3f}, {confs_2d.max():.3f}], {(confs_2d > args.conf_threshold).mean() * 100:.1f}% > {args.conf_threshold}")

    # Unproject depth maps using World-to-Camera (w2c) extrinsics
    world_pts = unproject_depth_map_to_point_map(depths_4d, extri_w2c_np, intri_np)  # [S, H, W, 3]

    # Save complete 3D Map file to disk
    out_file = export_point_cloud_map(
        world_points=world_pts,
        colors=images_np,
        confidence=confs_2d,
        filepath=args.out_map,
        conf_threshold=args.conf_threshold,
        voxel_size=args.voxel_size,
        extrinsics=extri_w2c_np,
        intrinsics=intri_np,
        depth=depths_4d,
        extrinsic_c2w=extri_c2w_np,
    )

    # Also automatically save .npz archive if out_map is .ply
    if args.out_map.endswith(".ply"):
        npz_file = args.out_map.replace(".ply", ".npz")
        save_npz(
            filepath=npz_file,
            images=images_np,
            depth=depths_4d,
            depth_conf=confs_2d,
            extrinsic_w2c=extri_w2c_np,
            extrinsic_c2w=extri_c2w_np,
            intrinsic=intri_np,
            world_points=world_pts,
        )

    print(f"\n[Done] 3D Map creation finished successfully!")
    print(f"To visualize the map using minimal memory, run:")
    print(f"    python view_map.py --map_file {out_file if out_file.endswith('.npz') else args.out_map.replace('.ply', '.npz')}")


if __name__ == "__main__":
    main()
