#!/usr/bin/env python3
"""
MLX LingBot-MAP: Apple Silicon Metal Native Streaming 3D Reconstruction Pipeline.

License: Apache License 2.0

Usage:
    python demo_mlx.py --image_folder example/loop --stride 2 --out_ply checkpoints/loop_mlx.ply
"""

import argparse
import glob
import os
import sys
import time
import cv2
import numpy as np
import mlx.core as mx
from PIL import Image
from tqdm.auto import tqdm

# Local package imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from lingbot_map_mlx.models import MLXGCTStream
from lingbot_map_mlx.utils import closed_form_inverse_se3_general, unproject_depth_map_to_point_map, resolve_weights
from visualize_ply import render_preview_image, start_viser_server

def load_images(image_folder=None, video_path=None, fps=None, first_k=None, stride=1, image_size=518):
    paths = []
    if image_folder:
        exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.JPG", "*.JPEG", "*.PNG"]
        for ext in exts:
            paths.extend(glob.glob(os.path.join(image_folder, ext)))
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
    if first_k and first_k > 0:
        paths = paths[:first_k]

    print(f"Loading {len(paths)} images for MLX inference...")
    images_list = []
    for p in tqdm(paths, desc="Loading images"):
        img = Image.open(p).convert("RGB")
        W_orig, H_orig = img.size
        
        # Canonical size (518x294)
        aspect = W_orig / H_orig
        new_w = image_size
        new_h = int(round(image_size / aspect / 14) * 14)
        img_resized = img.resize((new_w, new_h), Image.Resampling.BICUBIC)
        
        arr = np.array(img_resized, dtype=np.float32) / 255.0  # [H, W, 3]
        images_list.append(arr)

    images_np = np.stack(images_list, axis=0)  # [S, H, W, 3]
    images_np = np.expand_dims(images_np, axis=0)  # [1, S, H, W, 3]
    return images_np, paths

def main():
    parser = argparse.ArgumentParser(description="MLX LingBot-MAP Demo (Apple Silicon GPU)")
    parser.add_argument("--image_folder", type=str, default=None, help="Folder containing sequence images")
    parser.add_argument("--video_path", type=str, default=None, help="Input video file path")
    parser.add_argument("--fps", type=float, default=None, help="Target FPS for video frame extraction")
    parser.add_argument("--first_k", type=int, default=None, help="Process only first K frames")
    parser.add_argument("--stride", type=int, default=1, help="Stride interval for frame sampling")
    parser.add_argument("--model_path", type=str, default="checkpoints/lingbot-map-mlx.safetensors", help="Path to MLX safetensors weights")
    parser.add_argument("--image_size", type=int, default=518, help="Target image width size")
    parser.add_argument("--conf_threshold", type=float, default=1.5, help="Confidence threshold for 3D points")
    parser.add_argument("--port", type=int, default=8080, help="Viser 3D interactive viewer port")
    parser.add_argument("--out_ply", type=str, default="checkpoints/loop_mlx_output.ply", help="Output 3D PLY file path")
    parser.add_argument("--no_viewer", action="store_true", default=False, help="Disable interactive 3D Viser viewer")

    args = parser.parse_args()
    assert args.image_folder or args.video_path, "Specify --image_folder or --video_path"

    t0 = time.time()
    images_np, paths = load_images(
        image_folder=args.image_folder, video_path=args.video_path,
        fps=args.fps, first_k=args.first_k, stride=args.stride,
        image_size=args.image_size
    )

    print("Building MLX model architecture...")
    model = MLXGCTStream()
    weights_path = resolve_weights(args.model_path)
    model.load_weights(weights_path)
    print(f"MLX Model loaded in {time.time() - t0:.2f}s")

    images_mlx = mx.array(images_np)
    B, S, H, W, C = images_mlx.shape
    chunk_size = 16
    depth_list = []
    conf_list = []

    print(f"\nRunning MLX Metal GPU streaming inference on {S} frames (Chunk size: {chunk_size})...")
    t_inf = time.time()

    for s_idx in range(0, S, chunk_size):
        sub_images = images_mlx[:, s_idx:s_idx + chunk_size]
        sub_out = model(sub_images)
        mx.eval(sub_out["depth"], sub_out["confidence"])
        depth_list.append(np.array(sub_out["depth"]))
        conf_list.append(np.array(sub_out["confidence"]))

    print(f"MLX Metal GPU Inference completed in {time.time() - t_inf:.2f}s!")

    depths_full = np.concatenate(depth_list, axis=1)
    confs_full = np.concatenate(conf_list, axis=1)

    depths_np = depths_full[0]  # [S, H, W, 1]
    confs_np = confs_full[0, :, :, :, 0]  # [S, H, W]

    # Generate synthetic/canonical intrinsics and identity extrinsics for demo point cloud reconstruction
    fov_w = 60.0 * np.pi / 180.0
    fx = (W / 2.0) / np.tan(fov_w / 2.0)
    fy = fx
    cx = W / 2.0
    cy = H / 2.0

    intri_np = np.tile(np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32), (S, 1, 1))
    w2c_np = np.tile(np.eye(4, dtype=np.float32)[:3, :4], (S, 1, 1))

    print(f"Unprojecting depth maps to 3D point cloud...")
    world_pts = unproject_depth_map_to_point_map(depths_np, w2c_np, intri_np)  # [S, H, W, 3]

    conf_mask = confs_np >= args.conf_threshold
    pts_flat = world_pts[conf_mask]
    rgb_flat = (images_np[0] * 255.0).clip(0, 255).astype(np.uint8)[conf_mask]

    if args.out_ply:
        os.makedirs(os.path.dirname(args.out_ply), exist_ok=True)
        print(f"Exporting PLY point cloud to {args.out_ply}...")
        header = f"""ply
format binary_little_endian 1.0
element vertex {len(pts_flat)}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
"""
        vertices = np.empty(len(pts_flat), dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
        vertices['x'] = pts_flat[:, 0]
        vertices['y'] = pts_flat[:, 1]
        vertices['z'] = pts_flat[:, 2]
        vertices['red'] = rgb_flat[:, 0]
        vertices['green'] = rgb_flat[:, 1]
        vertices['blue'] = rgb_flat[:, 2]

        with open(args.out_ply, 'wb') as f:
            f.write(header.encode('ascii'))
            vertices.tofile(f)
        print(f"Successfully exported {len(pts_flat)} 3D points to {args.out_ply}")

        # Render preview image
        preview_img = os.path.splitext(args.out_ply)[0] + "_preview.png"
        render_preview_image(args.out_ply, preview_img)

    if not args.no_viewer and args.out_ply:
        start_viser_server(args.out_ply, port=args.port)

if __name__ == "__main__":
    main()
