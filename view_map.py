#!/usr/bin/env python3
"""
View Map — Lightweight Standalone 3D Map Visualizer

Loads saved 3D reconstruction map data (.npz / .ply) from disk
and launches the interactive Viser Web Viewer without loading any MLX/PyTorch models into memory.

Usage:
    python view_map.py --map_file checkpoints/reconstruction_map.npz --port 8080
"""

import argparse
import os
import sys
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="View 3D Map (Lightweight Viewer)")
    parser.add_argument("--map_file", type=str, default="maps/reconstruction_map.npz", help="Path to saved 3D map (.npz or .ply)")
    parser.add_argument("--port", type=int, default=8080, help="Viser 3D Web Viewer port")
    parser.add_argument("--conf_threshold", type=float, default=1.5, help="Confidence threshold for displaying 3D points")
    parser.add_argument("--downsample_factor", type=int, default=10, help="Point cloud downsample factor")
    parser.add_argument("--point_size", type=float, default=0.00001, help="Point size in 3D viewer")

    args = parser.parse_args()

    # Check if map file exists, fallback to .ply if .npz missing
    if not os.path.exists(args.map_file):
        if args.map_file.endswith(".npz") and os.path.exists(args.map_file.replace(".npz", ".ply")):
            args.map_file = args.map_file.replace(".npz", ".ply")
        elif args.map_file.endswith(".ply") and os.path.exists(args.map_file.replace(".ply", ".npz")):
            args.map_file = args.map_file.replace(".ply", ".npz")
        else:
            raise FileNotFoundError(f"Map file not found: {args.map_file}")

    print("=" * 60)
    print("Lightweight 3D Map Web Visualizer (Zero MLX Model Overhead)")
    print("=" * 60)
    print(f"Loading map file from {args.map_file}...")

    ext = os.path.splitext(args.map_file)[1].lower()

    if ext == ".npz":
        data = np.load(args.map_file)
        images_np = data["images"]
        depths_np = data["depth"]
        confs_np = data["depth_conf"]
        extri_c2w_np = data.get("extrinsic_c2w")
        if extri_c2w_np is None and "extrinsic_w2c" in data:
            extri_w2c = data["extrinsic_w2c"]
            R = extri_w2c[:, :3, :3]
            t = extri_w2c[:, :3, 3:]
            R_inv = np.transpose(R, (0, 2, 1))
            t_inv = -(R_inv @ t)
            extri_c2w_np = np.concatenate([R_inv, t_inv], axis=-1)

        intri_np = data["intrinsic"]

        if images_np.ndim == 4 and images_np.shape[-1] == 3:
            images_chw = images_np.transpose(0, 3, 1, 2)
        else:
            images_chw = images_np

        if depths_np.ndim == 3:
            depths_np = depths_np[..., None]
        if confs_np.ndim == 4 and confs_np.shape[-1] == 1:
            confs_np = confs_np.squeeze(-1)

        vis_preds = {
            'images': images_chw,
            'depth': depths_np,
            'depth_conf': confs_np,
            'extrinsic': extri_c2w_np,
            'intrinsic': intri_np,
        }

        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from lingbot_map_mlx.vis import PointCloudViewer

        print(f"\nLaunching 3D Web Visualizer at http://localhost:{args.port}...")
        viewer = PointCloudViewer(
            pred_dict=vis_preds,
            port=args.port,
            vis_threshold=args.conf_threshold,
            downsample_factor=args.downsample_factor,
            point_size=args.point_size
        )
        viewer.run()

    elif ext == ".ply":
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from lingbot_map_mlx.vis import viser_wrapper

        # Simple PLY parser for point visualization
        with open(args.map_file, 'rb') as f:
            header_lines = []
            while True:
                line = f.readline().decode('ascii', errors='ignore')
                header_lines.append(line)
                if line.startswith('end_header'):
                    break

            num_points = 0
            for line in header_lines:
                if line.startswith('element vertex'):
                    num_points = int(line.split()[-1])

            dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
            vertices = np.fromfile(f, dtype=dtype, count=num_points)

        pts = np.stack([vertices['x'], vertices['y'], vertices['z']], axis=-1)
        colors = np.stack([vertices['red'], vertices['green'], vertices['blue']], axis=-1).astype(np.float32) / 255.0

        print(f"Loaded {len(pts):,} points from PLY file.")

        import viser
        server = viser.ViserServer(host="0.0.0.0", port=args.port)
        server.scene.add_point_cloud("ply_map", points=pts, colors=colors, point_size=args.point_size * 50)
        print(f"\n3D PLY Map visualizer running at http://localhost:{args.port}")

        import time
        while True:
            time.sleep(1)


if __name__ == "__main__":
    main()
