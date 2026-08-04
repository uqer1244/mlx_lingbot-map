"""Point Cloud & 3D Map Exporter Utilities for MLX LingBot-Map."""

import os
import numpy as np


def voxel_grid_downsample(points: np.ndarray, colors: np.ndarray, voxel_size: float) -> tuple[np.ndarray, np.ndarray]:
    """Downsample point cloud using a 3D voxel grid filter."""
    if voxel_size <= 0:
        return points, colors

    voxel_indices = np.floor(points / voxel_size).astype(np.int64)
    _, unique_indices = np.unique(voxel_indices, axis=0, return_index=True)
    return points[unique_indices], colors[unique_indices]


def save_ply(points: np.ndarray, colors: np.ndarray, filepath: str, binary: bool = True) -> str:
    """Save point cloud as PLY file (binary or ASCII)."""
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    
    if colors.max() <= 1.0:
        colors = (colors * 255.0).clip(0, 255).astype(np.uint8)
    else:
        colors = colors.clip(0, 255).astype(np.uint8)

    num_points = len(points)
    
    if binary:
        header = f"""ply
format binary_little_endian 1.0
element vertex {num_points}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
"""
        vertices = np.empty(num_points, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
        vertices['x'] = points[:, 0].astype(np.float32)
        vertices['y'] = points[:, 1].astype(np.float32)
        vertices['z'] = points[:, 2].astype(np.float32)
        vertices['red'] = colors[:, 0]
        vertices['green'] = colors[:, 1]
        vertices['blue'] = colors[:, 2]

        with open(filepath, 'wb') as f:
            f.write(header.encode('ascii'))
            vertices.tofile(f)
    else:
        header = f"""ply
format ascii 1.0
element vertex {num_points}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
"""
        with open(filepath, 'w') as f:
            f.write(header)
            for i in range(num_points):
                f.write(f"{points[i,0]:.4f} {points[i,1]:.4f} {points[i,2]:.4f} {colors[i,0]} {colors[i,1]} {colors[i,2]}\n")

    print(f"[Exporter] Successfully saved {num_points:,} points to {filepath}")
    return filepath


def save_pcd(points: np.ndarray, colors: np.ndarray, filepath: str) -> str:
    """Save point cloud as PCD (Point Cloud Data) ASCII format."""
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)

    if colors.max() <= 1.0:
        colors = (colors * 255.0).clip(0, 255).astype(np.uint8)

    num_points = len(points)
    rgb_packed = (colors[:, 0].astype(np.uint32) << 16) | (colors[:, 1].astype(np.uint32) << 8) | colors[:, 2].astype(np.uint32)
    rgb_float = rgb_packed.view(np.float32)

    header = f"""# .PCD v0.7 - Point Cloud Data
VERSION 0.7
FIELDS x y z rgb
SIZE 4 4 4 4
TYPE F F F F
COUNT 1 1 1 1
WIDTH {num_points}
HEIGHT 1
VIEWPOINT 0 0 0 1 0 0 0
POINTS {num_points}
DATA ascii
"""
    with open(filepath, 'w') as f:
        f.write(header)
        for i in range(num_points):
            f.write(f"{points[i,0]:.4f} {points[i,1]:.4f} {points[i,2]:.4f} {rgb_float[i]:.8e}\n")

    print(f"[Exporter] Successfully saved {num_points:,} points to {filepath}")
    return filepath


def save_npz(
    filepath: str = "checkpoints/reconstruction_map.npz",
    images: np.ndarray = None,
    depth: np.ndarray = None,
    depth_conf: np.ndarray = None,
    extrinsic_w2c: np.ndarray = None,
    extrinsic_c2w: np.ndarray = None,
    intrinsic: np.ndarray = None,
    world_points: np.ndarray = None,
) -> str:
    """Save full 3D reconstruction map data to compressed NPZ archive."""
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)

    data = {}
    if images is not None:
        data["images"] = images.astype(np.float32)
    if depth is not None:
        data["depth"] = depth.astype(np.float32)
    if depth_conf is not None:
        data["depth_conf"] = depth_conf.astype(np.float32)
    if extrinsic_w2c is not None:
        data["extrinsic_w2c"] = extrinsic_w2c.astype(np.float32)
    if extrinsic_c2w is not None:
        data["extrinsic_c2w"] = extrinsic_c2w.astype(np.float32)
    if intrinsic is not None:
        data["intrinsic"] = intrinsic.astype(np.float32)
    if world_points is not None:
        data["world_points"] = world_points.astype(np.float32)

    np.savez_compressed(filepath, **data)
    print(f"[Exporter] Successfully saved 3D map archive to {filepath}")
    return filepath


def export_point_cloud_map(
    world_points: np.ndarray,
    colors: np.ndarray,
    confidence: np.ndarray = None,
    filepath: str = "checkpoints/map.ply",
    conf_threshold: float = 1.5,
    voxel_size: float = 0.0,
    extrinsics: np.ndarray = None,
    intrinsics: np.ndarray = None,
    depth: np.ndarray = None,
    extrinsic_c2w: np.ndarray = None,
) -> str:
    """Export reconstructed 3D map into requested file format (.ply, .pcd, .npz, .obj)."""
    ext = os.path.splitext(filepath)[1].lower()
    
    if ext == ".npz":
        return save_npz(
            filepath=filepath,
            images=colors,
            depth=depth,
            depth_conf=confidence,
            extrinsic_w2c=extrinsics,
            extrinsic_c2w=extrinsic_c2w,
            intrinsic=intrinsics,
            world_points=world_points
        )

    # Filter points by confidence if confidence array is provided
    if confidence is not None:
        if confidence.ndim == 4 and confidence.shape[-1] == 1:
            confidence = confidence.squeeze(-1)
        conf_mask = confidence >= conf_threshold
        pts = world_points[conf_mask]
        rgb = colors[conf_mask]
    else:
        pts = world_points.reshape(-1, 3)
        rgb = colors.reshape(-1, 3)

    # Filter NaN / Inf
    valid_mask = np.isfinite(pts).all(axis=-1)
    pts = pts[valid_mask]
    rgb = rgb[valid_mask]

    # Voxel grid downsampling
    if voxel_size > 0:
        pts, rgb = voxel_grid_downsample(pts, rgb, voxel_size)

    if ext == ".ply":
        return save_ply(pts, rgb, filepath, binary=True)
    elif ext == ".pcd":
        return save_pcd(pts, rgb, filepath)
    elif ext == ".obj":
        return save_ply(pts, rgb, filepath.replace(".obj", ".ply"), binary=True)
    else:
        return save_ply(pts, rgb, filepath, binary=True)
