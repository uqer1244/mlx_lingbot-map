#!/usr/bin/env python3
import os
import sys
import time
import numpy as np
import trimesh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import viser

def render_preview_image(ply_path, out_img_path):
    print(f"Loading PLY for preview rendering: {ply_path}")
    mesh = trimesh.load(ply_path)
    pts = mesh.vertices
    cols = mesh.visual.vertex_colors[:, :3] / 255.0

    # Subsample for matplotlib rendering speed
    if len(pts) > 50000:
        idx = np.random.choice(len(pts), 50000, replace=False)
        pts_sub = pts[idx]
        cols_sub = cols[idx]
    else:
        pts_sub = pts
        cols_sub = cols

    fig = plt.figure(figsize=(12, 8), dpi=150)
    fig.patch.set_facecolor('#111111')

    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('#111111')
    ax.scatter(pts_sub[:, 0], pts_sub[:, 1], pts_sub[:, 2], c=cols_sub, s=0.5, alpha=0.8)

    ax.set_axis_off()
    ax.view_init(elev=20, azim=45)

    plt.tight_layout()
    plt.savefig(out_img_path, facecolor='#111111', bbox_inches='tight')
    plt.close()
    print(f"Saved preview image to {out_img_path}")

def start_viser_server(ply_path, port=8080):
    print(f"Loading {ply_path} into Viser server...")
    mesh = trimesh.load(ply_path)
    pts = mesh.vertices
    cols = mesh.visual.vertex_colors[:, :3]

    server = viser.ViserServer(port=port)
    server.scene.add_point_cloud(
        name="/reconstruction/point_cloud",
        points=pts,
        colors=cols,
        point_size=0.015,
    )
    print(f"\n=======================================================")
    print(f"Interactive 3D Viser Viewer started at:")
    print(f"http://localhost:{port}")
    print(f"=======================================================\n")
    
    # Keep server running
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Viewer stopped.")

if __name__ == "__main__":
    ply_file = "checkpoints/loop_output_test.ply"
    if len(sys.argv) > 1:
        ply_file = sys.argv[1]

    img_out = os.path.splitext(ply_file)[0] + "_preview.png"
    render_preview_image(ply_file, img_out)
    
    start_viser_server(ply_file, port=8080)
