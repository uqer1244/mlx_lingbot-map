# MLX LingBot-MAP

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-uqer1244%2Fmlx__lingbot--map-yellow)](https://huggingface.co/uqer1244/mlx_lingbot-map)
[![Original Repo](https://img.shields.io/badge/Original-Robbyant%2Flingbot--map-black)](https://github.com/Robbyant/lingbot-map)
[![Python](https://img.shields.io/badge/Python-3.10%2B-brightgreen.svg)](https://python.org)
[![Platform](https://img.shields.io/badge/Platform-Apple%20Silicon%20(M1%2FM2%2FM3%2FM4)-black.svg)](https://github.com/ml-explore/mlx)

**Native Apple Silicon (Metal) GPU-Accelerated 3D Reconstruction Pipeline powered by Apple MLX.**

This repository provides a native Apple Silicon port of [LingBot-MAP / Geometric Context Transformer (GCT)](https://github.com/Robbyant/lingbot-map) for streaming 3D reconstruction, point cloud generation, camera pose estimation, and depth map prediction.

---

## 🔗 Links & Resources

- **MLX GitHub Repository**: [uqer1244/mlx_lingbot-map](https://github.com/uqer1244/mlx_lingbot-map)
- **Converted MLX Model Weights (Hugging Face)**: [uqer1244/mlx_lingbot-map](https://huggingface.co/uqer1244/mlx_lingbot-map)
- **Original Model Weights (Hugging Face)**: [robbyant/lingbot-map](https://huggingface.co/robbyant/lingbot-map)
- **Original GitHub Repository**: [Robbyant/lingbot-map](https://github.com/Robbyant/lingbot-map)

---

## 🛠️ Technical Highlights

- **Native MLX Port**: Re-implemented the Geometric Context Transformer (GCT) and DINOv2 vision trunk using `mlx.core` and `mlx.nn` for Metal GPU acceleration.
- **Tensor Layout Alignment**: Transposed 4D Conv weights from PyTorch `OIHW` to MLX `OHWI` layout for memory efficiency while preserving numerical parity (`max_diff < 1e-5`).
- **Multi-Precision & Quantization**: Supports FP32, FP16, BF16, as well as 8-bit and 4-bit group-wise affine quantization (`scales`, `biases`).
- **Streaming KV Cache**: Implements sliding-window KV caching for efficient camera tracking and depth estimation over long image sequences.
- **Decoupled Architecture**: Separates GPU inference (`create_map.py`) from 3D visualization (`view_map.py`) to ensure immediate RAM/VRAM release after mapping.

---

## 📦 Model Weight Variants

Pre-converted MLX safetensors weights are hosted on Hugging Face at [uqer1244/mlx_lingbot-map](https://huggingface.co/uqer1244/mlx_lingbot-map) in 5 precision formats:

| Format / Precision | File Name | Size | Target Hardware / Description |
| :--- | :--- | :--- | :--- |
| **FP16 (Recommended)** | `lingbot-map-fp16.safetensors` | **2.16 GB** | Default precision for Apple Silicon GPUs (Fastest & Balanced) |
| **FP32** | `lingbot-map-fp32.safetensors` | **4.31 GB** | Full precision reference weights |
| **BF16** | `lingbot-map-bf16.safetensors` | **2.16 GB** | BFloat16 precision for M2 / M3 / M4 chips |
| **INT8** | `lingbot-map-int8.safetensors` | **1.24 GB** | 8-bit Group-wise Affine Quantization for low-memory devices |
| **INT4** | `lingbot-map-int4.safetensors` | **0.72 GB** | 4-bit Group-wise Affine Quantization for minimal RAM usage |

---

## ⚡ Key Features

- **Apple Silicon Metal GPU Acceleration**: Optimized MLX graph execution for Apple M-series chips.
- **Zero-RAM-Leak 2-Process Architecture**:
  - `create_map.py`: Runs MLX GPU inference, saves 3D map data, and **immediately exits to free 100% of GPU & system RAM**.
  - `view_map.py`: Standalone 3D Web Visualizer (**~150MB RAM**) without loading any MLX/PyTorch models into memory.
- **Original LingBot-Map 3D Web Visualizer**: Includes native 1:1 integration of the original `PointCloudViewer` Viser web visualizer on `http://localhost:8080`.
- **Multi-Format 3D Map Exporter**: Export point cloud & camera tracking data to `.ply`, `.pcd`, `.npz`, `.obj`.
- **Voxel Downsampling & Noise Filtering**: Spatial 3D voxel grid downsampling and confidence thresholding.

---

## 🚀 Quick Start Guide

### 1. Installation

```bash
# Clone repository
git clone https://github.com/uqer1244/mlx_lingbot-map.git
cd mlx_lingbot-map

# Create virtual environment & install dependencies
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

---

### 2. High-Efficiency 2-Process Workflow (Recommended)

#### Step 1: Create & Export 3D Map (`create_map.py`)
Runs MLX GPU inference on images or video, exports the 3D map data, and frees GPU memory immediately upon completion:

```bash
# Process image sequence and save to .npz map archive in maps/
python3 create_map.py --image_folder path/to/images --max_frames 20 --out_map maps/my_map.npz

# Process video file with stride
python3 create_map.py --video_path path/to/video.mp4 --fps 10 --out_map maps/video_map.ply --voxel_size 0.02
```

#### Step 2: Visualize 3D Map (`view_map.py`)
Launches the interactive 3D Web Viewer using negligible RAM (0% MLX model overhead):

```bash
# Visualize using default port 8080
python3 view_map.py --map_file maps/my_map.npz --port 8080
```

Open your browser at: **[http://localhost:8080](http://localhost:8080)**

---

### 3. All-in-One Single Command (`demo_mlx.py`)

If you want to run inference and launch the 3D web viewer in a single command:

```bash
python3 demo_mlx.py --image_folder path/to/images --out_ply maps/reconstruction.ply --port 8080
```

---

## 📂 Project Architecture

```text
mlx_lingbot-map/
├── create_map.py         # [Process 1] MLX GPU inference & 3D map exporter (exits to release RAM)
├── view_map.py           # [Process 2] Standalone lightweight 3D web visualizer (~150MB RAM)
├── demo_mlx.py           # Single-command unified pipeline
├── lingbot_map_mlx/      # Core MLX model package
│   ├── aggregator/       # Feature extraction & streaming KV cache
│   ├── heads/            # Camera pose & DPT depth heads
│   ├── layers/           # Vision Transformer & RoPE layers
│   ├── models/           # GCTBase & GCTStream models
│   ├── utils/            # 3D geometry, pose encoding, & multi-format exporter
│   └── vis/              # Original LingBot-Map PointCloudViewer web visualizer
├── maps/                 # Storage for generated 3D map files (.npz, .ply)
├── checkpoints/          # MLX safetensors weights (FP16, FP32, BF16, INT8, INT4)
├── tools/                # Conversion and Hugging Face upload utilities
├── pyproject.toml
└── requirements.txt
```

---

## 📄 Export Formats

| Format | Description | Target Use Case |
| :--- | :--- | :--- |
| **`.npz`** | Full compressed archive (Points, RGB, Depth, Conf, Extrinsics, Intrinsics) | Unity Sentis, AR/VR, Python downstream pipelines |
| **`.ply`** | Standard 3D Point Cloud with RGB colors | MeshLab, CloudCompare, Blender |
| **`.pcd`** | Point Cloud Data (ASCII / Binary) | PCL, ROS 1/2 |
| **`.obj`** | Standard 3D Geometry file | 3D Modeling software |

---

## 📜 License

Apache 2.0 License. Free for commercial and research use.
