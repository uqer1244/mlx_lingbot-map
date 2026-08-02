# MLX LingBot-MAP
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-uqer1244%2Fmlx__lingbot--map-yellow)](https://huggingface.co/uqer1244/mlx_lingbot-map)
[![Python](https://img.shields.io/badge/Python-3.10%2B-brightgreen.svg)](https://python.org)
[![Platform](https://img.shields.io/badge/Platform-Apple%20Silicon%20(M1%2FM2%2FM3%2FM4)-black.svg)](https://github.com/ml-explore/mlx)

**Native Apple Silicon (Metal) GPU-Accelerated 3D Reconstruction Pipeline powered by Apple MLX.**

This repository provides an ultra-fast, native Apple Silicon port of LingBot-MAP / Geometric Context Transformer (GCT) for streaming 3D reconstruction, point cloud generation, and depth estimation.

Model weights are hosted on Hugging Face: [uqer1244/mlx_lingbot-map](https://huggingface.co/uqer1244/mlx_lingbot-map)

---

## Key Features & Performance Benchmark

- **122x Faster than PyTorch MPS**: Reconstructs 119 video frames in **~1.7 minutes** on Apple Silicon Metal GPU (compared to **~3.5 hours** on PyTorch MPS float32).
- **100% Numerical Accuracy**: Includes weight converter verifying zero numerical loss (`Max Absolute Difference: 0.00e+00`, `Cosine Similarity: 1.00000000`) between PyTorch weights and MLX `.safetensors`.
- **Metal Buffer Safety**: Chunked streaming inference prevents Metal GPU single buffer allocation limits (`10.7 GB ceiling`) even on long sequences.
- **Automatic Hugging Face Integration**: Downloads weights automatically from [uqer1244/mlx_lingbot-map](https://huggingface.co/uqer1244/mlx_lingbot-map) if not present locally.
- **Interactive 3D Web Viewer**: Live browser-based 3D point cloud visualization powered by [Viser](https://github.com/nerfstudio-project/viser).
- **Apache 2.0 License**: Fully open-source and free for commercial and non-commercial use.

---

## Benchmark Comparison (119 Video Frames)

| Metrics | PyTorch (MPS) | MLX LingBot-MAP (Metal GPU) | Performance Gain |
| :--- | :--- | :--- | :--- |
| **Inference Time** | `12,516.4s` (~3.5 Hours) | **`102.26s` (~1.7 Minutes)** | **122.4x Faster** |
| **Reconstructed Points** | 1,430,801 pts | **18,122,748 pts** | **12.6x Higher Density** |
| **Metal GPU Efficiency** | OOM / Watermark Limits | **Native Metal Graph (~100% GPU)** | Safe & Stable |
| **Weight Format** | `lingbot-map.pt` (4.63 GB) | `lingbot-map-mlx.safetensors` (4.41 GB) | Identical Weights |

---

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/mlx-lingbot-map.git
cd mlx-lingbot-map

# Create & activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -e .
```

---

## Quick Start Demo

Run streaming 3D reconstruction on an image folder or video file (weights will be automatically fetched from Hugging Face if needed):

```bash
# Run on image sequence with stride 2
python demo_mlx.py --image_folder path/to/images --stride 2 --out_ply checkpoints/reconstruction.ply

# Run on video file
python demo_mlx.py --video_path path/to/video.mp4 --fps 10 --out_ply checkpoints/video_reconstruction.ply
```

After processing, an interactive 3D Viser viewer automatically launches at:
**http://localhost:8080**

---

## Model Weights & Hugging Face

Weights are available on Hugging Face at:
[https://huggingface.co/uqer1244/mlx_lingbot-map](https://huggingface.co/uqer1244/mlx_lingbot-map)

To upload converted weights to your Hugging Face repository:

```bash
huggingface-cli upload uqer1244/mlx_lingbot-map \
  checkpoints/lingbot-map-mlx.safetensors \
  lingbot-map-mlx.safetensors
```

---

## PyTorch to MLX Weight Conversion

If you have a original PyTorch `.pt` model checkpoint, convert it to MLX `.safetensors` with 100% loss-free validation:

```bash
python convert_weights.py
```

Validation output:
```text
Verification Summary
 - Total Layers: 1342 / 1342 (100.0% Matched)
 - Shape Validation: 100.0% Passed
 - Max Absolute Difference: 0.00e+00
 - Average Cosine Similarity: 1.00000000
[SUCCESS] MLX converted weights are 100% identical to original PyTorch weights.
```

---

## Repository Structure

```text
mlx_lingbot-map/
├── LICENSE                     # Apache License 2.0
├── README.md                   # Project Documentation
├── pyproject.toml              # Build & Package Setup
├── requirements.txt            # Dependencies List
├── demo_mlx.py                 # Main CLI Streaming 3D Reconstruction Executable
├── convert_weights.py          # PyTorch -> MLX Weight Converter & Verifier
├── visualize_ply.py            # Viser 3D Web Viewer & Preview Image Renderer
├── checkpoints/
│   └── lingbot-map-mlx.safetensors # Converted MLX Weights (~4.4 GB)
└── lingbot_map_mlx/
    ├── models/
    │   └── gct_stream_mlx.py   # MLX Native GCT Stream Model Architecture
    └── utils/
        ├── geometry.py          # Pose Inversion & Point Cloud Unprojection
        ├── hf_weights.py        # Automatic Hugging Face Weight Downloader
        ├── load_fn.py           # Image/Video Frame Preprocessing
        ├── pose_enc.py          # Intrinsic/Extrinsic Matrix Encoders
        └── rotation.py          # Quaternion & Rotation Matrix Utils
```

---

## License

This project is licensed under the **Apache License 2.0**. See the [LICENSE](LICENSE) file for details.
