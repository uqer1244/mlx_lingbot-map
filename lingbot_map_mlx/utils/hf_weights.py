#!/usr/bin/env python3
"""
Hugging Face Weight Downloader Helper for MLX LingBot-MAP.
"""
import os
import sys

def resolve_weights(model_path: str = "checkpoints/lingbot-map-mlx.safetensors", repo_id: str = None) -> str:
    """
    Check if weight file exists locally; if not, download from Hugging Face repo.
    """
    if os.path.exists(model_path):
        return model_path

    if repo_id is None:
        repo_id = os.environ.get("MLX_LINGBOT_MAP_REPO", "uqer1244/mlx_lingbot-map")

    filename = os.path.basename(model_path)
    target_dir = os.path.dirname(os.path.abspath(model_path)) or "checkpoints"
    os.makedirs(target_dir, exist_ok=True)

    print(f"Local weight file not found at '{model_path}'.")
    print(f"Downloading '{filename}' from Hugging Face repository '{repo_id}'...")

    try:
        from huggingface_hub import hf_hub_download
        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=target_dir,
        )
        print(f"Weight download completed: {downloaded_path}")
        return downloaded_path
    except Exception as e:
        print(f"Failed to download weights from Hugging Face: {e}")
        print(f"Please place your '{filename}' file manually inside '{target_dir}/'.")
        raise e
