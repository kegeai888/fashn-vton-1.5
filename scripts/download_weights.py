#!/usr/bin/env python3
"""
Download all model weights required for FASHN VTON.

Usage:
    python scripts/download_weights.py --weights-dir ./models

This will download:
    - TryOnModel weights (model.safetensors) from HuggingFace
    - DWPose ONNX models (yolox_l.onnx, dw-ll_ucoco_384.onnx)
    - FashnHumanParser weights (auto-cached to models/huggingface)
"""

import argparse
import os
from pathlib import Path

from huggingface_hub import hf_hub_download


def get_default_models_dir() -> str:
    """获取默认模型目录（项目根目录下的 models）"""
    return str(Path(__file__).parent.parent / "models")


def download_tryon_model(weights_dir: str) -> str:
    """Download TryOnModel weights from HuggingFace."""
    print("Downloading TryOnModel weights...")
    path = hf_hub_download(
        repo_id="fashn-ai/fashn-vton-1.5",
        filename="model.safetensors",
        local_dir=weights_dir,
    )
    print(f"  Saved to: {path}")
    return path


def download_dwpose_models(weights_dir: str) -> str:
    """Download DWPose ONNX models from HuggingFace."""
    dwpose_dir = os.path.join(weights_dir, "dwpose")
    os.makedirs(dwpose_dir, exist_ok=True)

    repo_id = "fashn-ai/DWPose"
    filenames = ["yolox_l.onnx", "dw-ll_ucoco_384.onnx"]

    for filename in filenames:
        print(f"Downloading DWPose/{filename}...")
        path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=dwpose_dir,
        )
        print(f"  Saved to: {path}")

    return dwpose_dir


def download_human_parser(weights_dir: str) -> None:
    """Initialize FashnHumanParser to trigger weight download."""
    print("Downloading FashnHumanParser weights...")

    # 设置 HF_HOME 到项目 models 目录下的 huggingface 子目录
    hf_cache_dir = os.path.join(weights_dir, "huggingface")
    os.makedirs(hf_cache_dir, exist_ok=True)
    os.environ["HF_HOME"] = hf_cache_dir

    from fashn_human_parser import FashnHumanParser

    # This will auto-download weights to HuggingFace cache if not present
    _ = FashnHumanParser(device="cpu")
    print(f"  Cached in: {hf_cache_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Download all model weights for FASHN VTON",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    python scripts/download_weights.py --weights-dir ./models

After downloading, use the pipeline:
    from fashn_vton import TryOnPipeline
    pipeline = TryOnPipeline(weights_dir="./models")
        """,
    )
    parser.add_argument(
        "--weights-dir",
        type=str,
        default=None,
        help="Directory to save model weights (default: ./models)",
    )
    args = parser.parse_args()

    weights_dir = args.weights_dir or get_default_models_dir()
    weights_dir = os.path.abspath(weights_dir)
    os.makedirs(weights_dir, exist_ok=True)

    print(f"\nDownloading weights to: {weights_dir}\n")

    # Download all models
    download_tryon_model(weights_dir)
    print()
    download_dwpose_models(weights_dir)
    print()
    download_human_parser(weights_dir)

    print(f"""
Download complete!

Weights directory structure:
    {weights_dir}/
    ├── model.safetensors
    └── dwpose/
        ├── yolox_l.onnx
        └── dw-ll_ucoco_384.onnx

Usage:
    from fashn_vton import TryOnPipeline
    pipeline = TryOnPipeline(weights_dir="{weights_dir}")
""")


if __name__ == "__main__":
    main()
