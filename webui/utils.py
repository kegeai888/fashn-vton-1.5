"""
WebUI 工具函数模块
文件保存、时间戳生成等
"""

import os
from datetime import datetime
from pathlib import Path
from typing import Optional

from PIL import Image


def get_project_root() -> Path:
    """获取项目根目录"""
    return Path(__file__).parent.parent


def get_models_dir() -> Path:
    """获取模型目录路径"""
    return get_project_root() / "models"


def get_outputs_dir() -> Path:
    """获取输出目录路径"""
    outputs_dir = get_project_root() / "outputs"
    outputs_dir.mkdir(exist_ok=True)
    return outputs_dir


def generate_output_filename(prefix: str = "outputs", ext: str = "png") -> str:
    """
    生成带时间戳的输出文件名
    格式: outputs_数字年月日时分秒.png
    """
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    return f"{prefix}_{timestamp}.{ext}"


def save_output_image(
    image: Image.Image,
    prefix: str = "outputs",
    ext: str = "png",
) -> str:
    """
    保存输出图像到 outputs 目录
    返回保存的文件路径
    """
    outputs_dir = get_outputs_dir()
    filename = generate_output_filename(prefix, ext)
    filepath = outputs_dir / filename
    image.save(filepath)
    return str(filepath)


def save_multiple_images(
    images: list,
    prefix: str = "outputs",
    ext: str = "png",
) -> list:
    """
    批量保存多张图像
    返回保存的文件路径列表
    """
    saved_paths = []
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

    outputs_dir = get_outputs_dir()
    for i, img in enumerate(images):
        if len(images) > 1:
            filename = f"{prefix}_{timestamp}_{i+1}.{ext}"
        else:
            filename = f"{prefix}_{timestamp}.{ext}"
        filepath = outputs_dir / filename
        img.save(filepath)
        saved_paths.append(str(filepath))

    return saved_paths


def format_file_size(size_bytes: int) -> str:
    """格式化文件大小显示"""
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


def check_weights_exist(weights_dir: Optional[str] = None) -> dict:
    """
    检查模型权重是否存在
    返回检查结果字典
    """
    if weights_dir is None:
        weights_dir = get_models_dir()
    else:
        weights_dir = Path(weights_dir)

    result = {
        "tryon_model": False,
        "dwpose_yolox": False,
        "dwpose_pose": False,
        "all_ready": False,
    }

    tryon_path = weights_dir / "model.safetensors"
    yolox_path = weights_dir / "dwpose" / "yolox_l.onnx"
    pose_path = weights_dir / "dwpose" / "dw-ll_ucoco_384.onnx"

    result["tryon_model"] = tryon_path.exists()
    result["dwpose_yolox"] = yolox_path.exists()
    result["dwpose_pose"] = pose_path.exists()
    result["all_ready"] = all([
        result["tryon_model"],
        result["dwpose_yolox"],
        result["dwpose_pose"],
    ])

    return result
