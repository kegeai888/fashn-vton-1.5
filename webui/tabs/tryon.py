"""
虚拟试衣 Tab 模块
核心功能界面
"""

import gradio as gr
from PIL import Image
from typing import Optional, List
import traceback

from ..components import (
    create_image_upload,
    create_category_dropdown,
    create_garment_type_radio,
    create_advanced_params,
)
from ..utils import save_multiple_images, check_weights_exist, get_models_dir


# 全局 pipeline 实例（延迟加载）
_pipeline = None


def get_pipeline():
    """获取或创建 pipeline 实例"""
    global _pipeline
    if _pipeline is None:
        from fashn_vton import TryOnPipeline
        models_dir = str(get_models_dir())
        _pipeline = TryOnPipeline(weights_dir=models_dir)
    return _pipeline


def run_tryon(
    person_image: Optional[Image.Image],
    garment_image: Optional[Image.Image],
    category: str,
    garment_photo_type: str,
    num_timesteps: int,
    guidance_scale: float,
    seed: int,
    num_samples: int,
    segmentation_free: bool,
    progress=gr.Progress(),
) -> tuple:
    """
    执行虚拟试衣推理
    返回: (结果图像列表, 状态信息, 保存路径列表)
    """
    # 输入验证
    if person_image is None:
        return None, "❌ 请上传人物图像", None
    if garment_image is None:
        return None, "❌ 请上传服装图像", None

    # 检查模型权重
    weights_status = check_weights_exist()
    if not weights_status["all_ready"]:
        missing = []
        if not weights_status["tryon_model"]:
            missing.append("TryOn 模型")
        if not weights_status["dwpose_yolox"]:
            missing.append("YOLOX 检测器")
        if not weights_status["dwpose_pose"]:
            missing.append("DWPose 模型")
        return None, f"❌ 缺少模型权重: {', '.join(missing)}", None

    try:
        progress(0.1, desc="加载模型...")
        pipeline = get_pipeline()

        progress(0.3, desc="处理图像...")
        # 确保图像是 RGB 模式
        person_image = person_image.convert("RGB")
        garment_image = garment_image.convert("RGB")

        progress(0.5, desc="生成中...")
        result = pipeline(
            person_image=person_image,
            garment_image=garment_image,
            category=category,
            garment_photo_type=garment_photo_type,
            num_timesteps=int(num_timesteps),
            guidance_scale=float(guidance_scale),
            seed=int(seed),
            num_samples=int(num_samples),
            segmentation_free=segmentation_free,
        )

        progress(0.9, desc="保存结果...")
        # 保存结果图像
        saved_paths = save_multiple_images(result.images)

        progress(1.0, desc="完成!")
        status = f"✅ 生成成功！已保存 {len(saved_paths)} 张图像"
        paths_info = "\n".join(saved_paths)

        return result.images, status, paths_info

    except Exception as e:
        error_msg = f"❌ 生成失败: {str(e)}"
        traceback.print_exc()
        return None, error_msg, None


def create_tryon_tab() -> gr.Tab:
    """创建虚拟试衣 Tab"""
    with gr.Tab("👕 虚拟试衣", id="tryon") as tab:
        with gr.Row(equal_height=False):
            # 左侧：输入区域（稍窄）
            with gr.Column(scale=5, min_width=400):
                gr.Markdown("### 📸 输入图像")

                person_image = create_image_upload("人物图像")
                garment_image = create_image_upload("服装图像")

                gr.Markdown("### ⚙️ 基础设置")
                category = create_category_dropdown()
                garment_type = create_garment_type_radio()

                # 高级参数
                params = create_advanced_params()

                # 生成按钮
                generate_btn = gr.Button(
                    "🚀 开始生成",
                    variant="primary",
                    size="lg",
                    elem_classes=["generate-btn"],
                )

            # 右侧：输出区域（稍宽）
            with gr.Column(scale=6, min_width=500):
                gr.Markdown("### 🎨 生成结果")

                output_gallery = gr.Gallery(
                    label="结果图像",
                    columns=2,
                    rows=2,
                    height=600,
                    object_fit="contain",
                    elem_classes=["result-image"],
                )

                status_text = gr.Textbox(
                    label="状态",
                    interactive=False,
                    lines=1,
                )

                saved_paths = gr.Textbox(
                    label="保存路径",
                    interactive=False,
                    lines=2,
                    visible=True,
                )

        # 绑定事件
        generate_btn.click(
            fn=run_tryon,
            inputs=[
                person_image,
                garment_image,
                category,
                garment_type,
                params["num_timesteps"],
                params["guidance_scale"],
                params["seed"],
                params["num_samples"],
                params["segmentation_free"],
            ],
            outputs=[output_gallery, status_text, saved_paths],
        )

    return tab
