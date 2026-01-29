#!/usr/bin/env python3
"""
FASHN VTON v1.5 WebUI 主程序
现代简约风格 Gradio 界面
"""

import os
import sys

# 添加项目根目录到 Python 路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import gradio as gr

from webui.theme import create_custom_theme, CUSTOM_CSS
from webui.components import create_header, create_footer
from webui.tabs import create_tryon_tab
from webui.utils import check_weights_exist, get_models_dir


def check_environment():
    """检查运行环境"""
    print("=" * 50)
    print("FASHN VTON v1.5 WebUI")
    print("=" * 50)

    # 检查模型权重
    weights_status = check_weights_exist()
    models_dir = get_models_dir()

    print(f"\n模型目录: {models_dir}")
    print(f"TryOn 模型: {'✓' if weights_status['tryon_model'] else '✗'}")
    print(f"YOLOX 检测器: {'✓' if weights_status['dwpose_yolox'] else '✗'}")
    print(f"DWPose 模型: {'✓' if weights_status['dwpose_pose'] else '✗'}")

    if not weights_status["all_ready"]:
        print("\n⚠️  警告: 部分模型权重缺失")
        print(f"请运行: python scripts/download_weights.py --weights-dir {models_dir}")

    print("=" * 50)


def create_app() -> gr.Blocks:
    """创建 Gradio 应用"""
    theme = create_custom_theme()

    with gr.Blocks(
        theme=theme,
        css=CUSTOM_CSS,
        title="FASHN VTON v1.5 虚拟试衣",
    ) as app:
        # 页面头部
        create_header()

        # 功能 Tabs
        with gr.Tabs():
            create_tryon_tab()

        # 页面底部
        create_footer()

    return app


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="FASHN VTON v1.5 WebUI")
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="服务器地址 (默认: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="服务器端口 (默认: 7860)",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="创建公共分享链接",
    )
    args = parser.parse_args()

    # 检查环境
    check_environment()

    # 创建并启动应用
    app = create_app()
    app.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        show_error=True,
    )


if __name__ == "__main__":
    main()
