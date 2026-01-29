"""
WebUI 通用组件模块
Header、Footer 等公共组件
"""

import gradio as gr


def create_header() -> gr.HTML:
    """创建页面头部组件"""
    header_html = """
    <div class="header-container">
        <h1 class="header-title">FASHN VTON v1.5 虚拟试衣</h1>
        <p class="header-subtitle">
            webUI二次开发 by 科哥 | 微信：312088415 公众号：科哥玩AI
        </p>
        <p class="header-copyright">
            承诺永远开源使用 但是需要保留本人版权信息！
        </p>
    </div>
    """
    return gr.HTML(header_html)


def create_footer() -> gr.HTML:
    """创建页面底部组件"""
    footer_html = """
    <div style="text-align: center; padding: 16px; color: #6B7280; font-size: 13px;">
        <p>基于 FASHN VTON v1.5 | Apache-2.0 License</p>
        <p>Powered by Gradio</p>
    </div>
    """
    return gr.HTML(footer_html)


def create_image_upload(label: str, elem_classes: list = None) -> gr.Image:
    """创建图片上传组件"""
    return gr.Image(
        label=label,
        type="pil",
        sources=["upload", "clipboard"],
        elem_classes=elem_classes or ["image-upload"],
        height=280,
    )


def create_category_dropdown() -> gr.Dropdown:
    """创建服装类别下拉框"""
    return gr.Dropdown(
        label="服装类别",
        choices=[
            ("上装 (T恤、衬衫、夹克)", "tops"),
            ("下装 (裤子、裙子、短裤)", "bottoms"),
            ("连体装 (连衣裙、连体裤)", "one-pieces"),
        ],
        value="tops",
        interactive=True,
    )


def create_garment_type_radio() -> gr.Radio:
    """创建服装图片类型选择"""
    return gr.Radio(
        label="服装图片类型",
        choices=[
            ("模特穿着照", "model"),
            ("平铺产品图", "flat-lay"),
        ],
        value="model",
        interactive=True,
    )


def create_advanced_params() -> dict:
    """创建高级参数组件，返回组件字典"""
    components = {}

    with gr.Accordion("高级参数", open=False):
        with gr.Row():
            components["num_timesteps"] = gr.Slider(
                label="采样步数",
                minimum=10,
                maximum=50,
                value=30,
                step=1,
                info="步数越多质量越高，但速度越慢",
            )
            components["guidance_scale"] = gr.Slider(
                label="引导强度",
                minimum=1.0,
                maximum=3.0,
                value=1.5,
                step=0.1,
                info="控制生成图像与输入的一致性",
            )

        with gr.Row():
            components["seed"] = gr.Number(
                label="随机种子",
                value=42,
                precision=0,
                info="相同种子产生相同结果",
            )
            components["num_samples"] = gr.Slider(
                label="生成数量",
                minimum=1,
                maximum=4,
                value=1,
                step=1,
                info="一次生成多张图片",
            )

        with gr.Row():
            components["segmentation_free"] = gr.Checkbox(
                label="无掩码模式",
                value=True,
                info="推荐开启，更好保留身体特征",
            )

    return components
