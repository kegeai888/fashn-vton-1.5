"""
WebUI 主题样式模块
紫蓝渐变 + 白色卡片设计
"""

import gradio as gr


def create_custom_theme() -> gr.Theme:
    """创建自定义 Gradio 主题"""
    return gr.themes.Soft(
        primary_hue=gr.themes.colors.violet,
        secondary_hue=gr.themes.colors.blue,
        neutral_hue=gr.themes.colors.gray,
        font=gr.themes.GoogleFont("Inter"),
    ).set(
        # 主容器
        body_background_fill="#F3F4F6",
        # 按钮
        button_primary_background_fill="linear-gradient(135deg, #8B5CF6, #3B82F6)",
        button_primary_background_fill_hover="linear-gradient(135deg, #7C3AED, #2563EB)",
        button_primary_text_color="white",
        button_primary_border_color="transparent",
        # 卡片/块
        block_background_fill="white",
        block_border_color="#E5E7EB",
        block_border_width="1px",
        block_radius="12px",
        block_shadow="0 4px 6px -1px rgba(0, 0, 0, 0.1)",
        # 输入框
        input_background_fill="white",
        input_border_color="#D1D5DB",
        input_border_width="1px",
        input_radius="8px",
        # 标签
        block_label_background_fill="transparent",
        block_label_text_color="#374151",
        block_label_text_weight="500",
    )


# 自定义 CSS 样式
CUSTOM_CSS = """
/* 全局容器 - 适配电脑浏览器 */
.gradio-container {
    max-width: 1600px !important;
    width: 95% !important;
    margin: 0 auto !important;
    padding: 20px !important;
}

/* 主内容区域 */
.main {
    max-width: 100% !important;
}

/* 标题区域 */
.header-container {
    background: linear-gradient(135deg, #8B5CF6, #3B82F6);
    padding: 24px 32px;
    border-radius: 12px;
    margin-bottom: 24px;
    text-align: center;
}

.header-title {
    color: white;
    font-size: 28px;
    font-weight: 700;
    margin: 0 0 12px 0;
    text-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
}

.header-subtitle {
    color: white;
    font-size: 14px;
    margin: 0 0 8px 0;
    text-shadow: 0 1px 2px rgba(0, 0, 0, 0.15);
}

.header-copyright {
    color: rgba(255, 255, 255, 0.9);
    font-size: 13px;
    margin: 0;
    text-shadow: 0 1px 2px rgba(0, 0, 0, 0.15);
}

/* Tab 样式 */
.tabs {
    background: white;
    border-radius: 12px;
    padding: 20px;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
}

/* Row 布局优化 - 减少间距 */
.row {
    gap: 24px !important;
}

/* Column 布局优化 */
.column {
    gap: 16px !important;
}

/* 图片上传区域 */
.image-upload {
    border: 2px dashed #D1D5DB;
    border-radius: 12px;
    transition: border-color 0.2s;
    min-height: 300px;
}

.image-upload:hover {
    border-color: #8B5CF6;
}

/* 生成按钮 */
.generate-btn {
    background: linear-gradient(135deg, #8B5CF6, #3B82F6) !important;
    font-size: 16px !important;
    font-weight: 600 !important;
    padding: 12px 32px !important;
    width: 100% !important;
    margin-top: 16px !important;
}

/* 参数面板 */
.params-panel {
    background: #F9FAFB;
    border-radius: 8px;
    padding: 16px;
}

/* 结果展示 */
.result-image {
    border-radius: 12px;
    overflow: hidden;
}

/* 优化组件间距 */
.block {
    margin-bottom: 12px !important;
}

/* Accordion 样式 */
.accordion {
    margin-top: 16px !important;
}

/* Gallery 优化 */
.gallery {
    min-height: 400px !important;
}

/* Markdown 标题优化 */
.markdown h3 {
    margin-top: 20px !important;
    margin-bottom: 12px !important;
    font-size: 18px !important;
    font-weight: 600 !important;
    color: #374151 !important;
}

/* 输入组件优化 */
.input-text, .dropdown, .radio {
    margin-bottom: 8px !important;
}

/* 响应式布局 */
@media (max-width: 1400px) {
    .gradio-container {
        max-width: 1200px !important;
    }
}

@media (max-width: 1200px) {
    .gradio-container {
        max-width: 100% !important;
        width: 98% !important;
    }
}
"""
