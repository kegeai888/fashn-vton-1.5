# WebUI 模块化架构设计

## 目录结构
```
webui/
├── __init__.py          # 模块导出
├── theme.py             # 主题样式（紫蓝渐变、白色卡片）
├── components.py        # 通用组件（Header、Footer）
├── utils.py             # 工具函数（文件保存、时间戳）
└── tabs/
    ├── __init__.py
    └── tryon.py         # 虚拟试衣 Tab
```

## 模块职责

### theme.py (~50行)
- 自定义 Gradio 主题
- 紫蓝渐变色定义
- 白色卡片样式

### components.py (~100行)
- `create_header()`: 标题区域（渐变背景、副标题、版权）
- `create_footer()`: 页脚

### utils.py (~80行)
- `generate_output_filename()`: 生成时间戳文件名
- `save_output_image()`: 保存到 outputs 目录
- `get_models_dir()`: 获取模型目录路径

### tabs/tryon.py (~300行)
- 人物图像上传
- 服装图像上传
- 类别选择（tops/bottoms/one-pieces）
- 服装类型选择（model/flat-lay）
- 高级参数（采样步数、引导强度、种子等）
- 生成按钮
- 结果展示

### app.py (~150行)
- 主入口
- 组装所有组件
- 启动 Gradio 服务

## 样式规范

### 颜色
- 主色（紫）: #8B5CF6
- 辅色（蓝）: #3B82F6
- 渐变: linear-gradient(135deg, #8B5CF6, #3B82F6)
- 卡片背景: #FFFFFF
- 页面背景: #F3F4F6

### 布局
- 容器宽度: 90%
- 左右留白: 各 5%
- 卡片圆角: 12px
- 卡片阴影: 0 4px 6px rgba(0,0,0,0.1)

### 字体
- 标题: 24px, bold, white, text-shadow
- 副标题: 14px, white
- 正文: 14px, #374151
