# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FASHN VTON v1.5 是一个虚拟试衣模型，在像素空间直接生成逼真图像，无需分割掩码。基于 diffusion model 架构（改编自 FLUX.1），使用 DWPose 进行姿态检测，FashnHumanParser 进行人体解析。

**核心能力**: 给定人物图像和服装图像，生成穿着该服装的逼真图像。支持模特照片和平铺产品图作为服装输入。

## Setup & Installation

```bash
# 安装依赖（推荐使用虚拟环境）
python -m venv .venv && source .venv/bin/activate
pip install -e .

# GPU 加速（可选）
pip uninstall onnxruntime && pip install onnxruntime-gpu

# 下载模型权重（~2GB）
python scripts/download_weights.py --weights-dir ./weights
```

**注意**: Human parser 权重（~244MB）首次使用时自动下载到 HuggingFace cache。可通过 `HF_HOME` 环境变量自定义位置。

## Development Commands

### 运行推理
```bash
# 基础推理
python examples/basic_inference.py \
    --weights-dir ./weights \
    --person-image examples/data/model.webp \
    --garment-image examples/data/garment.webp \
    --category tops

# 支持的 category: tops | bottoms | one-pieces
# 支持的 garment-photo-type: model | flat-lay
```

### 测试
```bash
# 运行所有测试
pytest tests/

# 运行单个测试文件
pytest tests/test_pose_shapes.py

# 运行特定测试
pytest tests/test_pose_shapes.py::TestPoseShapes::test_draw_pose_grayscale_returns_2d_array
```

### 代码质量
```bash
# 格式化代码（Black）
black src/ tests/ examples/ scripts/ --line-length 120

# Lint 检查（Ruff）
ruff check src/ tests/ examples/ scripts/
```

### 调试工具
```bash
# 可视化掩码生成过程
python scripts/debug_masks.py \
    --weights-dir ./weights \
    --person-image examples/data/model.webp \
    --garment-image examples/data/garment.webp \
    --category tops
```

## Architecture

### 核心组件层次

```
TryOnPipeline (pipeline.py)
├── TryOnModel (tryon_mmdit.py)          # MMDiT 架构，改编自 FLUX.1
├── DWposeDetector (dwpose/)             # 姿态检测（ONNX）
├── FashnHumanParser                     # 人体解析（外部依赖）
└── Preprocessing (preprocessing/)
    ├── agnostic.py                      # 服装无关图像生成
    ├── masks.py                         # 掩码处理
    └── transforms.py                    # 图像变换（resize/pad）
```

### 推理流程（Pipeline.__call__）

1. **预处理阶段**:
   - 图像预缩放（AspectPreserveResize）保持姿态检测质量
   - DWPose 检测人物和服装姿态（flat-lay 使用 dummy keypoints）
   - FashnHumanParser 生成分割掩码

2. **图像准备**:
   - `create_clothing_agnostic_image()`: 生成服装无关图像（CA image）
     - `segmentation_free=True` 时不掩盖人物，保留身体结构
   - `create_garment_image()`: 处理服装图像
   - ResizePad 统一尺寸到模型输入（768x576）

3. **采样阶段**:
   - Euler sampling + CFG（Classifier-Free Guidance）
   - Rectified Flow schedule（时间从 0→1）
   - 最后 N 步跳过 CFG 防止颜色过饱和

4. **后处理**: Unpad 恢复原始宽高比

### 关键设计决策

- **Maskless 模式** (`segmentation_free=True`): 不掩盖人物图像，允许服装体积超出原始轮廓，更好保留身体特征
- **bfloat16 精度**: Ampere+ GPU（RTX 30xx/40xx, A100）自动使用 bf16，旧硬件/CPU 回退到 float32
- **RoPE 位置编码**: 2D 旋转位置嵌入，支持任意分辨率
- **QKNorm**: Query/Key 归一化提升训练稳定性

## Code Conventions

### 张量形状约定
- 图像: `(B, C, H, W)` - PyTorch 标准
- 姿态: `(B, 1, H, W)` - 灰度图
- 注意力: `(B, H, L, D)` - Batch, Heads, Length, Dim

### 数值范围
- 输入图像: `[-1, 1]` 归一化（`normalize_uint8_to_neg1_1`）
- 输出图像: `[-1, 1]` clamp 后转 PIL

### 模块职责
- `utils/`: 纯函数工具（checkpoint 加载、张量转换、采样调度）
- `preprocessing/`: 图像预处理管道（无状态变换）
- `dwpose/`: 姿态检测封装（ONNX 推理）
- `tryon_mmdit.py`: 核心模型定义（不含推理逻辑）
- `pipeline.py`: 端到端推理编排

## Important Files

- `src/fashn_vton/pipeline.py`: 主推理入口，理解整体流程从这里开始
- `src/fashn_vton/tryon_mmdit.py`: MMDiT 模型架构（~500 行）
- `src/fashn_vton/preprocessing/agnostic.py`: 服装无关图像生成核心逻辑
- `src/fashn_vton/utils/sampling.py`: Rectified Flow 采样调度
- `pyproject.toml`: 依赖管理和工具配置（Black, Ruff）

## Dependencies

**核心依赖**:
- `torch>=2.0.0`: 主框架
- `safetensors`: 权重加载
- `onnxruntime`: DWPose 推理（CPU/GPU）
- `fashn-human-parser>=0.1.1`: 人体解析（自动下载权重）

**第三方组件许可**:
- DWPose (Apache-2.0)
- YOLOX (Apache-2.0)
- fashn-human-parser (见其仓库 LICENSE)

## Python Version

支持 Python 3.10, 3.11, 3.12

## Notes

- 模型权重存储为 bfloat16，首次加载时自动转换到目标设备/精度
- DWPose 期望 BGR 输入（OpenCV 格式），需要 `[..., ::-1]` 转换
- `mem_padding=True` 用于 CA image，确保内存对齐优化
- 测试主要覆盖张量形状和数据流，不包含端到端推理（需要权重）
