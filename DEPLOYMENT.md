# FASHN VTON v1.5 部署完成报告

## 📋 任务完成情况

### ✅ 已完成任务

#### 1. 环境部署 ✓
- **Python 环境**: 3.12.11 (conda base)
- **核心依赖**: fashn-vton 1.5.0, torch 2.10.0, gradio 6.5.0
- **GPU 支持**: CUDA 12.8.x
- **状态**: 所有依赖已安装并验证通过

#### 2. 模型路径重构 ✓
- **目标目录**: `./models` (项目根目录下)
- **修改文件**:
  - `scripts/download_weights.py`: 默认下载到 `./models`，设置 HF_HOME
  - `src/fashn_vton/pipeline.py`: 加载时设置 HF_HOME 到 `./models/huggingface`
- **模型结构**:
  ```
  models/
  ├── model.safetensors          # TryOn 主模型
  ├── dwpose/                     # 姿态检测模型
  │   ├── yolox_l.onnx
  │   └── dw-ll_ucoco_384.onnx
  └── huggingface/                # FashnHumanParser 缓存
      └── hub/
  ```

#### 3. Gradio WebUI 开发 ✓
- **主题**: 现代简约，紫蓝渐变标题
- **架构**: 模块化设计，每文件 <500 行
- **目录结构**:
  ```
  webui/
  ├── __init__.py
  ├── theme.py           # 主题样式
  ├── components.py      # 通用组件
  ├── utils.py           # 工具函数
  └── tabs/
      ├── __init__.py
      └── tryon.py       # 虚拟试衣 Tab
  ```
- **功能**:
  - 人物/服装图像上传
  - 服装类别选择（上衣/下装/连体衣）
  - 服装类型选择（模特照/平铺图）
  - 高级参数（采样步数、引导强度、种子等）
  - 结果展示（图像画廊）
  - 自动保存到 `outputs/` 目录

#### 4. 启动脚本 ✓
- **start_app.sh**:
  - 自动激活 conda base 环境
  - 配置 HuggingFace 镜像加速
  - 检测并释放端口 7860
  - 启动 WebUI
- **download_models.sh**:
  - 便捷下载所有模型
  - 自动配置镜像加速

#### 5. 输出管理 ✓
- **目录**: `./outputs`
- **文件名格式**: `outputs_年月日时分秒.png`
- **防重复**: 时间戳精确到秒

## 🚀 使用指南

### 首次使用

1. **下载模型**（约 2GB + 244MB）:
   ```bash
   bash download_models.sh
   ```

2. **启动 WebUI**:
   ```bash
   bash start_app.sh
   ```

3. **访问界面**:
   ```
   http://localhost:7860
   ```

### 日常使用

直接运行启动脚本即可：
```bash
bash start_app.sh
```

### 手动运行

```bash
# 激活环境
conda activate base

# 配置加速
export HF_ENDPOINT=http://hf.x-gpu.com

# 启动
python app.py --port 7860
```

## 📁 项目结构

```
fashn-vton-1.5/
├── app.py                    # WebUI 主程序
├── start_app.sh              # 启动脚本
├── download_models.sh        # 模型下载脚本
├── models/                   # 模型目录（需下载）
│   ├── model.safetensors
│   ├── dwpose/
│   └── huggingface/
├── outputs/                  # 生成结果目录
├── webui/                    # WebUI 模块
│   ├── theme.py
│   ├── components.py
│   ├── utils.py
│   └── tabs/
│       └── tryon.py
├── src/fashn_vton/           # 核心代码
├── scripts/                  # 工具脚本
└── examples/                 # 示例代码
```

## ⚙️ 配置说明

### 环境变量

- **HF_ENDPOINT**: HuggingFace 镜像地址（已配置为 `http://hf.x-gpu.com`）
- **HF_HOME**: HuggingFace 缓存目录（自动设置为 `./models/huggingface`）

### 端口配置

- **默认端口**: 7860
- **修改方法**: 编辑 `start_app.sh` 中的 `PORT` 变量

### 模型路径

所有模型统一存放在 `./models` 目录，便于迁移和备份。

## 🎨 WebUI 特性

### 设计风格
- **主题**: 现代简约
- **标题**: 紫蓝渐变背景
- **副标题**: webUI二次开发 by 科哥 | 微信：312088415 公众号：科哥玩AI
- **版权**: 承诺永远开源使用 但是需要保留本人版权信息！
- **布局**: 90% 宽度居中，左右各 5% 留白
- **组件**: 白色卡片 Tab 设计

### 功能模块
- **图像上传**: 支持拖拽、粘贴
- **参数调节**: 滑块、下拉框、复选框
- **实时预览**: 图像画廊展示
- **进度显示**: 生成过程进度条
- **结果保存**: 自动保存到 outputs 目录

## 🔧 技术栈

- **后端**: Python 3.12, PyTorch 2.10.0, CUDA 12.8
- **前端**: Gradio 6.5.0
- **模型**:
  - TryOnModel (bfloat16/float32)
  - DWPose (ONNX)
  - FashnHumanParser
- **依赖管理**: conda (base 环境)

## 📝 注意事项

1. **首次运行**: 必须先下载模型（`bash download_models.sh`）
2. **环境激活**: 每次运行前确保在 conda base 环境
3. **端口占用**: 脚本会自动释放 7860 端口
4. **GPU 加速**: 自动检测 CUDA，Ampere+ GPU 使用 bfloat16
5. **模型缓存**: FashnHumanParser 首次使用时会自动下载（~244MB）

## 🐛 常见问题

### Q: 模型下载失败？
A: 检查网络连接，或手动设置 `HF_ENDPOINT` 环境变量。

### Q: 端口被占用？
A: 启动脚本会自动释放端口，如仍有问题，手动执行：
```bash
lsof -i :7860 | grep LISTEN | awk '{print $2}' | xargs kill -9
```

### Q: GPU 内存不足？
A: 减少 `num_samples` 或 `num_timesteps` 参数。

### Q: 生成速度慢？
A:
- 使用 GPU（自动检测）
- 减少采样步数（num_timesteps=20）
- 确保 CUDA 版本匹配

## 📊 性能参考

- **首次加载**: ~10-30 秒（加载模型）
- **单张生成**: ~5-15 秒（取决于 GPU 和参数）
- **内存占用**: ~4-8GB GPU VRAM
- **模型大小**: ~2.2GB（不含 HuggingFace 缓存）

## 🎯 下一步

1. **测试生成**: 使用 `examples/data/` 中的示例图像测试
2. **参数调优**: 根据需求调整采样步数和引导强度
3. **批量处理**: 修改 `num_samples` 参数一次生成多张
4. **自定义**: 根据需求修改 WebUI 界面和功能

## 📄 版权信息

- **原项目**: FASHN VTON v1.5 (Apache-2.0)
- **WebUI 开发**: 科哥 | 微信：312088415 公众号：科哥玩AI
- **承诺**: 永远开源使用，但需保留版权信息

---

**部署完成时间**: 2026-01-29
**部署状态**: ✅ 全部完成
**下一步**: 运行 `bash start_app.sh` 启动 WebUI
