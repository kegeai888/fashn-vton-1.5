# 🚀 快速启动指南

## 一键启动（推荐）

```bash
# 1. 下载模型（首次使用）
bash download_models.sh

# 2. 启动 WebUI
bash start_app.sh
```

访问: http://localhost:7860

## 详细说明

### 环境要求
- ✅ Python 3.12.11 (conda base)
- ✅ CUDA 12.8.x
- ✅ 所有依赖已安装

### 目录说明
- `models/` - 模型存储（首次需下载）
- `outputs/` - 生成结果自动保存
- `webui/` - WebUI 界面代码

### 常用命令

```bash
# 下载模型
bash download_models.sh

# 启动 WebUI
bash start_app.sh

# 手动启动（高级）
conda activate base
export HF_ENDPOINT=http://hf.x-gpu.com
python app.py --port 7860
```

### WebUI 功能
1. 上传人物图像
2. 上传服装图像
3. 选择服装类别（上衣/下装/连体衣）
4. 调整参数（可选）
5. 点击"开始生成"
6. 查看结果（自动保存到 outputs/）

### 参数说明
- **采样步数**: 20=快速, 30=平衡, 50=高质量
- **引导强度**: 1.0-3.0，越高越贴合输入
- **随机种子**: 固定种子可复现结果
- **生成数量**: 1-4 张

### 故障排除
- **端口占用**: 脚本自动处理
- **模型缺失**: 运行 `bash download_models.sh`
- **GPU 内存不足**: 减少采样步数或生成数量

---

**开发者**: 科哥 | 微信：312088415 公众号：科哥玩AI
**版权**: 永远开源，保留版权信息
