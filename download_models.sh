#!/bin/bash
#
# 下载 FASHN VTON 模型到 models 目录
# 使用 HuggingFace 镜像加速
#

set -e

echo "=========================================="
echo "FASHN VTON v1.5 模型下载脚本"
echo "=========================================="

# 设置 HuggingFace 镜像加速
export HF_ENDPOINT=http://hf.x-gpu.com

# 激活 conda base 环境
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
elif [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
    source "/opt/conda/etc/profile.d/conda.sh"
fi
conda activate base 2>/dev/null || echo "已在 base 环境中"

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 下载模型
echo ""
echo "开始下载模型到 ./models 目录..."
echo ""

python scripts/download_weights.py

echo ""
echo "=========================================="
echo "模型下载完成！"
echo "=========================================="
echo ""
echo "现在可以运行 WebUI："
echo "  bash start_app.sh"
echo ""
