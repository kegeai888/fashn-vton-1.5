#!/bin/bash
#
# FASHN VTON v1.5 WebUI 启动脚本
# 自动检测并释放端口 7860，然后启动 WebUI
#

set -e

# 配置
PORT=7860
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=========================================="
echo "FASHN VTON v1.5 WebUI 启动脚本"
echo "=========================================="

# 激活 conda base 环境
echo "[1/4] 激活 conda base 环境..."
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
elif [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
    source "/opt/conda/etc/profile.d/conda.sh"
fi
conda activate base 2>/dev/null || echo "已在 base 环境中"

# 设置 HuggingFace 镜像加速
echo "[2/4] 配置 HuggingFace 镜像加速..."
export HF_ENDPOINT=http://hf.x-gpu.com

# 检测并释放端口
echo "[3/4] 检测端口 $PORT..."
if lsof -i :$PORT -t >/dev/null 2>&1; then
    echo "端口 $PORT 被占用，正在释放..."
    # 获取占用端口的进程 PID 并终止
    PIDS=$(lsof -i :$PORT -t 2>/dev/null)
    for PID in $PIDS; do
        echo "终止进程 PID: $PID"
        kill -9 $PID 2>/dev/null || true
    done
    echo "等待端口释放..."
    sleep 2
    echo "端口 $PORT 已释放"
else
    echo "端口 $PORT 可用"
fi

# 启动 WebUI
echo "[4/4] 启动 WebUI..."
echo "=========================================="
echo "访问地址: http://localhost:$PORT"
echo "=========================================="

cd "$SCRIPT_DIR"
python app.py --port $PORT --host 0.0.0.0
