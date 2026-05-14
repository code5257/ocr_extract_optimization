#!/bin/bash
#
# 安装 Triton Python Backend 需要的依赖
#
# 如果用 Docker 部署，这些需要在 Dockerfile 中安装
# 如果用原生部署，直接运行此脚本

set -e

echo "安装 Triton Python Backend 依赖..."

pip install \
    numpy \
    opencv-python-headless \
    shapely \
    pyclipper \
    --no-cache-dir

echo "✓ 依赖安装完成"
