#!/bin/bash
#
# 阶段4：启动 Triton Server
#
# 两种方式：Docker Compose（推荐）或原生启动

set -e

echo "============================================"
echo "  启动 Triton Inference Server"
echo "============================================"

# ============================================
# 方式一：Docker Compose（推荐）
# ============================================
start_with_docker() {
    echo ""
    echo "[方式一] Docker Compose 启动"
    echo ""
    
    # 检查 Docker
    if ! command -v docker &> /dev/null; then
        echo "错误：未安装 Docker"
        exit 1
    fi
    
    # 检查 NVIDIA Container Toolkit
    if ! docker info 2>/dev/null | grep -q "nvidia"; then
        echo "警告：可能未安装 NVIDIA Container Toolkit"
        echo "安装命令："
        echo "  distribution=\$(. /etc/os-release;echo \$ID\$VERSION_ID)"
        echo "  curl -s -L https://nvidia.github.io/libnvidia-container/gpgkey | sudo apt-key add -"
        echo "  curl -s -L https://nvidia.github.io/libnvidia-container/\$distribution/libnvidia-container.list | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list"
        echo "  sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit"
        echo "  sudo systemctl restart docker"
    fi
    
    # 启动
    docker-compose up -d
    
    echo ""
    echo "等待 Triton 启动..."
    sleep 10
    
    # 检查健康状态
    for i in $(seq 1 30); do
        if curl -s http://localhost:8000/v2/health/ready | grep -q "true"; then
            echo "✓ Triton Server 启动成功！"
            echo ""
            echo "服务地址："
            echo "  HTTP:    http://localhost:8000"
            echo "  gRPC:    localhost:8001"
            echo "  Metrics: http://localhost:8002/metrics"
            echo ""
            echo "查看已加载模型："
            curl -s http://localhost:8000/v2/models | python3 -m json.tool
            return 0
        fi
        echo "  等待中... (${i}/30)"
        sleep 2
    done
    
    echo "错误：Triton 启动超时"
    docker-compose logs triton-ocr
    exit 1
}

# ============================================
# 方式二：原生启动（不用 Docker）
# ============================================
start_native() {
    echo ""
    echo "[方式二] 原生启动（需要预装 Triton）"
    echo ""
    
    MODEL_REPO="../step3_triton_config/model_repository"
    
    tritonserver \
        --model-repository=${MODEL_REPO} \
        --strict-model-config=false \
        --log-verbose=1 \
        --grpc-port=8001 \
        --http-port=8000 \
        --metrics-port=8002
}

# ============================================
# 选择启动方式
# ============================================
if [ "${1}" == "native" ]; then
    start_native
else
    start_with_docker
fi
