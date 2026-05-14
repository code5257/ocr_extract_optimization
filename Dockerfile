# Triton OCR Server Dockerfile
#
# 基于 NVIDIA Triton Server 官方镜像
# 添加 Python Backend 所需的依赖
#
# 构建：
#   docker build -t triton-ocr:latest .
#
# 运行：
#   docker run --rm --gpus all \
#       -p 8000:8000 -p 8001:8001 -p 8002:8002 \
#       -v $(pwd)/step3_triton_config/model_repository:/models \
#       triton-ocr:latest

FROM nvcr.io/nvidia/tritonserver:24.05-py3

# 安装 Python Backend 依赖
RUN pip install --no-cache-dir \
    numpy \
    opencv-python-headless \
    shapely \
    pyclipper

# 创建模型仓库目录
RUN mkdir -p /models

# 健康检查
HEALTHCHECK --interval=10s --timeout=5s --retries=5 --start-period=30s \
    CMD curl -f http://localhost:8000/v2/health/ready || exit 1

# 启动命令
ENTRYPOINT ["tritonserver"]
CMD ["--model-repository=/models", "--strict-model-config=false", "--log-verbose=1"]
