#!/bin/bash
#
# 阶段2：将 ONNX 模型转换为 TensorRT Engine
#
# 前置条件：
#   - 已完成阶段1，onnx_models/ 目录下有 det.onnx, cls.onnx, rec.onnx
#   - 已安装 TensorRT（推荐使用 NGC TensorRT 容器）
#
# 使用方法：
#   chmod +x convert_to_trt.sh
#   ./convert_to_trt.sh
#
# 注意事项：
#   - TensorRT engine 与 GPU 型号绑定，A100 上转的不能在 4090 上用
#   - 需要在目标部署机器上执行转换
#   - FP16 精度损失极小（<0.1%），推荐开启

set -e

ONNX_DIR="../step1_export_onnx/onnx_models"
TRT_DIR="./trt_engines"
mkdir -p ${TRT_DIR}

echo "============================================"
echo "  PP-OCRv4 ONNX → TensorRT 转换"
echo "============================================"
echo ""
echo "目标：生成 FP16 优化的 TensorRT Engine"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo ""

# ============================================
# 1. Det 模型转换（文本检测）
# ============================================
echo "[1/3] 转换 Det 模型（文本检测）..."
echo "  输入 shape: 动态 batch + 动态分辨率"
echo "  优化策略: FP16 + 动态 shape"

trtexec \
    --onnx=${ONNX_DIR}/det.onnx \
    --saveEngine=${TRT_DIR}/det.plan \
    --fp16 \
    --minShapes=x:1x3x736x736 \
    --optShapes=x:8x3x736x1280 \
    --maxShapes=x:16x3x736x1280 \
    --workspace=4096 \
    --verbose \
    2>&1 | tee logs/det_convert.log

echo "  ✓ Det 模型转换完成: ${TRT_DIR}/det.plan"
echo ""

# ============================================
# 2. Cls 模型转换（方向分类）
# ============================================
echo "[2/3] 转换 Cls 模型（方向分类）..."
echo "  输入 shape: batch x 3 x 48 x 192"
echo "  优化策略: FP16 + 动态 batch"

trtexec \
    --onnx=${ONNX_DIR}/cls.onnx \
    --saveEngine=${TRT_DIR}/cls.plan \
    --fp16 \
    --minShapes=x:1x3x48x192 \
    --optShapes=x:16x3x48x192 \
    --maxShapes=x:64x3x48x192 \
    --workspace=2048 \
    --verbose \
    2>&1 | tee logs/cls_convert.log

echo "  ✓ Cls 模型转换完成: ${TRT_DIR}/cls.plan"
echo ""

# ============================================
# 3. Rec 模型转换（文本识别）
# ============================================
echo "[3/3] 转换 Rec 模型（文本识别）..."
echo "  输入 shape: batch x 3 x 48 x 动态宽度"
echo "  优化策略: FP16 + 动态 batch + 动态宽度"

trtexec \
    --onnx=${ONNX_DIR}/rec.onnx \
    --saveEngine=${TRT_DIR}/rec.plan \
    --fp16 \
    --minShapes=x:1x3x48x48 \
    --optShapes=x:32x3x48x320 \
    --maxShapes=x:64x3x48x640 \
    --workspace=4096 \
    --verbose \
    2>&1 | tee logs/rec_convert.log

echo "  ✓ Rec 模型转换完成: ${TRT_DIR}/rec.plan"
echo ""

# ============================================
# 验证
# ============================================
echo "============================================"
echo "  转换完成！"
echo "============================================"
echo ""
echo "生成的 TensorRT Engine 文件："
ls -lh ${TRT_DIR}/
echo ""
echo "下一步：将这些 .plan 文件复制到 Triton model_repository 中"
echo "  cp ${TRT_DIR}/det.plan ../step3_triton_config/model_repository/det_model/1/model.plan"
echo "  cp ${TRT_DIR}/cls.plan ../step3_triton_config/model_repository/cls_model/1/model.plan"
echo "  cp ${TRT_DIR}/rec.plan ../step3_triton_config/model_repository/rec_model/1/model.plan"
