# 快速开始（5 分钟跑通）

如果你想最快验证效果，按这个顺序执行：

## 1. 准备 ONNX 模型（2 分钟）

```bash
pip install rapidocr_onnxruntime

# 找到模型路径
RAPIDOCR_PATH=$(python -c "import rapidocr_onnxruntime; import os; print(os.path.dirname(rapidocr_onnxruntime.__file__))")

# 复制模型
mkdir -p onnx_models
cp ${RAPIDOCR_PATH}/models/*det*.onnx onnx_models/det.onnx
cp ${RAPIDOCR_PATH}/models/*cls*.onnx onnx_models/cls.onnx
cp ${RAPIDOCR_PATH}/models/*rec*.onnx onnx_models/rec.onnx

ls -lh onnx_models/
```

## 2. 快速验证：先用 ONNX Backend（不转 TensorRT）

如果你想跳过 TensorRT 转换，可以先用 ONNX 模型直接跑 Triton：

```bash
# 把 ONNX 模型放入 model_repository
mkdir -p step3_triton_config/model_repository/det_model/1/
mkdir -p step3_triton_config/model_repository/cls_model/1/
mkdir -p step3_triton_config/model_repository/rec_model/1/

cp onnx_models/det.onnx step3_triton_config/model_repository/det_model/1/model.onnx
cp onnx_models/cls.onnx step3_triton_config/model_repository/cls_model/1/model.onnx
cp onnx_models/rec.onnx step3_triton_config/model_repository/rec_model/1/model.onnx
```

修改 config.pbtxt 中的 platform：
```protobuf
# 从 "tensorrt_plan" 改为 "onnxruntime_onnx"
platform: "onnxruntime_onnx"
```

## 3. 启动 Triton（1 分钟）

```bash
docker run --rm --gpus all \
    -p 8000:8000 -p 8001:8001 -p 8002:8002 \
    -v $(pwd)/step3_triton_config/model_repository:/models \
    nvcr.io/nvidia/tritonserver:24.05-py3 \
    tritonserver --model-repository=/models --strict-model-config=false

# 等待输出 "Started GRPCInferenceService" 即为启动成功
```

## 4. 测试（1 分钟）

```bash
pip install tritonclient[grpc]

python -c "
import tritonclient.grpc as grpcclient
client = grpcclient.InferenceServerClient('localhost:8001')
print('Server live:', client.is_server_live())
print('Models:', client.get_model_repository_index())
"
```

## 5. 后续优化

确认跑通后，再做：
1. ONNX → TensorRT 转换（再快 2-3x）
2. 调整 Dynamic Batching 参数
3. 跑 benchmark.py 对比
