# PP-OCRv4 极致速度优化：Triton + TensorRT + Dynamic Batching 完整部署方案

## 目标

将现有 RapidOCR (PyTorch) 逐帧串行推理，迁移到 Triton + TensorRT + Dynamic Batching，实现 **6-10x 速度提升**。

---

## 整体架构

```
改造前：
  process_video → RapidOCR(torch) → 逐帧串行 det→cls→rec → 结果
  
改造后：
  process_video → 批量抽帧 → gRPC 异步请求 → Triton Server → 批量结果
                                                    │
                                          ┌─────────┴─────────┐
                                          │   Triton Server    │
                                          │                    │
                                          │  ┌──────────────┐  │
                                          │  │ det (TensorRT)│  │
                                          │  │ batch=16      │  │
                                          │  │ instance=2    │  │
                                          │  └──────┬───────┘  │
                                          │         │          │
                                          │  ┌──────┴───────┐  │
                                          │  │ cls (TensorRT)│  │
                                          │  │ batch=32      │  │
                                          │  └──────┬───────┘  │
                                          │         │          │
                                          │  ┌──────┴───────┐  │
                                          │  │ rec (TensorRT)│  │
                                          │  │ batch=64      │  │
                                          │  └──────┬───────┘  │
                                          │         │          │
                                          │  ┌──────┴───────┐  │
                                          │  │ ensemble      │  │
                                          │  │ (编排流水线)   │  │
                                          │  └──────────────┘  │
                                          └────────────────────┘
```

---

## 预期收益

| 指标 | 改造前 (torch 逐帧) | 改造后 (Triton+TRT+Batch) |
|------|---------------------|--------------------------|
| Det 吞吐 | ~30-50 帧/秒 | ~300-500 帧/秒 |
| Rec 吞吐 | ~100 框/秒 | ~1000-2000 框/秒 |
| 30min视频OCR耗时 | ~30-60s | ~3-6s |
| GPU利用率 | 30-50% | 85-95% |
| 显存占用 | ~2-3GB (torch开销) | ~800MB-1.5GB |

---

## 环境要求

| 依赖 | 版本要求 | 说明 |
|------|---------|------|
| GPU | 算力 >= 7.0 (V100/T4/A10/A100/4090 等) | |
| CUDA | 12.x | |
| cuDNN | 8.9+ | |
| TensorRT | 8.6+ 或 10.x | |
| Triton Server | 24.xx+ (NGC容器) | |
| Python | 3.8+ | 客户端 |
| tritonclient | 最新版 | pip install tritonclient[grpc] |

---

## 目录结构

```
triton_ocr_deployment/
├── README.md                          ← 你正在看的文档
├── step1_export_onnx/                 ← 阶段1：导出 ONNX 模型
│   └── export_det.py
├── step2_convert_tensorrt/            ← 阶段2：转换 TensorRT
│   ├── convert_to_trt.sh
│   └── verify_trt.py
├── step3_triton_config/               ← 阶段3：Triton 配置
│   └── model_repository/
│       ├── det_preprocess/            Python Backend - 检测预处理
│       ├── det_model/                 TensorRT - 文本检测
│       ├── det_postprocess/           Python Backend - 检测后处理
│       ├── cls_model/                 TensorRT - 方向分类
│       ├── rec_model/                 TensorRT - 文本识别
│       ├── rec_postprocess/           Python Backend - 识别后处理
│       ├── ocr_orchestrator/          Python Backend - BLS 编排（核心）
│       └── ocr_pipeline/              Ensemble 配置（参考）
├── step4_launch_triton/               ← 阶段4：启动服务
│   ├── docker-compose.yml
│   └── start_triton.sh
├── step5_client_code/                 ← 阶段5：客户端代码
│   ├── triton_ocr_client.py          替换 RapidOCR 的客户端
│   └── process_video_optimized.py    优化后的完整处理流程
└── step6_benchmark/                   ← 阶段6：性能对比
    └── benchmark.py
```

---

## 完整步骤（共 6 个阶段）

---

### 阶段1：导出 ONNX 模型

**目标**：获取 PP-OCRv4 的 ONNX 格式模型

**最简方式**（推荐）：
```bash
# 安装 rapidocr_onnxruntime，它自带 ONNX 模型
pip install rapidocr_onnxruntime

# 找到模型位置
python -c "import rapidocr_onnxruntime; import os; print(os.path.dirname(rapidocr_onnxruntime.__file__))"

# 复制模型到工作目录
mkdir -p onnx_models
cp <上面输出的路径>/models/ch_PP-OCRv4_det_server_infer.onnx onnx_models/det.onnx
cp <上面输出的路径>/models/ch_ppocr_mobile_v2.0_cls_infer.onnx onnx_models/cls.onnx
cp <上面输出的路径>/models/ch_PP-OCRv4_rec_server_infer.onnx onnx_models/rec.onnx
```

**备选方式**（从 Paddle 模型转）：
```bash
pip install paddlepaddle paddleocr paddle2onnx

# 下载 + 转换（详见 step1_export_onnx/export_det.py）
paddle2onnx --model_dir ch_PP-OCRv4_det_server_infer \
    --model_filename inference.pdmodel \
    --params_filename inference.pdiparams \
    --save_file onnx_models/det.onnx \
    --opset_version 16
```

**验证**：
```bash
python -c "
import onnx
model = onnx.load('onnx_models/det.onnx')
onnx.checker.check_model(model)
print('输入:', [(i.name, [d.dim_value for d in i.type.tensor_type.shape.dim]) for i in model.graph.input])
print('输出:', [(o.name, [d.dim_value for d in o.type.tensor_type.shape.dim]) for o in model.graph.output])
"
```

---

### 阶段2：转换 TensorRT Engine

**目标**：将 ONNX 模型编译为 TensorRT Engine（FP16），获得 2-3x 加速

**前提**：
```bash
# 确认 TensorRT 和 trtexec 可用
trtexec --help | head -5

# 如果没有，用 NGC 容器
docker run --rm --gpus all -v $(pwd):/workspace nvcr.io/nvidia/tensorrt:24.05-py3 bash
```

**执行转换**：
```bash
cd step2_convert_tensorrt
chmod +x convert_to_trt.sh
./convert_to_trt.sh
```

**关键参数说明**：
```bash
trtexec \
    --onnx=det.onnx \           # 输入 ONNX
    --saveEngine=det.plan \     # 输出 TRT Engine
    --fp16 \                    # 开启 FP16（精度损失 <0.1%，速度翻倍）
    --minShapes=x:1x3x736x736 \    # 最小 batch/shape
    --optShapes=x:8x3x736x1280 \   # 最优 batch/shape（TRT 会针对这个优化）
    --maxShapes=x:16x3x736x1280    # 最大 batch/shape
```

**注意**：
- ⚠️ TensorRT Engine 与 GPU 型号绑定！A100 上转的不能在 4090 上用
- ⚠️ 必须在目标部署 GPU 上执行转换
- 转换过程需要 5-15 分钟（TRT 在搜索最优 kernel）

---

### 阶段3：配置 Triton Model Repository

**目标**：配置 Triton 模型仓库，定义 Dynamic Batching 等参数

**放置模型文件**：
```bash
# 将 TensorRT engine 放入对应目录
cp trt_engines/det.plan step3_triton_config/model_repository/det_model/1/model.plan
cp trt_engines/cls.plan step3_triton_config/model_repository/cls_model/1/model.plan
cp trt_engines/rec.plan step3_triton_config/model_repository/rec_model/1/model.plan

# 放置字典文件（CTC 解码需要）
cp ppocr_keys_v1.txt step3_triton_config/model_repository/ocr_orchestrator/
cp ppocr_keys_v1.txt step3_triton_config/model_repository/rec_postprocess/
```

**关键配置参数解释**：

```protobuf
# Dynamic Batching（核心加速机制）
dynamic_batching {
  preferred_batch_size: [8, 16, 32]     # Triton 尝试攒到这些 batch 大小
  max_queue_delay_microseconds: 50000   # 最多等 50ms 来攒批
}
# 含义：Triton 收到请求后，最多等 50ms 凑够 16 个请求一起推理
# 效果：1 次推理处理 16 帧，GPU 利用率从 30% → 90%

# 多实例（并行推理）
instance_group [
  { count: 2, kind: KIND_GPU, gpus: [0] }
]
# 含义：在 GPU 0 上开 2 个模型实例，可以同时推理 2 个 batch
# 效果：吞吐再翻倍
```

---

### 阶段4：启动 Triton Server

**目标**：用 Docker 启动 Triton 服务

```bash
cd step4_launch_triton

# 启动（Docker Compose，推荐）
docker-compose up -d

# 或者直接用 docker run
docker run --rm --gpus all \
    -p 8000:8000 -p 8001:8001 -p 8002:8002 \
    -v $(pwd)/../step3_triton_config/model_repository:/models \
    nvcr.io/nvidia/tritonserver:24.05-py3 \
    tritonserver --model-repository=/models --strict-model-config=false
```

**验证服务状态**：
```bash
# 健康检查
curl http://localhost:8000/v2/health/ready
# 期望输出: true

# 查看已加载模型
curl http://localhost:8000/v2/models | python3 -m json.tool

# 查看模型详情
curl http://localhost:8000/v2/models/ocr_orchestrator | python3 -m json.tool
```

**查看 GPU 指标**：
```bash
curl http://localhost:8002/metrics | grep "nv_gpu_utilization"
```

---

### 阶段5：改造客户端代码

**目标**：将你的 process_video 中的 OCR 调用从 RapidOCR 改为 Triton 客户端

**安装客户端依赖**：
```bash
pip install tritonclient[grpc] numpy opencv-python
```

**最小改动方式**（推荐，改动最少）：

```python
# ============ 原来的代码 ============
from rapidocr import RapidOCR

class YourProcessor:
    def __init__(self, config=None):
        if config is None:
            config = "tools/rec/1.yaml"
        self.ocr = RapidOCR(config_path=config)
    
    def recognize(self, img_path):
        result = self.engine(img_path)
        return result.txts


# ============ 改为 ============
from triton_ocr_client import TritonOCRWrapper

class YourProcessor:
    def __init__(self, triton_url="localhost:8001"):
        self.ocr = TritonOCRWrapper(url=triton_url)
    
    def recognize(self, img_path):
        result = self.ocr(img_path)
        return result.txts
    
    # process_video 里的 self.ocr.recognize(gray_frame) 不需要改！
```

**完整优化方式**（批量异步，速度最快）：
```python
# 见 step5_client_code/process_video_optimized.py
# 核心改动：逐帧串行 → 批量异步
```

---

### 阶段6：性能对比验证

**目标**：确认优化效果，对比前后性能

```bash
cd step6_benchmark

# 运行对比测试
python benchmark.py \
    --video /path/to/test_video.mp4 \
    --frame-interval 25 \
    --triton-url localhost:8001 \
    --config tools/rec/1.yaml
```

**期望输出**：
```
============================================================
  性能对比结果
============================================================

指标                      原始方案              Triton方案            提升
----------------------------------------------------------------------
OCR 总耗时                35000                4500                 7.8x
吞吐 (帧/秒)             51.4                 400.0                7.8x
平均延迟 (ms/帧)          19.4                 2.5
处理帧数                  1800                 1800
----------------------------------------------------------------------

总结: Triton 方案比原始方案快 7.8 倍
============================================================
```

---

## 常见问题

### Q: 模型输入输出的 tensor name 不对怎么办？

```bash
# 用 netron 查看 ONNX 模型的输入输出名称
pip install netron
netron onnx_models/det.onnx
# 浏览器打开 http://localhost:8080 查看
```

然后修改 config.pbtxt 中的 input/output name。

### Q: 动态 shape 怎么确定 min/opt/max？

- **min**: 最小可能的输入（通常 batch=1）
- **opt**: 最常见的输入（TRT 会针对这个做最优优化）
- **max**: 最大可能的输入（超过会报错）

对于你的字幕场景：
- Det: 字幕区域裁剪后约 181x1280，resize 后约 736x1280
- Rec: 单个文本框 resize 后约 48x320

### Q: 精度有损失怎么办？

1. 先确认是 ONNX 导出导致的还是 TensorRT FP16 导致的
2. 如果是 FP16，尝试去掉 `--fp16` 用 FP32（会慢但精度一致）
3. 或者用 `--fp16 --layerPrecisions=<某层>:fp32` 指定关键层用 FP32

### Q: 显存不够怎么办？

```protobuf
# 减少实例数
instance_group [
  { count: 1, kind: KIND_GPU }  # 从 2 改为 1
]

# 减少 max_batch_size
max_batch_size: 8  # 从 16 改为 8
```

### Q: 如何在多 GPU 上跑？

```protobuf
instance_group [
  { count: 1, kind: KIND_GPU, gpus: [0] },
  { count: 1, kind: KIND_GPU, gpus: [1] }
]
```

---

## 回滚方案

如果 Triton 出问题，随时可以切回原来的 RapidOCR：

```python
# 只需要改初始化
# self.ocr = TritonOCRWrapper(url="localhost:8001")  # Triton
self.ocr = RapidOCR(config_path="tools/rec/1.yaml")   # 回滚
```

业务代码（process_video）完全不用改。
