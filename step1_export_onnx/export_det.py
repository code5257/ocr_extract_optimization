"""
阶段1：将 PP-OCRv4 Det 模型导出为 ONNX

PP-OCRv4 server 版文本检测模型导出。
RapidOCR 的 torch 模型底层就是 PaddleOCR 转的 PyTorch 权重，
我们需要把它导出为 ONNX 格式。

方法一（推荐）：直接下载 PaddleOCR 官方提供的 ONNX 模型
方法二：从 PaddlePaddle 模型手动导出
"""

import subprocess
import os
import sys


def method1_download_official_onnx():
    """
    方法一：直接使用 PaddleOCR 官方或 RapidOCR 提供的 ONNX 模型
    
    这是最稳定的方式，模型已经验证过精度一致性。
    """
    
    print("=" * 60)
    print("方法一：下载官方 ONNX 模型（推荐）")
    print("=" * 60)
    
    # RapidOCR 提供的模型下载地址
    models = {
        "det": {
            "name": "ch_PP-OCRv4_det_server_infer.onnx",
            "url": "https://github.com/RapidAI/RapidOCR/releases/download/v1.x.x/ch_PP-OCRv4_det_server_infer.onnx",
            "desc": "PP-OCRv4 Server 文本检测模型",
        },
        "cls": {
            "name": "ch_ppocr_mobile_v2.0_cls_infer.onnx",
            "url": "https://github.com/RapidAI/RapidOCR/releases/download/v1.x.x/ch_ppocr_mobile_v2.0_cls_infer.onnx",
            "desc": "方向分类模型",
        },
        "rec": {
            "name": "ch_PP-OCRv4_rec_server_infer.onnx",
            "url": "https://github.com/RapidAI/RapidOCR/releases/download/v1.x.x/ch_PP-OCRv4_rec_server_infer.onnx",
            "desc": "PP-OCRv4 Server 文本识别模型",
        },
    }
    
    print("""
实际操作步骤：

1. 安装 rapidocr_onnxruntime（它自带 ONNX 模型）：
   pip install rapidocr_onnxruntime
   
2. 模型文件位置（安装后）：
   python -c "import rapidocr_onnxruntime; import os; print(os.path.dirname(rapidocr_onnxruntime.__file__))"
   
   模型在该目录下的 models/ 文件夹中：
   - models/ch_PP-OCRv4_det_server_infer.onnx
   - models/ch_ppocr_mobile_v2.0_cls_infer.onnx  
   - models/ch_PP-OCRv4_rec_server_infer.onnx

3. 或者从 GitHub 直接下载：
   https://github.com/RapidAI/RapidOCR/tree/main/assets/models
   
4. 将下载的 .onnx 文件复制到指定目录：
   mkdir -p onnx_models/
   cp <path>/ch_PP-OCRv4_det_server_infer.onnx onnx_models/det.onnx
   cp <path>/ch_ppocr_mobile_v2.0_cls_infer.onnx onnx_models/cls.onnx
   cp <path>/ch_PP-OCRv4_rec_server_infer.onnx onnx_models/rec.onnx
""")


def method2_export_from_paddle():
    """
    方法二：从 PaddleOCR 模型手动导出 ONNX
    
    如果你需要自定义模型或官方没有提供 ONNX 版本时使用。
    """
    
    print("=" * 60)
    print("方法二：从 PaddlePaddle 手动导出 ONNX")
    print("=" * 60)
    
    print("""
步骤：

1. 安装依赖：
   pip install paddlepaddle-gpu paddleocr paddle2onnx onnx onnxruntime

2. 下载 PaddleOCR inference 模型：
   # 检测模型
   wget https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_det_server_infer.tar
   tar -xf ch_PP-OCRv4_det_server_infer.tar
   
   # 分类模型
   wget https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar
   tar -xf ch_ppocr_mobile_v2.0_cls_infer.tar
   
   # 识别模型
   wget https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_rec_server_infer.tar
   tar -xf ch_PP-OCRv4_rec_server_infer.tar

3. 使用 paddle2onnx 转换：
   # Det 模型
   paddle2onnx \\
       --model_dir ch_PP-OCRv4_det_server_infer \\
       --model_filename inference.pdmodel \\
       --params_filename inference.pdiparams \\
       --save_file onnx_models/det.onnx \\
       --opset_version 16 \\
       --enable_onnx_checker True
   
   # Cls 模型
   paddle2onnx \\
       --model_dir ch_ppocr_mobile_v2.0_cls_infer \\
       --model_filename inference.pdmodel \\
       --params_filename inference.pdiparams \\
       --save_file onnx_models/cls.onnx \\
       --opset_version 16 \\
       --enable_onnx_checker True
   
   # Rec 模型
   paddle2onnx \\
       --model_dir ch_PP-OCRv4_rec_server_infer \\
       --model_filename inference.pdmodel \\
       --params_filename inference.pdiparams \\
       --save_file onnx_models/rec.onnx \\
       --opset_version 16 \\
       --enable_onnx_checker True

4. 验证 ONNX 模型：
   python -c "
   import onnx
   model = onnx.load('onnx_models/det.onnx')
   onnx.checker.check_model(model)
   print('Det 模型验证通过')
   print(f'输入: {[i.name for i in model.graph.input]}')
   print(f'输出: {[o.name for o in model.graph.output]}')
   "
""")


def verify_onnx_models():
    """验证导出的 ONNX 模型精度一致性"""
    
    print("=" * 60)
    print("验证：对比 ONNX 与原始 torch 模型输出一致性")
    print("=" * 60)
    
    print("""
验证脚本：

```python
import cv2
import numpy as np
from rapidocr import RapidOCR

# 原始 torch 引擎
engine_torch = RapidOCR(config_path="tools/rec/1.yaml")

# ONNX 引擎（改配置为 onnxruntime）
engine_onnx = RapidOCR(config_path="tools/rec/1_onnx.yaml")

# 测试图片
img = cv2.imread("test_subtitle.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 对比结果
result_torch = engine_torch(gray)
result_onnx = engine_onnx(gray)

print(f"Torch 结果: {result_torch.txts}")
print(f"ONNX  结果: {result_onnx.txts}")

# 确认一致
assert result_torch.txts == result_onnx.txts, "精度不一致！"
print("验证通过：ONNX 模型输出与 Torch 一致")
```
""")


if __name__ == "__main__":
    method1_download_official_onnx()
    print("\n" + "=" * 60 + "\n")
    method2_export_from_paddle()
    print("\n" + "=" * 60 + "\n")
    verify_onnx_models()
