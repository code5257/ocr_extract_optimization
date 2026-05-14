"""
最小改动示例：展示如何用最少的代码改动接入 Triton

你只需要改 2 行代码：
1. import 语句
2. 初始化语句

其他所有代码（process_video、字幕组装等）完全不动。
"""

# ================================================================
# ========== 你原来的代码 ==========================================
# ================================================================

# from rapidocr import RapidOCR
#
# class OCREngine:
#     def __init__(self, config=None):
#         if config is None:
#             config = "tools/rec/1.yaml"
#         self.engine = RapidOCR(config_path=config)
#
#     def recognize(self, img_path):
#         result = self.engine(img_path)
#         return result.txts


# ================================================================
# ========== 改动后的代码（只改了 2 行）==============================
# ================================================================

from triton_ocr_client import TritonOCRWrapper  # ← 改动1: import


class OCREngine:
    def __init__(self, config=None, triton_url="localhost:8001"):
        # 改动2: 初始化从 RapidOCR 改为 TritonOCRWrapper
        self.engine = TritonOCRWrapper(url=triton_url)

    def recognize(self, img_path):
        # 接口完全一致，不需要改
        result = self.engine(img_path)
        return result.txts


# ================================================================
# ========== 你的 process_video 完全不用改 ==========================
# ================================================================
#
# 原来怎么调用：
#   results = self.ocr.recognize(gray_frame)
#
# 现在还是：
#   results = self.ocr.recognize(gray_frame)
#
# 区别：底层从 PyTorch 本地推理 → gRPC 调用 Triton → TensorRT 加速推理
# 你的业务代码感知不到这个变化。


# ================================================================
# ========== 环境变量控制切换（生产环境推荐）=========================
# ================================================================

import os


def create_ocr_engine():
    """
    通过环境变量控制使用哪个引擎
    
    生产环境：
        export OCR_ENGINE=triton
        export TRITON_URL=localhost:8001
    
    开发环境 / 回滚：
        export OCR_ENGINE=local
    """
    engine_type = os.getenv("OCR_ENGINE", "local")
    
    if engine_type == "triton":
        triton_url = os.getenv("TRITON_URL", "localhost:8001")
        print(f"[OCR] 使用 Triton 引擎: {triton_url}")
        return TritonOCRWrapper(url=triton_url)
    else:
        from rapidocr import RapidOCR
        config = os.getenv("OCR_CONFIG", "tools/rec/1.yaml")
        print(f"[OCR] 使用本地引擎: {config}")
        return RapidOCR(config_path=config)


# 在你的 Processor 初始化中：
# self.ocr = create_ocr_engine()
