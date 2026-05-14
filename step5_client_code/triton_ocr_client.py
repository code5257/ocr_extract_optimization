"""
阶段5：Triton OCR 客户端

替换你原来的 self.ocr.recognize(gray_frame) 调用。
通过 gRPC 发送图片到 Triton Server，获取 OCR 结果。

支持：
- 单帧同步调用
- 批量异步调用（推荐，速度最快）
- 连接池管理
"""

import numpy as np
import cv2
import time
from typing import Any
from concurrent.futures import Future
from dataclasses import dataclass

import tritonclient.grpc as grpcclient
from tritonclient.utils import np_to_triton_dtype


@dataclass
class OCRResult:
    """OCR 识别结果"""
    text: str
    confidence: float


class TritonOCRClient:
    """
    Triton OCR 客户端
    
    用法（替换你原来的 RapidOCR）：
        # 原来
        self.ocr = RapidOCR(config_path="tools/rec/1.yaml")
        result = self.ocr(gray_frame)
        texts = result.txts
        
        # 改为
        self.ocr = TritonOCRClient(url="localhost:8001")
        texts = self.ocr.recognize(gray_frame)
    """
    
    def __init__(
        self,
        url: str = "localhost:8001",
        model_name: str = "ocr_orchestrator",
        timeout: float = 10.0,
        verbose: bool = False,
    ):
        """
        Args:
            url: Triton gRPC 地址
            model_name: 模型名称（ocr_orchestrator）
            timeout: 超时时间（秒）
            verbose: 是否打印调试信息
        """
        self.url = url
        self.model_name = model_name
        self.timeout = timeout
        self.verbose = verbose
        
        # 创建 gRPC 客户端
        self.client = grpcclient.InferenceServerClient(
            url=url,
            verbose=verbose,
        )
        
        # 验证服务可用
        if not self.client.is_server_live():
            raise ConnectionError(f"Triton Server 不可用: {url}")
        
        if not self.client.is_model_ready(model_name):
            raise RuntimeError(f"模型 {model_name} 未就绪")
        
        print(f"[TritonOCRClient] 连接成功: {url}, 模型: {model_name}")

    def recognize(self, gray_frame: np.ndarray) -> list[str] | None:
        """
        同步识别单帧（直接替换你原来的 self.ocr.recognize()）
        
        Args:
            gray_frame: 灰度图 (H, W) 或 (H, W, 1)
            
        Returns:
            识别的文本列表，如 ["你好世界"]，无文本返回 None
        """
        # 确保是 (H, W, 1) 格式
        if gray_frame.ndim == 2:
            gray_frame = gray_frame[:, :, np.newaxis]
        
        # 构建输入
        input_tensor = grpcclient.InferInput(
            "IMAGE",
            gray_frame.shape,
            np_to_triton_dtype(np.uint8),
        )
        input_tensor.set_data_from_numpy(gray_frame.astype(np.uint8))
        
        # 请求输出
        output_text = grpcclient.InferRequestedOutput("TEXT")
        output_conf = grpcclient.InferRequestedOutput("CONFIDENCE")
        
        # 同步推理
        response = self.client.infer(
            model_name=self.model_name,
            inputs=[input_tensor],
            outputs=[output_text, output_conf],
            client_timeout=self.timeout,
        )
        
        # 解析结果
        text_data = response.as_numpy("TEXT")
        conf_data = response.as_numpy("CONFIDENCE")
        
        text = text_data[0].decode("utf-8") if text_data.size > 0 else ""
        confidence = float(conf_data[0]) if conf_data.size > 0 else 0.0
        
        if text.strip():
            return [text.strip()]
        return None

    def recognize_batch(self, frames: list[np.ndarray]) -> list[str | None]:
        """
        批量识别多帧（推荐，速度更快）
        
        Args:
            frames: 灰度图列表
            
        Returns:
            每帧的识别结果列表
        """
        results = []
        
        # Triton 会通过 Dynamic Batching 自动攒批
        # 我们用异步请求并发发送
        futures: list[Future] = []
        
        for frame in frames:
            if frame.ndim == 2:
                frame = frame[:, :, np.newaxis]
            
            input_tensor = grpcclient.InferInput(
                "IMAGE",
                frame.shape,
                np_to_triton_dtype(np.uint8),
            )
            input_tensor.set_data_from_numpy(frame.astype(np.uint8))
            
            output_text = grpcclient.InferRequestedOutput("TEXT")
            output_conf = grpcclient.InferRequestedOutput("CONFIDENCE")
            
            # 异步发送
            future = self.client.async_infer(
                model_name=self.model_name,
                inputs=[input_tensor],
                outputs=[output_text, output_conf],
                client_timeout=self.timeout,
            )
            futures.append(future)
        
        # 收集结果
        for future in futures:
            try:
                response = future.result()
                text_data = response.as_numpy("TEXT")
                text = text_data[0].decode("utf-8") if text_data.size > 0 else ""
                results.append(text.strip() if text.strip() else None)
            except Exception as e:
                if self.verbose:
                    print(f"[TritonOCRClient] 推理失败: {e}")
                results.append(None)
        
        return results

    def close(self):
        """关闭连接"""
        self.client.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class TritonOCRWrapper:
    """
    封装层：让 Triton 客户端的接口与你原来的 RapidOCR 完全一致
    
    用法：
        # 直接替换原来的初始化
        # self.ocr = RapidOCR(config_path="tools/rec/1.yaml")
        self.ocr = TritonOCRWrapper(url="localhost:8001")
        
        # 调用方式完全不变
        result = self.ocr(gray_frame)
        texts = result.txts
    """
    
    def __init__(self, url: str = "localhost:8001", model_name: str = "ocr_orchestrator"):
        self.client = TritonOCRClient(url=url, model_name=model_name)
    
    def __call__(self, img):
        """模拟 RapidOCR 的调用接口"""
        texts = self.client.recognize(img)
        return _FakeResult(texts)
    
    def recognize(self, img):
        """兼容你的 recognize 接口"""
        return self.client.recognize(img)


class _FakeResult:
    """模拟 RapidOCR 的返回对象"""
    def __init__(self, texts):
        self.txts = texts
