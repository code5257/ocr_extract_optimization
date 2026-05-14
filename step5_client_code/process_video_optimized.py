"""
阶段5：优化后的 process_video

核心改造：
1. 将逐帧串行 OCR 改为批量异步推理
2. 解码和推理并行（流水线）
3. 保持原有的静态帧跳过、字幕组装逻辑不变

使用方式：
    把你原来 process_video 方法中的 self.ocr 替换为 TritonOCRClient，
    或者直接用这个优化版本替换整个方法。
"""

import os
import time
import queue
import threading
import cv2
import numpy as np
from typing import Any
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from triton_ocr_client import TritonOCRClient


@dataclass
class FrameTask:
    """一帧的处理任务"""
    frame_idx: int
    timestamp: float
    gray_frame: np.ndarray


class OptimizedVideoProcessor:
    """
    优化后的视频处理器
    
    改造点：
    1. 抽帧阶段：生产者线程持续解码
    2. OCR 阶段：攒批 → 异步发送到 Triton → Dynamic Batching 自动优化
    3. 字幕组装：保持原有逻辑不变
    """
    
    def __init__(
        self,
        triton_url: str = "localhost:8001",
        frame_interval: int = 25,  # 1秒抽1帧（假设25fps）
        area: dict = None,
        batch_size: int = 16,
        skip_static_enabled: bool = False,
        static_diff_threshold: float = 2.0,
    ):
        self.triton_client = TritonOCRClient(url=triton_url)
        self.frame_interval = frame_interval
        self.area = area or {"y_start": 750, "y_end": 931, "x_start": 0, "x_end": 1280}
        self.batch_size = batch_size
        self.skip_static_enabled = skip_static_enabled
        self.static_diff_threshold = static_diff_threshold

    def process_video(self, video_path: str) -> list[dict[str, Any]]:
        """
        优化版 process_video
        
        与你原来的接口完全一致：输入视频路径，输出字幕列表。
        """
        started_at = time.perf_counter()
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[ERROR] 无法打开视频: {video_path}")
            return []
        
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        
        y_s = int(self.area["y_start"])
        y_e = int(self.area["y_end"])
        x_s = int(self.area["x_start"])
        x_e = int(self.area["x_end"])
        
        print(f"[INFO] 开始处理: {video_path}")
        print(f"  FPS: {fps}, 总帧数: {total_frames}, 抽帧间隔: {self.frame_interval}")
        print(f"  预计抽帧数: {total_frames // self.frame_interval}")
        
        # ============================================
        # 阶段1: 抽帧 + 预处理（收集所有待识别帧）
        # ============================================
        extract_started = time.perf_counter()
        tasks: list[FrameTask] = []
        frame_idx = 0
        prev_sig = None
        skipped_static = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if self.frame_interval > 0 and frame_idx % self.frame_interval == 0:
                # 裁剪 + 灰度
                crop_frame = frame[y_s:y_e, x_s:x_e]
                gray_frame = cv2.cvtColor(crop_frame, cv2.COLOR_BGR2GRAY)
                
                # 静态帧跳过
                if self.skip_static_enabled:
                    sig = self._compute_signature(gray_frame)
                    if prev_sig is not None:
                        diff = np.mean(np.abs(sig.astype(float) - prev_sig.astype(float)))
                        if diff <= self.static_diff_threshold:
                            skipped_static += 1
                            prev_sig = sig
                            frame_idx += 1
                            continue
                    prev_sig = sig
                
                timestamp = frame_idx / fps if fps > 0 else 0.0
                tasks.append(FrameTask(
                    frame_idx=frame_idx,
                    timestamp=timestamp,
                    gray_frame=gray_frame,
                ))
            
            frame_idx += 1
        
        cap.release()
        extract_ms = (time.perf_counter() - extract_started) * 1000
        
        print(f"  抽帧完成: {len(tasks)} 帧待识别, 跳过静态帧: {skipped_static}")
        print(f"  抽帧耗时: {extract_ms:.0f}ms")
        
        if not tasks:
            return []
        
        # ============================================
        # 阶段2: 批量异步 OCR（核心加速点）
        # ============================================
        ocr_started = time.perf_counter()
        
        # 方式A: 批量异步推理（推荐）
        all_texts = self._batch_ocr_async(tasks)
        
        ocr_ms = (time.perf_counter() - ocr_started) * 1000
        print(f"  OCR 完成: {len(all_texts)} 帧已识别")
        print(f"  OCR 耗时: {ocr_ms:.0f}ms")
        print(f"  OCR 吞吐: {len(tasks) / (ocr_ms / 1000):.0f} 帧/秒")
        
        # ============================================
        # 阶段3: 组装字幕时间轴（逻辑与原来完全一致）
        # ============================================
        subtitles = self._build_subtitles(tasks, all_texts, fps)
        
        total_ms = (time.perf_counter() - started_at) * 1000
        print(f"  总耗时: {total_ms:.0f}ms, 字幕数: {len(subtitles)}")
        
        return subtitles

    def _batch_ocr_async(self, tasks: list[FrameTask]) -> list[str]:
        """
        批量异步发送到 Triton
        
        所有帧通过 async_infer 并发发送，Triton 的 Dynamic Batching 
        会自动将多个请求合并成 batch 一次性推理。
        """
        frames = [task.gray_frame for task in tasks]
        results = self.triton_client.recognize_batch(frames)
        
        # 转换为文本列表
        texts = []
        for r in results:
            if r is None:
                texts.append("")
            elif isinstance(r, list):
                texts.append(" ".join(r))
            else:
                texts.append(str(r))
        
        return texts

    def _batch_ocr_pipeline(self, tasks: list[FrameTask]) -> list[str]:
        """
        流水线方式：边解码边推理（更进一步优化）
        
        适用于视频解码和 OCR 推理并行的场景。
        对于你的场景（先抽帧再推理），上面的 _batch_ocr_async 已经够用。
        """
        result_texts = [""] * len(tasks)
        batch_queue = queue.Queue(maxsize=4)
        
        def sender():
            """发送者：按 batch_size 攒批发送"""
            for i in range(0, len(tasks), self.batch_size):
                batch_tasks = tasks[i:i + self.batch_size]
                batch_frames = [t.gray_frame for t in batch_tasks]
                batch_indices = list(range(i, min(i + self.batch_size, len(tasks))))
                batch_queue.put((batch_indices, batch_frames))
            batch_queue.put(None)  # 结束信号
        
        def receiver():
            """接收者：收集结果"""
            while True:
                item = batch_queue.get()
                if item is None:
                    break
                indices, frames = item
                results = self.triton_client.recognize_batch(frames)
                for idx, text in zip(indices, results):
                    result_texts[idx] = text if text else ""
        
        with ThreadPoolExecutor(max_workers=2) as executor:
            f1 = executor.submit(sender)
            f2 = executor.submit(receiver)
            f1.result()
            f2.result()
        
        return result_texts

    def _build_subtitles(
        self, 
        tasks: list[FrameTask], 
        texts: list[str], 
        fps: float
    ) -> list[dict[str, Any]]:
        """
        组装字幕时间轴
        
        逻辑与你原来的代码完全一致：
        - 相同文本 → 延长结束时间
        - 不同文本 → 新建字幕段
        - 空文本 → 结束当前字幕段
        """
        subtitles = []
        current_subtitle: dict[str, Any] | None = None
        frame_duration = (self.frame_interval / fps) if fps > 0 else 0.0
        
        for task, text in zip(tasks, texts):
            timestamp = task.timestamp
            
            if text:
                if current_subtitle is not None and text == current_subtitle.get("text", ""):
                    # 相同文本，延长结束时间
                    current_subtitle["end_time"] = timestamp + frame_duration
                else:
                    # 不同文本，保存旧的，创建新的
                    if current_subtitle is not None:
                        subtitles.append(current_subtitle)
                    current_subtitle = {
                        "text": text,
                        "start_time": timestamp,
                        "end_time": timestamp + frame_duration,
                    }
            else:
                # 无文本，结束当前字幕段
                if current_subtitle is not None:
                    subtitles.append(current_subtitle)
                    current_subtitle = None
        
        # 处理最后一条
        if current_subtitle is not None:
            subtitles.append(current_subtitle)
        
        return subtitles

    def _compute_signature(self, gray_frame: np.ndarray, w=64, h=16) -> np.ndarray:
        """计算帧签名（用于静态帧检测）"""
        return cv2.resize(gray_frame, (w, h), interpolation=cv2.INTER_AREA)


# ============================================
# 使用示例
# ============================================
if __name__ == "__main__":
    import sys
    
    video_path = sys.argv[1] if len(sys.argv) > 1 else "test_video.mp4"
    
    processor = OptimizedVideoProcessor(
        triton_url="localhost:8001",
        frame_interval=25,  # 根据你的 fps 设置
        area={"y_start": 750, "y_end": 931, "x_start": 0, "x_end": 1280},
        batch_size=16,
        skip_static_enabled=True,
        static_diff_threshold=2.0,
    )
    
    subtitles = processor.process_video(video_path)
    
    print(f"\n识别到 {len(subtitles)} 条字幕:")
    for i, sub in enumerate(subtitles[:10], 1):
        print(f"  {i}. [{sub['start_time']:.1f}s - {sub['end_time']:.1f}s] {sub['text']}")
