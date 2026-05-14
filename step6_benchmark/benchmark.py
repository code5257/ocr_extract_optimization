"""
阶段6：性能对比基准测试

对比：
1. 原始方案 - RapidOCR torch 逐帧
2. Triton 方案 - TensorRT + Dynamic Batching

使用方法：
    python benchmark.py --video test_video.mp4 --triton-url localhost:8001
"""

import argparse
import time
import cv2
import numpy as np
from typing import Any


def benchmark_original(video_path: str, frame_interval: int, area: dict, config: str) -> dict:
    """测试原始方案性能"""
    from rapidocr import RapidOCR
    
    engine = RapidOCR(config_path=config)
    
    cap = cv2.VideoCapture(video_path)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    
    y_s, y_e = int(area["y_start"]), int(area["y_end"])
    x_s, x_e = int(area["x_start"]), int(area["x_end"])
    
    frames_processed = 0
    total_ocr_ms = 0
    frame_idx = 0
    
    started = time.perf_counter()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_interval > 0 and frame_idx % frame_interval == 0:
            crop = frame[y_s:y_e, x_s:x_e]
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            
            ocr_start = time.perf_counter()
            result = engine(gray)
            total_ocr_ms += (time.perf_counter() - ocr_start) * 1000
            
            frames_processed += 1
        
        frame_idx += 1
    
    cap.release()
    total_ms = (time.perf_counter() - started) * 1000
    
    return {
        "method": "原始方案 (RapidOCR torch)",
        "total_ms": total_ms,
        "ocr_ms": total_ocr_ms,
        "frames_processed": frames_processed,
        "throughput_fps": frames_processed / (total_ocr_ms / 1000) if total_ocr_ms > 0 else 0,
        "avg_latency_ms": total_ocr_ms / frames_processed if frames_processed > 0 else 0,
    }


def benchmark_triton(video_path: str, frame_interval: int, area: dict, triton_url: str) -> dict:
    """测试 Triton 方案性能"""
    from triton_ocr_client import TritonOCRClient
    
    client = TritonOCRClient(url=triton_url)
    
    cap = cv2.VideoCapture(video_path)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    
    y_s, y_e = int(area["y_start"]), int(area["y_end"])
    x_s, x_e = int(area["x_start"]), int(area["x_end"])
    
    # 先抽所有帧
    frames = []
    frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_interval > 0 and frame_idx % frame_interval == 0:
            crop = frame[y_s:y_e, x_s:x_e]
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            frames.append(gray)
        
        frame_idx += 1
    
    cap.release()
    
    # 批量推理
    started = time.perf_counter()
    results = client.recognize_batch(frames)
    total_ocr_ms = (time.perf_counter() - started) * 1000
    
    total_ms = total_ocr_ms  # 这里只计 OCR 时间，公平对比
    
    return {
        "method": "Triton 方案 (TensorRT + Batching)",
        "total_ms": total_ms,
        "ocr_ms": total_ocr_ms,
        "frames_processed": len(frames),
        "throughput_fps": len(frames) / (total_ocr_ms / 1000) if total_ocr_ms > 0 else 0,
        "avg_latency_ms": total_ocr_ms / len(frames) if len(frames) > 0 else 0,
    }


def print_comparison(result_original: dict, result_triton: dict):
    """打印对比结果"""
    print("\n" + "=" * 70)
    print("  性能对比结果")
    print("=" * 70)
    print(f"\n{'指标':<25} {'原始方案':<20} {'Triton方案':<20} {'提升':<10}")
    print("-" * 70)
    
    # OCR 总耗时
    orig_ocr = result_original["ocr_ms"]
    tri_ocr = result_triton["ocr_ms"]
    speedup_ocr = orig_ocr / tri_ocr if tri_ocr > 0 else 0
    print(f"{'OCR 总耗时':<25} {orig_ocr:<20.0f} {tri_ocr:<20.0f} {speedup_ocr:<10.1f}x")
    
    # 吞吐
    orig_fps = result_original["throughput_fps"]
    tri_fps = result_triton["throughput_fps"]
    speedup_fps = tri_fps / orig_fps if orig_fps > 0 else 0
    print(f"{'吞吐 (帧/秒)':<25} {orig_fps:<20.1f} {tri_fps:<20.1f} {speedup_fps:<10.1f}x")
    
    # 平均延迟
    orig_lat = result_original["avg_latency_ms"]
    tri_lat = result_triton["avg_latency_ms"]
    print(f"{'平均延迟 (ms/帧)':<25} {orig_lat:<20.2f} {tri_lat:<20.2f}")
    
    # 处理帧数
    print(f"{'处理帧数':<25} {result_original['frames_processed']:<20} {result_triton['frames_processed']:<20}")
    
    print("-" * 70)
    print(f"\n总结: Triton 方案比原始方案快 {speedup_ocr:.1f} 倍")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="OCR 性能对比基准测试")
    parser.add_argument("--video", required=True, help="测试视频路径")
    parser.add_argument("--frame-interval", type=int, default=25, help="抽帧间隔")
    parser.add_argument("--triton-url", default="localhost:8001", help="Triton gRPC 地址")
    parser.add_argument("--config", default="tools/rec/1.yaml", help="RapidOCR 配置文件")
    parser.add_argument("--y-start", type=int, default=750)
    parser.add_argument("--y-end", type=int, default=931)
    parser.add_argument("--x-start", type=int, default=0)
    parser.add_argument("--x-end", type=int, default=1280)
    
    args = parser.parse_args()
    
    area = {
        "y_start": args.y_start,
        "y_end": args.y_end,
        "x_start": args.x_start,
        "x_end": args.x_end,
    }
    
    print("=" * 70)
    print("  OCR 性能对比基准测试")
    print("=" * 70)
    print(f"  视频: {args.video}")
    print(f"  抽帧间隔: {args.frame_interval}")
    print(f"  字幕区域: y[{area['y_start']}:{area['y_end']}] x[{area['x_start']}:{area['x_end']}]")
    print("")
    
    # 测试原始方案
    print("[1/2] 测试原始方案 (RapidOCR torch)...")
    try:
        result_original = benchmark_original(args.video, args.frame_interval, area, args.config)
        print(f"  完成: {result_original['ocr_ms']:.0f}ms, {result_original['throughput_fps']:.1f} 帧/秒")
    except Exception as e:
        print(f"  跳过: {e}")
        result_original = {
            "method": "原始方案",
            "total_ms": 0, "ocr_ms": 0,
            "frames_processed": 0, "throughput_fps": 0, "avg_latency_ms": 0,
        }
    
    # 测试 Triton 方案
    print(f"\n[2/2] 测试 Triton 方案 (url={args.triton_url})...")
    try:
        result_triton = benchmark_triton(args.video, args.frame_interval, area, args.triton_url)
        print(f"  完成: {result_triton['ocr_ms']:.0f}ms, {result_triton['throughput_fps']:.1f} 帧/秒")
    except Exception as e:
        print(f"  跳过: {e}")
        result_triton = {
            "method": "Triton方案",
            "total_ms": 0, "ocr_ms": 0,
            "frames_processed": 0, "throughput_fps": 0, "avg_latency_ms": 0,
        }
    
    # 对比
    if result_original["ocr_ms"] > 0 and result_triton["ocr_ms"] > 0:
        print_comparison(result_original, result_triton)
    else:
        print("\n部分测试未完成，无法对比")


if __name__ == "__main__":
    main()
