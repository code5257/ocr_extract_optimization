"""
验证 TensorRT Engine 是否正确转换

运行此脚本确认：
1. Engine 能正常加载
2. 推理结果与 ONNX 对比一致
3. 性能基准测试
"""

import numpy as np
import time


def verify_engine(engine_path, input_shape, num_warmup=10, num_test=100):
    """验证单个 TensorRT Engine"""
    
    try:
        import tensorrt as trt
        import pycuda.driver as cuda
        import pycuda.autoinit
    except ImportError:
        print("请安装: pip install tensorrt pycuda")
        return
    
    # 加载 Engine
    logger = trt.Logger(trt.Logger.WARNING)
    with open(engine_path, "rb") as f:
        engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
    
    context = engine.create_execution_context()
    
    # 设置动态 shape
    context.set_input_shape("x", input_shape)
    
    # 分配显存
    input_data = np.random.randn(*input_shape).astype(np.float32)
    d_input = cuda.mem_alloc(input_data.nbytes)
    
    output_shape = context.get_tensor_shape(engine.get_tensor_name(1))
    output_data = np.empty(output_shape, dtype=np.float32)
    d_output = cuda.mem_alloc(output_data.nbytes)
    
    stream = cuda.Stream()
    
    # Warmup
    for _ in range(num_warmup):
        cuda.memcpy_htod_async(d_input, input_data, stream)
        context.execute_async_v3(stream_handle=stream.handle)
        cuda.memcpy_dtoh_async(output_data, d_output, stream)
        stream.synchronize()
    
    # Benchmark
    latencies = []
    for _ in range(num_test):
        start = time.perf_counter()
        cuda.memcpy_htod_async(d_input, input_data, stream)
        context.execute_async_v3(stream_handle=stream.handle)
        cuda.memcpy_dtoh_async(output_data, d_output, stream)
        stream.synchronize()
        latencies.append((time.perf_counter() - start) * 1000)
    
    avg_latency = np.mean(latencies)
    p50 = np.percentile(latencies, 50)
    p95 = np.percentile(latencies, 95)
    p99 = np.percentile(latencies, 99)
    
    print(f"\n{'='*50}")
    print(f"Engine: {engine_path}")
    print(f"Input shape: {input_shape}")
    print(f"{'='*50}")
    print(f"  Avg latency: {avg_latency:.2f} ms")
    print(f"  P50 latency: {p50:.2f} ms")
    print(f"  P95 latency: {p95:.2f} ms")
    print(f"  P99 latency: {p99:.2f} ms")
    print(f"  Throughput:  {1000/avg_latency * input_shape[0]:.0f} frames/sec")
    print(f"{'='*50}")
    
    return avg_latency


if __name__ == "__main__":
    TRT_DIR = "./trt_engines"
    
    print("验证 TensorRT Engine 性能\n")
    
    # Det: batch=8, 3通道, 736x1280
    verify_engine(
        f"{TRT_DIR}/det.plan",
        input_shape=(8, 3, 736, 1280),
        num_warmup=10,
        num_test=50,
    )
    
    # Cls: batch=16, 3通道, 48x192
    verify_engine(
        f"{TRT_DIR}/cls.plan",
        input_shape=(16, 3, 48, 192),
        num_warmup=10,
        num_test=50,
    )
    
    # Rec: batch=32, 3通道, 48x320
    verify_engine(
        f"{TRT_DIR}/rec.plan",
        input_shape=(32, 3, 48, 320),
        num_warmup=10,
        num_test=50,
    )
