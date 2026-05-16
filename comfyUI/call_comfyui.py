"""
ComfyUI 工作流调用示例

使用 ComfyUI 的 HTTP API 提交工作流任务并等待结果。

前置条件：
    - ComfyUI 服务已启动（默认端口 8188）
    - 启动命令: python main.py --listen 0.0.0.0 --port 8188

依赖：
    pip install requests websocket-client
"""

import json
import time
import uuid
import urllib.parse
import urllib.request

import requests
import websocket  # pip install websocket-client


COMFYUI_HOST = "127.0.0.1:8188"
WORKFLOW_FILE = "workflow_optimized.json"


def load_workflow(workflow_path: str = WORKFLOW_FILE) -> dict:
    """加载工作流 JSON"""
    with open(workflow_path, "r", encoding="utf-8") as f:
        return json.load(f)


def queue_prompt(workflow: dict, client_id: str) -> str:
    """提交工作流任务"""
    payload = {
        "prompt": workflow,
        "client_id": client_id,
    }
    response = requests.post(
        f"http://{COMFYUI_HOST}/prompt",
        json=payload,
    )
    response.raise_for_status()
    return response.json()["prompt_id"]


def get_history(prompt_id: str) -> dict:
    """获取任务执行历史"""
    response = requests.get(f"http://{COMFYUI_HOST}/history/{prompt_id}")
    response.raise_for_status()
    return response.json()


def get_image(filename: str, subfolder: str, folder_type: str) -> bytes:
    """下载生成的图片"""
    params = {
        "filename": filename,
        "subfolder": subfolder,
        "type": folder_type,
    }
    url = f"http://{COMFYUI_HOST}/view?{urllib.parse.urlencode(params)}"
    return urllib.request.urlopen(url).read()


def wait_for_completion(prompt_id: str, client_id: str, timeout: int = 600) -> dict:
    """通过 WebSocket 等待任务完成"""
    ws = websocket.WebSocket()
    ws.connect(f"ws://{COMFYUI_HOST}/ws?clientId={client_id}")
    
    start_time = time.time()
    
    try:
        while True:
            if time.time() - start_time > timeout:
                raise TimeoutError(f"任务执行超时 ({timeout}s)")
            
            message = ws.recv()
            
            # 二进制消息（中间预览图，跳过）
            if not isinstance(message, str):
                continue
            
            data = json.loads(message)
            
            if data["type"] == "executing":
                payload = data["data"]
                # 当 node 为 None 且 prompt_id 匹配时表示完成
                if payload["node"] is None and payload["prompt_id"] == prompt_id:
                    print(f"[完成] prompt_id={prompt_id}")
                    break
                
                if payload["prompt_id"] == prompt_id:
                    print(f"  执行节点: {payload['node']}")
            
            elif data["type"] == "execution_error":
                raise RuntimeError(f"执行错误: {data['data']}")
    
    finally:
        ws.close()
    
    # 获取结果
    return get_history(prompt_id)


def run_workflow(input_image: str, output_dir: str = "./outputs") -> list[str]:
    """
    执行完整工作流
    
    Args:
        input_image: 输入图片文件名（必须已上传到 ComfyUI 的 input 目录）
        output_dir: 输出目录
    
    Returns:
        生成的图片文件路径列表
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    client_id = str(uuid.uuid4())
    
    # 1. 加载并修改工作流
    workflow = load_workflow()
    workflow["36"]["inputs"]["image"] = input_image
    
    # 可选：随机化种子（避免使用同样的种子）
    # import random
    # workflow["2"]["inputs"]["noise_seed"] = random.randint(0, 2**63 - 1)
    
    # 2. 提交任务
    print(f"[提交] 输入图片: {input_image}")
    prompt_id = queue_prompt(workflow, client_id)
    print(f"[排队] prompt_id={prompt_id}")
    
    # 3. 等待完成
    history = wait_for_completion(prompt_id, client_id)
    
    # 4. 下载结果图片
    output_paths = []
    if prompt_id in history:
        outputs = history[prompt_id]["outputs"]
        for node_id, node_output in outputs.items():
            if "images" not in node_output:
                continue
            for img_info in node_output["images"]:
                img_data = get_image(
                    img_info["filename"],
                    img_info["subfolder"],
                    img_info["type"],
                )
                output_path = os.path.join(output_dir, img_info["filename"])
                with open(output_path, "wb") as f:
                    f.write(img_data)
                output_paths.append(output_path)
                print(f"[保存] {output_path}")
    
    return output_paths


def upload_image(image_path: str) -> str:
    """上传图片到 ComfyUI input 目录"""
    with open(image_path, "rb") as f:
        files = {"image": f}
        data = {"overwrite": "true"}
        response = requests.post(
            f"http://{COMFYUI_HOST}/upload/image",
            files=files,
            data=data,
        )
    response.raise_for_status()
    return response.json()["name"]


# ========================================
# 使用示例
# ========================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("使用方法: python call_comfyui.py <输入图片路径>")
        print("示例: python call_comfyui.py /path/to/image.jpg")
        sys.exit(1)
    
    input_path = sys.argv[1]
    
    # 1. 上传图片到 ComfyUI
    print(f"[上传] {input_path}")
    uploaded_name = upload_image(input_path)
    print(f"[上传完成] {uploaded_name}")
    
    # 2. 执行工作流
    output_paths = run_workflow(uploaded_name)
    
    print(f"\n生成完成！共 {len(output_paths)} 张图片：")
    for path in output_paths:
        print(f"  - {path}")
