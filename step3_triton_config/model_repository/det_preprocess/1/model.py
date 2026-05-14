"""
Det 预处理：Python Backend

功能：
1. 接收灰度图（H x W x 1）
2. 转 3 通道
3. Resize 到 det 模型输入尺寸（保持比例，limit_side_len=736）
4. 归一化（mean=0.5, std=0.5）
5. 输出 float32 tensor
"""

import numpy as np
import cv2
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    def initialize(self, args):
        self.limit_side_len = 736
        self.limit_type = "min"  # PP-OCRv4 det server 用 min
        self.mean = np.array([0.5, 0.5, 0.5], dtype=np.float32).reshape(3, 1, 1)
        self.std = np.array([0.5, 0.5, 0.5], dtype=np.float32).reshape(3, 1, 1)

    def execute(self, requests):
        responses = []
        
        for request in requests:
            raw_image = pb_utils.get_input_tensor_by_name(request, "RAW_IMAGE").as_numpy()
            
            # 灰度 → 3通道
            if raw_image.ndim == 2:
                img = cv2.cvtColor(raw_image, cv2.COLOR_GRAY2BGR)
            elif raw_image.shape[2] == 1:
                img = cv2.cvtColor(raw_image, cv2.COLOR_GRAY2BGR)
            else:
                img = raw_image
            
            original_h, original_w = img.shape[:2]
            
            # Resize（保持比例，limit_side_len）
            img_resized = self._resize_image(img)
            
            # HWC → CHW，归一化
            img_float = img_resized.astype(np.float32) / 255.0
            img_chw = img_float.transpose(2, 0, 1)  # CHW
            img_norm = (img_chw - self.mean) / self.std
            
            # 输出
            preprocessed = pb_utils.Tensor("PREPROCESSED_IMAGE", img_norm.astype(np.float32))
            orig_shape = pb_utils.Tensor(
                "ORIGINAL_SHAPE", 
                np.array([original_h, original_w], dtype=np.int32)
            )
            
            responses.append(pb_utils.InferenceResponse([preprocessed, orig_shape]))
        
        return responses

    def _resize_image(self, img):
        """PP-OCRv4 det 的 resize 逻辑"""
        h, w = img.shape[:2]
        
        if self.limit_type == "min":
            if min(h, w) < self.limit_side_len:
                if h < w:
                    ratio = self.limit_side_len / h
                else:
                    ratio = self.limit_side_len / w
            else:
                ratio = 1.0
        else:  # max
            if max(h, w) > self.limit_side_len:
                if h > w:
                    ratio = self.limit_side_len / h
                else:
                    ratio = self.limit_side_len / w
            else:
                ratio = 1.0
        
        new_h = int(h * ratio)
        new_w = int(w * ratio)
        
        # 对齐到 32 的倍数
        new_h = max(32, int(np.ceil(new_h / 32) * 32))
        new_w = max(32, int(np.ceil(new_w / 32) * 32))
        
        resized = cv2.resize(img, (new_w, new_h))
        return resized

    def finalize(self):
        pass
