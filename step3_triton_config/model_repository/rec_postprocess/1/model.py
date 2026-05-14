"""
Rec 后处理：Python Backend

功能：
1. 接收 rec 模型输出的概率矩阵
2. CTC 解码：argmax → 去重 → 去blank → 查字典 → 得到文字
3. 输出识别文本和置信度
"""

import numpy as np
import os
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    def initialize(self, args):
        """加载字符字典"""
        # PP-OCRv4 字典文件路径
        dict_path = os.path.join(
            args["model_repository"],
            "rec_postprocess",
            "ppocr_keys_v1.txt"
        )
        
        self.character = []
        # 添加 blank 符号（CTC 解码需要）
        self.character.append("")  # blank
        
        with open(dict_path, "r", encoding="utf-8") as f:
            for line in f:
                char = line.strip("\n")
                self.character.append(char)
        
        # 添加空格
        self.character.append(" ")
        
        print(f"[rec_postprocess] 字典加载完成，共 {len(self.character)} 个字符")

    def execute(self, requests):
        responses = []
        
        for request in requests:
            rec_output = pb_utils.get_input_tensor_by_name(request, "REC_OUTPUT").as_numpy()
            
            # CTC 解码
            text, confidence = self._ctc_decode(rec_output)
            
            # 输出
            text_tensor = pb_utils.Tensor(
                "TEXT",
                np.array([text.encode("utf-8")], dtype=np.object_)
            )
            conf_tensor = pb_utils.Tensor(
                "CONFIDENCE",
                np.array([confidence], dtype=np.float32)
            )
            
            responses.append(pb_utils.InferenceResponse([text_tensor, conf_tensor]))
        
        return responses

    def _ctc_decode(self, pred):
        """
        CTC 贪心解码
        
        pred: (seq_len, dict_size) 概率矩阵
        """
        # argmax 得到每个时间步最可能的字符索引
        pred_indices = pred.argmax(axis=1)
        pred_probs = pred.max(axis=1)
        
        # 去重 + 去 blank
        text = ""
        confidence_list = []
        prev_idx = -1
        
        for i, idx in enumerate(pred_indices):
            if idx == 0:  # blank
                prev_idx = idx
                continue
            if idx == prev_idx:  # 重复
                continue
            
            if idx < len(self.character):
                text += self.character[idx]
                confidence_list.append(float(pred_probs[i]))
            
            prev_idx = idx
        
        # 计算平均置信度
        confidence = float(np.mean(confidence_list)) if confidence_list else 0.0
        
        return text, confidence

    def finalize(self):
        pass
