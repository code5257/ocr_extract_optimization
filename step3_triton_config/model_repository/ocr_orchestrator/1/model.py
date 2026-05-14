"""
OCR Orchestrator：使用 Triton BLS (Business Logic Scripting) 编排整个 OCR 流程

这是核心编排模块，负责：
1. 接收灰度图
2. 预处理 → 调用 det_model（文本检测）
3. 从原图中裁剪文本框
4. 批量调用 rec_model（文本识别）
5. 拼接结果返回

使用 BLS 的优势：
- 在 Triton 内部调用其他模型，零网络开销
- det 和 rec 各自享受 Dynamic Batching
- 一帧检测出多个框时，rec 自动攒批
"""

import numpy as np
import cv2
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    def initialize(self, args):
        # 模型参数
        self.det_model_name = "det_model"
        self.rec_model_name = "rec_model"
        self.cls_model_name = "cls_model"
        
        # Det 预处理参数
        self.limit_side_len = 736
        self.det_mean = np.array([0.5, 0.5, 0.5], dtype=np.float32).reshape(3, 1, 1)
        self.det_std = np.array([0.5, 0.5, 0.5], dtype=np.float32).reshape(3, 1, 1)
        
        # Det 后处理参数
        self.det_thresh = 0.3
        self.det_box_thresh = 0.5
        self.unclip_ratio = 1.6
        
        # Rec 参数
        self.rec_image_shape = [3, 48, 320]
        self.text_score = 0.5
        
        # 加载字典
        self._load_dictionary(args)
        
        print("[ocr_orchestrator] 初始化完成")

    def _load_dictionary(self, args):
        """加载 PP-OCR 字符字典"""
        import os
        dict_path = os.path.join(
            args["model_repository"],
            "ocr_orchestrator",
            "ppocr_keys_v1.txt"
        )
        
        self.character = [""]  # blank for CTC
        if os.path.exists(dict_path):
            with open(dict_path, "r", encoding="utf-8") as f:
                for line in f:
                    self.character.append(line.strip("\n"))
            self.character.append(" ")
            print(f"[ocr_orchestrator] 字典加载完成，{len(self.character)} 字符")
        else:
            print(f"[ocr_orchestrator] 警告：字典文件不存在 {dict_path}")

    def execute(self, requests):
        responses = []
        
        for request in requests:
            image = pb_utils.get_input_tensor_by_name(request, "IMAGE").as_numpy()
            
            # 执行完整 OCR 流程
            text, confidence = self._full_ocr(image)
            
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

    def _full_ocr(self, gray_image):
        """完整 OCR 流程"""
        
        # 1. 灰度 → BGR
        if gray_image.ndim == 2:
            img = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
        elif gray_image.shape[2] == 1:
            img = cv2.cvtColor(gray_image.squeeze(2), cv2.COLOR_GRAY2BGR)
        else:
            img = gray_image
        
        original_h, original_w = img.shape[:2]
        
        # 2. Det 预处理
        det_input = self._det_preprocess(img)
        
        # 3. 调用 Det 模型 (BLS)
        det_output = self._infer_det(det_input)
        
        # 4. Det 后处理 → 得到文本框
        boxes = self._det_postprocess(det_output, original_h, original_w, det_input.shape[2], det_input.shape[3])
        
        if len(boxes) == 0:
            return "", 0.0
        
        # 5. 裁剪文本框 + Rec 预处理
        rec_inputs = []
        for box in boxes:
            crop = self._crop_and_resize(img, box)
            if crop is not None:
                rec_inputs.append(crop)
        
        if len(rec_inputs) == 0:
            return "", 0.0
        
        # 6. 批量调用 Rec 模型 (BLS)
        texts, confidences = self._infer_rec_batch(rec_inputs)
        
        # 7. 拼接文本
        valid_texts = []
        valid_confs = []
        for t, c in zip(texts, confidences):
            if c >= self.text_score and t.strip():
                valid_texts.append(t)
                valid_confs.append(c)
        
        final_text = " ".join(valid_texts)
        final_conf = float(np.mean(valid_confs)) if valid_confs else 0.0
        
        return final_text, final_conf

    def _det_preprocess(self, img):
        """Det 预处理"""
        h, w = img.shape[:2]
        
        # Resize
        ratio = 1.0
        if min(h, w) < self.limit_side_len:
            ratio = self.limit_side_len / min(h, w)
        
        new_h = max(32, int(np.ceil(h * ratio / 32) * 32))
        new_w = max(32, int(np.ceil(w * ratio / 32) * 32))
        
        resized = cv2.resize(img, (new_w, new_h))
        
        # HWC → CHW, normalize
        img_float = resized.astype(np.float32) / 255.0
        img_chw = img_float.transpose(2, 0, 1)
        img_norm = (img_chw - self.det_mean) / self.det_std
        
        return img_norm[np.newaxis, ...]  # (1, 3, H, W)

    def _infer_det(self, det_input):
        """通过 BLS 调用 det_model"""
        input_tensor = pb_utils.Tensor("x", det_input.astype(np.float32))
        
        infer_request = pb_utils.InferenceRequest(
            model_name=self.det_model_name,
            requested_output_names=["sigmoid_0.tmp_0"],
            inputs=[input_tensor],
        )
        
        infer_response = infer_request.exec()
        
        if infer_response.has_error():
            raise Exception(f"Det 推理失败: {infer_response.error().message()}")
        
        output = pb_utils.get_output_tensor_by_name(infer_response, "sigmoid_0.tmp_0")
        return output.as_numpy()

    def _det_postprocess(self, pred, original_h, original_w, det_h, det_w):
        """DB 后处理：概率图 → 文本框"""
        # pred shape: (1, 1, H, W)
        bitmap = pred[0][0]
        
        mask = (bitmap > self.det_thresh).astype(np.uint8)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        boxes = []
        for contour in contours[:1000]:
            if cv2.contourArea(contour) < 10:
                continue
            
            # 计算区域内平均得分
            x, y, w, h = cv2.boundingRect(contour)
            roi = bitmap[y:y+h, x:x+w]
            mask_roi = np.zeros_like(roi, dtype=np.uint8)
            contour_shifted = contour - np.array([x, y])
            cv2.fillPoly(mask_roi, [contour_shifted], 1)
            score = roi[mask_roi > 0].mean() if mask_roi.sum() > 0 else 0
            
            if score < self.det_box_thresh:
                continue
            
            # 最小外接矩形
            rect = cv2.minAreaRect(contour)
            if min(rect[1]) < 3:
                continue
            
            box = cv2.boxPoints(rect)
            
            # Unclip
            box = self._unclip(box, self.unclip_ratio)
            if box is None:
                continue
            
            # 映射回原图
            box[:, 0] = box[:, 0] * original_w / det_w
            box[:, 1] = box[:, 1] * original_h / det_h
            
            boxes.append(box)
        
        return boxes

    def _unclip(self, box, unclip_ratio):
        """扩大检测框"""
        try:
            import pyclipper
            from shapely.geometry import Polygon
            
            poly = Polygon(box)
            if poly.area == 0:
                return None
            distance = poly.area * unclip_ratio / poly.length
            
            offset = pyclipper.PyclipperOffset()
            offset.AddPath(
                [tuple(int(x) for x in p) for p in box],
                pyclipper.JT_ROUND,
                pyclipper.ET_CLOSEDPOLYGON,
            )
            expanded = offset.Execute(distance)
            if not expanded:
                return None
            
            points = np.array(expanded[0], dtype=np.float32)
            rect = cv2.minAreaRect(points)
            new_box = cv2.boxPoints(rect)
            return new_box.astype(np.float32)
        except Exception:
            return None

    def _crop_and_resize(self, img, box):
        """从原图裁剪文本框并 resize 到 rec 输入尺寸"""
        # 透视变换裁剪
        box = box.astype(np.float32)
        
        # 排序四个角点
        box = self._order_points(box)
        
        width = max(
            np.linalg.norm(box[0] - box[1]),
            np.linalg.norm(box[2] - box[3])
        )
        height = max(
            np.linalg.norm(box[0] - box[3]),
            np.linalg.norm(box[1] - box[2])
        )
        
        if width < 3 or height < 3:
            return None
        
        # 目标矩形
        dst = np.array([
            [0, 0],
            [width, 0],
            [width, height],
            [0, height],
        ], dtype=np.float32)
        
        M = cv2.getPerspectiveTransform(box, dst)
        crop = cv2.warpPerspective(img, M, (int(width), int(height)))
        
        # 如果高度大于宽度，旋转
        if height > width * 1.5:
            crop = cv2.rotate(crop, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        # Resize 到 rec 输入：(3, 48, W)
        target_h = self.rec_image_shape[1]  # 48
        ratio = target_h / crop.shape[0]
        target_w = min(int(crop.shape[1] * ratio), self.rec_image_shape[2])
        target_w = max(target_w, 10)
        
        resized = cv2.resize(crop, (target_w, target_h))
        
        # 归一化 + CHW
        img_float = resized.astype(np.float32) / 255.0
        img_float = (img_float - 0.5) / 0.5
        
        if img_float.ndim == 2:
            img_float = np.stack([img_float] * 3, axis=0)
        else:
            img_float = img_float.transpose(2, 0, 1)
        
        # Pad 到固定宽度
        _, h, w = img_float.shape
        if w < self.rec_image_shape[2]:
            pad_w = self.rec_image_shape[2] - w
            img_float = np.pad(img_float, ((0,0), (0,0), (0, pad_w)), constant_values=0)
        
        return img_float

    def _order_points(self, pts):
        """排序四个点：左上、右上、右下、左下"""
        rect = np.zeros((4, 2), dtype=np.float32)
        s = pts.sum(axis=1)
        rect[0] = pts[s.argmin()]
        rect[2] = pts[s.argmax()]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[diff.argmin()]
        rect[3] = pts[diff.argmax()]
        return rect

    def _infer_rec_batch(self, rec_inputs):
        """批量调用 rec_model (BLS)"""
        texts = []
        confidences = []
        
        # 分批推理（每批最多 batch_size 个）
        batch_size = 32
        
        for i in range(0, len(rec_inputs), batch_size):
            batch = rec_inputs[i:i + batch_size]
            
            # Stack 成一个 batch
            batch_array = np.stack(batch, axis=0).astype(np.float32)
            
            input_tensor = pb_utils.Tensor("x", batch_array)
            
            infer_request = pb_utils.InferenceRequest(
                model_name=self.rec_model_name,
                requested_output_names=["softmax_0.tmp_0"],
                inputs=[input_tensor],
            )
            
            infer_response = infer_request.exec()
            
            if infer_response.has_error():
                # 如果批量失败，逐个重试
                for single_input in batch:
                    texts.append("")
                    confidences.append(0.0)
                continue
            
            output = pb_utils.get_output_tensor_by_name(
                infer_response, "softmax_0.tmp_0"
            ).as_numpy()
            
            # 逐个解码
            for j in range(len(batch)):
                if j < output.shape[0]:
                    text, conf = self._ctc_decode(output[j])
                    texts.append(text)
                    confidences.append(conf)
                else:
                    texts.append("")
                    confidences.append(0.0)
        
        return texts, confidences

    def _ctc_decode(self, pred):
        """CTC 贪心解码"""
        pred_indices = pred.argmax(axis=1)
        pred_probs = pred.max(axis=1)
        
        text = ""
        conf_list = []
        prev_idx = -1
        
        for i, idx in enumerate(pred_indices):
            if idx == 0:  # blank
                prev_idx = idx
                continue
            if idx == prev_idx:  # 重复
                continue
            if idx < len(self.character):
                text += self.character[idx]
                conf_list.append(float(pred_probs[i]))
            prev_idx = idx
        
        confidence = float(np.mean(conf_list)) if conf_list else 0.0
        return text, confidence

    def finalize(self):
        pass
