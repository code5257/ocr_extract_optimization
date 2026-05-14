"""
Det 后处理：Python Backend

功能：
1. 接收 det 模型输出的概率图
2. DB 后处理：二值化 → 找轮廓 → 计算文本框
3. 将检测框映射回原图坐标
4. 输出文本框坐标列表
"""

import numpy as np
import cv2
import triton_python_backend_utils as pb_utils
from shapely.geometry import Polygon
import pyclipper


class TritonPythonModel:
    def initialize(self, args):
        self.thresh = 0.3
        self.box_thresh = 0.5
        self.max_candidates = 1000
        self.unclip_ratio = 1.6
        self.min_size = 3

    def execute(self, requests):
        responses = []
        
        for request in requests:
            det_output = pb_utils.get_input_tensor_by_name(request, "DET_OUTPUT").as_numpy()
            orig_shape = pb_utils.get_input_tensor_by_name(request, "ORIGINAL_SHAPE").as_numpy()
            
            original_h, original_w = int(orig_shape[0]), int(orig_shape[1])
            
            # 获取概率图 (1, H, W) → (H, W)
            pred = det_output[0]
            
            # DB 后处理
            boxes = self._boxes_from_bitmap(pred, original_h, original_w)
            
            # 转为固定格式输出
            if len(boxes) > 0:
                boxes_array = np.array(boxes, dtype=np.float32)
            else:
                boxes_array = np.zeros((0, 4, 2), dtype=np.float32)
            
            num_boxes = np.array([len(boxes)], dtype=np.int32)
            
            boxes_tensor = pb_utils.Tensor("BOXES", boxes_array)
            num_tensor = pb_utils.Tensor("NUM_BOXES", num_boxes)
            
            responses.append(pb_utils.InferenceResponse([boxes_tensor, num_tensor]))
        
        return responses

    def _boxes_from_bitmap(self, pred, original_h, original_w):
        """从概率图提取文本框"""
        bitmap = (pred > self.thresh).astype(np.uint8)
        
        height, width = bitmap.shape
        
        contours, _ = cv2.findContours(
            bitmap, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
        )
        
        boxes = []
        for contour in contours[:self.max_candidates]:
            epsilon = 0.002 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            points = approx.reshape(-1, 2)
            
            if len(points) < 4:
                continue
            
            score = self._box_score(pred, contour)
            if score < self.box_thresh:
                continue
            
            # 最小外接矩形
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.array(box)
            
            if min(rect[1]) < self.min_size:
                continue
            
            # Unclip 扩大框
            box = self._unclip(box)
            if box is None:
                continue
            
            # 映射回原图坐标
            box[:, 0] = box[:, 0] * original_w / width
            box[:, 1] = box[:, 1] * original_h / height
            
            boxes.append(box)
        
        return boxes

    def _box_score(self, pred, contour):
        """计算文本框内平均概率"""
        h, w = pred.shape
        box = contour.reshape(-1, 2)
        xmin = np.clip(box[:, 0].min(), 0, w - 1).astype(np.int32)
        xmax = np.clip(box[:, 0].max(), 0, w - 1).astype(np.int32)
        ymin = np.clip(box[:, 1].min(), 0, h - 1).astype(np.int32)
        ymax = np.clip(box[:, 1].max(), 0, h - 1).astype(np.int32)
        
        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
        box[:, 0] = box[:, 0] - xmin
        box[:, 1] = box[:, 1] - ymin
        cv2.fillPoly(mask, [box.astype(np.int32)], 1)
        
        return pred[ymin:ymax + 1, xmin:xmax + 1][mask.astype(bool)].mean()

    def _unclip(self, box):
        """Unclip 扩大检测框"""
        try:
            poly = Polygon(box)
            distance = poly.area * self.unclip_ratio / poly.length
            offset = pyclipper.PyclipperOffset()
            offset.AddPath(
                [tuple(p) for p in box.astype(int)],
                pyclipper.JT_ROUND,
                pyclipper.ET_CLOSEDPOLYGON,
            )
            expanded = offset.Execute(distance)
            if not expanded:
                return None
            
            expanded_box = np.array(expanded[0], dtype=np.float32)
            rect = cv2.minAreaRect(expanded_box)
            box = cv2.boxPoints(rect)
            return np.array(box, dtype=np.float32)
        except Exception:
            return None

    def finalize(self):
        pass
