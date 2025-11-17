"""MSLite YOLO predictor class wrapper.

封装 MSLite YOLO 推理流程：
1. 从 mindir_path 加载模型（只加载一次）。
2. 传入 np.ndarray(BGR) 图像，执行检测并返回 result_dict。
3. 支持自定义参数（img_size、conf_thres、iou_thres、conf_free、nms_time_limit 等）。

依赖：
    mindspore_lite
    mindyolo.utils.metrics: non_max_suppression, scale_coords, xyxy2xywh
    mindyolo.data: COCO80_TO_COCO91_CLASS

示例：
    from mslite_yolo_predictor import MSLiteYOLODetector
    import cv2
    det = MSLiteYOLODetector('model.mindir', img_size=640)
    img = cv2.imread('image.jpg')
    result = det.predict(img)
    print(result)
"""

from __future__ import annotations

import os
import time
from typing import List, Dict, Any

import cv2
import numpy as np
import mindspore_lite as mslite

from mindyolo.data import COCO80_TO_COCO91_CLASS
from mindyolo.utils.metrics import non_max_suppression, scale_coords, xyxy2xywh
from mindyolo.utils import logger


class MSLiteYOLODetector:
    """封装后的推理类。

    参数:
        mindir_path: 模型 mindir 文件路径。
        img_size: 推理输入尺寸，保持长边等比例缩放后补边到 img_size。
        conf_thres: 置信度阈值。
        iou_thres: NMS IOU 阈值。
        conf_free: 模型输出是否不含 objectness（仅分类置信度）。
        nms_time_limit: NMS 耗时上限秒数。
        device_target: 设备类型，默认 Ascend，可改 CPU / GPU 等（需环境支持）。
        single_cls: 是否视为单类别（若需要外部 names 配合）。
        names: 类别名称列表（用于外部可视化，可选）。
    """

    def __init__(
        self,
        mindir_path: str,
        img_size: int = 640,
        conf_thres: float = 0.25,
        iou_thres: float = 0.7,
        conf_free: bool = True,
        nms_time_limit: float = 60.0,
        device_target: str = "Ascend",
        single_cls: bool = False,
        names: List[str] | None = None,
    ) -> None:
        self.mindir_path = mindir_path
        if not os.path.isfile(self.mindir_path):
            raise FileNotFoundError(f"mindir file not found: {self.mindir_path}")

        self.img_size = img_size
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.conf_free = conf_free
        self.nms_time_limit = nms_time_limit
        self.device_target = device_target
        self.single_cls = single_cls
        self.names = names or []

        # 构建上下文与模型
        logger.info("Initializing MSLite model context...")
        self.context = mslite.Context()
        self.context.target = [self.device_target]
        self.model = mslite.Model()
        logger.info("mslite model init...")
        logger.info("Building model from mindir...")
        self.model.build_from_file(self.mindir_path, mslite.ModelType.MINDIR, self.context)
        self.inputs = self.model.get_inputs()
        logger.info("Model loaded successfully.")

    def _resize_and_pad(self, img: np.ndarray) -> np.ndarray:
        """按原脚本逻辑进行缩放与补边，返回处理后的图像(BGR)。"""
        h_ori, w_ori = img.shape[:2]
        r = self.img_size / max(h_ori, w_ori)
        if r != 1:
            interp = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
            img = cv2.resize(img, (int(w_ori * r), int(h_ori * r)), interpolation=interp)
        h, w = img.shape[:2]
        if h < self.img_size or w < self.img_size:
            new_shape = (self.img_size, self.img_size)
            dh, dw = (new_shape[0] - h) / 2, (new_shape[1] - w) / 2
            top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
            left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
            img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        return img

    def _preprocess(self, img: np.ndarray) -> np.ndarray:
        """BGR np.ndarray -> 模型输入 (1,3,H,W) float32。"""
        img = self._resize_and_pad(img)
        img = img[:, :, ::-1].transpose(2, 0, 1) / 255.0  # BGR->RGB + CHW + 归一化
        img = np.array(img[None], dtype=np.float32)
        return np.ascontiguousarray(img)

    def predict(self, img: np.ndarray) -> Dict[str, Any]:
        """执行推理，返回 result_dict。

        参数:
            img: 输入图像 (BGR, np.ndarray, HxWx3)。
        返回:
            result_dict: {"category_id": [...], "bbox": [[x,y,w,h], ...], "score": [...]}
        """
        if not isinstance(img, np.ndarray) or img.ndim != 3:
            raise ValueError("img must be a HxWx3 numpy.ndarray")
        h_ori, w_ori = img.shape[:2]

        # 预处理
        processed = self._preprocess(img)

        # 推理
        t0 = time.time()
        self.model.resize(self.inputs, [list(processed.shape)])
        self.inputs[0].set_data_from_numpy(processed)
        outputs = self.model.predict(self.inputs)
        outputs = [o.get_data_to_numpy().copy() for o in outputs]
        out = outputs[0]
        infer_time = time.time() - t0

        # NMS
        t1 = time.time()
        logger.info("perform nms...")
        out = non_max_suppression(
            out,
            conf_thres=self.conf_thres,
            iou_thres=self.iou_thres,
            conf_free=self.conf_free,
            multi_label=True,
            time_limit=self.nms_time_limit,
        )
        nms_time = time.time() - t1

        result_dict = {"category_id": [], "bbox": [], "score": []}
        total_category_ids, total_bboxes, total_scores = [], [], []

        for pred in out:
            if len(pred) == 0:
                continue
            predn = np.copy(pred)
            # 缩放回原图尺寸
            scale_coords(processed.shape[2:], predn[:, :4], (h_ori, w_ori))
            box = xyxy2xywh(predn[:, :4])
            box[:, :2] -= box[:, 2:] / 2
            for p, b in zip(pred.tolist(), box.tolist()):
                cls_id = int(p[5])
                total_category_ids.append(cls_id)
                total_bboxes.append([int(x) for x in b])
                total_scores.append(round(p[4], 5))

        result_dict["category_id"].extend(total_category_ids)
        result_dict["bbox"].extend(total_bboxes)
        result_dict["score"].extend(total_scores)

        logger.info(
            "Speed: %.1f/%.1f/%.1f ms inference/NMS/total for %gx%g image." % (
                infer_time * 1e3,
                nms_time * 1e3,
                (infer_time + nms_time) * 1e3,
                self.img_size,
                self.img_size,
            )
        )
        logger.info(f"Predict result: {result_dict}")
        logger.info("Detect a image success.")
        return result_dict

    __call__ = predict  # 允许实例直接调用


__all__ = ["MSLiteYOLODetector"]


if __name__ == "__main__":
    # 简单自测示例（需替换真实路径）
    import argparse

    parser = argparse.ArgumentParser("MSLite YOLO Predictor Demo")
    parser.add_argument("--mindir", required=True, help="Path to mindir model")
    parser.add_argument("--image", required=True, help="Path to image file")
    parser.add_argument("--img_size", type=int, default=640)
    parser.add_argument("--conf_thres", type=float, default=0.25)
    parser.add_argument("--iou_thres", type=float, default=0.7)
    parser.add_argument("--nms_time_limit", type=float, default=60.0)
    parser.add_argument("--conf_free", type=bool, default=True)
    args = parser.parse_args()

    logger.setup_logging(logger_name="MindYOLO", log_level="INFO")
    det = MSLiteYOLODetector(
        mindir_path=args.mindir,
        img_size=args.img_size,
        conf_thres=args.conf_thres,
        iou_thres=args.iou_thres,
        conf_free=args.conf_free,
        nms_time_limit=args.nms_time_limit,
    )
    img = cv2.imread(args.image)
    if img is None:
        raise FileNotFoundError(f"Image not found: {args.image}")
    res = det.predict(img)
    print(res)