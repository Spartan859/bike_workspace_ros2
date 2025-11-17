#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Camera AI Utilities for RealSense D455 (MindSpore Lite YOLO)

本模块提供 `CameraAI` 类，集成 Intel RealSense D455 深度相机与基于 MindSpore Lite 的 YOLO 检测：

更新说明：
- 已移除旧的 acllite + .om 模型部署方式。
- 改用封装类 `MSLiteYOLODetector`（来自 `mslite_yolo_predictor.py`），加载 MINDIR 模型进行推理。
- 后台线程循环中直接调用封装类的 `predict` 获取类别、bbox、score，再计算人员距离。

主要功能：
1. 初始化 RealSense 彩色与深度流，并对齐深度到彩色。
2. 后台线程持续抓取图像并进行 YOLO 目标检测（仅关心 person）。
3. 为每个检测到的人员提供像素级包围框与深度中心点距离（米）。
4. 提供安全距离检测与是否需要后退的辅助方法。

依赖：
    pyrealsense2
    mindspore_lite (通过封装类间接使用)
    mslite_yolo_predictor.MSLiteYOLODetector
"""

import sys
import os
import threading
import time
import cv2
import numpy as np
from typing import Optional, Tuple, List

try:
    import pyrealsense2 as rs
except ImportError as e:
    print(f"Error importing pyrealsense2: {e}")
    sys.exit(1)

try:
    from mslite_yolo_predictor import MSLiteYOLODetector
except ImportError as e:
    print(f"Failed to import MSLiteYOLODetector: {e}")
    sys.exit(1)

# --- Constants ---
CLASS_NAMES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
    'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
    'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
    'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
    'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
    'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
    'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
    'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]
PERSON_CLASS_ID = CLASS_NAMES.index('person')

class CameraAI:
    """
    A class to manage RealSense camera, NPU-accelerated AI detection,
    and depth calculation in a separate thread.
    """
    def __init__(self, mindir_path: str = "./yolov8x.mindir", visualize: bool = False,
                 conf_thres: float = 0.25, iou_thres: float = 0.7,
                 nms_time_limit: float = 60.0, conf_free: bool = True,
                 device_target: str = "Ascend", detection_interval: int = 1,):
        """
        Initializes the CameraAI system.

        Args:
            mindir_path (str): MINDIR 模型路径，对应 MSLiteYOLODetector 的 mindir_path。
            visualize (bool): 是否生成深度热力图。
            conf_thres (float): 置信度阈值，默认 0.25（与 MSLiteYOLODetector 保持一致）。
            iou_thres (float): NMS IOU 阈值，默认 0.7（与 MSLiteYOLODetector 保持一致）。
            nms_time_limit (float): NMS 耗时上限秒数，默认 60.0（与 MSLiteYOLODetector 保持一致）。
            conf_free (bool): 模型是否为 conf-free 输出，默认 True（与 MSLiteYOLODetector 保持一致）。
            device_target (str): 推理设备，默认 "Ascend"。
        """
        self._visualize = visualize
        self._mindir_path = mindir_path
        self._detector: Optional[MSLiteYOLODetector] = None
        self._detector_args = dict(
            mindir_path=self._mindir_path,
            img_size=640,
            conf_thres=conf_thres,
            iou_thres=iou_thres,
            conf_free=conf_free,
            nms_time_limit=nms_time_limit,
            device_target=device_target,
        )
        
        self._pipeline = None
        self._align = None
        
        self._detection_thread = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        
        # Shared data between threads
        self._latest_detections = []
        self._latest_color_frame = None
        self._latest_depth_frame = None
        self._latest_depth_heatmap = None
        self.is_running = False
        self._detection_interval = detection_interval

        # 旧预处理参数已不再使用（封装类内部处理）

    def start(self) -> bool:
        """
        Initializes the camera and starts the background detection thread.
        NPU resources will be initialized within the background thread.

        Returns:
            bool: True if camera started successfully, False otherwise.
        """
        if self.is_running:
            print("CameraAI is already running.")
            return True
            
        print("Starting CameraAI service...")
        # 模型将在后台线程初始化
        if not self._initialize_camera():
            return False
        
        self._stop_event.clear()
        self._detection_thread = threading.Thread(target=self._run_detection_loop, daemon=True)
        self._detection_thread.start()
        self.is_running = True
        print("✅ CameraAI service started successfully (NPU initializing in background).")
        return True

    def stop(self):
        """Stops the detection thread and releases all resources."""
        if not self.is_running:
            return
            
        print("Stopping CameraAI service...")
        self._stop_event.set()
        if self._detection_thread:
            self._detection_thread.join(timeout=5)
        
        if self._pipeline:
            try:
                self._pipeline.stop()
            except Exception as e:
                print(f"Error stopping RealSense pipeline: {e}")
                
        # MSLite YOLO Detector 资源随对象释放
        print("🧹 Resources cleaned up.")
        self.is_running = False

    def get_latest_person_detections(self) -> list[dict]:
        """
        Get the latest list of detected persons with their distances.
        This method is thread-safe.

        Returns:
            list[dict]: A list of dictionaries, where each dictionary
                        represents a detected person and contains:
                        {'box': [x, y, w, h], 'distance_m': float}
        """
        with self._lock:
            return self._latest_detections.copy()

    def get_latest_visuals_and_detections(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], List[dict]]:
        """
        Get the latest color frame, depth heatmap, and the list of detected persons.
        This method is thread-safe.

        Returns:
            A tuple containing:
                - The latest color frame as a numpy array, or None.
                - The latest depth heatmap as a numpy array, or None.
                - A list of person detection dictionaries.
        """
        with self._lock:
            return (
                self._latest_color_frame.copy() if self._latest_color_frame is not None else None,
                self._latest_depth_heatmap.copy() if self._latest_depth_heatmap is not None else None,
                self._latest_detections.copy()
            )

    def is_safe(self, person_safe_dist: float = 1.5, obstacle_safe_dist: float = 1.0, obstacle_threshold_ratio: float = 0.05) -> Tuple[bool, str]:
        """
        Performs a comprehensive safety check.

        Returns True only if all of these conditions are met:
        1. All detected persons are farther than `person_safe_dist`.
        2. The central area in front of the camera is clear of any obstacles
           closer than `obstacle_safe_dist`.

        This method is thread-safe.

        Args:
            person_safe_dist (float): The minimum safe distance for persons (meters).
            obstacle_safe_dist (float): The minimum safe distance for general obstacles (meters).
            obstacle_threshold_ratio (float): The percentage of pixels in the central
                                            area that must be close to trigger an
                                            obstacle warning. Defaults to 5%.

        Returns:
            Tuple[bool, str]: A tuple containing a boolean (True if safe) and a string message.
        """
        with self._lock:
            # Condition 0: Check if data is available
            if self._latest_depth_frame is None:
                return False, "WARNING: No depth data available"

            # print("latest detections:", self._latest_detections)
            # Condition 1: Check for persons too close
            for person in self._latest_detections:
                dist = person.get('distance_m', 0.0)
                if 0.01 < dist < person_safe_dist:
                    return False, f"STOP: Person too close at {dist:.2f}m"

            # Condition 2: Check for general obstacles in front
            depth_frame = self._latest_depth_frame
            h, w = depth_frame.get_height(), depth_frame.get_width()
            
            # Define a Region of Interest (ROI) in the center-bottom of the view
            roi_x_start, roi_x_end = w // 4, w * 3 // 4
            roi_y_start, roi_y_end = h // 2, h
            
            roi = np.asanyarray(depth_frame.get_data())[roi_y_start:roi_y_end, roi_x_start:roi_x_end]
            
            # Find pixels in the ROI that are closer than the obstacle distance
            close_obstacle_pixels = roi[(roi < obstacle_safe_dist * 1000) & (roi > 10)] # Depth is in mm
            
            # If the number of close pixels exceeds a threshold, it's an obstacle
            roi_area = (roi_x_end - roi_x_start) * (roi_y_end - roi_y_start)
            if len(close_obstacle_pixels) > roi_area * obstacle_threshold_ratio:
                return False, "STOP: Obstacle detected ahead"

            # If all checks pass, it's safe
            return True, "Path Clear"

    #新增is_back方法
    def is_back(self, person_back_dist: float = 1.0) -> bool:
        """
        检测是否需要执行“后退”动作：
        若任意检测到的 person 距离小于 person_back_dist，则返回 True，否则 False。
        :param person_back_dist: 触发后退的距离阈值(米)
        :return: bool -> True 需要后退 / False 不需要
        """
        with self._lock:
            if self._latest_depth_frame is None:
                return False
            for person in self._latest_detections:
                dist = person.get('distance_m', 0.0)
                if 0.01 < dist < person_back_dist:
                    return True
            return False

    def _initialize_detector(self) -> bool:
        """初始化 MSLite YOLO MINDIR 模型。"""
        print("🧠 Initializing MindSpore Lite YOLO detector...")
        try:
            self._detector = MSLiteYOLODetector(**self._detector_args)
            print("✅ YOLO mindir model loaded successfully.")
            return True
        except Exception as e:
            print(f"❌ Failed to load mindir model: {e}")
            return False

    def _initialize_camera(self) -> bool:
        """Initializes the RealSense camera for color and depth streams."""
        print("📷 Initializing RealSense D455 camera...")
        try:
            self._pipeline = rs.pipeline()
            config = rs.config()
            # It's recommended to use the same resolution for color and depth
            # to simplify alignment and processing.
            # 优化：将分辨率从 640x480 降低到 424x240 以减少CPU负载
            config.enable_stream(rs.stream.depth, 424, 240, rs.format.z16, 30)
            config.enable_stream(rs.stream.color, 424, 240, rs.format.bgr8, 30)
            self._pipeline.start(config)
            
            # Create an alignment object (align depth to color)
            self._align = rs.align(rs.stream.color)
            
            # Create colorizer for depth visualization
            self._colorizer = rs.colorizer()
            self._colorizer.set_option(rs.option.color_scheme, 2)  # Jet colormap

            print("✅ RealSense camera initialized successfully.")
            return True
        except Exception as e:
            print(f"❌ Failed to initialize RealSense camera: {e}")
            return False

    def _run_detection_loop(self):
        """
        The main loop that runs in a background thread.
        It handles NPU initialization, frame grabbing, inference, and post-processing.
        """
        # 在后台线程初始化 MindSpore Lite 模型
        if not self._initialize_detector():
            print("❌ Detector initialization failed. Exiting detection loop.")
            self.is_running = False
            return

        # 优化：移除基于计数器的“忙等待”式跳帧，改为在循环末尾使用 time.sleep()
        while not self._stop_event.is_set():
            try:
                # 1. Get Frames
                # wait_for_frames 会阻塞直到新的一帧可用
                frames = self._pipeline.wait_for_frames(timeout_ms=2000)
                if not frames:
                    # 如果超时，短暂休眠后重试
                    time.sleep(0.1)
                    continue
                
                aligned_frames = self._align.process(frames)
                depth_frame = aligned_frames.get_depth_frame()
                color_frame = aligned_frames.get_color_frame()
                
                if not depth_frame or not color_frame:
                    continue
                
                color_image = np.asanyarray(color_frame.get_data())
                
                # Generate depth heatmap if visualization is enabled
                depth_heatmap = None
                if self._visualize:
                    try:
                        # Apply colormap to depth frame
                        depth_colormap_frame = self._colorizer.colorize(depth_frame)
                        depth_heatmap = np.asanyarray(depth_colormap_frame.get_data())
                        
                        # Ensure the heatmap has the same dimensions as color image
                        if depth_heatmap.shape != color_image.shape:
                            depth_heatmap = cv2.resize(depth_heatmap, 
                                                     (color_image.shape[1], color_image.shape[0]))
                    except Exception as e:
                        print(f"Warning: Failed to generate depth heatmap: {e}")
                        depth_heatmap = np.zeros_like(color_image)

                # 2. 直接调用封装类推理
                result = self._detector.predict(color_image)
                # print(result)
                boxes = result.get("bbox", [])  # [x,y,w,h]
                categories = result.get("category_id", [])
                scores = result.get("score", [])

                # 3. 过滤人员并计算距离
                person_detections = []
                for box, cat_id, score in zip(boxes, categories, scores):
                    # 直接与本地 CLASS_NAMES 对齐判断
                    if cat_id == PERSON_CLASS_ID:
                        distance = self._get_robust_distance(depth_frame, box)
                        person_detections.append({
                            'box': box,
                            'distance_m': round(distance, 2),
                            'score': score,
                            'category_id': cat_id
                        })

                # 5. Update shared data
                with self._lock:
                    self._latest_detections = person_detections
                    self._latest_color_frame = color_image
                    self._latest_depth_frame = depth_frame # Store raw depth frame
                    self._latest_depth_heatmap = depth_heatmap

            except Exception as e:
                print(f"Error in detection loop: {e}")
                time.sleep(1)

            # 优化：采用“闹钟”模式，处理完一帧后，线程主动休眠1秒，彻底释放CPU
            time.sleep(self._detection_interval)

    def _get_robust_distance(self, depth_frame: rs.depth_frame, box: list) -> float:
        """
        Calculates a more robust distance by averaging depth from multiple points.

        Args:
            depth_frame: The aligned depth frame from the camera.
            box: The bounding box [x, y, w, h] of the detected object.

        Returns:
            The calculated average distance in meters, or 0.0 if no valid
            depth point is found.
        """
        try:
            x, y, w, h = box
            # 优化：只使用中心点进行距离测量，而不是5个点，以减少计算量
            center_x = x + w // 2
            center_y = y + h // 2

            frame_h, frame_w = depth_frame.get_height(), depth_frame.get_width()
            # print("frame_w, frame_h, center_x, center_y:", frame_w, frame_h, center_x, center_y)
            # print("max in frame:", np.max(np.asanyarray(depth_frame.get_data())))

            # 确保坐标在图像范围内
            px = max(0, min(center_x, frame_w - 1))
            py = max(0, min(center_y, frame_h - 1))
            
            dist = depth_frame.get_distance(px, py)
            # print("dist: ", dist)
            
            # 过滤掉无效读数
            if 0.01 < dist < 20.0:
                return dist
            
            return 0.0
        except Exception as e:
            print("Error calculating robust distance: ", e)
            return 0.0

    # 旧的预处理与后处理函数已移除，使用封装类内部逻辑

    def __del__(self):
        """Ensures resources are released when the object is destroyed."""
        self.stop()
