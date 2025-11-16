# speedgun/vision_model.py

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from . import config


@dataclass
class VisionResult:
    """
    Result from the vision pipeline.
    Right now, just a placeholder:
    - `frame_vis` is the frame with visualization drawn on it.
    - `car_bbox` is (x1, y1, x2, y2) or None.
    """
    frame_vis: np.ndarray
    car_bbox: Optional[Tuple[int, int, int, int]] = None


class VisionModel:
    """
    Wraps an OpenCV-based model:
    - If USE_DNN_MODEL is False: uses simple Canny edges (no real detection).
    - If USE_DNN_MODEL is True: loads an ONNX model with cv2.dnn.
    """

    def __init__(self):
        self.use_dnn = bool(config.USE_DNN_MODEL)
        self.net = None

        if self.use_dnn:
            print(f"[Vision] Loading DNN model from {config.MODEL_PATH} ...")
            self.net = cv2.dnn.readNetFromONNX(config.MODEL_PATH)
            # Uncomment if you have CUDA and OpenCV built with it:
            # self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            # self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
            print("[Vision] Model loaded.")
        else:
            print("[Vision] Using simple Canny edge placeholder (no real detection).")

    def process_frame(self, frame_bgr: np.ndarray) -> VisionResult:
        """
        Run one frame through the vision pipeline.
        Returns VisionResult with visualization.
        """
        if self.use_dnn and self.net is not None:
            return self._process_frame_dnn(frame_bgr)
        else:
            return self._process_frame_simple(frame_bgr)

    # ---------- Simple placeholder model ----------

    def _process_frame_simple(self, frame_bgr: np.ndarray) -> VisionResult:
        """
        Simple visualization: edge detection + central box.
        Useful sanity check for AirSim + OpenCV wiring.
        """
        vis = frame_bgr.copy()
        gray = cv2.cvtColor(vis, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)

        # Overlay edges in green
        vis[edges != 0] = (0, 255, 0)

        h, w = vis.shape[:2]
        cx, cy = w // 2, h // 2
        box_w, box_h = w // 4, h // 4
        x1, y1 = cx - box_w // 2, cy - box_h // 2
        x2, y2 = cx + box_w // 2, cy + box_h // 2
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)

        return VisionResult(frame_vis=vis, car_bbox=(x1, y1, x2, y2))

    # ---------- DNN model path (skeleton) ----------

    def _process_frame_dnn(self, frame_bgr: np.ndarray) -> VisionResult:
        """
        Skeleton for a DNN-based model. You need to adapt decoding
        to your specific architecture (YOLO, SSD, custom, etc.).
        """
        vis = frame_bgr.copy()
        h, w = vis.shape[:2]

        inp_w, inp_h = config.MODEL_INPUT_SIZE
        blob = cv2.dnn.blobFromImage(
            frame_bgr,
            scalefactor=1 / 255.0,
            size=(inp_w, inp_h),
            mean=(0, 0, 0),
            swapRB=True,
            crop=False,
        )

        self.net.setInput(blob)
        outputs = self.net.forward()  # shape depends on your model

        # TODO: decode `outputs` for your specific model.
        # For now, we just draw a dummy box in the center as a placeholder.
        cx, cy = w // 2, h // 2
        box_w, box_h = w // 4, h // 4
        x1, y1 = cx - box_w // 2, cy - box_h // 2
        x2, y2 = cx + box_w // 2, cy + box_h // 2
        cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(vis, "DNN placeholder", (x1, max(0, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        return VisionResult(frame_vis=vis, car_bbox=(x1, y1, x2, y2))
