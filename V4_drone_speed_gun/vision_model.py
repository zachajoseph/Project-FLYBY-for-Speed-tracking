# V4_drone_speed_gun/vision_model.py
import math
import cv2
import numpy as np

COCO_CAR_LIKE_IDS = {2, 3, 5, 7}  # COCO: car, motorcycle, bus, truck

class VisionMeasurementSource:
    """
    Uses SSD-Mobilenet v3 (COCO) via OpenCV to detect vehicles and
    convert the best bounding box into (R, beta, gamma).
    """

    def __init__(
        self,
        model_path="V4_drone_speed_gun/models/ssd_mobilenet_v3_large_coco_2020_01_14.pb",
        config_path="V4_drone_speed_gun/models/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt",
        camera_fov_deg=60.0,
        conf_thresh=0.5,
        nms_thresh=0.3,
    ):
        self.camera_fov_deg = camera_fov_deg
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh

        net = cv2.dnn_DetectionModel(model_path, config_path)
        net.setInputSize(320, 320)
        net.setInputScale(1.0 / 127.5)
        net.setInputMean((127.5, 127.5, 127.5))
        net.setInputSwapRB(True)
        self.net = net

    def estimate_measurement(self, frame, drone_alt_m):
        """
        Given a BGR frame and drone altitude (m),
        return (R, beta, gamma) or None if no car-like object is found.

        beta: elevation from nadir (downward) [rad]
        gamma: azimuth around +Z [rad]
        """
        h, w, _ = frame.shape

        classes, confidences, boxes = self.net.detect(
            frame,
            confThreshold=self.conf_thresh,
            nmsThreshold=self.nms_thresh,
        )

        if classes is None or len(classes) == 0:
            return None

        # Pick the largest 'car-like' detection by area
        best_box = None
        best_area = 0.0

        for class_id, conf, box in zip(
            classes.flatten(), confidences.flatten(), boxes
        ):
            if int(class_id) not in COCO_CAR_LIKE_IDS:
                continue
            x, y, bw, bh = box
            area = bw * bh
            if area > best_area:
                best_area = area
                best_box = (x, y, bw, bh)

        if best_box is None:
            return None

        x, y, bw, bh = best_box
        cx = x + bw / 2.0
        cy = y + bh / 2.0

        # --- Pixel -> angles (very rough pinhole model) ---
        fov_rad = math.radians(self.camera_fov_deg)

        # Normalize offsets from center to [-1,1]
        vx = (cx - w / 2.0) / (w / 2.0)
        vy = (cy - h / 2.0) / (h / 2.0)

        # horizontal angle (gamma) left/right
        gamma = fov_rad * vx

        # vertical angle from image center -> turn into elevation from nadir
        # assume image center = straight ahead, bottom = closer to nadir
        # scale vy into beta crudely:
        beta = fov_rad * vy

        # --- Range estimate from altitude + elevation ---
        beta_down = math.pi / 2 - beta  # angle from horizontal down
        horizontal = drone_alt_m * math.tan(beta_down)
        R = math.sqrt(horizontal**2 + drone_alt_m**2)

        return R, beta, gamma
