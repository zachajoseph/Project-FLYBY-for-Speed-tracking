# V4_drone_speed_gun/vision_model.py
import numpy as np
import cv2
import math

class VisionMeasurementSource:
    """
    Wraps OpenCV model to produce (R, beta, gamma) given:
      - frame (BGR)
      - drone altitude (m)
      - camera intrinsics / FOV assumptions
    """
    def __init__(self, camera_fov_deg=60.0):
        self.camera_fov_deg = camera_fov_deg
        # TODO: load your actual model here (YOLO, etc.)

    def _dummy_detect_car_bbox(self, frame):
        """
        TEMP: returns a bbox in (x_min, y_min, x_max, y_max).
        Replace with real detection.
        """
        h, w, _ = frame.shape
        # For now, pretend the car is in the center quarter
        return int(w*0.25), int(h*0.5), int(w*0.75), int(h*0.9)

    def estimate_measurement(self, frame, drone_alt_m):
        """
        Returns (R, beta, gamma) or None if no detection.
        """
        h, w, _ = frame.shape

        x_min, y_min, x_max, y_max = self._dummy_detect_car_bbox(frame)
        cx = 0.5 * (x_min + x_max)
        cy = 0.5 * (y_min + y_max)

        # Simple pinhole-ish depth estimate:
        # assume ground plane and that vertical pixel angle maps to elevation
        fov_rad = math.radians(self.camera_fov_deg)
        # angle from image center in vertical direction
        vy = (cy - h / 2.0) / (h / 2.0)  # [-1,1]
        beta = fov_rad * vy              # very rough

        # range from altitude + elevation
        # model: tan(beta_down) = horizontal / altitude
        beta_down = math.pi/2 - beta
        horizontal = drone_alt_m * math.tan(beta_down)
        R = math.sqrt(horizontal**2 + drone_alt_m**2)

        # gamma: azimuth from center horizontally
        vx = (cx - w / 2.0) / (w / 2.0)
        gamma = fov_rad * vx

        return R, beta, gamma
