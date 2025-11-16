# speedgun/airsim_client.py

import time
from typing import Optional, Tuple

import airsim
import cv2
import numpy as np

from . import config
from .utils import ned_position_to_vec


class AirSimWrapper:
    """
    Simple wrapper around AirSim MultirotorClient:
    - Connect to server
    - Get drone pose (position/orientation)
    - Get car pose (by object name)
    - Get camera frames as OpenCV BGR images
    """

    def __init__(self):
        # You can pass ip/port via constructor if needed; using defaults here.
        self.client = airsim.MultirotorClient(
            ip=config.AIRSIM_IP,
            port=config.AIRSIM_PORT
        )
        self.client.confirmConnection()
        print(f"[AirSim] Connected to {config.AIRSIM_IP}:{config.AIRSIM_PORT}")

        # Optional: take control of the drone
        self.client.enableApiControl(True)
        self.client.armDisarm(True)

    # ---------- Pose helpers ----------

    def get_drone_state(self):
        """
        Returns the full AirSim MultirotorState object.
        """
        return self.client.getMultirotorState()

    def get_drone_position_vec(self) -> np.ndarray:
        """
        Returns drone position as np.array([x, y, z]) in NED (meters).
        """
        state = self.get_drone_state()
        pos = state.kinematics_estimated.position
        return ned_position_to_vec(pos)

    def get_car_position_vec(self) -> Optional[np.ndarray]:
        """
        Get car/world object pose from AirSim by name and convert to position vector.
        Returns None if object not found.
        """
        pose = self.client.simGetObjectPose(config.CAR_OBJECT_NAME)
        # AirSim returns Pose(0,0,0) with NaNs for orientation if not found
        if pose.position is None:
            return None

        return ned_position_to_vec(pose.position)

    # ---------- Image helpers ----------

    def _decode_image_bytes(self, image_bytes: bytes) -> Optional[np.ndarray]:
        """
        Convert AirSim simGetImage() response (PNG bytes) to an OpenCV BGR image.
        """
        if image_bytes is None:
            return None

        img1d = np.frombuffer(image_bytes, dtype=np.uint8)
        if img1d.size == 0:
            return None

        img_bgr = cv2.imdecode(img1d, cv2.IMREAD_COLOR)
        return img_bgr

    def get_rgb_frame(self) -> Optional[np.ndarray]:
        """
        Fetch an RGB frame from the configured camera.
        Returns an OpenCV BGR image or None on failure.
        """
        # Map config.IMAGE_TYPE string to AirSim enum
        if config.IMAGE_TYPE.lower() == "scene":
            img_type = airsim.ImageType.Scene
        else:
            # default to Scene if unknown
            img_type = airsim.ImageType.Scene

        img_bytes = self.client.simGetImage(config.CAMERA_NAME, img_type)
        if img_bytes is None:
            return None

        return self._decode_image_bytes(img_bytes)

    def shutdown(self):
        self.client.armDisarm(False)
        self.client.enableApiControl(False)
        print("[AirSim] API control released.")
        # no explicit close required; GC will handle client
