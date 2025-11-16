#!/usr/bin/env python3
"""
speed_gun.py

Vision-based speed gun:

- Drone pose comes from ArduPilot (MAVLink LOCAL_POSITION_NED) or AirSim.
- Camera frames come from:
    * OpenCV video source (e.g. USB cam, RTSP), or
    * AirSim RGB camera.
- Vision model produces a bounding box and (R, beta, gamma) measurement
  from the drone to the car.
- We back-project to 3D and estimate the car's speed along a known road
  direction using a ~2 second sliding window average.

Local frame convention:
  x = north, y = east, z = up

MAVLink LOCAL_POSITION_NED:
  x = north, y = east, z = down  (we flip z)
"""

import math
import time
from collections import deque
from typing import Optional, Tuple

import cv2
import numpy as np
from pymavlink import mavutil

try:
    import airsim
except ImportError:
    airsim = None

# ---------------------------------------------------------------------------
# Configuration (formerly config.py, now embedded here)
# ---------------------------------------------------------------------------

# --- Backends / modes -------------------------------------------------
# "airsim-vision":  AirSim drone + camera vision detection
# "camera-vision":  Real camera + MAVLink drone, OpenCV detection
MODE = "camera-vision"

# For non-AirSim camera (e.g. USB cam or RTSP URL)
VIDEO_SOURCE = 0  # cv2.VideoCapture index or RTSP URL

# --- MAVLink (for drone pose only) -----------------------------------
MAVLINK_CONNECTION = "udp:127.0.0.1:14550"

# --- Road / speed config ---------------------------------------------
ROAD_HEADING_DEG = 0.0        # 0° = north (+x), 90° = east (+y)
SPEED_LIMIT_MPH = 35.0
OVER_LIMIT_MARGIN_MPH = 5.0   # mph over the limit before we flag

# --- Estimator / loop timing -----------------------------------------
LOOP_HZ = 10.0
WINDOW_SIZE = 20              # samples kept; with LOOP_HZ=10 → ~2 s window
WINDOW_SEC = WINDOW_SIZE / LOOP_HZ

# --- Sensor model (kept for completeness; not heavily used here) -----
RANGE_NOISE_STD_M = 0.5
BETA_NOISE_STD_DEG = 0.5
GAMMA_NOISE_STD_DEG = 1.0


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def mps_to_mph(v_mps: float) -> float:
    return v_mps * 2.2369362920544


class LoopTimer:
    def __init__(self, hz: float):
        self.period = 1.0 / hz
        self.next_time = time.time()

    def sleep_to_rate(self):
        self.next_time += self.period
        dt = self.next_time - time.time()
        if dt > 0:
            time.sleep(dt)
        else:
            # If we fell behind badly, reset the schedule
            self.next_time = time.time()


# ---------------------------------------------------------------------------
# MAVLink client (embedded version of mavlink_client.py)
# ---------------------------------------------------------------------------

class MavlinkClient:
    def __init__(self, connection_str: str):
        print(f"[MAVLINK] Connecting to {connection_str} ...")
        self.master = mavutil.mavlink_connection(connection_str)
        self.master.wait_heartbeat()
        print(
            "[MAVLINK] Heartbeat received.",
            "System:", self.master.target_system,
            "Component:", self.master.target_component,
        )

        self.last_pos_vec = None
        self.origin_lat = None
        self.origin_lon = None

    def get_drone_position_vec(self) -> Optional[np.ndarray]:
        """
        Return drone position in local frame [x, y, z_up] (meters).

        Uses LOCAL_POSITION_NED: x=north, y=east, z=down and flips z.
        """
        msg = self.master.recv_match(
            type="LOCAL_POSITION_NED",
            blocking=True,
            timeout=1.0
        )
        if msg is None:
            return self.last_pos_vec

        self.last_pos_vec = np.array(
            [float(msg.x), float(msg.y), -float(msg.z)],
            dtype=float,
        )
        return self.last_pos_vec


# ---------------------------------------------------------------------------
# AirSim client (optional; only used if MODE == "airsim-vision")
# ---------------------------------------------------------------------------

class AirsimClient:
    def __init__(self, vehicle_name="Drone1"):
        if airsim is None:
            raise RuntimeError(
                "airsim package not installed, but MODE='airsim-vision'. "
                "Install airsim or change MODE."
            )

        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True, vehicle_name=vehicle_name)
        self.client.armDisarm(True, vehicle_name=vehicle_name)
        self.vehicle = vehicle_name
        print("[AIRSIM] Connected as", vehicle_name)

    def get_drone_position_vec(self) -> np.ndarray:
        state = self.client.getMultirotorState(vehicle_name=self.vehicle)
        pos = state.kinematics_estimated.position
        # AirSim: NED; flip z to get "up"
        return np.array([pos.x_val, pos.y_val, -pos.z_val], dtype=float)

    def get_rgb_frame(self, camera_name="0") -> Optional[np.ndarray]:
        resp = self.client.simGetImages(
            [airsim.ImageRequest(
                camera_name,
                airsim.ImageType.Scene,
                pixels_as_float=False,
                compress=False,
            )],
            vehicle_name=self.vehicle,
        )[0]
        if resp.width == 0 or resp.height == 0:
            return None

        img1d = np.frombuffer(resp.image_data_uint8, dtype=np.uint8)
        img_bgr = img1d.reshape(resp.height, resp.width, 3)
        return img_bgr


# ---------------------------------------------------------------------------
# Geometry helpers (embedded from geometry.py)
# ---------------------------------------------------------------------------

def unit_vector_from_heading_deg(heading_deg: float) -> np.ndarray:
    rad = math.radians(heading_deg)
    return np.array([math.cos(rad), math.sin(rad), 0.0], dtype=float)


def rel_vec_from_sensor(R: float, beta: float, gamma: float) -> np.ndarray:
    """
    Convert sensor measurement (R, beta, gamma) to a 3D vector in local frame.

    Convention (same as geometry.sensor_from_truth):
      - beta: 0 = nadir (straight down), pi/2 = horizontal
      - gamma: azimuth in the horizontal plane
    """
    sinb, cosb = math.sin(beta), math.cos(beta)
    cosg, sing = math.cos(gamma), math.sin(gamma)

    dx = R * sinb * cosg
    dy = R * sinb * sing
    dz = -R * cosb

    return np.array([dx, dy, dz], dtype=float)


# ---------------------------------------------------------------------------
# Vision model (embedded simplified version of vision_model.py)
# ---------------------------------------------------------------------------

class VisionMeasurementSource:
    """
    Very simple placeholder vision model:

    - Picks a bounding box near the bottom-center of the frame
      (pretend that's the car).
    - Uses camera FOV + drone altitude to estimate (R, beta, gamma).

    Replace _dummy_detect_car_bbox() and estimate_measurement()
    with your real detector (YOLO/TensorRT/etc) when ready.
    """

    def __init__(self, camera_fov_deg: float = 60.0):
        self.camera_fov_deg = camera_fov_deg

    def _dummy_detect_car_bbox(self, frame: np.ndarray):
        """
        TEMP: returns a bbox in (x_min, y_min, x_max, y_max).

        Right now: central bottom quarter of the frame.
        """
        h, w, _ = frame.shape
        return int(w * 0.25), int(h * 0.5), int(w * 0.75), int(h * 0.9)

    def estimate_measurement(self, frame: np.ndarray, drone_alt_m: float):
        """
        Returns (R, beta, gamma) or None if no detection.

        - R: range from drone to car [m]
        - beta: elevation relative to nadir (down) [rad]
        - gamma: azimuth around +Z [rad]
        """
        h, w, _ = frame.shape

        # In a real system you'd:
        #   - run detection model,
        #   - pick best car-like bounding box,
        #   - use that box for geometry.
        x_min, y_min, x_max, y_max = self._dummy_detect_car_bbox(frame)

        cx = 0.5 * (x_min + x_max)
        cy = 0.5 * (y_min + y_max)

        fov_rad = math.radians(self.camera_fov_deg)

        # Vertical offset from image center in [-1, 1]
        vy = (cy - h / 2.0) / (h / 2.0)
        # Approx elevation angle offset from optical axis
        beta = fov_rad * vy

        # Convert to a very rough range estimate:
        # model: tan(beta_down) = horizontal / altitude
        beta_down = math.pi / 2.0 - beta
        horizontal = drone_alt_m * math.tan(beta_down)
        R = math.sqrt(horizontal ** 2 + drone_alt_m ** 2)

        # Horizontal offset -> azimuth
        vx = (cx - w / 2.0) / (w / 2.0)
        gamma = fov_rad * vx

        return R, beta, gamma


# ---------------------------------------------------------------------------
# Speed estimator with ~2 s windowed average along the road
# ---------------------------------------------------------------------------

class PositionSpeedEstimator:
    """
    Keeps a sliding window of samples and returns the average speed along
    the road direction over that window:

        v = (s_last - s_first) / (t_last - t_first)

    where s = projection of car position onto the road direction.
    """

    def __init__(self, road_heading_deg: float, loop_hz: float, window_size: int):
        self.dir_vec = unit_vector_from_heading_deg(road_heading_deg)
        self.samples = deque()  # (t, s_along_road)
        self.window_sec = window_size / loop_hz

    def update(self, t: float, car_pos_vec: np.ndarray) -> Optional[float]:
        s = float(np.dot(car_pos_vec, self.dir_vec))
        self.samples.append((t, s))

        # Drop samples older than window_sec
        t_min = t - self.window_sec
        while len(self.samples) >= 2 and self.samples[0][0] < t_min:
            self.samples.popleft()

        if len(self.samples) < 2:
            return None

        t0, s0 = self.samples[0]
        t1, s1 = self.samples[-1]
        dt = t1 - t0
        if dt <= 0:
            return None

        return (s1 - s0) / dt  # m/s along the road


# ---------------------------------------------------------------------------
# Heading + drone speed from positions
# ---------------------------------------------------------------------------

def heading_and_speed_from_positions(
    prev_pos: Optional[np.ndarray],
    prev_t: Optional[float],
    cur_pos: np.ndarray,
    cur_t: float,
) -> Tuple[Optional[float], float]:
    """
    Estimate drone heading (deg) and horizontal speed (m/s)
    from consecutive position samples.
    """
    if prev_pos is None or prev_t is None:
        return None, 0.0

    dt = cur_t - prev_t
    if dt <= 0.0:
        return None, 0.0

    dpos = cur_pos - prev_pos
    vx = float(dpos[0]) / dt  # north
    vy = float(dpos[1]) / dt  # east
    speed = math.hypot(vx, vy)

    if speed < 1e-3:
        return None, 0.0

    heading_rad = math.atan2(vy, vx)  # atan2(east, north)
    heading_deg = (math.degrees(heading_rad) + 360.0) % 360.0
    return heading_deg, speed


# ---------------------------------------------------------------------------
# Drawing helper
# ---------------------------------------------------------------------------

def draw_bbox(frame: np.ndarray, bbox, color=(0, 255, 0), thickness=2):
    if bbox is None:
        return
    x_min, y_min, x_max, y_max = bbox
    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, thickness)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"[MAIN] MODE={MODE}, MAVLINK={MAVLINK_CONNECTION}, VIDEO_SOURCE={VIDEO_SOURCE}")
    timer = LoopTimer(LOOP_HZ)

    # Pose + frame source
    mav = None
    airsim_client = None
    cap = None

    if MODE == "camera-vision":
        mav = MavlinkClient(MAVLINK_CONNECTION)
        cap = cv2.VideoCapture(VIDEO_SOURCE)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open VIDEO_SOURCE={VIDEO_SOURCE}")

    elif MODE == "airsim-vision":
        airsim_client = AirsimClient()
    else:
        raise RuntimeError(f"Unsupported MODE={MODE}. Use 'camera-vision' or 'airsim-vision'.")

    vision = VisionMeasurementSource()
    estimator = PositionSpeedEstimator(
        road_heading_deg=ROAD_HEADING_DEG,
        loop_hz=LOOP_HZ,
        window_size=WINDOW_SIZE,
    )

    prev_drone_pos = None
    prev_t = None

    try:
        while True:
            t_now = time.time()

            # 1) Drone position
            if MODE == "camera-vision":
                drone_pos = mav.get_drone_position_vec()
            else:
                drone_pos = airsim_client.get_drone_position_vec()

            if drone_pos is None:
                print("[WARN] No drone position yet; skipping this cycle.")
                timer.sleep_to_rate()
                continue

            drone_alt = float(drone_pos[2])

            # 2) Drone heading + speed
            drone_heading_deg, drone_speed_mps = heading_and_speed_from_positions(
                prev_drone_pos, prev_t, drone_pos, t_now
            )
            prev_drone_pos = drone_pos.copy()
            prev_t = t_now

            # 3) Frame
            if MODE == "camera-vision":
                ret, frame = cap.read()
                if not ret:
                    print("[WARN] Could not read frame from camera; exiting.")
                    break
            else:
                frame = airsim_client.get_rgb_frame()
                if frame is None:
                    print("[WARN] AirSim returned empty frame; skipping.")
                    timer.sleep_to_rate()
                    continue

            # 4) Vision measurement
            meas = vision.estimate_measurement(frame, drone_alt_m=drone_alt)
            if meas is None:
                # No detection, just show frame
                cv2.imshow("SpeedGun", frame)
                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    break
                timer.sleep_to_rate()
                continue

            R_meas, beta_meas, gamma_meas = meas

            # 5) Back-project to car position in local frame
            rel_vec = rel_vec_from_sensor(R_meas, beta_meas, gamma_meas)
            car_pos_est = drone_pos + rel_vec

            # 6) Speed estimation along road
            v_road_mps = estimator.update(t_now, car_pos_est)
            est_mph = None
            over = False
            if v_road_mps is not None:
                est_mph = mps_to_mph(v_road_mps)
                over = est_mph > (SPEED_LIMIT_MPH + OVER_LIMIT_MARGIN_MPH)

            # 7) Draw bbox + HUD
            bbox = vision._dummy_detect_car_bbox(frame)
            draw_bbox(frame, bbox, color=(0, 255, 0) if not over else (0, 0, 255))

            hud_lines = []
            hud_lines.append(f"Alt: {drone_alt:4.1f} m")
            if drone_heading_deg is not None:
                hud_lines.append(f"Hdg: {drone_heading_deg:5.1f} deg")
            hud_lines.append(f"Drone spd: {drone_speed_mps:4.1f} m/s")
            hud_lines.append(f"Range: {R_meas:5.1f} m")

            if est_mph is not None:
                hud_lines.append(f"Car: {est_mph:5.1f} mph")
                if over:
                    hud_lines.append("** OVER LIMIT **")

            y0 = 20
            for i, text in enumerate(hud_lines):
                y = y0 + i * 20
                cv2.putText(
                    frame,
                    text,
                    (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0) if not over else (0, 0, 255),
                    1,
                    cv2.LINE_AA,
                )

            cv2.imshow("SpeedGun", frame)
            if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                break

            timer.sleep_to_rate()

    finally:
        print("[MAIN] Shutting down.")
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
