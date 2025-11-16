# V4_drone_speed_gun/speed_gun_v4.py
import time
import math
import numpy as np

from .config import *
from .mavlink_client import MavlinkClient
from .airsim_client import AirsimClient
from .estimators import PositionSpeedEstimator
from .geometry import rel_vec_from_sensor
from .vision_model import VisionMeasurementSource
from .utils import format_speed_line

import cv2  # for vision modes


def build_drone_source():
    if MODE.startswith("airsim"):
        return AirsimClient()
    else:
        return MavlinkClient(MAVLINK_CONNECTION)



def build_vision_source():
    if "vision" in MODE:
        return VisionMeasurementSource()
    return None


def main():
    """
    Main loop: acquire drone pose and vision measurements, estimate car speed.
    
    - Drone pose comes from AirsimClient or MavlinkClient depending on MODE.
    - Car position is estimated ONLY from vision detection (rel_est).
    - Speed is estimated along the road heading using only estimated car position.
    - No Python-side car simulation; no synthetic sensor noise applied to measurements.
    """
    drone = build_drone_source()

    # Note: origin_lat/lon is only used for potential coordinate transforms;
    # we use local NED coordinates throughout this script.
    if isinstance(drone, MavlinkClient):
        origin_lat, origin_lon = drone.get_origin_latlon()
        print(f"[SIM] Using origin lat/lon {origin_lat:.7f}, {origin_lon:.7f}")

    speed_estimator = PositionSpeedEstimator(
        road_heading_deg=ROAD_HEADING_DEG,
        window_size=WINDOW_SIZE,
    )

    vision = build_vision_source()
    cap = None
    if MODE == "camera-vision":
        cap = cv2.VideoCapture(VIDEO_SOURCE)

    print(f"[SIM] Mode = {MODE}")
    print("[SIM] Starting loop. Ctrl+C to quit.")
    dt_target = 1.0 / LOOP_HZ

    prev_drone_pos = None
    prev_drone_t = None
    t_start = time.time()

    try:
        while True:
            t_loop_start = time.time()

            # --- Drone pose ---
            drone_pos = drone.get_drone_position_vec()
            if drone_pos is None:
                print("[WARN] No drone position yet...")
                time.sleep(0.1)
                continue
            drone_alt = float(drone_pos[2])

            t_now = time.time()

            # --- Drone velocity / heading (for debug print only) ---
            drone_heading_deg = None
            drone_speed_mps = 0.0
            if prev_drone_pos is not None and prev_drone_t is not None:
                dt = t_now - prev_drone_t
                if dt > 0:
                    dpos = drone_pos - prev_drone_pos
                    vx, vy = dpos[0] / dt, dpos[1] / dt
                    drone_speed_mps = math.sqrt(vx*vx + vy*vy)
                    heading_rad = math.atan2(vy, vx)
                    drone_heading_deg = (math.degrees(heading_rad) + 360.0) % 360.0
            prev_drone_pos = drone_pos.copy()
            prev_drone_t = t_now

            # --- Vision measurement path (if enabled) ---
            if "vision" in MODE:
                if isinstance(drone, AirsimClient):
                    frame = drone.get_rgb_frame()
                else:
                    # Real camera mode
                    ret, frame = cap.read()
                    if not ret:
                        print("[VISION] No frame, stopping.")
                        break

                # Get (R, beta, gamma) measurement from vision
                meas = vision.estimate_measurement(frame, drone_alt)
                if meas is None:
                    # No valid detection this iteration; skip to next
                    continue
                R_meas, beta_meas, gamma_meas = meas
            else:
                # Non-vision modes are no longer supported (car only exists in Unreal)
                print("[ERROR] Non-vision modes are not supported. Car exists only in AirSim.")
                break

            # Convert measurement to relative position vector in local frame
            rel_est = rel_vec_from_sensor(R_meas, beta_meas, gamma_meas)
            car_pos_est = drone_pos + rel_est

            # --- Speed estimation using measured car position ---
            v_road_mps = speed_estimator.update(t_now, car_pos_est)

            if v_road_mps is not None:
                est_mph = v_road_mps * 2.23694
                over = est_mph > (SPEED_LIMIT_MPH + OVER_LIMIT_MARGIN_MPH)

                # Format and print status line
                status_line = format_speed_line(
                    drone_alt=drone_alt,
                    drone_heading_deg=drone_heading_deg,
                    drone_speed_mps=drone_speed_mps,
                    R_meas=R_meas,
                    est_mph=est_mph,
                    over=over,
                )
                print(status_line)

            elapsed = time.time() - t_loop_start
            sleep_time = dt_target - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n[SIM] Stopped by user.")
    finally:
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
