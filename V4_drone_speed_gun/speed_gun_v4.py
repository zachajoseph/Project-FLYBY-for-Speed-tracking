# V4_drone_speed_gun/speed_gun_v4.py
import time
import math
import numpy as np
from pymavlink import mavutil

from .config import *
from .mavlink_client import MavlinkClient
from .airsim_client import AirsimClient
from .sim_car import CarSim
from .estimators import PositionSpeedEstimator
from .geometry import (
    sensor_from_truth, apply_sensor_noise, rel_vec_from_sensor,
    local_to_gps,
)
from .vision_model import VisionMeasurementSource

import cv2  # for vision modes


def build_drone_source():
    if MODE.startswith("airsim"):
        return AirsimClient()
    else:
        return MavlinkClient(MAVLINK_CONNECTION)


def build_car_source(road_origin_vec):
    # For now, always CarSim (even in AirSim modes you can swap to GT later)
    return CarSim(
        road_origin_vec=road_origin_vec,
        road_heading_deg=ROAD_HEADING_DEG,
        speed_mps=CAR_SPEED_MPS,
    )


def build_vision_source():
    if "vision" in MODE:
        return VisionMeasurementSource()
    return None


def main():
    drone = build_drone_source()

    # Origin only exists in MAVLink/SITL world; AirSim doesn't need it.
    origin_lat, origin_lon = 0.0, 0.0
    if isinstance(drone, MavlinkClient):
        origin_lat, origin_lon = drone.get_origin_latlon()
        print(f"[SIM] Using origin lat/lon {origin_lat:.7f}, {origin_lon:.7f}")

    car_link = mavutil.mavlink_connection(
        f"udpout:{QGC_IP}:{QGC_PORT}",
        source_system=CAR_SYSID,
        source_component=1,
    )
    print(f"[CAR] Sending car MAVLink to {QGC_IP}:{QGC_PORT} as sysid {CAR_SYSID}")

    road_origin_vec = np.array([100.0, 0.0, 0.0], dtype=float)
    car = build_car_source(road_origin_vec)
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
    last_car_hb = 0.0
    car_heading_cdeg = int((ROAD_HEADING_DEG % 360.0) * 100)

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

            # --- Car state (truth) ---
            car_state = car.get_state()
            t_car = car_state["t"]
            car_pos_true = car_state["pos_vec"]
            t_now = t_car

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

            # --- Measurement path ---
            if "vision" in MODE:
                if isinstance(drone, AirsimClient):
                    frame = drone.get_rgb_frame()
                else:
                    ret, frame = cap.read()
                    if not ret:
                        print("[VISION] No frame, stopping.")
                        break

                meas = vision.estimate_measurement(frame, drone_alt)
                if meas is None:
                    # Just skip this loop, estimator will lag a bit
                    continue
                R_meas, beta_meas, gamma_meas = meas

            else:
                # synthetic sensor from truth
                R_true, beta_true, gamma_true = sensor_from_truth(drone_pos, car_pos_true)
                R_meas, beta_meas, gamma_meas = apply_sensor_noise(
                    R_true, beta_true, gamma_true
                )

            rel_est = rel_vec_from_sensor(R_meas, beta_meas, gamma_meas)
            car_pos_est = drone_pos + rel_est

            delta_true = car_pos_true - drone_pos
            range_true_m = float(np.linalg.norm(delta_true))

            # --- Speed estimation (est vs truth selector) ---
            if USE_SENSOR_BASED_ESTIMATE:
                v_road_mps = speed_estimator.update(t_car, car_pos_est)
            else:
                v_road_mps = speed_estimator.update(t_car, car_pos_true)

            if v_road_mps is not None:
                est_mph = v_road_mps * 2.23694
                true_mph = CAR_SPEED_MPS * 2.23694
                err_mph  = est_mph - true_mph
                over     = est_mph > (SPEED_LIMIT_MPH + OVER_LIMIT_MARGIN_MPH)

                heading_str = "n/a" if drone_heading_deg is None else f"{drone_heading_deg:5.1f}°"
                print(
                    f"[SpeedGun] Drone (alt {drone_alt:5.1f} m, heading {heading_str},"
                    f" {drone_speed_mps:4.1f} m/s)   "
                    f"Range ~{R_meas:6.1f} m (true {range_true_m:6.1f} m)   "
                    f"Car est {est_mph:5.1f} mph   true {true_mph:5.1f} mph   "
                    f"err {err_mph:5.1f} mph   OVER={over}"
                )

            # --- QGC visualization (always truth for now) ---
            if isinstance(drone, MavlinkClient):
                car_lat, car_lon = local_to_gps(car_pos_true, origin_lat, origin_lon)
                if t_now - last_car_hb > 1.0:
                    car_link.mav.heartbeat_send(
                        mavutil.mavlink.MAV_TYPE_GROUND_ROVER,
                        mavutil.mavlink.MAV_AUTOPILOT_GENERIC,
                        0, 0,
                        mavutil.mavlink.MAV_STATE_ACTIVE,
                    )
                    last_car_hb = t_now

                car_link.mav.global_position_int_send(
                    int((t_now - t_start) * 1000),
                    int(car_lat * 1e7),
                    int(car_lon * 1e7),
                    0, 0, 0, 0, 0,
                    car_heading_cdeg,
                )
                car_link.mav.vfr_hud_send(
                    float(CAR_SPEED_MPS),
                    float(CAR_SPEED_MPS),
                    int(ROAD_HEADING_DEG),
                    0, 0.0, 0.0,
                )

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
