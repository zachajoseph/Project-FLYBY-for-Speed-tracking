#!/usr/bin/env python3
import time
import math

import cv2
import numpy as np
from pymavlink import mavutil

from V4_drone_speed_gun.vision_model import VisionMeasurementSource
from V4_drone_speed_gun.estimators import PositionSpeedEstimator
from V4_drone_speed_gun.geometry import rel_vec_from_sensor

# ========================
# CONFIG
# ========================

# Where ArduPilot SITL or FC is broadcasting MAVLink
DRONE_MAVLINK_URL = "udp:127.0.0.1:14550"

# Where QGC is listening for MAVLink (often same machine/port)
QGC_IP = "127.0.0.1"
QGC_PORT = 14550

CAR_SYSID = 42
ROAD_HEADING_DEG = 0.0
SPEED_LIMIT_MPH = 35.0
OVER_LIMIT_MARGIN_MPH = 5.0
WINDOW_SIZE = 15
MISSED_FRAMES_TO_STOP = 15      # how many consecutive misses before stopping
CAMERA_INDEX = 0                # Orin camera index
FAKE_ALT_M = 30.0               # used if we don't trust GPS alt


# Simple helper: get lat/lon from drone
def get_drone_latlon(master):
    msg = master.recv_match(type="GLOBAL_POSITION_INT", blocking=False)
    if msg is None:
        return None, None
    return msg.lat * 1e-7, msg.lon * 1e-7


def main():
    # --- Connect to ArduPilot (for drone GPS) ---
    print("[DEMO] Connecting to ArduPilot MAVLink...")
    drone_master = mavutil.mavlink_connection(DRONE_MAVLINK_URL)
    drone_master.wait_heartbeat()
    print(f"[DEMO] Drone heartbeat: sys={drone_master.target_system} comp={drone_master.target_component}")

    # --- Create MAVLink connection to QGC for fake car ---
    print("[DEMO] Connecting to QGC for car rover...")
    car_link = mavutil.mavlink_connection(
        f"udpout:{QGC_IP}:{QGC_PORT}",
        source_system=CAR_SYSID,
        source_component=1,
    )

    # --- Set up camera ---
    print("[DEMO] Opening camera...")
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"[ERR] Could not open camera index {CAMERA_INDEX}")
        return

    # --- Vision + estimator ---
    vision = VisionMeasurementSource()
    speed_estimator = PositionSpeedEstimator(
        road_heading_deg=ROAD_HEADING_DEG,
        window_size=WINDOW_SIZE,
    )

    car_active = False
    last_car_hb = 0.0
    missed_frames = 0
    t_start = time.time()

    print("[DEMO] Ready. Press ESC to quit.")

    try:
        while True:
            # --- 1) Read camera frame ---
            ret, frame = cap.read()
            if not ret:
                print("[DEMO] Camera error, exiting.")
                break

            # --- 2) Optionally grab drone altitude/heading (for display) ---
            drone_alt = FAKE_ALT_M
            drone_heading_deg = None

            msg = drone_master.recv_match(type=["GLOBAL_POSITION_INT", "ATTITUDE"], blocking=False)
            drone_lat, drone_lon = None, None

            if msg is not None:
                if msg.get_type() == "GLOBAL_POSITION_INT":
                    drone_alt = msg.relative_alt / 1000.0
                    drone_lat = msg.lat * 1e-7
                    drone_lon = msg.lon * 1e-7
                elif msg.get_type() == "ATTITUDE":
                    yaw = msg.yaw
                    drone_heading_deg = (math.degrees(yaw) + 360.0) % 360.0

            # --- 3) Run CV to estimate measurement ---
            meas = vision.estimate_measurement(frame, drone_alt)

            if meas is None:
                # No detection this frame
                missed_frames += 1
                cv2.putText(frame, "No car detected", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                # If we had an active car and lose it for too long -> stop
                if car_active and missed_frames >= MISSED_FRAMES_TO_STOP:
                    print("[DEMO] Car lost from view. Stopping feed.")
                    break

                cv2.imshow("Orin QGC SpeedGun Demo", frame)
                if (cv2.waitKey(1) & 0xFF) == 27:
                    break
                continue

            # We have a detection
            missed_frames = 0
            R, beta, gamma = meas

            # --- 4) Build a simple local car position estimate ---
            drone_pos = np.array([0.0, 0.0, drone_alt], dtype=float)
            rel_est = rel_vec_from_sensor(R, beta, gamma)
            car_pos_est = drone_pos + rel_est

            # --- 5) Feed estimator ---
            t_now = time.time()
            v_road_mps = speed_estimator.update(t_now, car_pos_est)

            est_mph = None
            over = False
            if v_road_mps is not None:
                est_mph = v_road_mps * 2.23694
                over = est_mph > (SPEED_LIMIT_MPH + OVER_LIMIT_MARGIN_MPH)

            # Activate car on first proper estimate
            if not car_active and est_mph is not None:
                car_active = True
                print("[DEMO] Car detected; starting QGC rover feed.")

            # --- 6) If car active, send MAVLink to QGC ---
            if car_active and est_mph is not None:
                # Heartbeat once per second
                if t_now - last_car_hb > 1.0:
                    car_link.mav.heartbeat_send(
                        mavutil.mavlink.MAV_TYPE_GROUND_ROVER,
                        mavutil.mavlink.MAV_AUTOPILOT_GENERIC,
                        0, 0,
                        mavutil.mavlink.MAV_STATE_ACTIVE,
                    )
                    last_car_hb = t_now

                # Use drone lat/lon if available, else some fixed default
                if drone_lat is None or drone_lon is None:
                    # fallback near some dummy location
                    lat = 34.0
                    lon = -118.0
                else:
                    # place car a bit north of the drone
                    lat = drone_lat + 0.0001  # ~11 m north
                    lon = drone_lon

                car_link.mav.global_position_int_send(
                    int((t_now - t_start) * 1000),  # time_boot_ms
                    int(lat * 1e7),
                    int(lon * 1e7),
                    0,          # alt mm
                    0,          # relative alt mm
                    0, 0, 0,    # vx, vy, vz
                    int((ROAD_HEADING_DEG % 360.0) * 100),
                )

                car_link.mav.vfr_hud_send(
                    float(v_road_mps),  # airspeed m/s
                    float(v_road_mps),  # groundspeed m/s
                    int(ROAD_HEADING_DEG),
                    0,                  # throttle
                    0.0,                # alt
                    0.0,                # climb
                )

                print(f"[SpeedGun] Speed limit={SPEED_LIMIT_MPH:.1f} mph   "
                      f"est={est_mph:5.1f} mph   OVER={over}")

            # --- 7) Draw overlays for humans ---
            overlay = f"Speed limit={SPEED_LIMIT_MPH:.1f} mph"
            if est_mph is not None:
                overlay += f"  est={est_mph:5.1f} mph  OVER={over}"
            else:
                overlay += "  est=--.- mph"

            cv2.putText(frame, overlay, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            if drone_heading_deg is not None:
                cv2.putText(frame, f"Drone heading ~ {drone_heading_deg:5.1f} deg",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (255, 255, 255), 1)

            cv2.imshow("Orin QGC SpeedGun Demo", frame)
            if (cv2.waitKey(1) & 0xFF) == 27:
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[DEMO] Shutdown complete.")


if __name__ == "__main__":
    main()
