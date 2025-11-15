#!/usr/bin/env python3
import time
import math
from collections import deque

import numpy as np
from pymavlink import mavutil

# =========================
# CONFIG
# =========================

# Change LAPTOP_IP to the machine running ArduPilot Docker
MAVLINK_CONNECTION = "udp:127.0.0.1:14550"

# Road and car configuration
ROAD_HEADING_DEG = 0.0        # 0° = along +X in local frame (north in NED)
CAR_SPEED_MPS = 20.0          # ~44.7 mph
SPEED_LIMIT_MPH = 35.0
OVER_LIMIT_MARGIN_MPH = 5.0   # how much over before we flag

# Smoothing / regression settings
WINDOW_SIZE = 15              # number of samples in regression window
LOOP_HZ = 10.0                # how fast we run the loop (Hz)


# =========================
# UTILITIES
# =========================

def unit_vector_from_heading_deg(heading_deg: float) -> np.ndarray:
    """
    Heading in degrees -> unit vector in 2D (x, y, 0).
    Here we treat heading=0 as +X, heading=90 as +Y.
    """
    rad = math.radians(heading_deg)
    return np.array([math.cos(rad), math.sin(rad), 0.0], dtype=float)


# =========================
# MAVLINK DRONE CLIENT
# =========================

class MavlinkClient:
    """
    Connects to ArduPilot/PX4 over MAVLink and provides drone position
    as a 3D vector in LOCAL_POSITION_NED coordinates.
    """
    def __init__(self, connection_str: str):
        print(f"[MAVLINK] Connecting to {connection_str} ...")
        self.master = mavutil.mavlink_connection(connection_str)
        self.master.wait_heartbeat()
        print("[MAVLINK] Heartbeat received. System:",
              self.master.target_system, "Component:", self.master.target_component)

        self.last_pos_vec = None

    def get_drone_position_vec(self) -> np.ndarray | None:
        """
        Returns np.array([x, y, z]) in meters, where x/y/z are taken from
        LOCAL_POSITION_NED. z is flipped to be "up" positive.
        """
        msg = self.master.recv_match(type="LOCAL_POSITION_NED",
                                     blocking=True, timeout=1.0)
        if msg is None:
            return self.last_pos_vec

        # LOCAL_POSITION_NED: x=north, y=east, z=down (m)
        # We'll flip z so that positive is "up"
        self.last_pos_vec = np.array([
            float(msg.x),
            float(msg.y),
            -float(msg.z)
        ], dtype=float)

        return self.last_pos_vec


# =========================
# CAR SIMULATOR
# =========================

class CarSim:
    """
    Very simple car simulator:
    - Car moves along a straight road at constant speed.
    - Road defined by origin vector + heading.
    """
    def __init__(self,
                 road_origin_vec: np.ndarray,
                 road_heading_deg: float,
                 speed_mps: float):
        self.origin = road_origin_vec
        self.dir_vec = unit_vector_from_heading_deg(road_heading_deg)
        self.speed_mps = speed_mps
        self.t0 = time.time()

    def get_state(self) -> dict:
        """
        Returns:
        {
            "t": timestamp (s),
            "pos_vec": np.array([x,y,z]) in local frame
        }
        """
        t_now = time.time()
        dt = t_now - self.t0
        s = self.speed_mps * dt   # distance travelled along road
        pos_vec = self.origin + s * self.dir_vec
        return {"t": t_now, "pos_vec": pos_vec}


# =========================
# SPEED ESTIMATOR (VECTOR-BASED)
# =========================

class PositionSpeedEstimator:
    """
    Uses a vector of car positions over time to estimate speed along the road.
    - Projects car position onto road direction.
    - Runs linear regression on (t, projected_pos) over a window.
    """
    def __init__(self, road_heading_deg: float, window_size: int = 15):
        self.dir_vec = unit_vector_from_heading_deg(road_heading_deg)
        self.window_size = window_size
        self.samples = deque(maxlen=window_size)  # (t, s_along_road_m)

    def update(self, t: float, car_pos_vec: np.ndarray) -> float | None:
        """
        Add a new (t, car_pos_vec) sample and return estimated
        road speed in m/s, or None if not enough samples yet.
        """
        # Project car position onto road direction (scalar)
        s = float(np.dot(car_pos_vec, self.dir_vec))
        self.samples.append((t, s))

        if len(self.samples) < 3:
            return None

        ts = np.array([p[0] for p in self.samples], dtype=float)
        ss = np.array([p[1] for p in self.samples], dtype=float)

        # Linear regression: s(t) ≈ a * t + b
        A = np.vstack([ts, np.ones_like(ts)]).T
        a, b = np.linalg.lstsq(A, ss, rcond=None)[0]

        v_road_mps = float(a)
        return v_road_mps


# =========================
# MAIN LOOP
# =========================

def main():
    # Connect to MAVLink (ArduPilot Docker)
    mav = MavlinkClient(MAVLINK_CONNECTION)

    # Define a simple road and car in local frame
    # Road origin at (100, 0, 0): 100 m "ahead" in +X from NED origin.
    road_origin_vec = np.array([100.0, 0.0, 0.0], dtype=float)

    car = CarSim(
        road_origin_vec=road_origin_vec,
        road_heading_deg=ROAD_HEADING_DEG,
        speed_mps=CAR_SPEED_MPS
    )

    speed_estimator = PositionSpeedEstimator(
        road_heading_deg=ROAD_HEADING_DEG,
        window_size=WINDOW_SIZE
    )

    print("[SIM] Starting loop. Press Ctrl+C to quit.")
    dt_target = 1.0 / LOOP_HZ

    try:
        while True:
            t_loop_start = time.time()

            # 1) Get drone position vector from MAVLink
            drone_pos = mav.get_drone_position_vec()
            if drone_pos is None:
                print("[WARN] No drone position yet...")
                time.sleep(0.1)
                continue

            # 2) Get car position vector from simulator
            car_state = car.get_state()
            t_car = car_state["t"]
            car_pos = car_state["pos_vec"]

            # 3) Compute LOS range (for debug/info)
            delta = car_pos - drone_pos
            range_m = float(np.linalg.norm(delta))

            # 4) Estimate car speed along the road from positions
            v_road_mps = speed_estimator.update(t_car, car_pos)

            if v_road_mps is not None:
                speed_mph = v_road_mps * 2.23694
                over = speed_mph > (SPEED_LIMIT_MPH + OVER_LIMIT_MARGIN_MPH)

                print(
                    f"Range: {range_m:6.1f} m | "
                    f"Speed: {speed_mph:5.1f} mph | "
                    f"Limit: {SPEED_LIMIT_MPH} mph | "
                    f"OVER: {over}"
                )

            # Simple fixed loop rate
            elapsed = time.time() - t_loop_start
            sleep_time = dt_target - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n[SIM] Stopped by user.")


if __name__ == "__main__":
    main()
