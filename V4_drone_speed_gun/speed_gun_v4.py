#!/usr/bin/env python3
"""
sim_speed_gunv3.py

Simulate a ground vehicle (car) moving along a straight road in the local NED frame.
Use the drone as a "speed gun" sensor: derive synthetic range + angle measurements
from the drone to the car, add noise, reconstruct car position, and estimate speed.

Also spoof a GROUND_ROVER to QGroundControl using MAVLink so you can visualize
the car on the map.

Assumptions:
- Local frame: x = north, y = east, z = up.
- MAVLink LOCAL_POSITION_NED: x = north, y = east, z = down (we flip z).
"""

import math
import time
from collections import deque

import numpy as np
from pymavlink import mavutil


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# MAVLink connection to flight controller (SITL or real)
MAVLINK_CONNECTION = "udp:127.0.0.1:14550"

# MAVLink connection to QGroundControl (spoofed car)
QGC_CONNECTION = "udpout:127.0.0.1:14560"
CAR_SYSID = 50          # system ID for spoofed ground vehicle
CAR_COMPID = 1          # component ID for spoofed ground vehicle

# Earth radius for local→GPS conversion (simple flat-earth approx)
EARTH_RADIUS_M = 6_378_137.0

# Car / road config
CAR_SPEED_MPS = 20.0          # 20 m/s ≈ 44.7 mph
ROAD_HEADING_DEG = 0.0        # 0° = north (+x), 90° = east (+y)
ROAD_ORIGIN_VEC = np.array([100.0, 0.0, 0.0], dtype=float)  # start of road in local frame

# Speed limit logic
SPEED_LIMIT_MPH = 35.0
SPEED_LIMIT_MARGIN_MPH = 5.0   # over-limit if est >= limit + margin

# Sensor noise (synthetic LOS sensor)
RANGE_NOISE_STD_M = 2.0        # meters
BETA_NOISE_STD_DEG = 1.0       # elevation noise (deg)
GAMMA_NOISE_STD_DEG = 1.0      # azimuth noise (deg)
MIN_SENSOR_RANGE_M = 1.0       # ignore measurements below this

# Estimator behavior
USE_SENSOR_BASED_ESTIMATE = True   # True: use reconstructed car position; False: use ground truth
WINDOW_SIZE = 15                   # samples for regression window

# Loop
LOOP_RATE_HZ = 10.0                # target control loop rate (Hz)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def mps_to_mph(v_mps: float) -> float:
    return v_mps * 2.2369362920544


def clamp(x: float, xmin: float, xmax: float) -> float:
    return max(xmin, min(xmax, x))


# ---------------------------------------------------------------------------
# MAVLink client to FC (drone)
# ---------------------------------------------------------------------------

class MavlinkClient:
    def __init__(self, conn_str: str):
        self.conn = mavutil.mavlink_connection(conn_str)
        print(f"[MAV] Connecting to FC at {conn_str} ...")
        self.conn.wait_heartbeat()
        print(f"[MAV] Heartbeat from system {self.conn.target_system}, component {self.conn.target_component}")

        # Origin for local-to-GPS mapping
        self.origin_lat = None
        self.origin_lon = None
        self.origin_alt = None

        # Request data streams (just to be explicit; SITL often streams anyway)
        self._request_data_streams()

    def _request_data_streams(self):
        try:
            self.conn.mav.request_data_stream_send(
                self.conn.target_system,
                self.conn.target_component,
                mavutil.mavlink.MAV_DATA_STREAM_POSITION,
                10,  # 10 Hz
                1,
            )
        except Exception:
            # Not fatal; some FCs ignore this
            pass

    def get_origin_gps(self, timeout: float = 10.0):
        """Block until we see a GLOBAL_POSITION_INT to define the origin."""
        if self.origin_lat is not None:
            return self.origin_lat, self.origin_lon, self.origin_alt

        print("[MAV] Waiting for GLOBAL_POSITION_INT to set origin...")
        start = time.time()
        while True:
            msg = self.conn.recv_match(type="GLOBAL_POSITION_INT", blocking=True, timeout=1.0)
            if msg is not None:
                self.origin_lat = msg.lat / 1e7
                self.origin_lon = msg.lon / 1e7
                self.origin_alt = msg.alt / 1000.0
                print(f"[MAV] Origin set: lat={self.origin_lat:.7f}, lon={self.origin_lon:.7f}, alt={self.origin_alt:.1f} m")
                return self.origin_lat, self.origin_lon, self.origin_alt

            if time.time() - start > timeout:
                raise RuntimeError("Timeout waiting for GLOBAL_POSITION_INT for origin")

    def get_drone_state(self, timeout: float = 1.0):
        """
        Returns:
            pos_local: np.array([x, y, z]) in meters (x=north, y=east, z=up)
            vel_local: np.array([vx, vy, vz]) in m/s (x=north, y=east, z=up)
        """
        msg = self.conn.recv_match(type="LOCAL_POSITION_NED", blocking=True, timeout=timeout)
        if msg is None:
            raise RuntimeError("Timeout waiting for LOCAL_POSITION_NED")

        # LOCAL_POSITION_NED: x,y,z in NED (z down), vx,vy,vz in NED (vz down)
        x = float(msg.x)
        y = float(msg.y)
        z_up = -float(msg.z)
        vx = float(msg.vx)
        vy = float(msg.vy)
        vz_up = -float(msg.vz)

        pos = np.array([x, y, z_up], dtype=float)
        vel = np.array([vx, vy, vz_up], dtype=float)
        return pos, vel

    def local_to_gps(self, local_vec: np.ndarray):
        """
        Convert local (x=north, y=east, z=up) to approximate lat, lon, alt.
        Simple flat-earth approximation around origin.
        """
        if self.origin_lat is None:
            self.get_origin_gps()

        d_north = float(local_vec[0])
        d_east = float(local_vec[1])
        d_up = float(local_vec[2])

        d_lat = (d_north / EARTH_RADIUS_M) * (180.0 / math.pi)
        denom = EARTH_RADIUS_M * math.cos(math.radians(self.origin_lat))
        if abs(denom) < 1e-6:
            # Extremely unlikely in practice; avoid division by zero
            d_lon = 0.0
        else:
            d_lon = (d_east / denom) * (180.0 / math.pi)

        lat = self.origin_lat + d_lat
        lon = self.origin_lon + d_lon
        alt = self.origin_alt + d_up
        return lat, lon, alt


# ---------------------------------------------------------------------------
# QGC spoofed car sender
# ---------------------------------------------------------------------------

class QGCCarSender:
    def __init__(self, conn_str: str, sysid: int = CAR_SYSID, compid: int = CAR_COMPID):
        self.sysid = sysid
        self.compid = compid
        print(f"[QGC] Connecting spoofed car at {conn_str} (sysid={sysid}) ...")
        self.mav = mavutil.mavlink_connection(conn_str, source_system=sysid, source_component=compid)

        self.last_heartbeat_time = 0.0
        self.last_gps_time = 0.0
        self.last_vfr_time = 0.0

    def send_heartbeat(self, now: float):
        if now - self.last_heartbeat_time >= 1.0:
            self.last_heartbeat_time = now
            self.mav.mav.heartbeat_send(
                mavutil.mavlink.MAV_TYPE_GROUND_ROVER,
                mavutil.mavlink.MAV_AUTOPILOT_GENERIC,
                0, 0, 0
            )

    def send_state(self, now: float, lat_deg: float, lon_deg: float,
                   alt_m: float, heading_deg: float, groundspeed_mps: float):
        """
        Send GLOBAL_POSITION_INT + VFR_HUD at ~5 Hz to QGC.
        """
        # GLOBAL_POSITION_INT
        if now - self.last_gps_time >= 0.2:
            self.last_gps_time = now
            self.mav.mav.global_position_int_send(
                int(now * 1000),              # time_boot_ms (approx)
                int(lat_deg * 1e7),
                int(lon_deg * 1e7),
                int(alt_m * 1000),            # alt (mm, AMSL)
                int(alt_m * 1000),            # relative_alt (mm)
                int(groundspeed_mps * 100),   # vx (cm/s) - just pack total speed into x
                0,                            # vy
                0,                            # vz
                int(heading_deg * 100),       # hdg (cdeg)
                0
            )

        # VFR_HUD
        if now - self.last_vfr_time >= 0.2:
            self.last_vfr_time = now
            self.mav.mav.vfr_hud_send(
                airspeed=groundspeed_mps,
                groundspeed=groundspeed_mps,
                heading=int(heading_deg),
                throttle=0,
                alt=alt_m,
                climb=0.0
            )


# ---------------------------------------------------------------------------
# Car simulation
# ---------------------------------------------------------------------------

class CarSim:
    """
    Simulates a car driving at constant speed along a straight road in local frame.
    Road is defined by an origin point and a heading.
    """
    def __init__(self, origin_vec: np.ndarray, heading_deg: float, speed_mps: float):
        self.origin = np.array(origin_vec, dtype=float)
        yaw = math.radians(heading_deg)
        dir_vec = np.array([math.cos(yaw), math.sin(yaw), 0.0], dtype=float)
        norm = float(np.linalg.norm(dir_vec))
        if norm < 1e-6:
            raise ValueError("Invalid road heading: direction vector is zero")

        self.dir_vec = dir_vec / norm
        self.speed = float(speed_mps)

    def get_state(self, t: float):
        """
        Args:
            t: time since start (seconds)
        Returns:
            pos: np.array([x, y, z]) position in local frame
            vel: np.array([vx, vy, vz]) velocity in local frame
        """
        pos = self.origin + self.dir_vec * self.speed * t
        vel = self.dir_vec * self.speed
        return pos, vel


# ---------------------------------------------------------------------------
# Sensor geometry & noise
# ---------------------------------------------------------------------------

def compute_sensor_geometry(rel_vec: np.ndarray):
    """
    Compute range and angles from relative vector (car - drone).

    rel_vec: [x, y, z] in local (x=north, y=east, z=up)

    Returns:
        (R, beta, gamma)
        R     : range (m)
        beta  : elevation from nadir (rad). 0 = straight down, pi/2 = horizontal.
        gamma : azimuth around +Z (rad), 0 = north (+x), pi/2 = east (+y).
    """
    x, y, z = float(rel_vec[0]), float(rel_vec[1]), float(rel_vec[2])
    R = math.sqrt(x * x + y * y + z * z)
    if R < 1e-6:
        return None, None, None

    z_down = -z
    # ratio may have tiny numerical drift; clamp for acos
    ratio = clamp(z_down / R, -1.0, 1.0)
    beta = math.acos(ratio)          # angle from nadir
    gamma = math.atan2(y, x)         # azimuth around +Z

    return R, beta, gamma


def rel_vec_from_sensor(R: float, beta: float, gamma: float) -> np.ndarray:
    """
    Inverse of compute_sensor_geometry: recover relative vector from range & angles.
    """
    z_down = R * math.cos(beta)
    R_xy = R * math.sin(beta)

    x = R_xy * math.cos(gamma)
    y = R_xy * math.sin(gamma)
    z = -z_down  # back to z-up

    return np.array([x, y, z], dtype=float)


def apply_sensor_noise(R: float, beta: float, gamma: float):
    """
    Add Gaussian noise to R, beta, gamma. Clamp range to avoid non-physical values.
    """
    # Range noise
    R_noisy = R + np.random.normal(0.0, RANGE_NOISE_STD_M)
    # Avoid zero/negative ranges that would flip the vector direction
    R_noisy = max(0.01, R_noisy)

    # Angle noise
    beta_noisy = beta + np.random.normal(0.0, math.radians(BETA_NOISE_STD_DEG))
    gamma_noisy = gamma + np.random.normal(0.0, math.radians(GAMMA_NOISE_STD_DEG))

    return R_noisy, beta_noisy, gamma_noisy


# ---------------------------------------------------------------------------
# Speed estimator (regression over history window)
# ---------------------------------------------------------------------------

class PositionSpeedEstimator:
    """
    Estimate car speed along the road direction using a history of projected positions.

    For each sample we store (t, s) where s = dot(car_pos, dir_vec).
    We then do a least-squares linear fit s(t) ≈ a * t + b and use 'a' as speed.
    """
    def __init__(self, dir_vec: np.ndarray, window_size: int = WINDOW_SIZE):
        dir_vec = np.array(dir_vec, dtype=float)
        norm = float(np.linalg.norm(dir_vec))
        if norm < 1e-6:
            raise ValueError("Direction vector for speed estimator is zero")

        self.dir_vec = dir_vec / norm
        self.samples = deque(maxlen=window_size)

    def update(self, t: float, car_pos_vec: np.ndarray):
        s = float(np.dot(car_pos_vec, self.dir_vec))
        self.samples.append((t, s))
        return self.estimate_speed()

    def estimate_speed(self):
        if len(self.samples) < 2:
            return None

        ts = np.array([t for t, _ in self.samples], dtype=float)
        ss = np.array([s for _, s in self.samples], dtype=float)

        t0 = ts.mean()
        ts_centered = ts - t0
        denom = float((ts_centered ** 2).sum())
        if denom <= 0.0:
            return None

        slope = float((ts_centered * ss).sum() / denom)  # m/s along road
        return slope


# ---------------------------------------------------------------------------
# Drone heading / speed helper
# ---------------------------------------------------------------------------

def heading_and_speed_from_velocity(vel: np.ndarray):
    """
    Compute horizontal speed and heading from local velocity.

    vel: [vx, vy, vz] in local (x=north, y=east, z=up)

    Returns:
        heading_deg: 0° = north, 90° = east
        speed_mps: horizontal speed in m/s
    """
    vx = float(vel[0])
    vy = float(vel[1])
    speed = math.hypot(vx, vy)
    if speed < 1e-3:
        # If nearly stationary, keep heading defined but arbitrary
        heading_rad = 0.0
    else:
        heading_rad = math.atan2(vy, vx)
    heading_deg = (math.degrees(heading_rad) + 360.0) % 360.0
    return heading_deg, speed


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    # Connect to FC and QGC
    mav_client = MavlinkClient(MAVLINK_CONNECTION)
    mav_client.get_origin_gps()  # ensure we have origin for local_to_gps

    car_sim = CarSim(ROAD_ORIGIN_VEC, ROAD_HEADING_DEG, CAR_SPEED_MPS)
    estimator = PositionSpeedEstimator(car_sim.dir_vec, window_size=WINDOW_SIZE)

    qgc_sender = QGCCarSender(QGC_CONNECTION, sysid=CAR_SYSID, compid=CAR_COMPID)

    dt_target = 1.0 / LOOP_RATE_HZ
    t_start = time.time()

    print("[MAIN] Starting simulation loop...")
    print(f"[MAIN] Using {'sensor-based' if USE_SENSOR_BASED_ESTIMATE else 'ground-truth'} speed estimation")

    while True:
        loop_start = time.time()
        sim_t = loop_start - t_start

        # Get drone state
        drone_pos, drone_vel = mav_client.get_drone_state(timeout=1.0)
        drone_alt = float(drone_pos[2])
        drone_heading_deg, drone_speed_mps = heading_and_speed_from_velocity(drone_vel)

        # Simulated car state (truth)
        car_pos_true, car_vel_true = car_sim.get_state(sim_t)

        # Synthetic sensor from drone to car
        rel_true = car_pos_true - drone_pos
        R_true, beta_true, gamma_true = compute_sensor_geometry(rel_true)
        if R_true is None or R_true < MIN_SENSOR_RANGE_M:
            # Too close or undefined; skip this iteration
            # (estimator history still contains previous points)
            continue

        R_meas, beta_meas, gamma_meas = apply_sensor_noise(R_true, beta_true, gamma_true)
        if R_meas < MIN_SENSOR_RANGE_M:
            # Even after clamping/noise, treat as invalid
            continue

        rel_est = rel_vec_from_sensor(R_meas, beta_meas, gamma_meas)
        car_pos_est = drone_pos + rel_est

        # Choose what to feed into speed estimator
        if USE_SENSOR_BASED_ESTIMATE:
            pos_for_speed = car_pos_est
        else:
            pos_for_speed = car_pos_true

        speed_mps_est = estimator.update(sim_t, pos_for_speed)
        true_speed_mps = CAR_SPEED_MPS
        true_mph = mps_to_mph(true_speed_mps)

        est_mph_str = "N/A"
        over_limit = False
        speed_err_mph_str = "N/A"

        if speed_mps_est is not None:
            est_mph = mps_to_mph(speed_mps_est)
            est_mph_str = f"{est_mph:6.2f}"
            speed_err_mph = est_mph - true_mph
            speed_err_mph_str = f"{speed_err_mph:+6.2f}"
            over_limit = est_mph >= (SPEED_LIMIT_MPH + SPEED_LIMIT_MARGIN_MPH)

        # Console output
        print(
            f"[t={sim_t:6.2f}s] "
            f"Drone alt={drone_alt:6.1f} m, hdg={drone_heading_deg:6.1f}°, v={drone_speed_mps:5.2f} m/s | "
            f"Range true={R_true:6.1f} m | "
            f"Car true={true_mph:6.2f} mph, est={est_mph_str} mph, err={speed_err_mph_str} mph | "
            f"{'OVER LIMIT' if over_limit else ''}"
        )

        # Send car state to QGC (truth-based for clean visualization)
        now = loop_start
        car_lat, car_lon, car_alt = mav_client.local_to_gps(car_pos_true)
        car_heading_deg, _ = heading_and_speed_from_velocity(car_vel_true)
        qgc_sender.send_heartbeat(now)
        qgc_sender.send_state(now, car_lat, car_lon, car_alt, car_heading_deg, true_speed_mps)

        # Loop timing
        elapsed = time.time() - loop_start
        sleep_time = dt_target - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[MAIN] Exiting on user interrupt.")
