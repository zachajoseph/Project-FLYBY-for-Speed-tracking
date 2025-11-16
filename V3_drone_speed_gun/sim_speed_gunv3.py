#!/usr/bin/env python3
import time
import math
from collections import deque

import numpy as np
from pymavlink import mavutil

# =========================
# CONFIG
# =========================

# MAVLink connection from Jetson to SITL/MAVProxy
MAVLINK_CONNECTION = "udp:127.0.0.1:14550"

# QGC laptop IP (where QGroundControl is running)
QGC_IP = "192.168.1.220"   # change if your laptop IP changes
QGC_PORT = 14550
CAR_SYSID = 42             # arbitrary system id for the car

# Road and car configuration
ROAD_HEADING_DEG = 0.0        # 0° = +X (north in NED)
CAR_SPEED_MPS = 20.0          # ~44.7 mph
SPEED_LIMIT_MPH = 35.0
OVER_LIMIT_MARGIN_MPH = 5.0   # how much over before we flag

# Smoothing / estimator settings
WINDOW_SIZE = 15              # history length
LOOP_HZ = 10.0                # main loop rate

# Sensor model (synthetic range + angles derived from truth)
RANGE_NOISE_STD_M = 0.5       # m
BETA_NOISE_STD_DEG = 0.5      # deg (elevation from nadir)
GAMMA_NOISE_STD_DEG = 1.0     # deg (azimuth)

USE_SENSOR_BASED_ESTIMATE = True  # True = use reconstructed car_pos_est, False = use car truth


# =========================
# UTILITIES
# =========================

def unit_vector_from_heading_deg(heading_deg):
    """
    Heading in degrees -> unit vector in 2D (x, y, 0).
    Here we treat heading=0 as +X, heading=90 as +Y.
    """
    rad = math.radians(heading_deg)
    return np.array([math.cos(rad), math.sin(rad), 0.0], dtype=float)


def local_to_gps(pos_vec, origin_lat, origin_lon):
    """
    Convert local NED (x=north, y=east) in meters to lat/lon.
    Simple flat-earth approximation, fine for small demo distances.
    """
    x_north = float(pos_vec[0])
    y_east = float(pos_vec[1])

    dlat = x_north / 111_320.0
    dlon = y_east / (111_320.0 * math.cos(math.radians(origin_lat)) + 1e-9)

    lat = origin_lat + dlat
    lon = origin_lon + dlon
    return lat, lon


# ---------- Geometry helpers for drone->car LOS ----------

def sensor_from_truth(drone_pos_vec, car_pos_vec):
    """
    From true drone and car positions in local frame (x, y, z (up)),
    compute synthetic sensor readings:

    - R     : range [m]
    - beta  : elevation angle from straight down (nadir) [rad]
    - gamma : azimuth around vertical (Z) [rad]

    Coordinate assumption:
      - x: north, y: east, z: up (same as MavlinkClient.get_drone_position_vec).
    """
    rel = car_pos_vec - drone_pos_vec
    dx = float(rel[0])
    dy = float(rel[1])
    dz = float(rel[2])

    R = math.sqrt(dx*dx + dy*dy + dz*dz)
    if R <= 1e-6:
        return 0.0, 0.0, 0.0

    # Down vector = (0, 0, -1)
    # cos(beta) = (rel · down) / (|rel| * |down|) = (-dz) / R
    cos_beta = max(-1.0, min(1.0, -dz / R))
    beta = math.acos(cos_beta)   # 0 = straight down, pi/2 = horizontal

    # Azimuth in x–y plane
    gamma = math.atan2(dy, dx)   # [-pi, pi]

    return R, beta, gamma


def apply_sensor_noise(R, beta, gamma):
    """
    Add Gaussian noise to R, beta, gamma.
    """
    R_noisy = R + np.random.normal(0.0, RANGE_NOISE_STD_M)
    beta_noisy = beta + np.random.normal(
        0.0, math.radians(BETA_NOISE_STD_DEG)
    )
    gamma_noisy = gamma + np.random.normal(
        0.0, math.radians(GAMMA_NOISE_STD_DEG)
    )
    return R_noisy, beta_noisy, gamma_noisy


def rel_vec_from_sensor(R, beta, gamma):
    """
    Reconstruct drone->car vector in the same frame as drone_pos (x,y,z up)
    from range and angles:

      beta: elevation from straight down (nadir)
      gamma: azimuth around +Z

      u = [sin(beta)*cos(gamma),
           sin(beta)*sin(gamma),
           -cos(beta)]
      rel_vec = R * u
    """
    sinb = math.sin(beta)
    cosb = math.cos(beta)
    cosg = math.cos(gamma)
    sing = math.sin(gamma)

    dx = R * sinb * cosg
    dy = R * sinb * sing
    dz = -R * cosb

    return np.array([dx, dy, dz], dtype=float)


# =========================
# MAVLINK DRONE CLIENT
# =========================

class MavlinkClient:
    """
    Connects to ArduPilot/PX4 over MAVLink and provides drone position
    as a 3D vector in LOCAL_POSITION_NED coordinates.
    Also fetches origin lat/lon once for local->GPS conversion.
    """
    def __init__(self, connection_str):
        print(f"[MAVLINK] Connecting to {connection_str} ...")
        self.master = mavutil.mavlink_connection(connection_str)
        self.master.wait_heartbeat()
        print("[MAVLINK] Heartbeat received. System:",
              self.master.target_system, "Component:", self.master.target_component)

        self.last_pos_vec = None
        self.origin_lat = None
        self.origin_lon = None

    def get_drone_position_vec(self):
        """
        Returns np.array([x, y, z]) in meters, from LOCAL_POSITION_NED.
        z is flipped to be "up" positive.
        """
        msg = self.master.recv_match(type="LOCAL_POSITION_NED",
                                     blocking=True, timeout=1.0)
        if msg is None:
            return self.last_pos_vec

        # LOCAL_POSITION_NED: x=north, y=east, z=down (m)
        self.last_pos_vec = np.array([
            float(msg.x),
            float(msg.y),
            -float(msg.z)
        ], dtype=float)

        return self.last_pos_vec

    def get_origin_latlon(self):
        """
        Get the EKF origin lat/lon once using GLOBAL_POSITION_INT.
        """
        if self.origin_lat is not None and self.origin_lon is not None:
            return self.origin_lat, self.origin_lon

        print("[MAVLINK] Waiting for GLOBAL_POSITION_INT for origin lat/lon...")
        msg = self.master.recv_match(type="GLOBAL_POSITION_INT",
                                     blocking=True, timeout=10.0)
        if msg is None:
            raise RuntimeError("No GLOBAL_POSITION_INT received to set origin")

        self.origin_lat = msg.lat * 1e-7
        self.origin_lon = msg.lon * 1e-7

        print(f"[MAVLINK] Origin lat/lon: {self.origin_lat:.7f}, {self.origin_lon:.7f}")
        return self.origin_lat, self.origin_lon


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
                 road_origin_vec,
                 road_heading_deg,
                 speed_mps):
        self.origin = road_origin_vec
        self.dir_vec = unit_vector_from_heading_deg(road_heading_deg)
        self.speed_mps = speed_mps
        self.t0 = time.time()

    def get_state(self):
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
    - Uses finite difference on the last two samples.
    """
    def __init__(self, road_heading_deg, window_size=15):
        self.dir_vec = unit_vector_from_heading_deg(road_heading_deg)
        self.samples = deque(maxlen=window_size)  # (t, s_along_road_m)

    def update(self, t, car_pos_vec):
        """
        Add a new (t, car_pos_vec) sample and return estimated
        road speed in m/s, or None if not enough samples yet.
        """
        # Project car position onto road direction (scalar)
        s = float(np.dot(car_pos_vec, self.dir_vec))
        self.samples.append((t, s))

        if len(self.samples) < 2:
            return None

        (t1, s1), (t2, s2) = self.samples[-2], self.samples[-1]
        dt = t2 - t1
        if dt <= 0:
            return None

        v_road_mps = (s2 - s1) / dt
        return v_road_mps


# =========================
# MAIN LOOP
# =========================

def main():
    # Connect to MAVLink (SITL / real FC)
    mav = MavlinkClient(MAVLINK_CONNECTION)

    # Get origin lat/lon so we can put the car on the map
    origin_lat, origin_lon = mav.get_origin_latlon()
    print(f"[SIM] Using origin lat/lon {origin_lat:.7f}, {origin_lon:.7f}")

    # Create a MAVLink connection from Jetson directly to QGC for the car
    car_link = mavutil.mavlink_connection(
        f"udpout:{QGC_IP}:{QGC_PORT}",
        source_system=CAR_SYSID,
        source_component=1,
    )
    print(f"[CAR] Sending car MAVLink to {QGC_IP}:{QGC_PORT} as sysid {CAR_SYSID}")

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

    prev_drone_pos = None
    prev_drone_t = None

    t_start = time.time()
    last_car_hb = 0.0
    car_heading_cdeg = int((ROAD_HEADING_DEG % 360.0) * 100)

    try:
        while True:
            t_loop_start = time.time()

            # 1) Get drone position vector from MAVLink
            drone_pos = mav.get_drone_position_vec()
            if drone_pos is None:
                print("[WARN] No drone position yet...")
                time.sleep(0.1)
                continue

            drone_alt = float(drone_pos[2])  # z (up) in meters

            # 2) Get car position vector from simulator (truth)
            car_state = car.get_state()
            t_car = car_state["t"]
            car_pos_true = car_state["pos_vec"]

            # Use car timestamp as our "now" for motion estimates
            t_now = t_car

            # Estimate drone horizontal velocity and heading
            drone_heading_deg = None
            drone_speed_mps = 0.0

            if prev_drone_pos is not None and prev_drone_t is not None:
                dt = t_now - prev_drone_t
                if dt > 0:
                    dpos = drone_pos - prev_drone_pos
                    vx, vy = dpos[0] / dt, dpos[1] / dt   # x=north, y=east
                    drone_speed_mps = math.sqrt(vx * vx + vy * vy)
                    heading_rad = math.atan2(vy, vx)      # atan2(east, north)
                    drone_heading_deg = (math.degrees(heading_rad) + 360.0) % 360.0

            prev_drone_pos = drone_pos.copy()
            prev_drone_t = t_now

            # 3) Synthetic sensor: derive R, beta, gamma from truth, then add noise
            R_true, beta_true, gamma_true = sensor_from_truth(drone_pos, car_pos_true)
            R_meas, beta_meas, gamma_meas = apply_sensor_noise(R_true, beta_true, gamma_true)

            # Reconstruct car position estimate from "sensor" + drone pose
            rel_est = rel_vec_from_sensor(R_meas, beta_meas, gamma_meas)
            car_pos_est = drone_pos + rel_est

            # For debug/info: line-of-sight range (truth)
            delta_true = car_pos_true - drone_pos
            range_true_m = float(np.linalg.norm(delta_true))

            # 4) Estimate car speed along the road FROM ESTIMATED POSITIONS
            if USE_SENSOR_BASED_ESTIMATE:
                v_road_mps = speed_estimator.update(t_car, car_pos_est)
            else:
                # fallback: use ground-truth car position
                v_road_mps = speed_estimator.update(t_car, car_pos_true)

            if v_road_mps is not None:
                est_mph = v_road_mps * 2.23694
                true_mph = CAR_SPEED_MPS * 2.23694
                err_mph = est_mph - true_mph
                over = est_mph > (SPEED_LIMIT_MPH + OVER_LIMIT_MARGIN_MPH)

                heading_str = "n/a"
                if drone_heading_deg is not None:
                    heading_str = f"{drone_heading_deg:5.1f}°"

                print(
                    f"[SpeedGun] "
                    f"Drone (alt {drone_alt:5.1f} m, heading {heading_str}, {drone_speed_mps:4.1f} m/s)   "
                    f"Range ~{R_meas:6.1f} m (true {range_true_m:6.1f} m)   "
                    f"Car est {est_mph:5.1f} mph   "
                    f"true {true_mph:5.1f} mph   "
                    f"err {err_mph:5.1f} mph   "
                    f"OVER={over}"
                )

            # 5) Send car as a fake MAVLink ground vehicle to QGC
            #    a) heartbeat once per second
            if t_now - last_car_hb > 1.0:
                car_link.mav.heartbeat_send(
                    mavutil.mavlink.MAV_TYPE_GROUND_ROVER,
                    mavutil.mavlink.MAV_AUTOPILOT_GENERIC,
                    0, 0,
                    mavutil.mavlink.MAV_STATE_ACTIVE,
                )
                last_car_hb = t_now

            #    b) position as GLOBAL_POSITION_INT (use TRUE car position for visualization)
            car_lat, car_lon = local_to_gps(car_pos_true, origin_lat, origin_lon)

            car_link.mav.global_position_int_send(
                int((t_now - t_start) * 1000),   # time_boot_ms
                int(car_lat * 1e7),
                int(car_lon * 1e7),
                0,          # alt (mm)
                0,          # relative alt (mm)
                0, 0, 0,    # vx, vy, vz (cm/s)
                car_heading_cdeg,
            )

            #    c) VFR_HUD so QGC can show car speed (truth)
            car_link.mav.vfr_hud_send(
                float(CAR_SPEED_MPS),  # airspeed
                float(CAR_SPEED_MPS),  # groundspeed
                int(ROAD_HEADING_DEG),
                0,                     # throttle
                0.0,                   # alt
                0.0,                   # climb
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
