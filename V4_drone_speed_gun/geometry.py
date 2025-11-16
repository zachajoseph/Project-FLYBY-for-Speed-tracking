# V4_drone_speed_gun/geometry.py
import math
import numpy as np
from .config import (
    RANGE_NOISE_STD_M, BETA_NOISE_STD_DEG, GAMMA_NOISE_STD_DEG,
)

def unit_vector_from_heading_deg(heading_deg: float) -> np.ndarray:
    rad = math.radians(heading_deg)
    return np.array([math.cos(rad), math.sin(rad), 0.0], dtype=float)

def local_to_gps(pos_vec, origin_lat, origin_lon):
    x_north = float(pos_vec[0])
    y_east  = float(pos_vec[1])

    dlat = x_north / 111_320.0
    dlon = y_east / (111_320.0 * math.cos(math.radians(origin_lat)) + 1e-9)

    lat = origin_lat + dlat
    lon = origin_lon + dlon
    return lat, lon

def sensor_from_truth(drone_pos_vec, car_pos_vec):
    rel = car_pos_vec - drone_pos_vec
    dx, dy, dz = map(float, rel)

    R = math.sqrt(dx*dx + dy*dy + dz*dz)
    if R <= 1e-6:
        return 0.0, 0.0, 0.0

    cos_beta = max(-1.0, min(1.0, -dz / R))
    beta = math.acos(cos_beta)               # 0 = nadir, pi/2 = horiz
    gamma = math.atan2(dy, dx)              # azimuth in plane

    return R, beta, gamma

def apply_sensor_noise(R, beta, gamma):
    R_noisy = R + np.random.normal(0.0, RANGE_NOISE_STD_M)
    beta_noisy = beta + np.random.normal(
        0.0, math.radians(BETA_NOISE_STD_DEG)
    )
    gamma_noisy = gamma + np.random.normal(
        0.0, math.radians(GAMMA_NOISE_STD_DEG)
    )
    return R_noisy, beta_noisy, gamma_noisy

def rel_vec_from_sensor(R, beta, gamma):
    sinb, cosb = math.sin(beta), math.cos(beta)
    cosg, sing = math.cos(gamma), math.sin(gamma)

    dx = R * sinb * cosg
    dy = R * sinb * sing
    dz = -R * cosb

    return np.array([dx, dy, dz], dtype=float)
