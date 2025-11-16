# utils.py

import math
import numpy as np
import config


def unit_vector_from_heading_deg(heading_deg: float) -> np.ndarray:
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

def sensor_from_truth(drone_pos_vec: np.ndarray,
                      car_pos_vec: np.ndarray):
    """
    From true drone and car positions in local frame (x, y, z (up)),
    compute synthetic sensor readings:

    - R     : range [m]
    - beta  : elevation angle from straight down (nadir) [rad]
    - gamma : azimuth around vertical (Z) [rad]
    """
    rel = car_pos_vec - drone_pos_vec
    dx = float(rel[0])
    dy = float(rel[1])
    dz = float(rel[2])

    R = math.sqrt(dx * dx + dy * dy + dz * dz)
    if R <= 1e-6:
        return 0.0, 0.0, 0.0

    # Down vector = (0, 0, -1)
    # cos(beta) = (rel · down) / (|rel| * |down|) = (-dz) / R
    cos_beta = max(-1.0, min(1.0, -dz / R))
    beta = math.acos(cos_beta)   # 0 = straight down, pi/2 = horizontal

    # Azimuth in x–y plane
    gamma = math.atan2(dy, dx)   # [-pi, pi]

    return R, beta, gamma


def apply_sensor_noise(R: float, beta: float, gamma: float):
    """
    Add Gaussian noise to R, beta, gamma.
    """
    R_noisy = R + np.random.normal(0.0, config.RANGE_NOISE_STD_M)
    beta_noisy = beta + np.random.normal(
        0.0, math.radians(config.BETA_NOISE_STD_DEG)
    )
    gamma_noisy = gamma + np.random.normal(
        0.0, math.radians(config.GAMMA_NOISE_STD_DEG)
    )
    return R_noisy, beta_noisy, gamma_noisy


def rel_vec_from_sensor(R: float, beta: float, gamma: float) -> np.ndarray:
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
