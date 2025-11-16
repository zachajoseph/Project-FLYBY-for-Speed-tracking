# utils.py
import math
import numpy as np

def unit_vector_from_heading_deg(heading_deg: float) -> np.ndarray:
    rad = math.radians(heading_deg)
    return np.array([math.cos(rad), math.sin(rad), 0.0], dtype=float)

def local_to_gps(pos_vec, origin_lat, origin_lon):
    x_north = float(pos_vec[0])
    y_east = float(pos_vec[1])

    dlat = x_north / 111_320.0
    dlon = y_east / (111_320.0 * math.cos(math.radians(origin_lat)) + 1e-9)

    lat = origin_lat + dlat
    lon = origin_lon + dlon
    return lat, lon
