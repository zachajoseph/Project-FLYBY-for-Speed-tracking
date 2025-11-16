# V4_drone_speed_gun/estimators.py
from collections import deque
import numpy as np
from .geometry import unit_vector_from_heading_deg

class PositionSpeedEstimator:
    def __init__(self, road_heading_deg, window_size=15):
        self.dir_vec = unit_vector_from_heading_deg(road_heading_deg)
        self.samples = deque(maxlen=window_size)  # (t, s_along_road)

    def update(self, t, car_pos_vec):
        s = float(np.dot(car_pos_vec, self.dir_vec))
        self.samples.append((t, s))

        if len(self.samples) < 2:
            return None

        (t1, s1), (t2, s2) = self.samples[-2], self.samples[-1]
        dt = t2 - t1
        if dt <= 0:
            return None

        return (s2 - s1) / dt  # m/s along road
